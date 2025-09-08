#pragma once

#include "gcopter/banded_systems.hpp"
#include <Eigen/Eigen>

#include <cmath>
#include <vector>


namespace minco
{


    /**
     * @brief MINCO轨迹优化类 - S=3阶（最小化加加速度），非均匀时间分配
     * 
     * MINCO_S3NU类实现了针对S=3的最小控制轨迹优化算法。
     * 与S=2不同，S=3优化的是加加速度（jerk）的平方积分，
     * 生成更平滑的轨迹，特别适用于对舒适性要求较高的应用。
     * 
     * 算法特点：
     * 1. 轨迹表示：5次多项式，6个系数 (c0,c1,c2,c3,c4,c5)
     * 2. 优化目标：最小化 ∫||j(t)||²dt，其中j(t)为加加速度
     * 3. 约束条件：位置、速度、加速度连续性 + 边界条件
     * 4. 求解方法：带状线性系统求解 + 梯度传播
     * 
     * 边界条件：起始和终止的位置、速度、加速度 (PVA)
     * 连续性：相邻段间的位置、速度、加速度连续
     * 系统规模：6N×6N带状矩阵，带宽为6
     */
    class MINCO_S3NU
    {
    public:
        MINCO_S3NU() = default;                 // 默认构造函数
        ~MINCO_S3NU() { A.destroy(); }         // 析构函数，释放带状矩阵资源

    private:
        int N;                                  // 轨迹分段数量
        Eigen::Matrix3d headPVA;               // 起始状态 [位置; 速度; 加速度] (3×3)
        Eigen::Matrix3d tailPVA;               // 终止状态 [位置; 速度; 加速度] (3×3)
        BandedSystem A;                        // 约束矩阵 (6N×6N，带宽6)
        Eigen::MatrixX3d b;                    // 多项式系数矩阵 (6N×3)
        Eigen::VectorXd T1;                    // 时间向量 Ti
        Eigen::VectorXd T2;                    // 时间平方向量 Ti²
        Eigen::VectorXd T3;                    // 时间立方向量 Ti³
        Eigen::VectorXd T4;                    // 时间四次方向量 Ti⁴
        Eigen::VectorXd T5;                    // 时间五次方向量 Ti⁵

    public:
        /**
         * @brief 设置S=3轨迹边界条件和段数
         * @param headState 起始状态，包含位置、速度、加速度 [位置; 速度; 加速度] (3×3)
         * @param tailState 终止状态，包含位置、速度、加速度 [位置; 速度; 加速度] (3×3)
         * @param pieceNum 轨迹分段数量
         * 
         * 初始化MINCO_S3NU求解器，设置边界条件和矩阵维度。
         * 对于S=3的最小控制轨迹（最小加加速度），需要确保位置、速度、加速度连续性。
         * 每段轨迹需要6个系数，因此总共6N个未知数需要6N个约束方程。
         * 
         * 五次多项式形式：p_i(t) = c0 + c1*t + c2*t² + c3*t³ + c4*t⁴ + c5*t⁵
         * 其中各系数物理意义：c0(位置), c1(速度), c2(加速度/2), c3(加加速度/6), c4, c5
         */
        inline void setConditions(const Eigen::Matrix3d &headState,
                                  const Eigen::Matrix3d &tailState,
                                  const int &pieceNum)
        {
            N = pieceNum;                        // 保存轨迹段数
            headPVA = headState;                 // 保存起始位置、速度、加速度
            tailPVA = tailState;                 // 保存终止位置、速度、加速度
            A.create(6 * N, 6, 6);              // 创建带状线性系统矩阵 (6N×6N，带宽6)
            b.resize(6 * N, 3);                 // 右端向量 (6N×3，对应x,y,z三个维度)
            T1.resize(N);                       // 时间向量
            T2.resize(N);                       // 时间平方向量
            T3.resize(N);                       // 时间立方向量
            T4.resize(N);                       // 时间四次方向量
            T5.resize(N);                       // 时间五次方向量
            return;
        }

        /**
         * @brief 设置S=3轨迹参数并求解多项式系数
         * @param inPs 中间路径点矩阵 (3×(N-1))，不包含起始和终止点
         * @param ts 各段轨迹的时间分配向量 (N×1)
         * 
         * 根据给定的路径点和时间分配，构建约束矩阵并求解多项式系数。
         * 对于S=3的最小加加速度轨迹，约束条件包括：
         * 1. 边界条件：起始和终止的位置、速度、加速度
         * 2. 连续性条件：相邻段间的位置、速度、加速度连续
         * 3. 高阶连续性：加加速度和snap连续（S=3特有）
         * 4. 路径点约束：中间点必须经过指定位置
         * 
         * 五次多项式形式：p_i(t) = c0 + c1*t + c2*t² + c3*t³ + c4*t⁴ + c5*t⁵
         * 导数关系：
         * - 速度：v_i(t) = c1 + 2*c2*t + 3*c3*t² + 4*c4*t³ + 5*c5*t⁴
         * - 加速度：a_i(t) = 2*c2 + 6*c3*t + 12*c4*t² + 20*c5*t³
         * - 加加速度：j_i(t) = 6*c3 + 24*c4*t + 60*c5*t²
         * - Snap：s_i(t) = 24*c4 + 120*c5*t
         */
        inline void setParameters(const Eigen::Matrix3Xd &inPs,
                                  const Eigen::VectorXd &ts)
        {
            // 预计算时间的幂次方，用于构建约束矩阵
            T1 = ts;                            // t
            T2 = T1.cwiseProduct(T1);          // t²
            T3 = T2.cwiseProduct(T1);          // t³
            T4 = T2.cwiseProduct(T2);          // t⁴
            T5 = T4.cwiseProduct(T1);          // t⁵

            A.reset();                          // 重置约束矩阵为零
            b.setZero();                        // 重置右端向量为零

            // 设置起始边界条件约束
            // 第0段在t=0时的位置: c0 = headPVA.pos
            A(0, 0) = 1.0;
            // 第0段在t=0时的速度: c1 = headPVA.vel  
            A(1, 1) = 1.0;
            // 第0段在t=0时的加速度: 2*c2 = headPVA.acc
            A(2, 2) = 2.0;
            b.row(0) = headPVA.col(0).transpose();    // 起始位置
            b.row(1) = headPVA.col(1).transpose();    // 起始速度
            b.row(2) = headPVA.col(2).transpose();    // 起始加速度

            // 设置中间段连续性约束和路径点约束
            for (int i = 0; i < N - 1; i++)
            {
                // 加加速度连续性约束：段i终点加加速度 = 段i+1起点加加速度
                // j_i(Ti) = 6*c3 + 24*c4*Ti + 60*c5*Ti² = j_{i+1}(0) = 6*c_{i+1,3}
                A(6 * i + 3, 6 * i + 3) = 6.0;                      // 段i的加加速度系数
                A(6 * i + 3, 6 * i + 4) = 24.0 * T1(i);            // 段i的snap系数×时间
                A(6 * i + 3, 6 * i + 5) = 60.0 * T2(i);            // 段i的高阶系数×时间²
                A(6 * i + 3, 6 * i + 9) = -6.0;                     // 段i+1的加加速度系数

                // Snap连续性约束：段i终点snap = 段i+1起点snap
                // s_i(Ti) = 24*c4 + 120*c5*Ti = s_{i+1}(0) = 24*c_{i+1,4}
                A(6 * i + 4, 6 * i + 4) = 24.0;                     // 段i的snap系数
                A(6 * i + 4, 6 * i + 5) = 120.0 * T1(i);           // 段i的高阶系数×时间
                A(6 * i + 4, 6 * i + 10) = -24.0;                   // 段i+1的snap系数

                // 位置连续性约束：段i终点位置 = 段i+1起点位置 = 中间路径点
                A(6 * i + 5, 6 * i) = 1.0;                          // 位置系数
                A(6 * i + 5, 6 * i + 1) = T1(i);                   // 速度系数×时间
                A(6 * i + 5, 6 * i + 2) = T2(i);                   // 加速度系数×时间²
                A(6 * i + 5, 6 * i + 3) = T3(i);                   // 加加速度系数×时间³
                A(6 * i + 5, 6 * i + 4) = T4(i);                   // snap系数×时间⁴
                A(6 * i + 5, 6 * i + 5) = T5(i);                   // 高阶系数×时间⁵

                // 段i+1起点位置约束：c_{i+1,0} = inPs[i]
                A(6 * i + 6, 6 * i) = 1.0;
                A(6 * i + 6, 6 * i + 1) = T1(i);
                A(6 * i + 6, 6 * i + 2) = T2(i);
                A(6 * i + 6, 6 * i + 3) = T3(i);
                A(6 * i + 6, 6 * i + 4) = T4(i);
                A(6 * i + 6, 6 * i + 5) = T5(i);
                A(6 * i + 6, 6 * i + 6) = -1.0;

                // 速度连续性约束：段i终点速度 = 段i+1起点速度
                A(6 * i + 7, 6 * i + 1) = 1.0;                     // 段i的速度系数
                A(6 * i + 7, 6 * i + 2) = 2 * T1(i);              // 段i的加速度系数×时间
                A(6 * i + 7, 6 * i + 3) = 3 * T2(i);              // 段i的加加速度系数×时间²
                A(6 * i + 7, 6 * i + 4) = 4 * T3(i);              // 段i的snap系数×时间³
                A(6 * i + 7, 6 * i + 5) = 5 * T4(i);              // 段i的高阶系数×时间⁴
                A(6 * i + 7, 6 * i + 7) = -1.0;                    // 段i+1的速度系数

                // 加速度连续性约束：段i终点加速度 = 段i+1起点加速度
                A(6 * i + 8, 6 * i + 2) = 2.0;                     // 段i的加速度系数
                A(6 * i + 8, 6 * i + 3) = 6 * T1(i);              // 段i的加加速度系数×时间
                A(6 * i + 8, 6 * i + 4) = 12 * T2(i);             // 段i的snap系数×时间²
                A(6 * i + 8, 6 * i + 5) = 20 * T3(i);             // 段i的高阶系数×时间³
                A(6 * i + 8, 6 * i + 8) = -2.0;                    // 段i+1的加速度系数

                // 设置中间路径点约束的右端向量
                b.row(6 * i + 5) = inPs.col(i).transpose();
            }

            // 设置终止边界条件约束
            // 最后一段在t=T_{N-1}时的位置
            A(6 * N - 3, 6 * N - 6) = 1.0;                         // 位置系数
            A(6 * N - 3, 6 * N - 5) = T1(N - 1);                  // 速度系数×时间
            A(6 * N - 3, 6 * N - 4) = T2(N - 1);                  // 加速度系数×时间²
            A(6 * N - 3, 6 * N - 3) = T3(N - 1);                  // 加加速度系数×时间³
            A(6 * N - 3, 6 * N - 2) = T4(N - 1);                  // snap系数×时间⁴
            A(6 * N - 3, 6 * N - 1) = T5(N - 1);                  // 高阶系数×时间⁵

            // 最后一段在t=T_{N-1}时的速度
            A(6 * N - 2, 6 * N - 5) = 1.0;                         // 速度系数
            A(6 * N - 2, 6 * N - 4) = 2 * T1(N - 1);              // 加速度系数×时间
            A(6 * N - 2, 6 * N - 3) = 3 * T2(N - 1);              // 加加速度系数×时间²
            A(6 * N - 2, 6 * N - 2) = 4 * T3(N - 1);              // snap系数×时间³
            A(6 * N - 2, 6 * N - 1) = 5 * T4(N - 1);              // 高阶系数×时间⁴

            // 最后一段在t=T_{N-1}时的加速度
            A(6 * N - 1, 6 * N - 4) = 2;                           // 加速度系数
            A(6 * N - 1, 6 * N - 3) = 6 * T1(N - 1);              // 加加速度系数×时间
            A(6 * N - 1, 6 * N - 2) = 12 * T2(N - 1);             // snap系数×时间²
            A(6 * N - 1, 6 * N - 1) = 20 * T3(N - 1);             // 高阶系数×时间³

            b.row(6 * N - 3) = tailPVA.col(0).transpose();         // 终止位置
            b.row(6 * N - 2) = tailPVA.col(1).transpose();         // 终止速度
            b.row(6 * N - 1) = tailPVA.col(2).transpose();         // 终止加速度

            // 求解线性方程组 A*coeffs = b
            A.factorizeLU();                                        // LU分解
            A.solve(b);                                             // 求解系数矩阵

            return;
        }

        /**
         * @brief 构建S=3轨迹对象
         * @param traj 输出的轨迹对象，包含所有分段五次多项式
         * 
         * 将求解得到的多项式系数转换为Trajectory对象。
         * 每段轨迹由时间长度和6个系数（位置、速度、加速度、加加速度、snap、高阶）确定。
         * 
         * 注意：系数需要按照从高次到低次的顺序排列（c5, c4, c3, c2, c1, c0）
         * 这与Piece类的内部存储格式一致，支持五次多项式的快速评估。
         */
        inline void getTrajectory(Trajectory<5> &traj) const
        {
            traj.clear();                       // 清空现有轨迹
            traj.reserve(N);                    // 预分配内存空间
            for (int i = 0; i < N; i++)
            {
                // 为每段轨迹创建Piece<5>对象（五次多项式）
                // 时间长度：T1(i)
                // 系数矩阵：b的第6i到6i+5行，按列转置后行倒序（高次到低次）
                traj.emplace_back(T1(i),
                                  b.block<6, 3>(6 * i, 0)    // 提取6×3系数块
                                      .transpose()             // 转置为3×6 (x,y,z × c0,c1,c2,c3,c4,c5)
                                      .rowwise()               // 按行操作
                                      .reverse());             // 倒序为 (x,y,z × c5,c4,c3,c2,c1,c0)
            }
            return;
        }

        /**
         * @brief 计算S=3轨迹的总能量（加加速度平方积分）
         * @param energy 输出的总能量值
         * 
         * 对于S=3的MINCO轨迹，能量定义为加加速度的平方在时间上的积分：
         * E = ∫[0,T] ||j(t)||² dt
         * 
         * 对于五次多项式 p(t) = c0 + c1*t + c2*t² + c3*t³ + c4*t⁴ + c5*t⁵
         * 加加速度为 j(t) = 6*c3 + 24*c4*t + 60*c5*t²
         * 
         * 能量积分为：
         * E_i = ∫[0,Ti] (6*c3 + 24*c4*t + 60*c5*t²)² dt
         *     = ∫[0,Ti] (36*c3² + 288*c3*c4*t + 720*c3*c5*t² + 576*c4²*t² + 2880*c4*c5*t³ + 3600*c5²*t⁴) dt
         *     = 36*c3²*Ti + 144*c3*c4*Ti² + 240*c3*c5*Ti³ + 192*c4²*Ti³ + 720*c4*c5*Ti⁴ + 720*c5²*Ti⁵
         * 
         * 其中 c3 对应 b.row(6*i+3)，c4 对应 b.row(6*i+4)，c5 对应 b.row(6*i+5)
         */
        inline void getEnergy(double &energy) const
        {
            energy = 0.0;
            for (int i = 0; i < N; i++)
            {
                // 计算第i段的能量贡献
                energy += 36.0 * b.row(6 * i + 3).squaredNorm() * T1(i) +      // 36*||c3||²*Ti
                          144.0 * b.row(6 * i + 4).dot(b.row(6 * i + 3)) * T2(i) +  // 144*c3·c4*Ti²
                          192.0 * b.row(6 * i + 4).squaredNorm() * T3(i) +     // 192*||c4||²*Ti³
                          240.0 * b.row(6 * i + 5).dot(b.row(6 * i + 3)) * T3(i) +  // 240*c3·c5*Ti³
                          720.0 * b.row(6 * i + 5).dot(b.row(6 * i + 4)) * T4(i) +  // 720*c4·c5*Ti⁴
                          720.0 * b.row(6 * i + 5).squaredNorm() * T5(i);           // 720*||c5||²*Ti⁵
            }
            return;
        }

        /**
         * @brief 获取S=3多项式系数矩阵
         * @return 系数矩阵的常量引用 (6N×3)
         * 
         * 返回所有轨迹段的五次多项式系数。矩阵结构为：
         * 第6i+0行：第i段的位置系数 c_{i0}
         * 第6i+1行：第i段的速度系数 c_{i1}  
         * 第6i+2行：第i段的加速度系数 c_{i2}
         * 第6i+3行：第i段的加加速度系数 c_{i3}
         * 第6i+4行：第i段的snap系数 c_{i4}
         * 第6i+5行：第i段的高阶系数 c_{i5}
         * 列对应x, y, z三个坐标轴
         */
        inline const Eigen::MatrixX3d &getCoeffs(void) const
        {
            return b;
        }

        /**
         * @brief 计算S=3能量对多项式系数的偏导数
         * @param gdC 输出的梯度矩阵 (6N×3)
         * 
         * 计算总能量E对各段多项式系数的偏导数：∂E/∂c_{ij}
         * 
         * 对于S=3能量函数：
         * E_i = 36*c3²*Ti + 144*c3*c4*Ti² + 192*c4²*Ti³ + 240*c3*c5*Ti³ + 720*c4*c5*Ti⁴ + 720*c5²*Ti⁵
         * 
         * 偏导数为：
         * ∂E_i/∂c3 = 72*c3*Ti + 144*c4*Ti² + 240*c5*Ti³
         * ∂E_i/∂c4 = 144*c3*Ti² + 384*c4*Ti³ + 720*c5*Ti⁴
         * ∂E_i/∂c5 = 240*c3*Ti³ + 720*c4*Ti⁴ + 1440*c5*Ti⁵
         * ∂E_i/∂c0 = ∂E_i/∂c1 = ∂E_i/∂c2 = 0 (位置、速度、加速度系数不影响加加速度)
         * 
         * 这些梯度用于基于梯度的轨迹优化算法。
         */
        inline void getEnergyPartialGradByCoeffs(Eigen::MatrixX3d &gdC) const
        {
            gdC.resize(6 * N, 3);
            for (int i = 0; i < N; i++)
            {
                // ∂E_i/∂c5：对高阶系数的梯度
                gdC.row(6 * i + 5) = 240.0 * b.row(6 * i + 3) * T3(i) +       // 240*c3*Ti³
                                     720.0 * b.row(6 * i + 4) * T4(i) +       // 720*c4*Ti⁴
                                     1440.0 * b.row(6 * i + 5) * T5(i);       // 1440*c5*Ti⁵
                
                // ∂E_i/∂c4：对snap系数的梯度
                gdC.row(6 * i + 4) = 144.0 * b.row(6 * i + 3) * T2(i) +       // 144*c3*Ti²
                                     384.0 * b.row(6 * i + 4) * T3(i) +       // 384*c4*Ti³
                                     720.0 * b.row(6 * i + 5) * T4(i);        // 720*c5*Ti⁴
                
                // ∂E_i/∂c3：对加加速度系数的梯度
                gdC.row(6 * i + 3) = 72.0 * b.row(6 * i + 3) * T1(i) +        // 72*c3*Ti
                                     144.0 * b.row(6 * i + 4) * T2(i) +       // 144*c4*Ti²
                                     240.0 * b.row(6 * i + 5) * T3(i);        // 240*c5*Ti³
                
                // ∂E_i/∂c0 = ∂E_i/∂c1 = ∂E_i/∂c2 = 0：位置、速度、加速度系数的梯度为零
                gdC.block<3, 3>(6 * i, 0).setZero();
            }
            return;
        }

        /**
         * @brief 计算S=3能量对时间分配的偏导数
         * @param gdT 输出的时间梯度向量 (N×1)
         * 
         * 计算总能量E对各段时间分配的偏导数：∂E/∂Ti
         * 
         * 对于S=3能量函数：
         * E_i = 36*c3²*Ti + 144*c3*c4*Ti² + 192*c4²*Ti³ + 240*c3*c5*Ti³ + 720*c4*c5*Ti⁴ + 720*c5²*Ti⁵
         * 
         * 对时间的偏导数为：
         * ∂E_i/∂Ti = 36*||c3||² + 288*c3·c4*Ti + 576*||c4||²*Ti² + 720*c3·c5*Ti² + 2880*c4·c5*Ti³ + 3600*||c5||²*Ti⁴
         * 
         * 这个梯度表示增加第i段时间对总能量的影响，
         * 用于时间分配优化算法中。
         */
        inline void getEnergyPartialGradByTimes(Eigen::VectorXd &gdT) const
        {
            gdT.resize(N);
            for (int i = 0; i < N; i++)
            {
                // ∂E_i/∂Ti的计算
                gdT(i) = 36.0 * b.row(6 * i + 3).squaredNorm() +                          // 36*||c3||²
                         288.0 * b.row(6 * i + 4).dot(b.row(6 * i + 3)) * T1(i) +        // 288*c3·c4*Ti
                         576.0 * b.row(6 * i + 4).squaredNorm() * T2(i) +                 // 576*||c4||²*Ti²
                         720.0 * b.row(6 * i + 5).dot(b.row(6 * i + 3)) * T2(i) +        // 720*c3·c5*Ti²
                         2880.0 * b.row(6 * i + 5).dot(b.row(6 * i + 4)) * T3(i) +       // 2880*c4·c5*Ti³
                         3600.0 * b.row(6 * i + 5).squaredNorm() * T4(i);                 // 3600*||c5||²*Ti⁴
            }
            return;
        }

        /**
         * @brief 传播S=3梯度到设计变量（路径点和时间分配）
         * @param partialGradByCoeffs 目标函数对系数的偏导数 (6N×3)
         * @param partialGradByTimes 目标函数对时间的偏导数 (N×1)
         * @param gradByPoints 输出：目标函数对路径点的梯度 (3×(N-1))
         * @param gradByTimes 输出：目标函数对时间分配的总梯度 (N×1)
         * 
         * 使用链式法则和伴随方法将目标函数的梯度从多项式系数
         * 传播回原始设计变量（中间路径点和时间分配）。
         * 
         * 对于S=3五次多项式轨迹，需要考虑更高阶的连续性约束：
         * - 位置、速度、加速度连续性
         * - 加加速度和snap连续性（S=3特有）
         * 
         * 梯度传播考虑了所有约束条件对时间变化的敏感性。
         */
        inline void propogateGrad(const Eigen::MatrixX3d &partialGradByCoeffs,
                                  const Eigen::VectorXd &partialGradByTimes,
                                  Eigen::Matrix3Xd &gradByPoints,
                                  Eigen::VectorXd &gradByTimes)

        {
            gradByPoints.resize(3, N - 1);      // 中间路径点的梯度
            gradByTimes.resize(N);              // 时间分配的梯度
            
            // 求解伴随方程：A^T * adjGrad = partialGradByCoeffs
            Eigen::MatrixX3d adjGrad = partialGradByCoeffs;
            A.solveAdj(adjGrad);

            // 提取对路径点的梯度
            // 对于S=3，路径点约束对应第6i+5行的伴随变量
            for (int i = 0; i < N - 1; i++)
            {
                gradByPoints.col(i) = adjGrad.row(6 * i + 5).transpose();
            }

            // 计算对时间分配的梯度
            // 时间变化会影响约束矩阵A中的所有时间相关项
            Eigen::Matrix<double, 6, 3> B1;     // 中间段的时间梯度贡献 (6个约束)
            Eigen::Matrix3d B2;                  // 最后一段的时间梯度贡献 (3个边界约束)
            
            for (int i = 0; i < N - 1; i++)
            {
                // 计算约束矩阵A对时间Ti的偏导数与伴随变量的乘积
                
                // 对速度连续性约束的贡献：∂(v_i(Ti))/∂Ti
                B1.row(2) = -(b.row(i * 6 + 1) +                      // ∂(c1)/∂Ti = 0
                              2.0 * T1(i) * b.row(i * 6 + 2) +        // ∂(c2*Ti²)/∂Ti = 2*c2*Ti
                              3.0 * T2(i) * b.row(i * 6 + 3) +        // ∂(c3*Ti³)/∂Ti = 3*c3*Ti²
                              4.0 * T3(i) * b.row(i * 6 + 4) +        // ∂(c4*Ti⁴)/∂Ti = 4*c4*Ti³
                              5.0 * T4(i) * b.row(i * 6 + 5));        // ∂(c5*Ti⁵)/∂Ti = 5*c5*Ti⁴
                B1.row(3) = B1.row(2);                                 // 重复约束

                // 对加速度连续性约束的贡献：∂(a_i(Ti))/∂Ti
                B1.row(4) = -(2.0 * b.row(i * 6 + 2) +                // ∂(2*c2)/∂Ti = 0
                              6.0 * T1(i) * b.row(i * 6 + 3) +        // ∂(6*c3*Ti)/∂Ti = 6*c3
                              12.0 * T2(i) * b.row(i * 6 + 4) +       // ∂(12*c4*Ti²)/∂Ti = 24*c4*Ti
                              20.0 * T3(i) * b.row(i * 6 + 5));       // ∂(20*c5*Ti³)/∂Ti = 60*c5*Ti²

                // 对加加速度连续性约束的贡献：∂(j_i(Ti))/∂Ti
                B1.row(5) = -(6.0 * b.row(i * 6 + 3) +                // ∂(6*c3)/∂Ti = 0
                              24.0 * T1(i) * b.row(i * 6 + 4) +       // ∂(24*c4*Ti)/∂Ti = 24*c4
                              60.0 * T2(i) * b.row(i * 6 + 5));       // ∂(60*c5*Ti²)/∂Ti = 120*c5*Ti

                // 对snap连续性约束的贡献：∂(s_i(Ti))/∂Ti
                B1.row(0) = -(24.0 * b.row(i * 6 + 4) +               // ∂(24*c4)/∂Ti = 0
                              120.0 * T1(i) * b.row(i * 6 + 5));      // ∂(120*c5*Ti)/∂Ti = 120*c5

                // 对crackle连续性约束的贡献：∂(cr_i(Ti))/∂Ti = 0
                B1.row(1) = -120.0 * b.row(i * 6 + 5);                // ∂(120*c5)/∂Ti = 0

                // 计算第i段时间的梯度贡献
                gradByTimes(i) = B1.cwiseProduct(adjGrad.block<6, 3>(6 * i + 3, 0)).sum();
            }

            // 处理最后一段的边界条件
            // 最后一段的终止条件也依赖于时间T_{N-1}
            
            // 对终止速度约束的贡献
            B2.row(0) = -(b.row(6 * N - 5) +                          // 速度系数
                          2.0 * T1(N - 1) * b.row(6 * N - 4) +        // 加速度×时间
                          3.0 * T2(N - 1) * b.row(6 * N - 3) +        // 加加速度×时间²
                          4.0 * T3(N - 1) * b.row(6 * N - 2) +        // snap×时间³
                          5.0 * T4(N - 1) * b.row(6 * N - 1));        // 高阶×时间⁴

            // 对终止加速度约束的贡献
            B2.row(1) = -(2.0 * b.row(6 * N - 4) +                    // 加速度系数×2
                          6.0 * T1(N - 1) * b.row(6 * N - 3) +        // 加加速度×时间×6
                          12.0 * T2(N - 1) * b.row(6 * N - 2) +       // snap×时间²×12
                          20.0 * T3(N - 1) * b.row(6 * N - 1));       // 高阶×时间³×20

            // 对终止加加速度约束的贡献
            B2.row(2) = -(6.0 * b.row(6 * N - 3) +                    // 加加速度系数×6
                          24.0 * T1(N - 1) * b.row(6 * N - 2) +       // snap×时间×24
                          60.0 * T2(N - 1) * b.row(6 * N - 1));       // 高阶×时间²×60

            // 计算最后一段时间的梯度贡献
            gradByTimes(N - 1) = B2.cwiseProduct(adjGrad.block<3, 3>(6 * N - 3, 0)).sum();

            // 加上直接对时间的偏导数（如能量函数对时间的直接依赖）
            gradByTimes += partialGradByTimes;
        }
    };
}