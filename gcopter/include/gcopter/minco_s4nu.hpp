#pragma once

#include "gcopter/banded_systems.hpp"

#include <Eigen/Eigen>

#include <cmath>
#include <vector>
namespace minco
{
    /**
     * @brief MINCO轨迹优化类 - S=4阶（最小化snap），非均匀时间分配
     * 
     * MINCO_S4NU类实现了针对S=4的最小控制轨迹优化算法。
     * S=4优化的是snap（加加速度的导数）的平方积分，
     * 生成极其平滑的轨迹，适用于对平滑性要求极高的精密应用。
     * 
     * 算法特点：
     * 1. 轨迹表示：7次多项式，8个系数 (c0,c1,c2,c3,c4,c5,c6,c7)
     * 2. 优化目标：最小化 ∫||s(t)||²dt，其中s(t)为snap
     * 3. 约束条件：位置、速度、加速度、加加速度连续性 + 边界条件
     * 4. 求解方法：带状线性系统求解 + 梯度传播
     * 
     * 边界条件：起始和终止的位置、速度、加速度、加加速度 (PVAJ)
     * 连续性：相邻段间的位置、速度、加速度、加加速度连续
     * 系统规模：8N×8N带状矩阵，带宽为8
     * 
     * 应用场景：需要极高平滑性的轨迹，如精密机械臂、医疗机器人等
     */
    class MINCO_S4NU
    {
    public:
        MINCO_S4NU() = default;                 // 默认构造函数
        ~MINCO_S4NU() { A.destroy(); }         // 析构函数，释放带状矩阵资源

    private:
        int N;                                  // 轨迹分段数量
        Eigen::Matrix<double, 3, 4> headPVAJ;  // 起始状态 [位置; 速度; 加速度; 加加速度] (3×4)
        Eigen::Matrix<double, 3, 4> tailPVAJ;  // 终止状态 [位置; 速度; 加速度; 加加速度] (3×4)
        BandedSystem A;                        // 约束矩阵 (8N×8N，带宽8)
        Eigen::MatrixX3d b;                    // 多项式系数矩阵 (8N×3)
        Eigen::VectorXd T1;                    // 时间向量 Ti
        Eigen::VectorXd T2;                    // 时间平方向量 Ti²
        Eigen::VectorXd T3;                    // 时间立方向量 Ti³
        Eigen::VectorXd T4;                    // 时间四次方向量 Ti⁴
        Eigen::VectorXd T5;                    // 时间五次方向量 Ti⁵
        Eigen::VectorXd T6;                    // 时间六次方向量 Ti⁶
        Eigen::VectorXd T7;                    // 时间七次方向量 Ti⁷

    public:
        /**
         * @brief 设置S=4轨迹边界条件和段数
         * @param headState 起始状态，包含位置、速度、加速度、加加速度 [位置; 速度; 加速度; 加加速度] (3×4)
         * @param tailState 终止状态，包含位置、速度、加速度、加加速度 [位置; 速度; 加速度; 加加速度] (3×4)
         * @param pieceNum 轨迹分段数量
         * 
         * 初始化MINCO_S4NU求解器，设置边界条件和矩阵维度。
         * 对于S=4的最小控制轨迹（最小snap），需要确保位置、速度、加速度、加加速度连续性。
         * 每段轨迹需要8个系数，因此总共8N个未知数需要8N个约束方程。
         * 
         * 七次多项式形式：p_i(t) = c0 + c1*t + c2*t² + c3*t³ + c4*t⁴ + c5*t⁵ + c6*t⁶ + c7*t⁷
         * 其中各系数物理意义：
         * - c0: 位置
         * - c1: 速度
         * - c2: 加速度/2
         * - c3: 加加速度/6
         * - c4: snap/24
         * - c5, c6, c7: 高阶项
         */
        inline void setConditions(const Eigen::Matrix<double, 3, 4> &headState,
                                  const Eigen::Matrix<double, 3, 4> &tailState,
                                  const int &pieceNum)
        {
            N = pieceNum;                        // 保存轨迹段数
            headPVAJ = headState;               // 保存起始位置、速度、加速度、加加速度
            tailPVAJ = tailState;               // 保存终止位置、速度、加速度、加加速度
            A.create(8 * N, 8, 8);             // 创建带状线性系统矩阵 (8N×8N，带宽8)
            b.resize(8 * N, 3);                // 右端向量 (8N×3，对应x,y,z三个维度)
            T1.resize(N);                      // 时间向量
            T2.resize(N);                      // 时间平方向量
            T3.resize(N);                      // 时间立方向量
            T4.resize(N);                      // 时间四次方向量
            T5.resize(N);                      // 时间五次方向量
            T6.resize(N);                      // 时间六次方向量
            T7.resize(N);                      // 时间七次方向量
            return;
        }

        /**
         * @brief 设置S=4轨迹参数并求解多项式系数
         * @param inPs 中间路径点矩阵 (3×(N-1))，不包含起始和终止点
         * @param ts 各段轨迹的时间分配向量 (N×1)
         * 
         * 根据给定的路径点和时间分配，构建约束矩阵并求解多项式系数。
         * 对于S=4的最小snap轨迹，约束条件包括：
         * 1. 边界条件：起始和终止的位置、速度、加速度、加加速度
         * 2. 连续性条件：相邻段间的位置、速度、加速度、加加速度连续
         * 3. 高阶连续性：snap、crackle、pop连续（S=4特有）
         * 4. 路径点约束：中间点必须经过指定位置
         * 
         * 七次多项式形式：p_i(t) = c0 + c1*t + c2*t² + c3*t³ + c4*t⁴ + c5*t⁵ + c6*t⁶ + c7*t⁷
         * 导数关系：
         * - 速度：v_i(t) = c1 + 2*c2*t + 3*c3*t² + 4*c4*t³ + 5*c5*t⁴ + 6*c6*t⁵ + 7*c7*t⁶
         * - 加速度：a_i(t) = 2*c2 + 6*c3*t + 12*c4*t² + 20*c5*t³ + 30*c6*t⁴ + 42*c7*t⁵
         * - 加加速度：j_i(t) = 6*c3 + 24*c4*t + 60*c5*t² + 120*c6*t³ + 210*c7*t⁴
         * - Snap：s_i(t) = 24*c4 + 120*c5*t + 360*c6*t² + 840*c7*t³
         * - Crackle：cr_i(t) = 120*c5 + 720*c6*t + 2520*c7*t²
         * - Pop：p_i(t) = 720*c6 + 5040*c7*t
         */
        inline void setParameters(const Eigen::MatrixXd &inPs,
                                  const Eigen::VectorXd &ts)
        {
            // 预计算时间的幂次方，用于构建约束矩阵
            T1 = ts;                            // t
            T2 = T1.cwiseProduct(T1);          // t²
            T3 = T2.cwiseProduct(T1);          // t³
            T4 = T2.cwiseProduct(T2);          // t⁴
            T5 = T4.cwiseProduct(T1);          // t⁵
            T6 = T4.cwiseProduct(T2);          // t⁶
            T7 = T4.cwiseProduct(T3);          // t⁷

            A.reset();                          // 重置约束矩阵为零
            b.setZero();                        // 重置右端向量为零

            // 设置起始边界条件约束
            // 第0段在t=0时的位置: c0 = headPVAJ.pos
            A(0, 0) = 1.0;
            // 第0段在t=0时的速度: c1 = headPVAJ.vel
            A(1, 1) = 1.0;
            // 第0段在t=0时的加速度: 2*c2 = headPVAJ.acc
            A(2, 2) = 2.0;
            // 第0段在t=0时的加加速度: 6*c3 = headPVAJ.jerk
            A(3, 3) = 6.0;
            b.row(0) = headPVAJ.col(0).transpose();    // 起始位置
            b.row(1) = headPVAJ.col(1).transpose();    // 起始速度
            b.row(2) = headPVAJ.col(2).transpose();    // 起始加速度
            b.row(3) = headPVAJ.col(3).transpose();    // 起始加加速度

            // 设置中间段连续性约束和路径点约束
            for (int i = 0; i < N - 1; i++)
            {
                // Snap连续性约束：段i终点snap = 段i+1起点snap
                // s_i(Ti) = 24*c4 + 120*c5*Ti + 360*c6*Ti² + 840*c7*Ti³ = s_{i+1}(0) = 24*c_{i+1,4}
                A(8 * i + 4, 8 * i + 4) = 24.0;                      // 段i的snap系数
                A(8 * i + 4, 8 * i + 5) = 120.0 * T1(i);            // 段i的crackle系数×时间
                A(8 * i + 4, 8 * i + 6) = 360.0 * T2(i);            // 段i的pop系数×时间²
                A(8 * i + 4, 8 * i + 7) = 840.0 * T3(i);            // 段i的高阶系数×时间³
                A(8 * i + 4, 8 * i + 12) = -24.0;                    // 段i+1的snap系数

                // Crackle连续性约束：段i终点crackle = 段i+1起点crackle
                // cr_i(Ti) = 120*c5 + 720*c6*Ti + 2520*c7*Ti² = cr_{i+1}(0) = 120*c_{i+1,5}
                A(8 * i + 5, 8 * i + 5) = 120.0;                     // 段i的crackle系数
                A(8 * i + 5, 8 * i + 6) = 720.0 * T1(i);            // 段i的pop系数×时间
                A(8 * i + 5, 8 * i + 7) = 2520.0 * T2(i);           // 段i的高阶系数×时间²
                A(8 * i + 5, 8 * i + 13) = -120.0;                   // 段i+1的crackle系数

                // Pop连续性约束：段i终点pop = 段i+1起点pop
                // p_i(Ti) = 720*c6 + 5040*c7*Ti = p_{i+1}(0) = 720*c_{i+1,6}
                A(8 * i + 6, 8 * i + 6) = 720.0;                     // 段i的pop系数
                A(8 * i + 6, 8 * i + 7) = 5040.0 * T1(i);           // 段i的高阶系数×时间
                A(8 * i + 6, 8 * i + 14) = -720.0;                   // 段i+1的pop系数

                // 位置连续性约束：段i终点位置 = 段i+1起点位置 = 中间路径点
                A(8 * i + 7, 8 * i) = 1.0;                          // 位置系数
                A(8 * i + 7, 8 * i + 1) = T1(i);                   // 速度系数×时间
                A(8 * i + 7, 8 * i + 2) = T2(i);                   // 加速度系数×时间²
                A(8 * i + 7, 8 * i + 3) = T3(i);                   // 加加速度系数×时间³
                A(8 * i + 7, 8 * i + 4) = T4(i);                   // snap系数×时间⁴
                A(8 * i + 7, 8 * i + 5) = T5(i);                   // crackle系数×时间⁵
                A(8 * i + 7, 8 * i + 6) = T6(i);                   // pop系数×时间⁶
                A(8 * i + 7, 8 * i + 7) = T7(i);                   // 高阶系数×时间⁷

                // 段i+1起点位置约束：c_{i+1,0} = inPs[i]
                A(8 * i + 8, 8 * i) = 1.0;
                A(8 * i + 8, 8 * i + 1) = T1(i);
                A(8 * i + 8, 8 * i + 2) = T2(i);
                A(8 * i + 8, 8 * i + 3) = T3(i);
                A(8 * i + 8, 8 * i + 4) = T4(i);
                A(8 * i + 8, 8 * i + 5) = T5(i);
                A(8 * i + 8, 8 * i + 6) = T6(i);
                A(8 * i + 8, 8 * i + 7) = T7(i);
                A(8 * i + 8, 8 * i + 8) = -1.0;

                // 速度连续性约束：段i终点速度 = 段i+1起点速度
                A(8 * i + 9, 8 * i + 1) = 1.0;                     // 段i的速度系数
                A(8 * i + 9, 8 * i + 2) = 2.0 * T1(i);            // 段i的加速度系数×时间
                A(8 * i + 9, 8 * i + 3) = 3.0 * T2(i);            // 段i的加加速度系数×时间²
                A(8 * i + 9, 8 * i + 4) = 4.0 * T3(i);            // 段i的snap系数×时间³
                A(8 * i + 9, 8 * i + 5) = 5.0 * T4(i);            // 段i的crackle系数×时间⁴
                A(8 * i + 9, 8 * i + 6) = 6.0 * T5(i);            // 段i的pop系数×时间⁵
                A(8 * i + 9, 8 * i + 7) = 7.0 * T6(i);            // 段i的高阶系数×时间⁶
                A(8 * i + 9, 8 * i + 9) = -1.0;                    // 段i+1的速度系数

                // 加速度连续性约束：段i终点加速度 = 段i+1起点加速度
                A(8 * i + 10, 8 * i + 2) = 2.0;                    // 段i的加速度系数
                A(8 * i + 10, 8 * i + 3) = 6.0 * T1(i);           // 段i的加加速度系数×时间
                A(8 * i + 10, 8 * i + 4) = 12.0 * T2(i);          // 段i的snap系数×时间²
                A(8 * i + 10, 8 * i + 5) = 20.0 * T3(i);          // 段i的crackle系数×时间³
                A(8 * i + 10, 8 * i + 6) = 30.0 * T4(i);          // 段i的pop系数×时间⁴
                A(8 * i + 10, 8 * i + 7) = 42.0 * T5(i);          // 段i的高阶系数×时间⁵
                A(8 * i + 10, 8 * i + 10) = -2.0;                  // 段i+1的加速度系数

                // 加加速度连续性约束：段i终点加加速度 = 段i+1起点加加速度
                A(8 * i + 11, 8 * i + 3) = 6.0;                    // 段i的加加速度系数
                A(8 * i + 11, 8 * i + 4) = 24.0 * T1(i);          // 段i的snap系数×时间
                A(8 * i + 11, 8 * i + 5) = 60.0 * T2(i);          // 段i的crackle系数×时间²
                A(8 * i + 11, 8 * i + 6) = 120.0 * T3(i);         // 段i的pop系数×时间³
                A(8 * i + 11, 8 * i + 7) = 210.0 * T4(i);         // 段i的高阶系数×时间⁴
                A(8 * i + 11, 8 * i + 11) = -6.0;                  // 段i+1的加加速度系数

                // 设置中间路径点约束的右端向量
                b.row(8 * i + 7) = inPs.col(i).transpose();
            }

            // 设置终止边界条件约束
            // 最后一段在t=T_{N-1}时的位置
            A(8 * N - 4, 8 * N - 8) = 1.0;                         // 位置系数
            A(8 * N - 4, 8 * N - 7) = T1(N - 1);                  // 速度系数×时间
            A(8 * N - 4, 8 * N - 6) = T2(N - 1);                  // 加速度系数×时间²
            A(8 * N - 4, 8 * N - 5) = T3(N - 1);                  // 加加速度系数×时间³
            A(8 * N - 4, 8 * N - 4) = T4(N - 1);                  // snap系数×时间⁴
            A(8 * N - 4, 8 * N - 3) = T5(N - 1);                  // crackle系数×时间⁵
            A(8 * N - 4, 8 * N - 2) = T6(N - 1);                  // pop系数×时间⁶
            A(8 * N - 4, 8 * N - 1) = T7(N - 1);                  // 高阶系数×时间⁷

            // 最后一段在t=T_{N-1}时的速度
            A(8 * N - 3, 8 * N - 7) = 1.0;                         // 速度系数
            A(8 * N - 3, 8 * N - 6) = 2.0 * T1(N - 1);            // 加速度系数×时间
            A(8 * N - 3, 8 * N - 5) = 3.0 * T2(N - 1);            // 加加速度系数×时间²
            A(8 * N - 3, 8 * N - 4) = 4.0 * T3(N - 1);            // snap系数×时间³
            A(8 * N - 3, 8 * N - 3) = 5.0 * T4(N - 1);            // crackle系数×时间⁴
            A(8 * N - 3, 8 * N - 2) = 6.0 * T5(N - 1);            // pop系数×时间⁵
            A(8 * N - 3, 8 * N - 1) = 7.0 * T6(N - 1);            // 高阶系数×时间⁶

            // 最后一段在t=T_{N-1}时的加速度
            A(8 * N - 2, 8 * N - 6) = 2.0;                         // 加速度系数
            A(8 * N - 2, 8 * N - 5) = 6.0 * T1(N - 1);            // 加加速度系数×时间
            A(8 * N - 2, 8 * N - 4) = 12.0 * T2(N - 1);           // snap系数×时间²
            A(8 * N - 2, 8 * N - 3) = 20.0 * T3(N - 1);           // crackle系数×时间³
            A(8 * N - 2, 8 * N - 2) = 30.0 * T4(N - 1);           // pop系数×时间⁴
            A(8 * N - 2, 8 * N - 1) = 42.0 * T5(N - 1);           // 高阶系数×时间⁵

            // 最后一段在t=T_{N-1}时的加加速度
            A(8 * N - 1, 8 * N - 5) = 6.0;                         // 加加速度系数
            A(8 * N - 1, 8 * N - 4) = 24.0 * T1(N - 1);           // snap系数×时间
            A(8 * N - 1, 8 * N - 3) = 60.0 * T2(N - 1);           // crackle系数×时间²
            A(8 * N - 1, 8 * N - 2) = 120.0 * T3(N - 1);          // pop系数×时间³
            A(8 * N - 1, 8 * N - 1) = 210.0 * T4(N - 1);          // 高阶系数×时间⁴

            b.row(8 * N - 4) = tailPVAJ.col(0).transpose();        // 终止位置
            b.row(8 * N - 3) = tailPVAJ.col(1).transpose();        // 终止速度
            b.row(8 * N - 2) = tailPVAJ.col(2).transpose();        // 终止加速度
            b.row(8 * N - 1) = tailPVAJ.col(3).transpose();        // 终止加加速度

            // 求解线性方程组 A*coeffs = b
            A.factorizeLU();                                        // LU分解
            A.solve(b);                                             // 求解系数矩阵

            return;
        }

        // 获取优化后的S=4轨迹
        // 输出7次多项式轨迹，每段包含8个系数
        // 用于最小化snap(急动度)的轨迹表示
        inline void getTrajectory(Trajectory<7> &traj) const
        {
            traj.clear();                                     // 清空轨迹容器
            traj.reserve(N);                                  // 预分配N段轨迹空间
            for (int i = 0; i < N; i++)                      // 构建每段轨迹
            {
                traj.emplace_back(T1(i),                     // 时间长度
                                  b.block<8, 3>(8 * i, 0)    // 8个多项式系数(xyz维度)
                                      .transpose()            // 转置为维度×系数格式
                                      .rowwise()              // 按行操作
                                      .reverse());            // 反向排列(高次到低次)
            }
            return;
        }

        // 计算S=4轨迹的总能量
        // 基于snap(急动度)积分计算：∫₀ᵀ ||s⁽⁴⁾(t)||² dt
        // 7次多项式的snap积分解析式，用于评估轨迹平滑程度
        inline void getEnergy(double &energy) const
        {
            energy = 0.0;                                     // 初始化总能量
            for (int i = 0; i < N; i++)                      // 计算每段能量并累加
            {
                // snap积分的解析计算，包含所有交叉项
                // 对于p(t) = Σ cᵢt^i，snap = p⁽⁴⁾(t) = Σ i(i-1)(i-2)(i-3)cᵢt^(i-4)
                energy += 576.0 * b.row(8 * i + 4).squaredNorm() * T1(i) +      // c₄²项：4!²t
                          2880.0 * b.row(8 * i + 4).dot(b.row(8 * i + 5)) * T2(i) + // c₄c₅项：2×4!×5!/4!×t²
                          4800.0 * b.row(8 * i + 5).squaredNorm() * T3(i) +     // c₅²项：(5!/4!)²×t³/3
                          5760.0 * b.row(8 * i + 4).dot(b.row(8 * i + 6)) * T3(i) + // c₄c₆项
                          21600.0 * b.row(8 * i + 5).dot(b.row(8 * i + 6)) * T4(i) + // c₅c₆项
                          10080.0 * b.row(8 * i + 4).dot(b.row(8 * i + 7)) * T4(i) + // c₄c₇项
                          25920.0 * b.row(8 * i + 6).squaredNorm() * T5(i) +     // c₆²项
                          40320.0 * b.row(8 * i + 5).dot(b.row(8 * i + 7)) * T5(i) + // c₅c₇项
                          100800.0 * b.row(8 * i + 6).dot(b.row(8 * i + 7)) * T6(i) + // c₆c₇项
                          100800.0 * b.row(8 * i + 7).squaredNorm() * T7(i);     // c₇²项：(7!/3!)²×t⁷/7
            }
            return;
        }

        // 获取多项式系数矩阵
        // 返回8N×3矩阵，每行8个系数对应一个多项式段的xyz分量
        // 系数按c₀,c₁,...,c₇排列，用于构造7次多项式
        inline const Eigen::MatrixX3d &getCoeffs(void) const
        {
            return b;                                         // 返回系数矩阵引用
        }

        // 计算能量对多项式系数的偏导数
        // 用于基于梯度的轨迹优化算法
        // gdC输出：每个系数对snap积分能量的梯度
        inline void getEnergyPartialGradByCoeffs(Eigen::MatrixX3d &gdC) const
        {
            gdC.resize(8 * N, 3);                            // 分配梯度矩阵空间
            for (int i = 0; i < N; i++)                      // 计算每段系数的梯度
            {
                // 对c₇的偏导数：∂E/∂c₇ = 交叉项系数
                gdC.row(8 * i + 7) = 10080.0 * b.row(8 * i + 4) * T4(i) +    // c₄交叉项
                                     40320.0 * b.row(8 * i + 5) * T5(i) +     // c₅交叉项
                                     100800.0 * b.row(8 * i + 6) * T6(i) +    // c₆交叉项
                                     201600.0 * b.row(8 * i + 7) * T7(i);     // c₇自身项

                // 对c₆的偏导数
                gdC.row(8 * i + 6) = 5760.0 * b.row(8 * i + 4) * T3(i) +     // c₄交叉项
                                     21600.0 * b.row(8 * i + 5) * T4(i) +     // c₅交叉项
                                     51840.0 * b.row(8 * i + 6) * T5(i) +     // c₆自身项
                                     100800.0 * b.row(8 * i + 7) * T6(i);     // c₇交叉项

                // 对c₅的偏导数
                gdC.row(8 * i + 5) = 2880.0 * b.row(8 * i + 4) * T2(i) +     // c₄交叉项
                                     9600.0 * b.row(8 * i + 5) * T3(i) +      // c₅自身项
                                     21600.0 * b.row(8 * i + 6) * T4(i) +     // c₆交叉项
                                     40320.0 * b.row(8 * i + 7) * T5(i);      // c₇交叉项

                // 对c₄的偏导数
                gdC.row(8 * i + 4) = 1152.0 * b.row(8 * i + 4) * T1(i) +     // c₄自身项
                                     2880.0 * b.row(8 * i + 5) * T2(i) +      // c₅交叉项
                                     5760.0 * b.row(8 * i + 6) * T3(i) +      // c₆交叉项
                                     10080.0 * b.row(8 * i + 7) * T4(i);      // c₇交叉项

                gdC.block<4, 3>(8 * i, 0).setZero();         // c₀-c₃对snap无贡献，梯度为0
            }
            return;
        }

        // 计算能量对时间分配的偏导数
        // 用于时间优化：∂E/∂T_i，优化每段轨迹的时间分配
        // gdT输出：每段时间对总snap能量的梯度
        inline void getEnergyPartialGradByTimes(Eigen::VectorXd &gdT) const
        {
            gdT.resize(N);                                    // 分配时间梯度向量
            for (int i = 0; i < N; i++)                      // 计算每段时间的梯度
            {
                // 能量对时间的偏导数：∂E/∂T = ∂/∂T ∫₀ᵀ ||s⁽⁴⁾(t)||² dt
                gdT(i) = 576.0 * b.row(8 * i + 4).squaredNorm() +            // c₄²项的T导数
                         5760.0 * b.row(8 * i + 4).dot(b.row(8 * i + 5)) * T1(i) + // c₄c₅项
                         14400.0 * b.row(8 * i + 5).squaredNorm() * T2(i) +   // c₅²项
                         17280.0 * b.row(8 * i + 4).dot(b.row(8 * i + 6)) * T2(i) + // c₄c₆项
                         86400.0 * b.row(8 * i + 5).dot(b.row(8 * i + 6)) * T3(i) + // c₅c₆项
                         40320.0 * b.row(8 * i + 4).dot(b.row(8 * i + 7)) * T3(i) + // c₄c₇项
                         129600.0 * b.row(8 * i + 6).squaredNorm() * T4(i) +  // c₆²项
                         201600.0 * b.row(8 * i + 5).dot(b.row(8 * i + 7)) * T4(i) + // c₅c₇项
                         604800.0 * b.row(8 * i + 6).dot(b.row(8 * i + 7)) * T5(i) + // c₆c₇项
                         705600.0 * b.row(8 * i + 7).squaredNorm() * T6(i);   // c₇²项
            }
            return;
        }

        // 反向传播梯度到控制点和时间分配
        // 将系数梯度转换为优化变量(控制点位置和时间)的梯度
        // 用于梯度下降优化算法
        inline void propogateGrad(const Eigen::MatrixX3d &partialGradByCoeffs,
                                  const Eigen::VectorXd &partialGradByTimes,
                                  Eigen::Matrix3Xd &gradByPoints,
                                  Eigen::VectorXd &gradByTimes)
        {
            gradByPoints.resize(3, N - 1);                    // 控制点梯度矩阵
            gradByTimes.resize(N);                            // 时间梯度向量
            Eigen::MatrixX3d adjGrad = partialGradByCoeffs;   // 复制系数梯度
            A.solveAdj(adjGrad);                              // 求解伴随方程

            // 提取中间点(控制点)的梯度
            for (int i = 0; i < N - 1; i++)
            {
                gradByPoints.col(i) = adjGrad.row(8 * i + 7).transpose();  // 每个控制点的梯度
            }

            // 计算时间梯度的约束项
            Eigen::Matrix<double, 8, 3> B1;               // 临时约束矩阵
            Eigen::Matrix<double, 4, 3> B2;               // 边界约束矩阵
            for (int i = 0; i < N - 1; i++)              // 遍历每个连接点
            {
                // 负速度连续性约束的梯度贡献
                // ∂(-v(T_i))/∂T_i = -p'(T_i)，其中p'是速度
                B1.row(3) = -(b.row(i * 8 + 1) +                           // c₁项
                              2.0 * T1(i) * b.row(i * 8 + 2) +             // 2c₂T项
                              3.0 * T2(i) * b.row(i * 8 + 3) +             // 3c₃T²项
                              4.0 * T3(i) * b.row(i * 8 + 4) +             // 4c₄T³项
                              5.0 * T4(i) * b.row(i * 8 + 5) +             // 5c₅T⁴项
                              6.0 * T5(i) * b.row(i * 8 + 6) +             // 6c₆T⁵项
                              7.0 * T6(i) * b.row(i * 8 + 7));             // 7c₇T⁶项
                B1.row(4) = B1.row(3);                                      // 速度连续性约束重复

                // 负加速度连续性约束的梯度贡献
                // ∂(-a(T_i))/∂T_i = -p''(T_i)，其中p''是加速度
                B1.row(5) = -(2.0 * b.row(i * 8 + 2) +                     // 2c₂项
                              6.0 * T1(i) * b.row(i * 8 + 3) +             // 6c₃T项
                              12.0 * T2(i) * b.row(i * 8 + 4) +            // 12c₄T²项
                              20.0 * T3(i) * b.row(i * 8 + 5) +            // 20c₅T³项
                              30.0 * T4(i) * b.row(i * 8 + 6) +            // 30c₆T⁴项
                              42.0 * T5(i) * b.row(i * 8 + 7));            // 42c₇T⁵项

                // 负加加速度连续性约束的梯度贡献
                // ∂(-j(T_i))/∂T_i = -p'''(T_i)，其中p'''是加加速度
                B1.row(6) = -(6.0 * b.row(i * 8 + 3) +                     // 6c₃项
                              24.0 * T1(i) * b.row(i * 8 + 4) +            // 24c₄T项
                              60.0 * T2(i) * b.row(i * 8 + 5) +            // 60c₅T²项
                              120.0 * T3(i) * b.row(i * 8 + 6) +           // 120c₆T³项
                              210.0 * T4(i) * b.row(i * 8 + 7));           // 210c₇T⁴项

                // 负snap连续性约束的梯度贡献
                // ∂(-s(T_i))/∂T_i = -p⁽⁴⁾(T_i)，其中p⁽⁴⁾是snap
                B1.row(7) = -(24.0 * b.row(i * 8 + 4) +                    // 24c₄项
                              120.0 * T1(i) * b.row(i * 8 + 5) +           // 120c₅T项
                              360.0 * T2(i) * b.row(i * 8 + 6) +           // 360c₆T²项
                              840.0 * T3(i) * b.row(i * 8 + 7));           // 840c₇T³项

                // 负crackle连续性约束的梯度贡献
                // ∂(-c(T_i))/∂T_i = -p⁽⁵⁾(T_i)，其中p⁽⁵⁾是crackle
                B1.row(0) = -(120.0 * b.row(i * 8 + 5) +                   // 120c₅项
                              720.0 * T1(i) * b.row(i * 8 + 6) +           // 720c₆T项
                              2520.0 * T2(i) * b.row(i * 8 + 7));          // 2520c₇T²项

                // 负pop连续性约束的梯度贡献
                // ∂(-pop(T_i))/∂T_i = -p⁽⁶⁾(T_i)，其中p⁽⁶⁾是pop
                B1.row(1) = -(720.0 * b.row(i * 8 + 6) +                   // 720c₆项
                              5040.0 * T1(i) * b.row(i * 8 + 7));          // 5040c₇T项

                // 负高阶导数连续性约束的梯度贡献
                // ∂(-p⁽⁷⁾(T_i))/∂T_i = -p⁽⁷⁾(T_i)
                B1.row(2) = -5040.0 * b.row(i * 8 + 7);                    // 5040c₇项

                // 计算约束贡献与伴随梯度的内积
                gradByTimes(i) = B1.cwiseProduct(adjGrad.block<8, 3>(8 * i + 4, 0)).sum();
            }

            // 处理最后一段的边界条件梯度
            // 负速度边界条件的梯度贡献
            B2.row(0) = -(b.row(8 * N - 7) +                               // c₁项
                          2.0 * T1(N - 1) * b.row(8 * N - 6) +             // 2c₂T项
                          3.0 * T2(N - 1) * b.row(8 * N - 5) +             // 3c₃T²项
                          4.0 * T3(N - 1) * b.row(8 * N - 4) +             // 4c₄T³项
                          5.0 * T4(N - 1) * b.row(8 * N - 3) +             // 5c₅T⁴项
                          6.0 * T5(N - 1) * b.row(8 * N - 2) +             // 6c₆T⁵项
                          7.0 * T6(N - 1) * b.row(8 * N - 1));             // 7c₇T⁶项

            // 负加速度边界条件的梯度贡献
            B2.row(1) = -(2.0 * b.row(8 * N - 6) +                         // 2c₂项
                          6.0 * T1(N - 1) * b.row(8 * N - 5) +             // 6c₃T项
                          12.0 * T2(N - 1) * b.row(8 * N - 4) +            // 12c₄T²项
                          20.0 * T3(N - 1) * b.row(8 * N - 3) +            // 20c₅T³项
                          30.0 * T4(N - 1) * b.row(8 * N - 2) +            // 30c₆T⁴项
                          42.0 * T5(N - 1) * b.row(8 * N - 1));            // 42c₇T⁵项

            // 负加加速度边界条件的梯度贡献
            B2.row(2) = -(6.0 * b.row(8 * N - 5) +                         // 6c₃项
                          24.0 * T1(N - 1) * b.row(8 * N - 4) +            // 24c₄T项
                          60.0 * T2(N - 1) * b.row(8 * N - 3) +            // 60c₅T²项
                          120.0 * T3(N - 1) * b.row(8 * N - 2) +           // 120c₆T³项
                          210.0 * T4(N - 1) * b.row(8 * N - 1));           // 210c₇T⁴项

            // 负snap边界条件的梯度贡献
            B2.row(3) = -(24.0 * b.row(8 * N - 4) +                        // 24c₄项
                          120.0 * T1(N - 1) * b.row(8 * N - 3) +           // 120c₅T项
                          360.0 * T2(N - 1) * b.row(8 * N - 2) +           // 360c₆T²项
                          840.0 * T3(N - 1) * b.row(8 * N - 1));           // 840c₇T³项

            // 计算最后一段时间的梯度贡献
            gradByTimes(N - 1) = B2.cwiseProduct(adjGrad.block<4, 3>(8 * N - 4, 0)).sum();
            gradByTimes += partialGradByTimes;                              // 加上直接时间梯度
        }
    };
}