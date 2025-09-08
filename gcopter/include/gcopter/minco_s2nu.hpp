#pragma once


#include "gcopter/banded_systems.hpp"

#include <Eigen/Eigen>

#include <cmath>
#include <vector>
namespace minco
{

    /**
     * @class MINCO_S2NU  
     * @brief MINCO S=2阶非均匀时间轨迹生成器
     * 
     * MINCO (Minimum Control Effort) 算法的具体实现，用于生成S=2阶（即最小化加速度）
     * 的非均匀时间分配轨迹。该类专门处理3D空间中的轨迹优化问题。
     * 
     * 算法特点：
     * - S=2：最小化加速度的二次积分（控制输入为加速度）
     * - 非均匀时间：允许每段轨迹具有不同的时间长度
     * - 3D轨迹：支持三维空间中的位置、速度、加速度约束
     * - 连续性：保证轨迹在连接点处的位置和速度连续性
     * 
     * 数学基础：
     * - 最小化目标函数：∫||u(t)||²dt，其中u(t)是控制输入（加速度）
     * - 约束条件：位置、速度边界条件，连接点连续性条件
     * - 解决方案：分段多项式轨迹，通过二次规划求解系数
     */
    class MINCO_S2NU
    {
    public:
        MINCO_S2NU() = default;
        ~MINCO_S2NU() { A.destroy(); }

    private:
        int N;                                      ///< 轨迹段数量
        Eigen::Matrix<double, 3, 2> headPV;        ///< 起始点位置和速度 [3×2]
        Eigen::Matrix<double, 3, 2> tailPV;        ///< 终止点位置和速度 [3×2]  
        BandedSystem A;                             ///< 系数矩阵A（带状系统）
        Eigen::MatrixX3d b;                         ///< 右端向量b [N×3]
        Eigen::VectorXd T1;                         ///< 时间的一次幂 [N×1]
        Eigen::VectorXd T2;                         ///< 时间的二次幂 [N×1]
        Eigen::VectorXd T3;                         ///< 时间的三次幂 [N×1]

    public:
        /**
         * @brief 设置轨迹边界条件和段数
         * @param headState 起始状态，包含位置和速度 [位置; 速度] (3×2)
         * @param tailState 终止状态，包含位置和速度 [位置; 速度] (3×2)
         * @param pieceNum 轨迹分段数量
         * 
         * 初始化MINCO求解器，设置边界条件和矩阵维度。
         * 对于S=2的最小控制轨迹（最小加速度），需要确保位置和速度连续性。
         * 每段轨迹需要4个系数，因此总共4N个未知数需要4N个约束方程。
         */
        inline void setConditions(const Eigen::Matrix<double, 3, 2> &headState,
                                  const Eigen::Matrix<double, 3, 2> &tailState,
                                  const int &pieceNum)
        {
            N = pieceNum;                        // 保存轨迹段数
            headPV = headState;                  // 保存起始位置和速度
            tailPV = tailState;                  // 保存终止位置和速度
            A.create(4 * N, 4, 4);              // 创建带状线性系统矩阵 (4N×4N，带宽4)
            b.resize(4 * N, 3);                 // 右端向量 (4N×3，对应x,y,z三个维度)
            T1.resize(N);                       // 时间向量
            T2.resize(N);                       // 时间平方向量
            T3.resize(N);                       // 时间立方向量
            return;
        }

        /**
         * @brief 设置轨迹参数并求解多项式系数
         * @param inPs 中间路径点矩阵 (3×(N-1))，不包含起始和终止点
         * @param ts 各段轨迹的时间分配向量 (N×1)
         * 
         * 根据给定的路径点和时间分配，构建约束矩阵并求解多项式系数。
         * 约束条件包括：
         * 1. 边界条件：起始和终止的位置、速度
         * 2. 连续性条件：相邻段间的位置、速度、加速度连续
         * 3. 路径点约束：中间点必须经过指定位置
         * 
         * 多项式形式：p_i(t) = c_{i0} + c_{i1}*t + c_{i2}*t² + c_{i3}*t³
         * 其中 c_{i0}, c_{i1}, c_{i2}, c_{i3} 分别对应位置、速度、加速度、加加速度系数
         */
        inline void setParameters(const Eigen::Matrix3Xd &inPs,
                                  const Eigen::VectorXd &ts)
        {
            // 预计算时间的幂次方，避免重复计算
            T1 = ts;                            // t
            T2 = T1.cwiseProduct(T1);          // t²
            T3 = T2.cwiseProduct(T1);          // t³

            A.reset();                          // 重置矩阵为零
            b.setZero();                        // 重置右端向量为零

            // 设置起始条件约束
            // 第0段在t=0时的位置: c_{00} = headPV.pos
            A(0, 0) = 1.0;
            // 第0段在t=0时的速度: c_{01} = headPV.vel  
            A(1, 1) = 1.0;
            b.row(0) = headPV.col(0).transpose();    // 起始位置
            b.row(1) = headPV.col(1).transpose();    // 起始速度

            // 设置中间段连续性约束和路径点约束
            for (int i = 0; i < N - 1; i++)
            {
                // 加速度连续性约束：段i终点加速度 = 段i+1起点加速度
                // 2*c_{i2} + 6*c_{i3}*T_i = 2*c_{(i+1)2}
                A(4 * i + 2, 4 * i + 2) = 2.0;              // 段i的加速度系数
                A(4 * i + 2, 4 * i + 3) = 6.0 * T1(i);      // 段i的加加速度×时间
                A(4 * i + 2, 4 * i + 6) = -2.0;             // 段i+1的加速度系数

                // 位置连续性约束：段i终点位置 = 段i+1起点位置 = 中间路径点
                // c_{i0} + c_{i1}*T_i + c_{i2}*T_i² + c_{i3}*T_i³ = inPs[i]
                A(4 * i + 3, 4 * i) = 1.0;                  // 位置系数
                A(4 * i + 3, 4 * i + 1) = T1(i);           // 速度系数×时间
                A(4 * i + 3, 4 * i + 2) = T2(i);           // 加速度系数×时间²
                A(4 * i + 3, 4 * i + 3) = T3(i);           // 加加速度系数×时间³

                // 段i+1起点位置约束：c_{(i+1)0} = inPs[i]
                A(4 * i + 4, 4 * i + 4) = -1.0;

                // 速度连续性约束：段i终点速度 = 段i+1起点速度
                // c_{i1} + 2*c_{i2}*T_i + 3*c_{i3}*T_i² = c_{(i+1)1}
                A(4 * i + 5, 4 * i + 1) = 1.0;             // 段i的速度系数
                A(4 * i + 5, 4 * i + 2) = 2.0 * T1(i);     // 段i的加速度×时间
                A(4 * i + 5, 4 * i + 3) = 3.0 * T2(i);     // 段i的加加速度×时间²
                A(4 * i + 5, 4 * i + 5) = -1.0;            // 段i+1的速度系数

                // 设置中间路径点约束的右端向量
                b.row(4 * i + 3) = inPs.col(i).transpose();
            }

            // 设置终止条件约束
            // 最后一段在t=T_{N-1}时的位置: c_{(N-1)0} + ... = tailPV.pos
            A(4 * N - 2, 4 * N - 4) = 1.0;                 // 位置系数
            A(4 * N - 2, 4 * N - 3) = T1(N - 1);          // 速度系数×时间
            A(4 * N - 2, 4 * N - 2) = T2(N - 1);          // 加速度系数×时间²
            A(4 * N - 2, 4 * N - 1) = T3(N - 1);          // 加加速度系数×时间³
            
            // 最后一段在t=T_{N-1}时的速度: c_{(N-1)1} + ... = tailPV.vel
            A(4 * N - 1, 4 * N - 3) = 1.0;                 // 速度系数
            A(4 * N - 1, 4 * N - 2) = 2 * T1(N - 1);      // 加速度系数×时间
            A(4 * N - 1, 4 * N - 1) = 3 * T2(N - 1);      // 加加速度系数×时间²

            b.row(4 * N - 2) = tailPV.col(0).transpose();  // 终止位置
            b.row(4 * N - 1) = tailPV.col(1).transpose();  // 终止速度

            // 求解线性方程组 A*coeffs = b
            A.factorizeLU();                                // LU分解
            A.solve(b);                                     // 求解系数矩阵

            return;
        }

        /**
         * @brief 构建轨迹对象
         * @param traj 输出的轨迹对象，包含所有分段多项式
         * 
         * 将求解得到的多项式系数转换为Trajectory对象。
         * 每段轨迹由时间长度和4个系数（位置、速度、加速度、加加速度）确定。
         * 
         * 注意：系数需要按照从高次到低次的顺序排列（c3, c2, c1, c0）
         * 这与Piece类的内部存储格式一致。
         */
        inline void getTrajectory(Trajectory<3> &traj) const
        {
            traj.clear();                       // 清空现有轨迹
            traj.reserve(N);                    // 预分配内存空间
            for (int i = 0; i < N; i++)
            {
                // 为每段轨迹创建Piece对象
                // 时间长度：T1(i)
                // 系数矩阵：b的第4i到4i+3行，按列转置后行倒序（高次到低次）
                traj.emplace_back(T1(i),
                                  b.block<4, 3>(4 * i, 0)    // 提取4×3系数块
                                      .transpose()             // 转置为3×4 (x,y,z × c0,c1,c2,c3)
                                      .rowwise()               // 按行操作
                                      .reverse());             // 倒序为 (x,y,z × c3,c2,c1,c0)
            }
            return;
        }

        /**
         * @brief 计算轨迹的总能量（加速度平方积分）
         * @param energy 输出的总能量值
         * 
         * 对于S=2的MINCO轨迹，能量定义为加速度的平方在时间上的积分：
         * E = ∫[0,T] ||a(t)||² dt
         * 
         * 对于三次多项式 p(t) = c0 + c1*t + c2*t² + c3*t³
         * 加速度为 a(t) = 2*c2 + 6*c3*t
         * 
         * 能量积分为：
         * E_i = ∫[0,Ti] (2*c2 + 6*c3*t)² dt
         *     = ∫[0,Ti] (4*c2² + 24*c2*c3*t + 36*c3²*t²) dt  
         *     = 4*c2²*Ti + 12*c2*c3*Ti² + 12*c3²*Ti³
         * 
         * 其中 c2 对应 b.row(4*i+2)，c3 对应 b.row(4*i+3)
         */
        inline void getEnergy(double &energy) const
        {
            energy = 0.0;
            for (int i = 0; i < N; i++)
            {
                // 计算第i段的能量贡献
                // 4*||c2||²*Ti：加速度的常数项贡献
                energy += 4.0 * b.row(4 * i + 2).squaredNorm() * T1(i) +
                          // 12*c2·c3*Ti²：加速度的线性项贡献  
                          12.0 * b.row(4 * i + 2).dot(b.row(4 * i + 3)) * T2(i) +
                          // 12*||c3||²*Ti³：加速度的二次项贡献
                          12.0 * b.row(4 * i + 3).squaredNorm() * T3(i);
            }
            return;
        }

        /**
         * @brief 获取多项式系数矩阵
         * @return 系数矩阵的常量引用 (4N×3)
         * 
         * 返回所有轨迹段的多项式系数。矩阵结构为：
         * 第4i+0行：第i段的位置系数 c_{i0}
         * 第4i+1行：第i段的速度系数 c_{i1}  
         * 第4i+2行：第i段的加速度系数 c_{i2}
         * 第4i+3行：第i段的加加速度系数 c_{i3}
         * 列对应x, y, z三个坐标轴
         */
        inline const Eigen::MatrixX3d &getCoeffs(void) const
        {
            return b;
        }

        /**
         * @brief 计算能量对多项式系数的偏导数
         * @param gdC 输出的梯度矩阵 (4N×3)
         * 
         * 计算总能量E对各段多项式系数的偏导数：∂E/∂c_{ij}
         * 
         * 对于能量函数 E_i = 4*c2²*Ti + 12*c2*c3*Ti² + 12*c3²*Ti³
         * 偏导数为：
         * ∂E_i/∂c2 = 8*c2*Ti + 12*c3*Ti²
         * ∂E_i/∂c3 = 12*c2*Ti² + 24*c3*Ti³
         * ∂E_i/∂c0 = ∂E_i/∂c1 = 0 (位置和速度系数不影响加速度)
         * 
         * 这些梯度用于基于梯度的轨迹优化算法。
         */
        inline void getEnergyPartialGradByCoeffs(Eigen::MatrixX3d &gdC) const
        {
            gdC.resize(4 * N, 3);
            for (int i = 0; i < N; i++)
            {
                // ∂E_i/∂c3：对加加速度系数的梯度
                gdC.row(4 * i + 3) = 12.0 * b.row(4 * i + 2) * T2(i) +     // 12*c2*Ti²
                                     24.0 * b.row(4 * i + 3) * T3(i);       // 24*c3*Ti³
                
                // ∂E_i/∂c2：对加速度系数的梯度
                gdC.row(4 * i + 2) = 8.0 * b.row(4 * i + 2) * T1(i) +      // 8*c2*Ti
                                     12.0 * b.row(4 * i + 3) * T2(i);       // 12*c3*Ti²
                
                // ∂E_i/∂c0 = ∂E_i/∂c1 = 0：位置和速度系数的梯度为零
                gdC.block<2, 3>(4 * i, 0).setZero();
            }
            return;
        }

        /**
         * @brief 计算能量对时间分配的偏导数
         * @param gdT 输出的时间梯度向量 (N×1)
         * 
         * 计算总能量E对各段时间分配的偏导数：∂E/∂Ti
         * 
         * 对于能量函数 E_i = 4*c2²*Ti + 12*c2*c3*Ti² + 12*c3²*Ti³
         * 对时间的偏导数为：
         * ∂E_i/∂Ti = 4*||c2||² + 24*c2·c3*Ti + 36*||c3||²*Ti²
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
                gdT(i) = 4.0 * b.row(4 * i + 2).squaredNorm() +                    // 4*||c2||²
                         24.0 * b.row(4 * i + 2).dot(b.row(4 * i + 3)) * T1(i) +  // 24*c2·c3*Ti
                         36.0 * b.row(4 * i + 3).squaredNorm() * T2(i);            // 36*||c3||²*Ti²
            }
            return;
        }

        /**
         * @brief 传播梯度到设计变量（路径点和时间分配）
         * @param partialGradByCoeffs 目标函数对系数的偏导数 (4N×3)
         * @param partialGradByTimes 目标函数对时间的偏导数 (N×1)
         * @param gradByPoints 输出：目标函数对路径点的梯度 (3×(N-1))
         * @param gradByTimes 输出：目标函数对时间分配的总梯度 (N×1)
         * 
         * 使用链式法则和伴随方法将目标函数的梯度从多项式系数
         * 传播回原始设计变量（中间路径点和时间分配）。
         * 
         * 链式法则：∂J/∂x = (∂J/∂c)^T * (∂c/∂x)
         * 其中 J 是目标函数，c 是系数，x 是设计变量
         * 
         * 伴随方法避免了显式计算雅可比矩阵 ∂c/∂x，
         * 而是通过求解伴随方程 A^T * λ = ∂J/∂c 来高效计算梯度。
         */
        inline void propogateGrad(const Eigen::MatrixX3d &partialGradByCoeffs,
                                  const Eigen::VectorXd &partialGradByTimes,
                                  Eigen::Matrix3Xd &gradByPoints,
                                  Eigen::VectorXd &gradByTimes)

        {
            gradByPoints.resize(3, N - 1);      // 中间路径点的梯度
            gradByTimes.resize(N);              // 时间分配的梯度
            
            // 求解伴随方程：A^T * adjGrad = partialGradByCoeffs
            // 使用LU分解的转置求解，高效计算伴随变量
            Eigen::MatrixX3d adjGrad = partialGradByCoeffs;
            A.solveAdj(adjGrad);

            // 提取对路径点的梯度
            // 中间路径点直接对应约束方程中的右端向量，
            // 其梯度即为对应伴随变量的值
            for (int i = 0; i < N - 1; i++)
            {
                gradByPoints.col(i) = adjGrad.row(4 * i + 3).transpose();
            }

            // 计算对时间分配的梯度
            // 时间变化会影响约束矩阵A中的时间相关项
            Eigen::Matrix<double, 4, 3> B1;     // 中间段的时间梯度贡献
            Eigen::Matrix<double, 2, 3> B2;     // 最后一段的时间梯度贡献
            
            for (int i = 0; i < N - 1; i++)
            {
                // 计算约束矩阵A对时间Ti的偏导数与伴随变量的乘积
                
                // 对加速度连续性约束的贡献：∂(6*c3*Ti)/∂Ti = 6*c3
                B1.row(0) = -6.0 * b.row(i * 4 + 3);

                // 对速度连续性约束的贡献：∂(c1 + 2*c2*Ti + 3*c3*Ti²)/∂Ti
                B1.row(1) = -(b.row(i * 4 + 1) +                    // ∂(c1)/∂Ti = 0（但这里是负号）
                              2.0 * T1(i) * b.row(i * 4 + 2) +      // ∂(c2*Ti²)/∂Ti = 2*c2*Ti
                              3.0 * T2(i) * b.row(i * 4 + 3));      // ∂(c3*Ti³)/∂Ti = 3*c3*Ti²
                B1.row(2) = B1.row(1);                              // 重复使用

                // 对加速度连续性约束的贡献：∂(2*c2 + 6*c3*Ti)/∂Ti = 6*c3
                B1.row(3) = -(2.0 * b.row(i * 4 + 2) +              // ∂(2*c2)/∂Ti = 0（但这里是负号）
                              6.0 * T1(i) * b.row(i * 4 + 3));      // ∂(6*c3*Ti)/∂Ti = 6*c3

                // 计算第i段时间的梯度贡献
                gradByTimes(i) = B1.cwiseProduct(adjGrad.block<4, 3>(4 * i + 2, 0)).sum();
            }

            // 处理最后一段的边界条件
            // 最后一段的终止条件也依赖于时间T_{N-1}
            
            // 对终止速度约束的贡献
            B2.row(0) = -(b.row(4 * N - 3) +                        // 速度系数
                          2.0 * T1(N - 1) * b.row(4 * N - 2) +      // 加速度×时间
                          3.0 * T2(N - 1) * b.row(4 * N - 1));      // 加加速度×时间²

            // 对终止加速度约束的贡献
            B2.row(1) = -(2.0 * b.row(4 * N - 2) +                  // 加速度系数×2
                          6.0 * T1(N - 1) * b.row(4 * N - 1));      // 加加速度×时间×6

            // 计算最后一段时间的梯度贡献
            gradByTimes(N - 1) = B2.cwiseProduct(adjGrad.block<2, 3>(4 * N - 2, 0)).sum();

            // 加上直接对时间的偏导数（如能量函数对时间的直接依赖）
            gradByTimes += partialGradByTimes;
        }
    };
}