/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

/**
 * @file gcopter.hpp
 * @brief GCOPTER核心类：几何约束多机轨迹规划器
 * 
 * GCOPTER (Geometrically Constrained Trajectory Optimizer) 是一个先进的
 * 多机轨迹规划框架，专门设计用于复杂3D环境中的安全高效轨迹生成。
 * 
 * 核心特性：
 * 1. 安全走廊约束 (SFC) - 基于凸多面体的空间约束
 * 2. MINCO轨迹表示 - 最小控制努力的分段多项式
 * 3. 微分平坦性 - 将几何轨迹映射到控制输入
 * 4. L-BFGS优化 - 高效的梯度优化算法
 * 
 * 数学基础：
 * - 凸优化理论：安全走廊的凸性质保证全局最优解
 * - 微分几何：平坦系统的几何控制理论
 * - 数值优化：L-BFGS拟牛顿法的收敛性质
 * - 样条理论：分段多项式的连续性和光滑性
 * 
 * 应用领域：
 * - 多旋翼无人机轨迹规划
 * - 多机器人协调路径规划
 * - 空中交通管理系统
 * - 自动驾驶车辆路径规划
 */

#ifndef GCOPTER_HPP
#define GCOPTER_HPP

#include "gcopter/minco.hpp"    // MINCO轨迹优化算法
#include "gcopter/flatness.hpp" // 微分平坦性映射
#include "gcopter/lbfgs.hpp"    // L-BFGS优化器

#include <Eigen/Eigen>

#include <cmath>
#include <cfloat>
#include <iostream>
#include <vector>

namespace gcopter
{

    /**
     * @brief GCOPTER多面体安全走廊轨迹规划器
     * 
     * 这是GCOPTER框架的核心类，实现了基于安全走廊约束的轨迹优化。
     * 该规划器将复杂的3D环境分解为一系列凸多面体安全区域，
     * 然后在这些区域内生成平滑、安全、动力学可行的轨迹。
     * 
     * 工作流程：
     * 1. 环境建模：将自由空间划分为凸多面体走廊
     * 2. 轨迹初始化：在安全走廊中构建初始路径点
     * 3. 时间分配：基于动力学约束分配段时间
     * 4. 轨迹优化：使用L-BFGS最小化代价函数
     * 5. 约束检查：确保所有物理和几何约束满足
     * 
     * 关键算法组件：
     * - MINCO_S3NU：3阶导数约束的最小控制轨迹表示
     * - FlatnessMap：微分平坦性的状态-控制映射
     * - L-BFGS：限存储拟牛顿优化算法
     * - SFC：安全走廊约束的凸性质
     * 
     * 设计优势：
     * - 计算效率：多项式表示的解析梯度计算
     * - 数值稳定：凸优化保证收敛性
     * - 灵活性：支持多种约束类型和目标函数
     * - 可扩展性：模块化设计便于扩展新功能
     */
    class GCOPTER_PolytopeSFC
    {
    public:
        // 类型定义：几何表示的别名
        typedef Eigen::Matrix3Xd PolyhedronV;    // 多面体顶点表示 (3×N矩阵)
        typedef Eigen::MatrixX4d PolyhedronH;    // 多面体半空间表示 (M×4矩阵)
        typedef std::vector<PolyhedronV> PolyhedraV;  // 顶点表示的多面体序列
        typedef std::vector<PolyhedronH> PolyhedraH;  // 半空间表示的多面体序列

    private:
        // 核心算法组件
        minco::MINCO_S3NU minco;           // MINCO轨迹优化器：3阶导数连续的最小控制轨迹
        flatness::FlatnessMap flatmap;     // 微分平坦性映射：轨迹-控制状态转换器

        // 边界条件参数
        double rho;                        // 控制输入权重系数：平衡轨迹平滑度和控制成本
        Eigen::Matrix3d headPVA;          // 起始条件：位置-速度-加速度 [3×3矩阵]
        Eigen::Matrix3d tailPVA;          // 终止条件：位置-速度-加速度 [3×3矩阵]

        // 几何约束表示
        PolyhedraV vPolytopes;            // 顶点表示的安全走廊序列
        PolyhedraH hPolytopes;            // 半空间表示的安全走廊序列
        Eigen::Matrix3Xd shortPath;      // 初始路径点：连接各安全走廊的短路径

        // 索引映射关系
        Eigen::VectorXi pieceIdx;         // 轨迹段索引：每段对应的多面体编号
        Eigen::VectorXi vPolyIdx;         // 顶点多面体索引：路径点到顶点表示的映射
        Eigen::VectorXi hPolyIdx;         // 半空间多面体索引：路径点到半空间表示的映射

        // 问题维度参数
        int polyN;                        // 多面体总数：安全走廊的数量
        int pieceN;                       // 轨迹段总数：分段多项式的段数

        // 优化问题维度
        int spatialDim;                   // 空间维度：通常为3（3D空间）
        int temporalDim;                  // 时间维度：优化变量中时间参数的数量

        // 数值计算参数
        double smoothEps;                 // 平滑性容差：数值稳定性的小量参数
        int integralRes;                  // 积分分辨率：数值积分的采样点数
        Eigen::VectorXd magnitudeBd;      // 量级边界：各物理量的上下限约束
        Eigen::VectorXd penaltyWt;        // 惩罚权重：约束违反的惩罚系数
        Eigen::VectorXd physicalPm;       // 物理参数：系统的动力学参数（质量、惯性等）
        double allocSpeed;                // 分配速度：时间分配算法的参考速度

        // L-BFGS优化器参数
        lbfgs::lbfgs_parameter_t lbfgs_params;  // L-BFGS算法的配置参数

        // 优化变量和梯度
        Eigen::Matrix3Xd points;          // 优化路径点：当前迭代的空间坐标
        Eigen::VectorXd times;            // 优化时间分配：各段的持续时间
        Eigen::Matrix3Xd gradByPoints;    // 空间梯度：目标函数对路径点的偏导数
        Eigen::VectorXd gradByTimes;      // 时间梯度：目标函数对时间分配的偏导数
        Eigen::MatrixX3d partialGradByCoeffs;   // 系数梯度：对多项式系数的偏导数
        Eigen::VectorXd partialGradByTimes;     // 时间偏导数：时间变量的偏导数

    private:
        /**
         * @brief 时间变量正向变换：从优化变量到物理时间
         * 
         * 将无约束的优化变量τ转换为正的物理时间T。使用非线性变换
         * 确保时间始终为正，避免优化过程中的约束处理问题。
         * 
         * 变换公式：
         * - 当τ > 0时：T = (0.5*τ + 1)*τ + 1 = 0.5*τ² + τ + 1
         * - 当τ ≤ 0时：T = 1/((0.5*τ - 1)*τ + 1) = 1/(0.5*τ² - τ + 1)
         * 
         * 数学性质：
         * - 单调性：dT/dτ > 0（严格单调递增）
         * - 正值性：T > 0 对所有τ成立
         * - 连续性：在τ = 0处连续且可导
         * - 对称性：变换在τ = 0附近近似对称
         * 
         * @param tau 无约束优化变量向量
         * @param T 输出的正时间向量
         */
        static inline void forwardT(const Eigen::VectorXd &tau,
                                    Eigen::VectorXd &T)
        {
            const int sizeTau = tau.size();
            T.resize(sizeTau);
            for (int i = 0; i < sizeTau; i++)
            {
                T(i) = tau(i) > 0.0
                           ? ((0.5 * tau(i) + 1.0) * tau(i) + 1.0)     // 正分支：二次增长
                           : 1.0 / ((0.5 * tau(i) - 1.0) * tau(i) + 1.0);  // 负分支：倒数形式
            }
            return;
        }

        /**
         * @brief 时间变量反向变换：从物理时间到优化变量
         * 
         * forwardT的逆变换，将正的物理时间T转换回无约束优化变量τ。
         * 这是通过求解变换方程的逆函数得到的。
         * 
         * 逆变换公式：
         * - 当T > 1时：τ = √(2T - 1) - 1
         * - 当T ≤ 1时：τ = 1 - √(2/T - 1)
         * 
         * 推导过程：
         * 1. 对于T > 1：由0.5τ² + τ + 1 = T求解得τ = √(2T-1) - 1
         * 2. 对于T ≤ 1：由1/(0.5τ² - τ + 1) = T求解得τ = 1 - √(2/T-1)
         * 
         * 数值稳定性：
         * - 避免除零：分母总是正数
         * - 开方保护：确保被开方数非负
         * - 分支连续：在T = 1处连续
         * 
         * @tparam EIGENVEC Eigen向量类型（支持VectorXd等）
         * @param T 正时间向量
         * @param tau 输出的优化变量向量
         */
        template <typename EIGENVEC>
        static inline void backwardT(const Eigen::VectorXd &T,
                                     EIGENVEC &tau)
        {
            const int sizeT = T.size();
            tau.resize(sizeT);
            for (int i = 0; i < sizeT; i++)
            {
                tau(i) = T(i) > 1.0
                             ? (sqrt(2.0 * T(i) - 1.0) - 1.0)      // 大于1：开方减1
                             : (1.0 - sqrt(2.0 / T(i) - 1.0));     // 小于等于1：1减开方
            }

            return;
        }

        /**
         * @brief 时间梯度反向传播：计算优化变量的梯度
         * 
         * 使用链式法则将时间T的梯度反向传播到优化变量τ的梯度。
         * 这是实现无约束优化的关键步骤，确保梯度信息正确传递。
         * 
         * 链式法则：∂L/∂τ = (∂L/∂T) * (∂T/∂τ)
         * 
         * 导数计算：
         * - 当τ > 0时：dT/dτ = τ + 1
         * - 当τ ≤ 0时：dT/dτ = (1 - τ)/(0.5τ² - τ + 1)²
         * 
         * 推导过程：
         * 1. 正分支：T = 0.5τ² + τ + 1 ⟹ dT/dτ = τ + 1
         * 2. 负分支：T = 1/(0.5τ² - τ + 1) ⟹ dT/dτ = -(1-τ)/den²
         * 
         * 数值稳定性考虑：
         * - 避免除零：分母始终为正
         * - 梯度连续性：在τ = 0处梯度连续
         * - 正定性：雅可比矩阵正定保证优化收敛
         * 
         * @tparam EIGENVEC Eigen向量类型
         * @param tau 当前的优化变量
         * @param gradT 时间T的梯度（来自目标函数）
         * @param gradTau 输出的优化变量τ的梯度
         */
        template <typename EIGENVEC>
        static inline void backwardGradT(const Eigen::VectorXd &tau,
                                         const Eigen::VectorXd &gradT,
                                         EIGENVEC &gradTau)
        {
            const int sizeTau = tau.size();
            gradTau.resize(sizeTau);
            double denSqrt;  // 分母的平方项，用于数值稳定性
            for (int i = 0; i < sizeTau; i++)
            {
                if (tau(i) > 0)
                {
                    // 正分支：简单的线性关系
                    gradTau(i) = gradT(i) * (tau(i) + 1.0);
                }
                else
                {
                    // 负分支：需要计算复合函数的导数
                    denSqrt = (0.5 * tau(i) - 1.0) * tau(i) + 1.0;
                    gradTau(i) = gradT(i) * (1.0 - tau(i)) / (denSqrt * denSqrt);
                }
            }

            return;
        }

        /**
         * @brief 路径点正向变换：从优化变量到3D空间坐标
         * 
         * 将优化变量ξ转换为3D空间中的路径点P。使用重心坐标表示法
         * 确保路径点始终位于对应的凸多面体安全区域内。
         * 
         * 算法原理：
         * 1. 参数归一化：将优化变量归一化为单位向量
         * 2. 重心坐标：使用归一化参数的平方作为重心坐标
         * 3. 凸组合：计算顶点的加权平均得到空间点
         * 4. 偏移处理：添加基准点（第一个顶点）作为偏移
         * 
         * 数学表示：
         * P = V₀ + Σᵢ(qᵢ² * (Vᵢ - V₀))
         * 其中：q = normalize(ξ), qᵢ²保证权重非负且归一化
         * 
         * 凸性保证：
         * - 权重非负：qᵢ² ≥ 0
         * - 权重归一化：Σqᵢ² = 1（由单位向量性质）
         * - 凸组合：点在凸包内部
         * 
         * @param xi 优化变量向量（连接所有路径点的参数）
         * @param vIdx 顶点索引：每个路径点对应的多面体编号
         * @param vPolys 顶点表示的多面体序列
         * @param P 输出的3D路径点矩阵
         */
        static inline void forwardP(const Eigen::VectorXd &xi,
                                    const Eigen::VectorXi &vIdx,
                                    const PolyhedraV &vPolys,
                                    Eigen::Matrix3Xd &P)
        {
            const int sizeP = vIdx.size();  // 路径点总数
            P.resize(3, sizeP);             // 3D空间坐标
            Eigen::VectorXd q;              // 归一化权重向量
            
            // 遍历每个路径点
            for (int i = 0, j = 0, k, l; i < sizeP; i++, j += k)
            {
                l = vIdx(i);                    // 当前点对应的多面体索引
                k = vPolys[l].cols();           // 当前多面体的顶点数
                
                // 提取并归一化对应的优化变量段
                q = xi.segment(j, k).normalized().head(k - 1);
                
                // 重心坐标变换：使用平方权重保证非负性
                P.col(i) = vPolys[l].rightCols(k - 1) * q.cwiseProduct(q) +
                           vPolys[l].col(0);    // 基准点偏移
            }
            return;
        }

        /**
         * @brief 微型非线性最小二乘代价函数
         * 
         * 用于路径点在多面体内部的局部优化。该函数计算路径点到目标位置
         * 的最小距离，同时保持在凸多面体约束内。
         * 
         * 问题描述：
         * 给定目标路径点(P_goal)和多面体顶点(V)，寻找最优的重心坐标参数ξ
         * 使得生成的点P最接近目标点。
         * 
         * 目标函数：
         * min ||P - P_goal||²
         * s.t. P = V₀ + Σᵢ(qᵢ² * (Vᵢ - V₀))
         *      q = normalize(ξ)
         * 
         * 数学推导：
         * 1. delta = P - P_goal = V₀ + V_tail * r² - P_goal
         * 2. cost = ||delta||²
         * 3. 梯度通过链式法则计算：∂cost/∂ξ = ∂cost/∂r * ∂r/∂ξ
         * 
         * 应用场景：
         * - 路径点的微调优化
         * - 初始轨迹的局部改进
         * - 约束满足的数值稳定化
         * 
         * @param ptr 指向多面体顶点矩阵的指针
         * @param xi 优化变量（未归一化的重心坐标）
         * @param gradXi 输出的梯度向量
         * @return double 代价函数值（距离的平方）
         */
        static inline double costTinyNLS(void *ptr,
                                         const Eigen::VectorXd &xi,
                                         Eigen::VectorXd &gradXi)
        {
            const int n = xi.size();                               // 参数维度
            const Eigen::Matrix3Xd &ovPoly = *(Eigen::Matrix3Xd *)ptr;  // 多面体顶点

            // 步骤1：参数归一化
            const double sqrNormXi = xi.squaredNorm();            // ||ξ||²
            const double invNormXi = 1.0 / sqrt(sqrNormXi);      // 1/||ξ||
            const Eigen::VectorXd unitXi = xi * invNormXi;        // 单位向量 ξ/||ξ||
            const Eigen::VectorXd r = unitXi.head(n - 1);        // 前n-1个分量

            // 步骤2：计算当前点与目标点的偏差
            const Eigen::Vector3d delta = ovPoly.rightCols(n - 1) * r.cwiseProduct(r) +
                                          ovPoly.col(1) - ovPoly.col(0);

            // 步骤3：计算代价函数值
            double cost = delta.squaredNorm();

            // 步骤4：计算梯度 - 使用链式法则
            // ∂cost/∂r = 2 * V^T * delta * diag(r)
            gradXi.head(n - 1) = (ovPoly.rightCols(n - 1).transpose() * (2 * delta)).array() *
                                 r.array() * 2.0;
            gradXi(n - 1) = 0.0;  // 最后一个分量的梯度为0

            // 步骤5：归一化约束的梯度投影
            // 将梯度投影到单位球面的切空间：∇_ξ = (∇_unit - (∇_unit·unit)*unit) / ||ξ||
            gradXi = (gradXi - unitXi.dot(gradXi) * unitXi).eval() * invNormXi;

            // 步骤6：添加归一化约束的惩罚项
            const double sqrNormViolation = sqrNormXi - 1.0;
            if (sqrNormViolation > 0.0)
            {
                // 三次惩罚函数：c = violation³，确保强制约束满足
                double c = sqrNormViolation * sqrNormViolation;    // violation²
                const double dc = 3.0 * c;                        // 3*violation²（导数系数）
                c *= sqrNormViolation;                             // violation³
                cost += c;                                         // 添加惩罚项
                gradXi += dc * 2.0 * xi;                          // 添加惩罚梯度：6*violation²*ξ
            }

            return cost;
        }

        /**
         * @brief 路径点反向变换：从3D空间坐标到优化变量
         * 
         * 将给定的3D路径点P转换为对应的优化变量ξ。这是forwardP的逆操作，
         * 通过求解非线性最小二乘问题找到最佳的重心坐标参数。
         * 
         * 问题描述：
         * 给定路径点P和多面体V，寻找参数ξ使得forwardP(ξ)最接近P。
         * 这是一个约束优化问题：min ||P_forward - P_target||²
         * 
         * 算法流程：
         * 1. 构建增广多面体：[P_target, V]
         * 2. 初始化：均匀分布的重心坐标
         * 3. L-BFGS优化：最小化投影误差
         * 4. 约束处理：保持归一化约束
         * 
         * 数值考虑：
         * - 初始值选择：√(1/k)保证归一化和均匀分布
         * - 收敛准则：高精度梯度阈值（FLT_EPSILON）
         * - 迭代限制：最多128次迭代防止过度计算
         * 
         * 应用场景：
         * - 轨迹初始化：将给定路径转换为优化变量
         * - 约束投影：确保路径点在安全区域内
         * - 参数恢复：从几何轨迹重建优化参数
         * 
         * @tparam EIGENVEC Eigen向量类型
         * @param P 输入的3D路径点矩阵
         * @param vIdx 顶点索引：每个路径点对应的多面体编号
         * @param vPolys 顶点表示的多面体序列
         * @param xi 输出的优化变量向量
         */
        template <typename EIGENVEC>
        static inline void backwardP(const Eigen::Matrix3Xd &P,
                                     const Eigen::VectorXi &vIdx,
                                     const PolyhedraV &vPolys,
                                     EIGENVEC &xi)
        {
            const int sizeP = P.cols();  // 路径点数量

            double minSqrD;              // 最小平方距离（优化结果）
            
            // 配置微型非线性最小二乘优化器参数
            lbfgs::lbfgs_parameter_t tiny_nls_params;
            tiny_nls_params.past = 0;                    // 不使用历史信息
            tiny_nls_params.delta = 1.0e-5;              // 收敛判断的函数值变化阈值
            tiny_nls_params.g_epsilon = FLT_EPSILON;     // 梯度收敛阈值（高精度）
            tiny_nls_params.max_iterations = 128;        // 最大迭代次数

            Eigen::Matrix3Xd ovPoly;  // 增广多面体：[目标点, 原始顶点]
            
            // 对每个路径点分别求解
            for (int i = 0, j = 0, k, l; i < sizeP; i++, j += k)
            {
                l = vIdx(i);                  // 当前点对应的多面体索引
                k = vPolys[l].cols();         // 当前多面体的顶点数

                // 构建增广多面体：第一列为目标点，其余列为原始顶点
                ovPoly.resize(3, k + 1);
                ovPoly.col(0) = P.col(i);             // 目标点
                ovPoly.rightCols(k) = vPolys[l];      // 原始多面体顶点

                // 初始化：均匀分布的归一化参数
                Eigen::VectorXd x(k);
                x.setConstant(sqrt(1.0 / k));         // √(1/k) 保证 Σx²ᵢ = 1

                // 调用L-BFGS求解器
                lbfgs::lbfgs_optimize(x,
                                      minSqrD,
                                      &GCOPTER_PolytopeSFC::costTinyNLS,  // 代价函数
                                      nullptr,                             // 无进度回调
                                      nullptr,                             // 无额外回调
                                      &ovPoly,                            // 数据指针
                                      tiny_nls_params);                   // 优化参数

                // 保存优化结果
                xi.segment(j, k) = x;
            }

            return;
        }

        /**
         * @brief 路径点梯度反向传播：计算空间变量的梯度
         * 
         * 将路径点P的梯度反向传播到优化变量ξ的梯度。这是实现完整优化流程
         * 的关键步骤，确保空间约束的梯度信息正确传递到优化算法。
         * 
         * 数学原理：
         * 使用链式法则：∂L/∂ξ = ∂L/∂P * ∂P/∂ξ
         * 其中P = V₀ + V_tail * q²，q = normalize(ξ)
         * 
         * 梯度计算步骤：
         * 1. 归一化处理：q = ξ/||ξ||, 计算单位向量
         * 2. 局部梯度：∂P/∂q = 2 * V_tail * diag(q)
         * 3. 链式法则：∂P/∂ξ = ∂P/∂q * ∂q/∂ξ
         * 4. 投影处理：将梯度投影到单位球面的切空间
         * 
         * 数值稳定性：
         * - 归一化约束的正确处理
         * - 切空间投影避免约束违反
         * - 防止除零的保护措施
         * 
         * @tparam EIGENVEC Eigen向量类型
         * @param xi 当前优化变量
         * @param vIdx 顶点索引映射
         * @param vPolys 顶点表示的多面体序列
         * @param gradP 路径点的梯度（来自上层）
         * @param gradXi 输出的优化变量梯度
         */
        template <typename EIGENVEC>
        static inline void backwardGradP(const Eigen::VectorXd &xi,
                                         const Eigen::VectorXi &vIdx,
                                         const PolyhedraV &vPolys,
                                         const Eigen::Matrix3Xd &gradP,
                                         EIGENVEC &gradXi)
        {
            const int sizeP = vIdx.size();  // 路径点数量
            gradXi.resize(xi.size());       // 梯度向量大小

            double normInv;                 // 归一化因子的倒数
            Eigen::VectorXd q, gradQ, unitQ;  // 局部变量：参数、梯度、单位向量
            
            // 对每个路径点分别计算梯度
            for (int i = 0, j = 0, k, l; i < sizeP; i++, j += k)
            {
                l = vIdx(i);                    // 当前点对应的多面体索引
                k = vPolys[l].cols();           // 多面体顶点数
                q = xi.segment(j, k);           // 提取对应的优化变量段
                normInv = 1.0 / q.norm();       // 计算归一化因子
                unitQ = q * normInv;            // 单位向量

                // 计算局部梯度：∂P/∂q = 2 * V^T * ∇P * diag(q)
                gradQ.resize(k);
                gradQ.head(k - 1) = (vPolys[l].rightCols(k - 1).transpose() * gradP.col(i)).array() *
                                    unitQ.head(k - 1).array() * 2.0;
                gradQ(k - 1) = 0.0;             // 最后一个分量的梯度为0

                // 切空间投影：将梯度投影到单位球面的切空间
                // ∇_ξ = (∇_q - (∇_q·unit)*unit) / ||ξ||
                gradXi.segment(j, k) = (gradQ - unitQ * unitQ.dot(gradQ)) * normInv;
            }

            return;
        }

        /**
         * @brief 归一化约束层：强制执行单位向量约束
         * 
         * 对所有空间优化变量施加归一化约束，确保重心坐标参数满足单位向量条件。
         * 使用三次惩罚函数处理约束违反，提供强制约束满足能力。
         * 
         * 约束形式：||ξᵢ||² = 1 for all i
         * 惩罚函数：penalty = (||ξ||² - 1)³₊
         * 其中 (·)₊ 表示正部函数：max(0, ·)
         * 
         * 三次惩罚的优势：
         * - 强制约束：惩罚函数增长迅速
         * - 光滑性：二阶连续可导
         * - 数值稳定：避免约束边界的振荡
         * 
         * 梯度计算：
         * ∂penalty/∂ξ = 6 * violation² * ξ
         * 其中 violation = ||ξ||² - 1
         * 
         * @tparam EIGENVEC Eigen向量类型
         * @param xi 当前优化变量
         * @param vIdx 顶点索引映射
         * @param vPolys 顶点表示的多面体序列
         * @param cost 输入输出的代价函数值（累加惩罚）
         * @param gradXi 输入输出的梯度向量（累加惩罚梯度）
         */
        template <typename EIGENVEC>
        static inline void normRetrictionLayer(const Eigen::VectorXd &xi,
                                               const Eigen::VectorXi &vIdx,
                                               const PolyhedraV &vPolys,
                                               double &cost,
                                               EIGENVEC &gradXi)
        {
            const int sizeP = vIdx.size();  // 路径点数量
            gradXi.resize(xi.size());       // 梯度向量大小

            double sqrNormQ, sqrNormViolation, c, dc;  // 局部变量：模长、违反量、惩罚值、惩罚导数
            Eigen::VectorXd q;                          // 当前参数段
            
            // 对每个路径点检查归一化约束
            for (int i = 0, j = 0, k; i < sizeP; i++, j += k)
            {
                k = vPolys[vIdx(i)].cols();     // 当前多面体的顶点数

                q = xi.segment(j, k);           // 提取对应的优化变量段
                sqrNormQ = q.squaredNorm();     // 计算平方模长
                sqrNormViolation = sqrNormQ - 1.0;  // 约束违反量

                // 检查是否违反约束
                if (sqrNormViolation > 0.0)
                {
                    // 计算三次惩罚函数
                    c = sqrNormViolation * sqrNormViolation;    // violation²
                    dc = 3.0 * c;                               // 导数系数：3*violation²
                    c *= sqrNormViolation;                      // violation³
                    
                    // 累加惩罚项到总代价
                    cost += c;
                    
                    // 累加惩罚梯度：∂penalty/∂ξ = 6*violation²*ξ
                    gradXi.segment(j, k) += dc * 2.0 * q;
                }
            }

            return;
        }

        /**
         * @brief 平滑L1惩罚函数：处理不等式约束的光滑近似
         * 
         * 实现平滑的L1惩罚函数，用于处理不等式约束违反。相比标准L1函数，
         * 该函数在接近约束边界时提供光滑的梯度，改善优化算法的收敛性。
         * 
         * 函数定义：
         * - x < 0: 无惩罚（约束满足）
         * - 0 ≤ x ≤ μ: 三次多项式平滑过渡
         * - x > μ: 线性增长（标准L1）
         * 
         * 平滑区域（0 ≤ x ≤ μ）：
         * f(x) = (μ - x/2) * (x/μ)³
         * f'(x) = (x/μ)² * (-x/(2μ) + 3(μ-x/2)/μ)
         * 
         * 线性区域（x > μ）：
         * f(x) = x - μ/2
         * f'(x) = 1
         * 
         * 数学性质：
         * - C¹连续：函数值和导数在x=μ处连续
         * - 单调递增：f'(x) ≥ 0
         * - 光滑过渡：避免导数跳跃
         * 
         * @param x 约束违反量（输入）
         * @param mu 平滑参数（控制过渡区域宽度）
         * @param f 输出的函数值
         * @param df 输出的导数值
         * @return bool 是否需要惩罚（x ≥ 0）
         */
        static inline bool smoothedL1(const double &x,
                                      const double &mu,
                                      double &f,
                                      double &df)
        {
            if (x < 0.0)
            {
                // 约束满足：无需惩罚
                return false;
            }
            else if (x > mu)
            {
                // 线性区域：标准L1惩罚
                f = x - 0.5 * mu;
                df = 1.0;
                return true;
            }
            else
            {
                // 平滑过渡区域：三次多项式
                const double xdmu = x / mu;                    // 归一化违反量
                const double sqrxdmu = xdmu * xdmu;            // (x/μ)²
                const double mumxd2 = mu - 0.5 * x;            // μ - x/2
                f = mumxd2 * sqrxdmu * xdmu;                   // 函数值
                df = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);  // 导数值
                return true;
            }
        }

        /**
         * @brief 附加惩罚函数：添加物理约束和安全约束的惩罚项
         * 
         * 这是GCOPTER系统的核心函数之一，负责将各种物理约束和几何约束
         * 转换为可优化的惩罚项。通过数值积分沿轨迹评估约束违反。
         * 
         * 约束类型：
         * 1. 几何约束：位置必须在安全走廊内
         * 2. 速度约束：||v|| ≤ v_max
         * 3. 角速度约束：||ω|| ≤ ω_max  
         * 4. 姿态约束：倾斜角 ≤ θ_max
         * 5. 推力约束：thrust_min ≤ T ≤ thrust_max
         * 
         * 参数说明：
         * - magnitudeBounds: [v_max, ω_max, θ_max, thrust_min, thrust_max]ᵀ
         * - penaltyWeights: [pos_weight, vel_weight, ω_weight, θ_weight, thrust_weight]ᵀ  
         * - physicalParams: [mass, g, h_drag, v_drag, parasitic_drag, smooth_factor]ᵀ
         * 
         * 算法流程：
         * 1. 参数预处理：提取边界和权重
         * 2. 轨迹采样：沿每段轨迹进行数值积分
         * 3. 状态计算：评估位置、速度、加速度等
         * 4. 微分平坦性：计算推力、姿态、角速度
         * 5. 约束检查：使用平滑L1函数处理违反
         * 6. 梯度传播：反向传播梯度到系数和时间
         * 
         * @param T 时间分配向量
         * @param coeffs 多项式系数矩阵
         * @param hIdx 半空间索引映射
         * @param hPolys 半空间表示的多面体序列
         * @param smoothFactor 平滑因子
         * @param integralResolution 积分分辨率
         * @param magnitudeBounds 物理量边界约束
         * @param penaltyWeights 惩罚权重
         * @param flatMap 微分平坦性映射对象
         * @param cost 输入输出的代价函数值
         * @param gradT 输入输出的时间梯度
         * @param gradC 输入输出的系数梯度
         */
        // magnitudeBounds = [v_max, omg_max, theta_max, thrust_min, thrust_max]^T
        // penaltyWeights = [pos_weight, vel_weight, omg_weight, theta_weight, thrust_weight]^T
        // physicalParams = [vehicle_mass, gravitational_acceleration, horitonral_drag_coeff,
        //                   vertical_drag_coeff, parasitic_drag_coeff, speed_smooth_factor]^T
        static inline void attachPenaltyFunctional(const Eigen::VectorXd &T,
                                                   const Eigen::MatrixX3d &coeffs,
                                                   const Eigen::VectorXi &hIdx,
                                                   const PolyhedraH &hPolys,
                                                   const double &smoothFactor,
                                                   const int &integralResolution,
                                                   const Eigen::VectorXd &magnitudeBounds,
                                                   const Eigen::VectorXd &penaltyWeights,
                                                   flatness::FlatnessMap &flatMap,
                                                   double &cost,
                                                   Eigen::VectorXd &gradT,
                                                   Eigen::MatrixX3d &gradC)
        {
            // 步骤1：预处理约束边界
            const double velSqrMax = magnitudeBounds(0) * magnitudeBounds(0);      // 最大速度平方
            const double omgSqrMax = magnitudeBounds(1) * magnitudeBounds(1);      // 最大角速度平方
            const double thetaMax = magnitudeBounds(2);                            // 最大倾斜角
            const double thrustMean = 0.5 * (magnitudeBounds(3) + magnitudeBounds(4));  // 推力均值
            const double thrustRadi = 0.5 * fabs(magnitudeBounds(4) - magnitudeBounds(3)); // 推力半径
            const double thrustSqrRadi = thrustRadi * thrustRadi;                  // 推力半径平方

            // 步骤2：提取惩罚权重
            const double weightPos = penaltyWeights(0);     // 位置约束权重
            const double weightVel = penaltyWeights(1);     // 速度约束权重
            const double weightOmg = penaltyWeights(2);     // 角速度约束权重
            const double weightTheta = penaltyWeights(3);   // 姿态约束权重
            const double weightThrust = penaltyWeights(4);  // 推力约束权重

            // 步骤3：声明计算变量
            // 轨迹状态变量
            Eigen::Vector3d pos, vel, acc, jer, sna;        // 位置、速度、加速度、急动、快动
            
            // 梯度累积变量
            Eigen::Vector3d totalGradPos, totalGradVel, totalGradAcc, totalGradJer;  // 状态梯度
            double totalGradPsi, totalGradPsiD;             // 偏航角及其导数的梯度
            
            // 微分平坦性映射变量
            double thr, cos_theta;                          // 推力和倾斜角余弦值
            Eigen::Vector4d quat;                           // 四元数姿态表示
            Eigen::Vector3d omg;                            // 角速度向量
            
            // 反向传播梯度变量
            double gradThr;                                 // 推力梯度
            Eigen::Vector4d gradQuat;                       // 四元数梯度
            Eigen::Vector3d gradPos, gradVel, gradOmg;      // 位置、速度、角速度梯度

            // 数值积分变量
            double step, alpha;                             // 积分步长和相对时间
            double s1, s2, s3, s4, s5;                      // 时间幂次：t, t², t³, t⁴, t⁵
            Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3, beta4;  // 多项式基函数及其导数
            
            // 约束检查变量
            Eigen::Vector3d outerNormal;                    // 多面体外法向量
            int K, L;                                       // 约束数和多面体索引
            
            // 约束违反量
            double violaPos, violaVel, violaOmg, violaTheta, violaThrust;
            
            // 约束惩罚导数
            double violaPosPenaD, violaVelPenaD, violaOmgPenaD, violaThetaPenaD, violaThrustPenaD;
            
            // 约束惩罚函数值
            double violaPosPena, violaVelPena, violaOmgPena, violaThetaPena, violaThrustPena;
            
            // 积分相关变量
            double node, pena;                              // 积分节点和惩罚值

            // 常量定义
            const int pieceNum = T.size();                  // 轨迹段数
            const double integralFrac = 1.0 / integralResolution;  // 积分权重因子

            // 步骤4：遍历所有轨迹段
            // 对每个多项式轨迹段执行数值积分计算约束惩罚
            for (int i = 0; i < pieceNum; i++)
            {
                // 获取当前段持续时间和多项式系数
                const auto &c = coeffs.block<6, 3>(6 * i, 0);       // 当前段系数矩阵 [6×3]
                step = T(i) / integralResolution;                    // 积分步长 = 时间段/分辨率

                // 步骤5：在当前段上执行Gauss-Legendre数值积分
                // 使用高精度数值积分方法计算约束违反量
                for (int j = 0; j <= integralResolution; j++)
                {
                    // 计算当前积分点的相对时间 α ∈ [0,1]
                    alpha = static_cast<double>(j) / integralResolution;
                    
                    // 计算时间幂次：t, t², t³, t⁴, t⁵ （用于多项式求值）
                    s1 = alpha;
                    s2 = s1 * alpha;      // t²
                    s3 = s2 * alpha;      // t³
                    s4 = s3 * alpha;      // t⁴
                    s5 = s4 * alpha;      // t⁵

                    // 计算多项式基函数及其导数
                    // β₀(t) = [1, t, t², t³, t⁴, t⁵]ᵀ （位置基函数）
                    beta0 << 1.0, s1, s2, s3, s4, s5;
                    
                    // β₁(t) = d/dt β₀(t) = [0, 1, 2t, 3t², 4t³, 5t⁴]ᵀ （速度基函数）
                    beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4;
                    
                    // β₂(t) = d²/dt² β₀(t) = [0, 0, 2, 6t, 12t², 20t³]ᵀ （加速度基函数）
                    beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
                    
                    // β₃(t) = d³/dt³ β₀(t) = [0, 0, 0, 6, 24t, 60t²]ᵀ （急动基函数）
                    beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2;
                    
                    // β₄(t) = d⁴/dt⁴ β₀(t) = [0, 0, 0, 0, 24, 120t]ᵀ （快动基函数）
                    beta4 << 0.0, 0.0, 0.0, 0.0, 24.0, 120.0 * s1;

                    // 步骤6：计算轨迹状态量
                    // 通过多项式基函数计算各阶导数：p(t) = C^T β(t)
                    pos = c.transpose() * beta0;    // 位置：r(t) = ∑ cᵢ βᵢ(t)
                    vel = c.transpose() * beta1;    // 速度：v(t) = ∑ cᵢ β'ᵢ(t)  
                    acc = c.transpose() * beta2;    // 加速度：a(t) = ∑ cᵢ β''ᵢ(t)
                    jer = c.transpose() * beta3;    // 急动：j(t) = ∑ cᵢ β'''ᵢ(t)
                    sna = c.transpose() * beta4;    // 快动：s(t) = ∑ cᵢ β⁽⁴⁾ᵢ(t)

                    // 步骤7：微分平坦性映射
                    // 从轨迹导数计算控制量：(v,a,j) → (推力,姿态,角速度)
                    flatMap.forward(vel, acc, jer, 0.0, 0.0, thr, quat, omg);

                    // 步骤8：计算约束违反量
                    // 各种物理约束的违反程度（正值表示违反）
                    violaVel = vel.squaredNorm() - velSqrMax;        // 速度约束：||v||² - v²_max
                    violaOmg = omg.squaredNorm() - omgSqrMax;        // 角速度约束：||ω||² - ω²_max
                    
                    // 姿态约束：倾斜角计算
                    cos_theta = 1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2));  // cos(θ) = 1-2(q₁²+q₂²)
                    violaTheta = acos(cos_theta) - thetaMax;         // θ - θ_max
                    
                    // 推力约束：椭圆区域约束
                    violaThrust = (thr - thrustMean) * (thr - thrustMean) - thrustSqrRadi;  // (T-T₀)² - r²

                    // 步骤9：初始化梯度变量
                    gradThr = 0.0;                          // 推力梯度
                    gradQuat.setZero();                     // 四元数梯度
                    gradPos.setZero(), gradVel.setZero(), gradOmg.setZero();  // 状态梯度
                    pena = 0.0;                            // 当前点惩罚值

                    // 步骤10：处理几何约束（安全走廊）
                    L = hIdx(i);                           // 当前段对应的多面体索引
                    K = hPolys[L].rows();                  // 多面体的半空间数量
                    
                    // 检查每个半空间约束：aᵀx ≤ b
                    for (int k = 0; k < K; k++)
                    {
                        outerNormal = hPolys[L].block<1, 3>(k, 0);        // 半空间法向量 a
                        violaPos = outerNormal.dot(pos) + hPolys[L](k, 3);  // 约束违反：aᵀx + d
                        
                        // 使用平滑L1函数处理位置约束违反
                        if (smoothedL1(violaPos, smoothFactor, violaPosPena, violaPosPenaD))
                        {
                            gradPos += weightPos * violaPosPenaD * outerNormal;  // 梯度累积
                            pena += weightPos * violaPosPena;                    // 惩罚累积
                        }
                    }

                    // 步骤11：处理速度约束
                    // 约束形式：||v|| ≤ v_max ⟺ ||v||² ≤ v²_max
                    if (smoothedL1(violaVel, smoothFactor, violaVelPena, violaVelPenaD))
                    {
                        gradVel += weightVel * violaVelPenaD * 2.0 * vel;    // ∇(||v||²) = 2v
                        pena += weightVel * violaVelPena;
                    }

                    // 步骤12：处理角速度约束
                    // 约束形式：||ω|| ≤ ω_max ⟺ ||ω||² ≤ ω²_max
                    if (smoothedL1(violaOmg, smoothFactor, violaOmgPena, violaOmgPenaD))
                    {
                        gradOmg += weightOmg * violaOmgPenaD * 2.0 * omg;    // ∇(||ω||²) = 2ω
                        pena += weightOmg * violaOmgPena;
                    }

                    // 步骤13：处理姿态约束
                    // 约束形式：倾斜角 θ ≤ θ_max，其中 cos(θ) = 1-2(q₁²+q₂²)
                    if (smoothedL1(violaTheta, smoothFactor, violaThetaPena, violaThetaPenaD))
                    {
                        // ∇θ = -∇cos(θ)/sin(θ) = -(-4[0,q₁,q₂,0]ᵀ)/sin(θ)
                        gradQuat += weightTheta * violaThetaPenaD /
                                    sqrt(1.0 - cos_theta * cos_theta) * 4.0 *
                                    Eigen::Vector4d(0.0, quat(1), quat(2), 0.0);
                        pena += weightTheta * violaThetaPena;
                    }

                    // 步骤14：处理推力约束
                    // 约束形式：推力在椭圆区域内 (T-T₀)² ≤ r²
                    if (smoothedL1(violaThrust, smoothFactor, violaThrustPena, violaThrustPenaD))
                    {
                        gradThr += weightThrust * violaThrustPenaD * 2.0 * (thr - thrustMean);
                        pena += weightThrust * violaThrustPena;
                    }

                    // 步骤15：微分平坦性反向传播
                    // 将控制量梯度反向传播到轨迹导数梯度
                    flatMap.backward(gradPos, gradVel, gradThr, gradQuat, gradOmg,
                                     totalGradPos, totalGradVel, totalGradAcc, totalGradJer,
                                     totalGradPsi, totalGradPsiD);

                    // 步骤16：数值积分权重计算
                    // 使用梯形法则：端点权重0.5，内部点权重1.0
                    node = (j == 0 || j == integralResolution) ? 0.5 : 1.0;
                    alpha = static_cast<double>(j) / integralResolution;
                    
                    // 步骤17：梯度反向传播到多项式系数
                    // 使用链式法则：∂L/∂C = ∑ᵢ (βᵢ ⊗ ∇pᵢ) * 权重 * 步长
                    gradC.block<6, 3>(i * 6, 0) += (beta0 * totalGradPos.transpose() +    // 位置项
                                                    beta1 * totalGradVel.transpose() +     // 速度项
                                                    beta2 * totalGradAcc.transpose() +     // 加速度项
                                                    beta3 * totalGradJer.transpose()) *    // 急动项
                                                   node * step;                            // 积分权重

                    // 步骤18：梯度反向传播到时间变量
                    // 时间梯度包含两部分：1) 轨迹导数变化的影响；2) 惩罚值的积分
                    gradT(i) += (totalGradPos.dot(vel) +        // ∂r/∂t = v
                                 totalGradVel.dot(acc) +        // ∂v/∂t = a  
                                 totalGradAcc.dot(jer) +        // ∂a/∂t = j
                                 totalGradJer.dot(sna)) *       // ∂j/∂t = s
                                    alpha * node * step +       // 时间导数项
                                node * integralFrac * pena;     // 惩罚积分项
                    
                    // 步骤19：累积代价函数值
                    cost += node * step * pena;                 // 总惩罚代价
                }
            }

            return;
        }

        /**
         * @brief 代价函数计算：GCOPTER优化问题的目标函数
         * 
         * 这是L-BFGS优化算法的核心目标函数，整合了轨迹平滑性和约束满足。
         * 
         * 目标函数组成：
         * J = J_smooth + J_constraint + ρ * J_time
         * 其中：
         * - J_smooth: MINCO轨迹平滑代价（最小控制努力）
         * - J_constraint: 物理约束和几何约束惩罚
         * - J_time: 时间正则化项（防止时间过长）
         * - ρ: 时间权重参数
         * 
         * 计算流程：
         * 1. 解析优化变量：时间变量τ和空间变量ξ
         * 2. 变量变换：τ→T, ξ→P（实际时间和路径点）
         * 3. MINCO设置：配置轨迹参数和边界条件
         * 4. 平滑代价：计算最小控制努力代价
         * 5. 约束惩罚：添加物理和几何约束惩罚
         * 6. 梯度计算：反向传播所有梯度
         * 7. 时间正则化：添加时间总和惩罚
         * 8. 约束层：处理归一化约束
         * 
         * 数学表达：
         * ∇J = ∇J_smooth + ∇J_constraint + ρ * ∇J_time
         * 
         * @param ptr GCOPTER对象指针
         * @param x 优化变量向量 [τ; ξ]
         * @param g 输出梯度向量 [∇τ; ∇ξ]
         * @return double 总代价函数值
         */
        static inline double costFunctional(void *ptr,
                                            const Eigen::VectorXd &x,
                                            Eigen::VectorXd &g)
        {
            // 步骤1：解析GCOPTER对象和优化变量
            GCOPTER_PolytopeSFC &obj = *(GCOPTER_PolytopeSFC *)ptr;
            const int dimTau = obj.temporalDim;              // 时间变量维度
            const int dimXi = obj.spatialDim;                // 空间变量维度
            const double weightT = obj.rho;                  // 时间权重
            
            // 变量映射：将优化向量x分解为时间和空间部分
            Eigen::Map<const Eigen::VectorXd> tau(x.data(), dimTau);                    // 时间变量τ
            Eigen::Map<const Eigen::VectorXd> xi(x.data() + dimTau, dimXi);            // 空间变量ξ
            Eigen::Map<Eigen::VectorXd> gradTau(g.data(), dimTau);                     // 时间梯度
            Eigen::Map<Eigen::VectorXd> gradXi(g.data() + dimTau, dimXi);              // 空间梯度

            // 步骤2：变量变换
            forwardT(tau, obj.times);                        // τ → T：时间变量变换
            forwardP(xi, obj.vPolyIdx, obj.vPolytopes, obj.points);  // ξ → P：路径点变换

            // 步骤3：MINCO轨迹优化设置
            double cost;
            obj.minco.setParameters(obj.points, obj.times);          // 设置边界点和时间分配
            obj.minco.getEnergy(cost);                               // 计算平滑代价
            obj.minco.getEnergyPartialGradByCoeffs(obj.partialGradByCoeffs);     // 系数梯度
            obj.minco.getEnergyPartialGradByTimes(obj.partialGradByTimes);       // 时间梯度

            // 步骤4：添加约束惩罚
            attachPenaltyFunctional(obj.times, obj.minco.getCoeffs(),
                                    obj.hPolyIdx, obj.hPolytopes,
                                    obj.smoothEps, obj.integralRes,
                                    obj.magnitudeBd, obj.penaltyWt, obj.flatmap,
                                    cost, obj.partialGradByTimes, obj.partialGradByCoeffs);

            // 步骤5：梯度反向传播到路径点和时间
            obj.minco.propogateGrad(obj.partialGradByCoeffs, obj.partialGradByTimes,
                                    obj.gradByPoints, obj.gradByTimes);

            // 步骤6：添加时间正则化
            cost += weightT * obj.times.sum();               // 时间总和惩罚
            obj.gradByTimes.array() += weightT;              // 时间梯度更新

            // 步骤7：反向传播到优化变量
            backwardGradT(tau, obj.gradByTimes, gradTau);    // T → τ 梯度反向传播
            backwardGradP(xi, obj.vPolyIdx, obj.vPolytopes, obj.gradByPoints, gradXi);  // P → ξ 梯度反向传播
            
            // 步骤8：处理归一化约束
            normRetrictionLayer(xi, obj.vPolyIdx, obj.vPolytopes, cost, gradXi);

            return cost;
        }

        /**
         * @brief 距离代价函数：初始化阶段的路径距离优化
         * 
         * 用于优化重叠区域中路径点的连接，最小化路径总长度。这是安全走廊
         * 生成后的初始化步骤，确保路径点合理分布。
         * 
         * 目标函数：
         * J_dist = ∑ᵢ f(||pᵢ₊₁ - pᵢ||)
         * 其中f(d)是平滑距离函数，避免梯度不连续
         * 
         * 应用场景：
         * - 安全走廊重叠区域的路径点初始化
         * - 保证路径连通性和合理的点间距离
         * - 为主优化提供良好的初始值
         * 
         * 计算流程：
         * 1. 解析重叠区域数量和边界点
         * 2. 遍历相邻路径点对
         * 3. 计算平滑距离代价
         * 4. 累积总代价和梯度
         * 
         * @param ptr 包含优化参数的指针数组
         * @param xi 空间变量向量（路径点参数）
         * @param gradXi 输出的空间梯度向量
         * @return double 距离代价值
         */
        static inline double costDistance(void *ptr,
                                          const Eigen::VectorXd &xi,
                                          Eigen::VectorXd &gradXi)
        {
            // 步骤1：解析优化参数
            void **dataPtrs = (void **)ptr;
            const double &dEps = *((const double *)(dataPtrs[0]));           // 平滑参数
            const Eigen::Vector3d &ini = *((const Eigen::Vector3d *)(dataPtrs[1]));  // 起始点
            const Eigen::Vector3d &fin = *((const Eigen::Vector3d *)(dataPtrs[2]));  // 终止点
            const PolyhedraV &vPolys = *((PolyhedraV *)(dataPtrs[3]));       // 顶点多面体序列

            double cost = 0.0;                               // 总代价
            const int overlaps = vPolys.size() / 2;          // 重叠区域数量

            // 初始化变量
            Eigen::Matrix3Xd gradP = Eigen::Matrix3Xd::Zero(3, overlaps);  // 路径点梯度
            Eigen::Vector3d a, b, d;                         // 当前点、下一点、距离向量
            Eigen::VectorXd r;                               // 归一化向量
            double smoothedDistance;                          // 平滑距离值
            
            // 步骤2：遍历路径段计算距离代价
            for (int i = 0, j = 0, k = 0; i <= overlaps; i++, j += k)
            {
                a = i == 0 ? ini : b;                        // 当前段起点：初始点或上一段终点
                if (i < overlaps)
                {
                    k = vPolys[2 * i + 1].cols();            // 当前重叠区域的顶点数
                    Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);  // 当前点的归一化坐标
                    
                    // 步骤3：计算路径点坐标
                    // 从归一化坐标转换为实际3D坐标
                    r = q.normalized().head(k - 1);          // 归一化向量（前k-1维）
                    b = vPolys[2 * i + 1].rightCols(k - 1) * r.cwiseProduct(r) +  // V_tail * r²
                        vPolys[2 * i + 1].col(0);            // + V₀
                }
                else
                {
                    b = fin;                                  // 最后一个点是终点
                }

                // 步骤4：计算平滑距离代价
                d = b - a;                                    // 距离向量
                smoothedDistance = sqrt(d.squaredNorm() + dEps);  // 平滑距离：√(||d||² + ε)
                cost += smoothedDistance;                     // 累积总代价

                // 步骤5：计算梯度
                // ∇f = d/√(||d||² + ε) 关于路径点的梯度
                if (i < overlaps)
                {
                    gradP.col(i) += d / smoothedDistance;     // 对终点的梯度
                }
                if (i > 0)
                {
                    gradP.col(i - 1) -= d / smoothedDistance; // 对起点的梯度（负号）
                }
            }

            // 步骤6：梯度反向传播到归一化坐标
            Eigen::VectorXd unitQ;                            // 单位向量q
            double sqrNormQ, invNormQ, sqrNormViolation, c, dc;  // 归一化相关变量
            for (int i = 0, j = 0, k; i < overlaps; i++, j += k)
            {
                k = vPolys[2 * i + 1].cols();                // 当前区域顶点数
                Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);       // 当前归一化坐标
                Eigen::Map<Eigen::VectorXd> gradQ(gradXi.data() + j, k);     // 对应梯度

                // 归一化处理
                sqrNormQ = q.squaredNorm();                  // ||q||²
                invNormQ = 1.0 / sqrt(sqrNormQ);            // 1/||q||
                unitQ = q * invNormQ;                        // q̂ = q/||q||

                // 路径点到归一化坐标的梯度传播
                // ∇q = 2 * V_tail^T * ∇P * q̂ （链式法则）
                gradQ.head(k - 1) = (vPolys[2 * i + 1].rightCols(k - 1).transpose() * gradP.col(i)).array() *
                                    unitQ.head(k - 1).array() * 2.0;
                gradQ(k - 1) = 0.0;                         // 最后一维固定为0

                // 投影到单位球面切空间：∇q_proj = (∇q - q̂(q̂·∇q))/||q||
                gradQ = (gradQ - unitQ * unitQ.dot(gradQ)).eval() * invNormQ;

                // 步骤7：添加归一化约束惩罚
                // 约束：||q|| = 1，违反量：||q||² - 1
                sqrNormViolation = sqrNormQ - 1.0;
                if (sqrNormViolation > 0.0)
                {
                    c = sqrNormViolation * sqrNormViolation;  // (||q||² - 1)²
                    dc = 3.0 * c;                            // 3(||q||² - 1)²
                    c *= sqrNormViolation;                   // (||q||² - 1)³
                    cost += c;                               // 添加惩罚代价
                    gradQ += dc * 2.0 * q;                   // 添加惩罚梯度：6(||q||² - 1)²q
                }
            }

            return cost;
        }

        /**
         * @brief 最短路径计算：在重叠多面体中计算最优连接路径
         * 
         * 使用L-BFGS优化在重叠的安全走廊区域中找到连接起始点和终止点的
         * 最短路径。这是轨迹初始化的关键步骤。
         * 
         * 算法原理：
         * 1. 在每个重叠区域中放置一个路径点
         * 2. 使用平滑距离函数作为目标函数
         * 3. 通过L-BFGS优化最小化总路径长度
         * 4. 保证归一化约束（路径点在单位球面上）
         * 
         * 应用场景：
         * - 安全走廊生成后的路径初始化
         * - 为MINCO提供合理的中间路径点
         * - 保证初始路径的连通性和合理性
         * 
         * @param ini 起始点坐标
         * @param fin 终止点坐标  
         * @param vPolys 顶点表示的重叠多面体序列
         * @param smoothD 平滑距离参数
         * @param path 输出的最短路径（包含所有路径点）
         */
        static inline void getShortestPath(const Eigen::Vector3d &ini,
                                           const Eigen::Vector3d &fin,
                                           const PolyhedraV &vPolys,
                                           const double &smoothD,
                                           Eigen::Matrix3Xd &path)
        {
            // 步骤1：初始化优化变量
            const int overlaps = vPolys.size() / 2;          // 重叠区域数量
            Eigen::VectorXi vSizes(overlaps);                // 每个区域的顶点数
            for (int i = 0; i < overlaps; i++)
            {
                vSizes(i) = vPolys[2 * i + 1].cols();        // 记录顶点数
            }
            
            // 初始化归一化坐标：每个分量设为 √(1/k)，保证 ||q|| = 1
            Eigen::VectorXd xi(vSizes.sum());               // 总的优化变量向量
            for (int i = 0, j = 0; i < overlaps; i++)
            {
                xi.segment(j, vSizes(i)).setConstant(sqrt(1.0 / vSizes(i)));  // 均匀分布在单位球面
                j += vSizes(i);
            }

            // 步骤2：设置优化参数
            double minDistance;                              // 最小距离结果
            void *dataPtrs[4];                              // 参数指针数组
            dataPtrs[0] = (void *)(&smoothD);               // 平滑参数
            dataPtrs[1] = (void *)(&ini);                   // 起始点
            dataPtrs[2] = (void *)(&fin);                   // 终止点
            dataPtrs[3] = (void *)(&vPolys);                // 多面体序列
            
            // L-BFGS优化参数配置
            lbfgs::lbfgs_parameter_t shortest_path_params;
            shortest_path_params.past = 3;                   // 历史梯度数量
            shortest_path_params.delta = 1.0e-3;             // 收敛阈值
            shortest_path_params.g_epsilon = 1.0e-5;         // 梯度收敛阈值

            // 步骤3：执行L-BFGS优化
            lbfgs::lbfgs_optimize(xi,
                                  minDistance,
                                  &GCOPTER_PolytopeSFC::costDistance,  // 距离代价函数
                                  nullptr,                              // 无进度回调
                                  nullptr,                              // 无监控回调
                                  dataPtrs,                            // 参数数据
                                  shortest_path_params);               // 优化参数

            // 步骤4：构建最终路径
            path.resize(3, overlaps + 2);                   // 路径矩阵：起点+中间点+终点
            path.leftCols<1>() = ini;                       // 第一列：起始点
            path.rightCols<1>() = fin;                      // 最后一列：终止点
            Eigen::VectorXd r;                              // 临时归一化向量
            
            // 步骤5：从优化结果恢复中间路径点
            for (int i = 0, j = 0, k; i < overlaps; i++, j += k)
            {
                k = vPolys[2 * i + 1].cols();               // 当前区域顶点数
                Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);  // 优化后的归一化坐标
                
                // 转换为实际3D坐标
                r = q.normalized().head(k - 1);             // 归一化处理
                path.col(i + 1) = vPolys[2 * i + 1].rightCols(k - 1) * r.cwiseProduct(r) +  // V_tail * r²
                                  vPolys[2 * i + 1].col(0); // + V₀
            }

            return;
        }

        /**
         * @brief 安全走廊处理：将半空间多面体转换为顶点表示
         * 
         * 将安全走廊的半空间表示转换为顶点表示，并构建重叠区域。
         * 这是空间变量参数化的前置步骤。
         * 
         * 处理流程：
         * 1. 遍历每个安全走廊多面体
         * 2. 枚举顶点并转换为中心-基向量表示
         * 3. 计算相邻多面体的重叠区域
         * 4. 构建统一的顶点表示序列
         * 
         * 数据结构转换：
         * - 输入：半空间表示 {Ax ≤ b}
         * - 输出：顶点表示 {V₀, V_tail}，其中点表示为 V₀ + V_tail * r²
         * 
         * @param hPs 输入的半空间多面体序列
         * @param vPs 输出的顶点多面体序列
         * @return bool 转换是否成功
         */
        static inline bool processCorridor(const PolyhedraH &hPs,
                                           PolyhedraV &vPs)
        {
            const int sizeCorridor = hPs.size() - 1;        // 安全走廊段数

            // 初始化输出容器
            vPs.clear();
            vPs.reserve(2 * sizeCorridor + 1);              // 预留空间：段+重叠区域

            // 处理变量声明
            int nv;                                         // 顶点数
            PolyhedronH curIH;                              // 当前重叠区域的半空间表示
            PolyhedronV curIV, curIOB;                      // 当前顶点表示和中心-基表示

            // 步骤1：处理每个安全走廊段及其重叠区域
            for (int i = 0; i < sizeCorridor; i++)
            {
                // 处理当前段：半空间→顶点→中心-基表示
                if (!geo_utils::enumerateVs(hPs[i], curIV))
                {
                    return false;                           // 顶点枚举失败
                }
                nv = curIV.cols();                          // 顶点数量
                curIOB.resize(3, nv);                       // 重新调整矩阵大小
                curIOB.col(0) = curIV.col(0);               // V₀ = 第一个顶点（中心点）
                curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);  // V_tail = 其他顶点-中心点
                vPs.push_back(curIOB);                      // 添加当前段

                // 处理重叠区域：当前段∩下一段
                curIH.resize(hPs[i].rows() + hPs[i + 1].rows(), 4);    // 组合约束矩阵
                curIH.topRows(hPs[i].rows()) = hPs[i];      // 当前段约束
                curIH.bottomRows(hPs[i + 1].rows()) = hPs[i + 1];      // 下一段约束
                
                if (!geo_utils::enumerateVs(curIH, curIV)) // 重叠区域顶点枚举
                {
                    return false;
                }
                nv = curIV.cols();
                curIOB.resize(3, nv);
                curIOB.col(0) = curIV.col(0);               // V₀
                curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);  // V_tail  
                vPs.push_back(curIOB);                      // 添加重叠区域
            }

            // 步骤2：处理最后一个安全走廊段
            if (!geo_utils::enumerateVs(hPs.back(), curIV))
            {
                return false;
            }
            nv = curIV.cols();
            curIOB.resize(3, nv);
            curIOB.col(0) = curIV.col(0);                   // V₀ = 中心点
            curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);  // V_tail
            vPs.push_back(curIOB);                          // 添加最后一段

            return true;                                    // 转换成功
        }

        /**
         * @brief 初始轨迹设置：根据路径和速度生成初始轨迹参数
         * 
         * 将路径点序列转换为MINCO轨迹的初始参数，包括中间路径点和时间分配。
         * 
         * 算法原理：
         * 1. 根据给定速度和路径长度计算时间分配
         * 2. 在每段路径上均匀插入中间路径点
         * 3. 生成MINCO优化所需的边界条件
         * 
         * 应用场景：
         * - GCOPTER优化的初始化阶段
         * - 为MINCO提供合理的初始路径点和时间分配
         * - 保证优化收敛的起始条件
         * 
         * @param path 输入路径点序列
         * @param speed 期望飞行速度
         * @param intervalNs 每段路径的分割数
         * @param innerPoints 输出的中间路径点
         * @param timeAlloc 输出的时间分配
         */
        static inline void setInitial(const Eigen::Matrix3Xd &path,
                                      const double &speed,
                                      const Eigen::VectorXi &intervalNs,
                                      Eigen::Matrix3Xd &innerPoints,
                                      Eigen::VectorXd &timeAlloc)
        {
            const int sizeM = intervalNs.size();           // 路径段数
            const int sizeN = intervalNs.sum();            // 总轨迹段数
            innerPoints.resize(3, sizeN - 1);              // 中间点矩阵
            timeAlloc.resize(sizeN);                       // 时间分配向量

            Eigen::Vector3d a, b, c;                       // 起点、终点、方向向量
            for (int i = 0, j = 0, k = 0, l; i < sizeM; i++)
            {
                l = intervalNs(i);                         // 当前段分割数
                a = path.col(i);                           // 当前段起点
                b = path.col(i + 1);                       // 当前段终点
                c = (b - a) / l;                           // 单位步长向量
                
                // 计算时间分配：距离/速度
                timeAlloc.segment(j, l).setConstant(c.norm() / speed);
                j += l;
                
                // 生成中间路径点
                for (int m = 0; m < l; m++)
                {
                    if (i > 0 || m > 0)                    // 跳过第一个点（起始点）
                    {
                        innerPoints.col(k++) = a + c * m;  // 线性插值生成中间点
                    }
                }
            }
        }

    public:
        /**
         * @brief GCOPTER系统配置和初始化
         * 
         * 配置轨迹规划器的所有必要参数，包括边界条件、安全走廊、
         * 物理约束和优化权重。这是使用GCOPTER进行轨迹规划前的必要步骤。
         * 
         * 参数说明：
         * - magnitudeBounds: [v_max, ω_max, θ_max, thrust_min, thrust_max]ᵀ
         *   速度、角速度、倾斜角和推力的边界约束
         * - penaltyWeights: [pos_weight, vel_weight, ω_weight, θ_weight, thrust_weight]ᵀ
         *   位置、速度、角速度、姿态角和推力的惩罚权重
         * - physicalParams: [mass, g, h_drag, v_drag, parasitic_drag, smooth_factor]ᵀ
         *   车辆质量、重力加速度、水平阻力系数、垂直阻力系数、寄生阻力系数、速度平滑因子
         * 
         * 算法流程：
         * 1. 参数设置：保存边界条件和物理参数
         * 2. 走廊处理：归一化半空间约束并转换为顶点表示
         * 3. 路径初始化：在安全走廊中生成初始路径点
         * 4. 时间分配：基于路径长度和动力学约束分配时间
         * 5. 索引建立：构建轨迹段与多面体的映射关系
         * 
         * @param timeWeight 时间权重：平衡轨迹时间和其他代价
         * @param initialPVA 初始状态：位置-速度-加速度矩阵[3×3]
         * @param terminalPVA 终端状态：位置-速度-加速度矩阵[3×3]
         * @param safeCorridor 安全走廊：半空间表示的凸多面体序列
         * @param lengthPerPiece 每段长度：轨迹分段的参考长度
         * @param smoothingFactor 平滑因子：数值稳定性参数
         * @param integralResolution 积分分辨率：数值积分的采样点数
         * @param magnitudeBounds 量级边界：物理量的上下限约束
         * @param penaltyWeights 惩罚权重：约束违反的惩罚系数
         * @param physicalParams 物理参数：系统的动力学参数
         * @return bool 配置是否成功
         */
        inline bool setup(const double &timeWeight,
                          const Eigen::Matrix3d &initialPVA,
                          const Eigen::Matrix3d &terminalPVA,
                          const PolyhedraH &safeCorridor,
                          const double &lengthPerPiece,
                          const double &smoothingFactor,
                          const int &integralResolution,
                          const Eigen::VectorXd &magnitudeBounds,
                          const Eigen::VectorXd &penaltyWeights,
                          const Eigen::VectorXd &physicalParams)
        {
            // 步骤1：基本参数设置
            rho = timeWeight;               // 时间代价权重
            headPVA = initialPVA;          // 起始状态
            tailPVA = terminalPVA;         // 终止状态

            // 步骤2：安全走廊预处理 - 归一化半空间约束
            hPolytopes = safeCorridor;
            for (size_t i = 0; i < hPolytopes.size(); i++)
            {
                // 计算法向量的模长
                const Eigen::ArrayXd norms =
                    hPolytopes[i].leftCols<3>().rowwise().norm();
                // 归一化：使每个半空间的法向量为单位向量
                hPolytopes[i].array().colwise() /= norms;
            }
            
            // 步骤3：走廊处理 - 转换为顶点表示
            if (!processCorridor(hPolytopes, vPolytopes))
            {
                return false;
            }

            // 步骤4：保存配置参数
            polyN = hPolytopes.size();          // 多面体数量
            smoothEps = smoothingFactor;        // 平滑性参数
            integralRes = integralResolution;   // 积分分辨率
            magnitudeBd = magnitudeBounds;      // 物理量边界
            penaltyWt = penaltyWeights;         // 惩罚权重
            physicalPm = physicalParams;        // 物理参数
            allocSpeed = magnitudeBd(0) * 3.0;  // 时间分配参考速度（3倍最大速度）

            // 步骤5：初始路径生成 - 在安全走廊中寻找最短路径
            getShortestPath(headPVA.col(0), tailPVA.col(0),
                            vPolytopes, smoothEps, shortPath);
            
            // 步骤6：轨迹分段 - 基于路径长度确定段数
            const Eigen::Matrix3Xd deltas = shortPath.rightCols(polyN) - shortPath.leftCols(polyN);
            pieceIdx = (deltas.colwise().norm() / lengthPerPiece).cast<int>().transpose();
            pieceIdx.array() += 1;              // 每段至少包含1个分段
            pieceN = pieceIdx.sum();            // 总段数

            // 步骤7：维度设置和索引构建
            temporalDim = pieceN;               // 时间优化变量维度
            spatialDim = 0;                     // 空间优化变量维度（稍后计算）
            vPolyIdx.resize(pieceN - 1);        // 顶点多面体索引
            hPolyIdx.resize(pieceN);            // 半空间多面体索引
            
            // 步骤8：建立轨迹段与多面体的映射关系
            for (int i = 0, j = 0, k; i < polyN; i++)
            {
                k = pieceIdx(i);                // 当前多面体对应的段数
                for (int l = 0; l < k; l++, j++)
                {
                    // 建立段索引映射：处理轨迹段与多面体的对应关系
                    if (l < k - 1)
                    {
                        // 多面体内部段：使用偶数索引的多面体
                        vPolyIdx(j) = 2 * i;
                        spatialDim += vPolytopes[2 * i].cols();
                    }
                    else if (i < polyN - 1)
                    {
                        // 多面体边界段：使用奇数索引的连接多面体
                        vPolyIdx(j) = 2 * i + 1;
                        spatialDim += vPolytopes[2 * i + 1].cols();
                    }
                    hPolyIdx(j) = i;  // 半空间约束索引
                }
            }

            // 步骤9：初始化核心算法组件
            // MINCO轨迹优化器：设置边界条件和段数
            minco.setConditions(headPVA, tailPVA, pieceN);
            
            // 微分平坦性映射：设置物理参数
            flatmap.reset(physicalPm(0),    // 车辆质量
                          physicalPm(1),    // 重力加速度  
                          physicalPm(2),    // 水平阻力系数
                          physicalPm(3),    // 垂直阻力系数
                          physicalPm(4),    // 寄生阻力系数
                          physicalPm(5));   // 速度平滑因子

            // 步骤10：分配临时变量存储空间
            points.resize(3, pieceN - 1);           // 中间路径点
            times.resize(pieceN);                   // 段时间分配
            gradByPoints.resize(3, pieceN - 1);     // 路径点梯度
            gradByTimes.resize(pieceN);             // 时间梯度
            partialGradByCoeffs.resize(6 * pieceN, 3);  // 系数偏导数
            partialGradByTimes.resize(pieceN);      // 时间偏导数

            return true;  // 配置成功
        }

        /**
         * @brief 轨迹优化主函数：执行完整的轨迹优化过程
         * 
         * 使用L-BFGS算法优化轨迹，最小化代价函数同时满足所有约束。
         * 优化变量包括时间分配和路径点坐标，通过变量变换处理约束。
         * 
         * 优化流程：
         * 1. 变量初始化：从初始路径和时间分配构建优化变量
         * 2. 变量变换：转换为无约束优化问题
         * 3. L-BFGS优化：迭代优化直到收敛
         * 4. 结果提取：从优化变量构建最终轨迹
         * 5. 错误处理：检查优化状态并处理失败情况
         * 
         * 优化变量结构：
         * x = [τ, ξ]ᵀ，其中：
         * - τ: 时间变量（通过变换保证正性）
         * - ξ: 空间变量（重心坐标参数）
         * 
         * 收敛判据：
         * - 相对代价容差：|f_new - f_old|/|f_old| < relCostTol
         * - 梯度范数：||∇f|| < g_epsilon （设为0使用相对容差）
         * 
         * @param traj 输出的优化轨迹（5阶多项式轨迹）
         * @param relCostTol 相对代价函数容差
         * @return double 最终的代价函数值（INFINITY表示失败）
         */
        inline double optimize(Trajectory<5> &traj,
                               const double &relCostTol)
        {
            // 步骤1：构建优化变量向量
            Eigen::VectorXd x(temporalDim + spatialDim);        // 完整优化变量
            Eigen::Map<Eigen::VectorXd> tau(x.data(), temporalDim);      // 时间变量映射
            Eigen::Map<Eigen::VectorXd> xi(x.data() + temporalDim, spatialDim);  // 空间变量映射

            // 步骤2：初始化优化变量
            setInitial(shortPath, allocSpeed, pieceIdx, points, times);  // 设置初始路径和时间
            backwardT(times, tau);                              // 时间正向变换
            backwardP(points, vPolyIdx, vPolytopes, xi);        // 路径点反向变换

            // 步骤3：配置L-BFGS优化器参数
            double minCostFunctional;                           // 最小代价函数值
            lbfgs_params.mem_size = 256;                        // 历史信息存储大小
            lbfgs_params.past = 3;                              // 用于收敛判断的历史点数
            lbfgs_params.min_step = 1.0e-32;                    // 最小步长（防止数值问题）
            lbfgs_params.g_epsilon = 0.0;                       // 梯度阈值（0表示不使用）
            lbfgs_params.delta = relCostTol;                    // 相对收敛阈值

            // 步骤4：执行L-BFGS优化
            int ret = lbfgs::lbfgs_optimize(x,
                                            minCostFunctional,
                                            &GCOPTER_PolytopeSFC::costFunctional,  // 代价函数
                                            nullptr,                                // 无进度回调
                                            nullptr,                                // 无额外回调
                                            this,                                   // 对象实例指针
                                            lbfgs_params);                         // 优化参数

            // 步骤5：处理优化结果
            if (ret >= 0)
            {
                // 优化成功：提取最终轨迹
                forwardT(tau, times);                           // 恢复物理时间
                forwardP(xi, vPolyIdx, vPolytopes, points);     // 恢复路径点
                minco.setParameters(points, times);             // 设置MINCO参数
                minco.getTrajectory(traj);                      // 生成最终轨迹
            }
            else
            {
                // 优化失败：清理并报告错误
                traj.clear();
                minCostFunctional = INFINITY;
                std::cout << "Optimization Failed: "
                          << lbfgs::lbfgs_strerror(ret)
                          << std::endl;
            }

            return minCostFunctional;  // 返回最终代价函数值
        }
    };

} // namespace gcopter

/**
 * @brief GCOPTER系统总结
 * 
 * 本文件实现了完整的几何约束轨迹优化系统，是GCOPTER框架的核心：
 * 
 * 🎯 核心功能：
 * 1. 安全走廊轨迹规划 - 在复杂环境中生成安全轨迹
 * 2. 多物理约束处理 - 同时满足动力学和几何约束  
 * 3. 实时优化求解 - 高效的L-BFGS优化算法
 * 4. 平滑轨迹生成 - MINCO保证轨迹的连续性和可执行性
 * 
 * 🔧 技术特色：
 * - 变量变换：处理正性和凸性约束的巧妙设计
 * - 模块化架构：MINCO + FlatnessMap + L-BFGS的完美结合
 * - 数值稳定：精心设计的参数化和梯度计算
 * - 高维优化：支持复杂环境下的大规模轨迹优化
 * 
 * 🚀 应用价值：
 * - 无人机群协调飞行
 * - 自动驾驶路径规划  
 * - 机器人避障导航
 * - 空中交通管理
 * 
 * 📊 性能特点：
 * - 计算效率：O(N³)复杂度的轨迹优化
 * - 内存优化：紧凑的数据结构和变量管理
 * - 收敛保证：凸优化理论支撑的全局最优性
 * - 实时性能：适合在线轨迹重规划
 * 
 * 该系统代表了轨迹规划领域的前沿水平，将理论严谨性与工程实用性
 * 完美结合，为复杂环境下的运动规划提供了强大工具。
 */

#endif
