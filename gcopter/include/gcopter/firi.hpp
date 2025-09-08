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
 * @file firi.hpp
 * @brief FIRI (Fast Inscribed Radius-based Inner approximation) 快速内接半径椭球逼近算法
 * 
 * FIRI算法是一种高效的凸包内接椭球计算方法，主要用于：
 * 1. 轨迹规划中的安全走廊生成
 * 2. 障碍物约束的椭球逼近表示
 * 3. 最大体积内接椭球(MVIE)问题求解
 * 
 * 算法特点：
 * - 快速收敛：通过迭代优化快速找到最优椭球
 * - 精确逼近：提供高质量的内接椭球近似
 * - 鲁棒性强：适用于各种复杂凸包形状
 * 
 * 这是FIRI算法的旧版本，用于临时使用
 */

/* This is an old version of FIRI for temporary usage here. */

#ifndef FIRI_HPP
#define FIRI_HPP

#include "lbfgs.hpp"    // L-BFGS优化算法
#include "sdlp.hpp"     // 简单线性规划求解器

#include <Eigen/Eigen>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <vector>

namespace firi
{

    /**
     * @brief 3x3矩阵的Cholesky分解
     * @param A 输入的3x3正定矩阵
     * @param L 输出的下三角Cholesky因子，满足A = L*L^T
     * 
     * 专门为3x3矩阵优化的Cholesky分解，避免通用算法的开销
     * 用于椭球参数化和协方差矩阵分解
     */

    inline void chol3d(const Eigen::Matrix3d &A,
                       Eigen::Matrix3d &L)
    {
        // 计算下三角Cholesky分解：A = L*L^T
        // 第一列：L11 = sqrt(A11)
        L(0, 0) = sqrt(A(0, 0));                        // L₁₁ = √A₁₁
        L(0, 1) = 0.0;                                  // 上三角部分置零
        L(0, 2) = 0.0;
        
        // 第二列：L21 = A21/L11, L22 = sqrt(A22 - L21²)
        L(1, 0) = 0.5 * (A(0, 1) + A(1, 0)) / L(0, 0); // L₂₁ = A₂₁/L₁₁
        L(1, 1) = sqrt(A(1, 1) - L(1, 0) * L(1, 0));   // L₂₂ = √(A₂₂-L₂₁²)
        L(1, 2) = 0.0;                                  // 上三角部分置零
        
        // 第三列：L31 = A31/L11, L32 = (A32-L31*L21)/L22, L33 = sqrt(A33-L31²-L32²)
        L(2, 0) = 0.5 * (A(0, 2) + A(2, 0)) / L(0, 0); // L₃₁ = A₃₁/L₁₁
        L(2, 1) = (0.5 * (A(1, 2) + A(2, 1)) - L(2, 0) * L(1, 0)) / L(1, 1); // L₃₂
        L(2, 2) = sqrt(A(2, 2) - L(2, 0) * L(2, 0) - L(2, 1) * L(2, 1));     // L₃₃
        return;
    }

    /**
     * @brief 平滑L1函数及其梯度计算
     * @param mu 平滑参数，控制平滑程度
     * @param x 输入变量
     * @param f 输出函数值
     * @param df 输出梯度值
     * @return 是否计算成功
     * 
     * 实现平滑的L1正则化函数，用于约束优化中的惩罚项
     * 当x < 0时无效，当x > mu时为线性，当0 ≤ x ≤ mu时为光滑过渡
     */
    inline bool smoothedL1(const double &mu,
                           const double &x,
                           double &f,
                           double &df)
    {
        if (x < 0.0)                                    // 无效输入
        {
            return false;
        }
        else if (x > mu)                                // 线性区域：f = x - 0.5*mu
        {
            f = x - 0.5 * mu;                           // 函数值
            df = 1.0;                                   // 梯度为1
            return true;
        }
        else                                            // 平滑区域：[0, mu]
        {
            const double xdmu = x / mu;                 // 归一化变量
            const double sqrxdmu = xdmu * xdmu;         // 平方项
            const double mumxd2 = mu - 0.5 * x;         // 辅助变量
            f = mumxd2 * sqrxdmu * xdmu;                // 三次多项式形式
            df = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu); // 导数
            return true;
        }
    }

    /**
     * @brief MVIE(最大体积内接椭球)优化的目标函数
     * @param data 优化数据指针，包含约束矩阵和参数
     * @param x 优化变量：[位置(3), 对角元素(3), 非对角元素(3)]
     * @param grad 输出梯度向量
     * @return 目标函数值
     * 
     * 目标函数组成：
     * 1. 约束违反惩罚项：通过平滑L1函数惩罚椭球超出边界
     * 2. 体积最大化项：通过负对数行列式最大化椭球体积
     * 
     * 椭球参数化：使用Cholesky分解Q = L*L^T表示椭球矩阵
     */
    inline double costMVIE(void *data,
                           const Eigen::VectorXd &x,
                           Eigen::VectorXd &grad)
    {
        // 解析数据指针结构
        const int64_t *pM = (int64_t *)data;                    // 约束数量
        const double *pSmoothEps = (double *)(pM + 1);          // 平滑参数
        const double *pPenaltyWt = pSmoothEps + 1;              // 惩罚权重
        const double *pA = pPenaltyWt + 1;                      // 约束矩阵

        const int M = *pM;                                      // 约束数量
        const double smoothEps = *pSmoothEps;                   // 平滑参数
        const double penaltyWt = *pPenaltyWt;                   // 惩罚权重
        Eigen::Map<const Eigen::MatrixX3d> A(pA, M, 3);        // 约束矩阵映射
        
        // 优化变量映射
        Eigen::Map<const Eigen::Vector3d> p(x.data());         // 椭球中心位置
        Eigen::Map<const Eigen::Vector3d> rtd(x.data() + 3);   // 对角元素开方
        Eigen::Map<const Eigen::Vector3d> cde(x.data() + 6);   // 非对角元素
        
        // 梯度变量映射
        Eigen::Map<Eigen::Vector3d> gdp(grad.data());          // 位置梯度
        Eigen::Map<Eigen::Vector3d> gdrtd(grad.data() + 3);    // 对角元素梯度
        Eigen::Map<Eigen::Vector3d> gdcde(grad.data() + 6);    // 非对角元素梯度

        double cost = 0;                                        // 初始化目标函数值
        gdp.setZero();                                          // 初始化梯度
        gdrtd.setZero();
        gdcde.setZero();

        // 构造Cholesky分解矩阵L，表示椭球形状矩阵Q = L*L^T
        Eigen::Matrix3d L;
        L(0, 0) = rtd(0) * rtd(0) + DBL_EPSILON;               // L₁₁²，添加数值稳定项
        L(0, 1) = 0.0;                                         // 上三角置零
        L(0, 2) = 0.0;
        L(1, 0) = cde(0);                                      // L₂₁
        L(1, 1) = rtd(1) * rtd(1) + DBL_EPSILON;               // L₂₂²
        L(1, 2) = 0.0;
        L(2, 0) = cde(2);                                      // L₃₁
        L(2, 1) = cde(1);                                      // L₃₂
        L(2, 2) = rtd(2) * rtd(2) + DBL_EPSILON;               // L₃₃²

        // 计算约束违反程度
        const Eigen::MatrixX3d AL = A * L;                     // 约束法向量与椭球矩阵相乘
        const Eigen::VectorXd normAL = AL.rowwise().norm();    // 每行的模长
        const Eigen::Matrix3Xd adjNormAL = (AL.array().colwise() / normAL.array()).transpose(); // 归一化
        const Eigen::VectorXd consViola = (normAL + A * p).array() - 1.0; // 约束违反量

        double c, dc;                                          // 平滑L1函数值和导数
        Eigen::Vector3d vec;                                   // 临时向量
        for (int i = 0; i < M; ++i)                           // 遍历所有约束
        {
            if (smoothedL1(smoothEps, consViola(i), c, dc))    // 计算平滑L1惩罚
            {
                cost += c;                                     // 累加惩罚项
                vec = dc * A.row(i).transpose();               // 梯度向量
                gdp += vec;                                    // 位置梯度累加
                gdrtd += adjNormAL.col(i).cwiseProduct(vec);   // 对角元素梯度
                gdcde(0) += adjNormAL(0, i) * vec(1);          // 非对角元素梯度
                gdcde(1) += adjNormAL(1, i) * vec(2);
                gdcde(2) += adjNormAL(0, i) * vec(2);
            }
        }
        
        // 应用惩罚权重
        cost *= penaltyWt;                                     // 惩罚项乘以权重
        gdp *= penaltyWt;
        gdrtd *= penaltyWt;
        gdcde *= penaltyWt;

        // 体积最大化项：-log(det(L)) = -log(L₁₁*L₂₂*L₃₃)
        cost -= log(L(0, 0)) + log(L(1, 1)) + log(L(2, 2));   // 负对数行列式
        gdrtd(0) -= 1.0 / L(0, 0);                            // 对L₁₁的梯度
        gdrtd(1) -= 1.0 / L(1, 1);                            // 对L₂₂的梯度
        gdrtd(2) -= 1.0 / L(2, 2);                            // 对L₃₃的梯度

        // 链式法则：L_ii = rtd_i², 所以∂f/∂rtd_i = ∂f/∂L_ii * 2*rtd_i
        gdrtd(0) *= 2.0 * rtd(0);                             // 对角元素梯度修正
        gdrtd(1) *= 2.0 * rtd(1);
        gdrtd(2) *= 2.0 * rtd(2);

        return cost;                                           // 返回总目标函数值
    }

    /**
     * @brief 计算给定凸多面体的最大体积内接椭球(MVIE)
     * @param hPoly 凸多面体约束矩阵，每行格式为[h0, h1, h2, h3]，约束为h0*x+h1*y+h2*z+h3≤0
     * @param R 输入输出：椭球旋转矩阵(初值和最优解)
     * @param p 输入输出：椭球中心位置(初值和最优解)
     * @param r 输入输出：椭球半轴长度(初值和最优解)
     * @return 是否成功计算MVIE
     * 
     * 算法流程：
     * 1. 找到凸多面体内部最深点(Chebyshev中心)
     * 2. 归一化约束并变换到单位球坐标系
     * 3. 使用L-BFGS优化椭球参数
     * 4. 通过SVD分解还原最终椭球参数
     * 
     * 注意：R假设为旋转矩阵，所有参数既是初值也是输出
     */
    inline bool maxVolInsEllipsoid(const Eigen::MatrixX4d &hPoly,
                                   Eigen::Matrix3d &R,
                                   Eigen::Vector3d &p,
                                   Eigen::Vector3d &r)
    {
        // 第一步：寻找最深内部点(Chebyshev中心)
        const int M = hPoly.rows();                            // 约束数量
        Eigen::MatrixX4d Alp(M, 4);                           // 归一化约束矩阵
        Eigen::VectorXd blp(M);                               // 约束右端项
        Eigen::Vector4d clp, xlp;                             // 线性规划目标和解
        
        // 归一化约束：将约束法向量归一化为单位向量
        const Eigen::ArrayXd hNorm = hPoly.leftCols<3>().rowwise().norm(); // 法向量模长
        Alp.leftCols<3>() = hPoly.leftCols<3>().array().colwise() / hNorm; // 归一化法向量
        Alp.rightCols<1>().setConstant(1.0);                 // 深度变量系数
        blp = -hPoly.rightCols<1>().array() / hNorm;          // 归一化右端项
        
        // 设置线性规划问题：max depth s.t. normalized_constraints
        clp.setZero();                                        // 目标函数系数
        clp(3) = -1.0;                                        // 最大化深度(最小化-depth)
        const double maxdepth = -sdlp::linprog<4>(clp, Alp, blp, xlp); // 求解线性规划
        
        if (!(maxdepth > 0.0) || std::isinf(maxdepth))        // 检查可行性
        {
            return false;                                      // 无内部点，失败
        }
        const Eigen::Vector3d interior = xlp.head<3>();       // 提取最深内部点

        // 第二步：准备MVIE优化数据
        uint8_t *optData = new uint8_t[sizeof(int64_t) + (2 + 3 * M) * sizeof(double)];
        int64_t *pM = (int64_t *)optData;                     // 约束数量指针
        double *pSmoothEps = (double *)(pM + 1);              // 平滑参数指针
        double *pPenaltyWt = pSmoothEps + 1;                  // 惩罚权重指针
        double *pA = pPenaltyWt + 1;                          // 约束矩阵指针

        *pM = M;                                              // 设置约束数量
        Eigen::Map<Eigen::MatrixX3d> A(pA, M, 3);            // 映射约束矩阵
        
        // 变换约束到以内部点为原点的坐标系
        A = Alp.leftCols<3>().array().colwise() /
            (blp - Alp.leftCols<3>() * interior).array();    // 归一化约束矩阵

        // 第三步：初始化优化变量
        Eigen::VectorXd x(9);                                // 9维优化变量
        const Eigen::Matrix3d Q = R * (r.cwiseProduct(r)).asDiagonal() * R.transpose(); // 椭球矩阵
        Eigen::Matrix3d L;                                   // Cholesky分解
        chol3d(Q, L);                                        // 计算Cholesky分解

        x.head<3>() = p - interior;                          // 相对位置
        x(3) = sqrt(L(0, 0));                               // L₁₁ = √L(0,0)
        x(4) = sqrt(L(1, 1));                               // L₂₂ = √L(1,1)
        x(5) = sqrt(L(2, 2));                               // L₃₃ = √L(2,2)
        x(6) = L(1, 0);                                     // L₂₁
        x(7) = L(2, 1);                                     // L₃₂
        x(8) = L(2, 0);                                     // L₃₁

        // 第四步：配置L-BFGS优化参数
        double minCost;                                      // 最小目标函数值
        lbfgs::lbfgs_parameter_t paramsMVIE;                // L-BFGS参数
        paramsMVIE.mem_size = 18;                           // 历史信息存储量
        paramsMVIE.g_epsilon = 0.0;                         // 梯度收敛阈值
        paramsMVIE.min_step = 1.0e-32;                      // 最小步长
        paramsMVIE.past = 3;                                // 历史比较窗口
        paramsMVIE.delta = 1.0e-7;                          // 相对改进阈值
        *pSmoothEps = 1.0e-2;                               // 平滑L1参数
        *pPenaltyWt = 1.0e+3;                               // 约束惩罚权重

        // 第五步：执行L-BFGS优化
        int ret = lbfgs::lbfgs_optimize(x,                   // 优化变量
                                        minCost,             // 输出最小值
                                        &costMVIE,           // 目标函数
                                        nullptr,             // 进度回调(未使用)
                                        nullptr,             // 线搜索回调(未使用)
                                        optData,             // 用户数据
                                        paramsMVIE);         // 优化参数

        if (ret < 0)                                         // 检查优化结果
        {
            printf("FIRI WARNING: %s\n", lbfgs::lbfgs_strerror(ret)); // 输出警告
        }

        // 第六步：提取优化结果并重构椭球参数
        p = x.head<3>() + interior;                          // 恢复绝对位置
        
        // 重构Cholesky分解矩阵L
        L(0, 0) = x(3) * x(3);                              // L₁₁²
        L(0, 1) = 0.0;                                      // 上三角置零
        L(0, 2) = 0.0;
        L(1, 0) = x(6);                                     // L₂₁
        L(1, 1) = x(4) * x(4);                              // L₂₂²
        L(1, 2) = 0.0;
        L(2, 0) = x(8);                                     // L₃₁
        L(2, 1) = x(7);                                     // L₃₂
        L(2, 2) = x(5) * x(5);                              // L₃₃²
        
        // 第七步：通过SVD分解提取椭球主轴和旋转
        Eigen::JacobiSVD<Eigen::Matrix3d, Eigen::FullPivHouseholderQRPreconditioner> svd(L, Eigen::ComputeFullU);
        const Eigen::Matrix3d U = svd.matrixU();            // 左奇异向量(旋转矩阵)
        const Eigen::Vector3d S = svd.singularValues();     // 奇异值(半轴长度)
        
        // 确保旋转矩阵的行列式为正(右手坐标系)
        if (U.determinant() < 0.0)                          // 检查行列式符号
        {
            R.col(0) = U.col(1);                            // 交换前两列
            R.col(1) = U.col(0);
            R.col(2) = U.col(2);
            r(0) = S(1);                                    // 对应交换半轴长度
            r(1) = S(0);
            r(2) = S(2);                                    // 第三轴保持不变
        }
        else                                                // 行列式为正，直接使用
        {
            R = U;                                          // 旋转矩阵
            r = S;                                          // 半轴长度
        }

        delete[] optData;                                   // 释放优化数据内存

        return ret >= 0;                                    // 返回优化是否成功
    }

    /**
     * @brief FIRI主算法：快速内接半径椭球逼近
     * @param bd 边界约束矩阵，每行格式为[n_x, n_y, n_z, d]，约束为n·x + d ≤ 0
     * @param pc 点云矩阵，每列为一个3D点
     * @param a 线段起点
     * @param b 线段终点
     * @param hPoly 输出：逼近的凸多面体约束
     * @param iterations 迭代次数，默认4次
     * @param epsilon 数值精度阈值，默认1e-6
     * @return 是否成功生成逼近
     * 
     * FIRI算法核心思想：
     * 1. 迭代优化椭球参数以最大化体积
     * 2. 在每次迭代中，椭球坐标系下生成切平面约束
     * 3. 通过贪心策略选择最重要的约束平面
     * 4. 最终输出包含线段且避开点云的凸多面体
     * 
     * 应用场景：轨迹规划中的安全走廊生成
     */
    inline bool firi(const Eigen::MatrixX4d &bd,
                     const Eigen::Matrix3Xd &pc,
                     const Eigen::Vector3d &a,
                     const Eigen::Vector3d &b,
                     Eigen::MatrixX4d &hPoly,
                     const int iterations = 4,
                     const double epsilon = 1.0e-6)
    {
        // 预处理：检查线段端点是否在边界约束内
        const Eigen::Vector4d ah(a(0), a(1), a(2), 1.0);        // 起点齐次坐标
        const Eigen::Vector4d bh(b(0), b(1), b(2), 1.0);        // 终点齐次坐标

        if ((bd * ah).maxCoeff() > 0.0 ||                        // 检查起点可行性
            (bd * bh).maxCoeff() > 0.0)                          // 检查终点可行性
        {
            return false;                                        // 线段端点违反边界约束
        }

        const int M = bd.rows();                                 // 边界约束数量
        const int N = pc.cols();                                 // 点云数量

        // 初始化椭球参数
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();         // 初始旋转矩阵(单位矩阵)
        Eigen::Vector3d p = 0.5 * (a + b);                      // 初始中心(线段中点)
        Eigen::Vector3d r = Eigen::Vector3d::Ones();             // 初始半轴长度(单位球)
        Eigen::MatrixX4d forwardH(M + N, 4);                    // 前向变换约束矩阵
        int nH = 0;                                              // 有效约束数量

        // FIRI主迭代循环
        for (int loop = 0; loop < iterations; ++loop)
        {
            // 第一步：计算坐标变换矩阵
            const Eigen::Matrix3d forward = r.cwiseInverse().asDiagonal() * R.transpose(); // 世界→椭球
            const Eigen::Matrix3d backward = R * r.asDiagonal();                          // 椭球→世界
            
            // 第二步：变换边界约束到椭球坐标系
            const Eigen::MatrixX3d forwardB = bd.leftCols<3>() * backward;               // 变换约束法向量
            const Eigen::VectorXd forwardD = bd.rightCols<1>() + bd.leftCols<3>() * p;   // 变换约束常数项
            
            // 第三步：变换点云和线段端点到椭球坐标系
            const Eigen::Matrix3Xd forwardPC = forward * (pc.colwise() - p);            // 变换点云
            const Eigen::Vector3d fwd_a = forward * (a - p);                            // 变换起点
            const Eigen::Vector3d fwd_b = forward * (b - p);                            // 变换终点

            // 第四步：计算边界约束到椭球表面的距离
            const Eigen::VectorXd distDs = forwardD.cwiseAbs().cwiseQuotient(forwardB.rowwise().norm());
            
            // 第五步：为每个点云点生成切平面约束
            Eigen::MatrixX4d tangents(N, 4);                    // 切平面约束矩阵
            Eigen::VectorXd distRs(N);                          // 点到椭球表面距离

            for (int i = 0; i < N; i++)                         // 遍历每个点云点
            {
                distRs(i) = forwardPC.col(i).norm();            // 点到椭球中心距离
                tangents(i, 3) = -distRs(i);                    // 切平面常数项
                tangents.block<1, 3>(i, 0) = forwardPC.col(i).transpose() / distRs(i); // 切平面法向量
                
                // 检查切平面是否与起点冲突，若冲突则调整
                if (tangents.block<1, 3>(i, 0).dot(fwd_a) + tangents(i, 3) > epsilon)
                {
                    const Eigen::Vector3d delta = forwardPC.col(i) - fwd_a;             // 点到起点向量
                    tangents.block<1, 3>(i, 0) = fwd_a - (delta.dot(fwd_a) / delta.squaredNorm()) * delta; // 投影修正
                    distRs(i) = tangents.block<1, 3>(i, 0).norm();          // 重新计算距离
                    tangents(i, 3) = -distRs(i);                            // 更新常数项
                    tangents.block<1, 3>(i, 0) /= distRs(i);                // 归一化法向量
                }
                
                // 检查切平面是否与终点冲突，若冲突则调整
                if (tangents.block<1, 3>(i, 0).dot(fwd_b) + tangents(i, 3) > epsilon)
                {
                    const Eigen::Vector3d delta = forwardPC.col(i) - fwd_b;             // 点到终点向量
                    tangents.block<1, 3>(i, 0) = fwd_b - (delta.dot(fwd_b) / delta.squaredNorm()) * delta; // 投影修正
                    distRs(i) = tangents.block<1, 3>(i, 0).norm();          // 重新计算距离
                    tangents(i, 3) = -distRs(i);                            // 更新常数项
                    tangents.block<1, 3>(i, 0) /= distRs(i);                // 归一化法向量
                }
                
                // 如果仍与起点冲突，使用叉积生成垂直平面
                if (tangents.block<1, 3>(i, 0).dot(fwd_a) + tangents(i, 3) > epsilon)
                {
                    // 使用起点-点和终点-点向量的叉积作为法向量
                    tangents.block<1, 3>(i, 0) = (fwd_a - forwardPC.col(i)).cross(fwd_b - forwardPC.col(i)).normalized();
                    tangents(i, 3) = -tangents.block<1, 3>(i, 0).dot(fwd_a); // 使平面通过起点
                    tangents.row(i) *= tangents(i, 3) > 0.0 ? -1.0 : 1.0;   // 调整符号确保正确方向
                }
            }

            // 第六步：贪心选择约束平面
            Eigen::Matrix<uint8_t, -1, 1> bdFlags = Eigen::Matrix<uint8_t, -1, 1>::Constant(M, 1); // 边界约束标志
            Eigen::Matrix<uint8_t, -1, 1> pcFlags = Eigen::Matrix<uint8_t, -1, 1>::Constant(N, 1); // 点云约束标志

            nH = 0;                                              // 重置约束计数

            bool completed = false;                              // 完成标志
            int bdMinId = 0, pcMinId = 0;                       // 最小距离约束索引
            double minSqrD = distDs.minCoeff(&bdMinId);         // 边界约束最小距离
            double minSqrR = INFINITY;                          // 点云约束最小距离
            if (distRs.size() != 0)
            {
                minSqrR = distRs.minCoeff(&pcMinId);            // 找到最近点云约束
            }
            
            // 贪心选择循环：每次选择距离椭球最近的约束
            for (int i = 0; !completed && i < (M + N); ++i)
            {
                if (minSqrD < minSqrR)                          // 边界约束更近
                {
                    forwardH.block<1, 3>(nH, 0) = forwardB.row(bdMinId); // 添加边界约束
                    forwardH(nH, 3) = forwardD(bdMinId);
                    bdFlags(bdMinId) = 0;                       // 标记已使用
                }
                else                                            // 点云约束更近
                {
                    forwardH.row(nH) = tangents.row(pcMinId);   // 添加点云切平面约束
                    pcFlags(pcMinId) = 0;                       // 标记已使用
                }

                // 更新剩余约束的最小距离
                completed = true;                               // 假设完成
                minSqrD = INFINITY;                             // 重置边界最小距离
                for (int j = 0; j < M; ++j)                    // 查找剩余边界约束
                {
                    if (bdFlags(j))                             // 未使用的约束
                    {
                        completed = false;                      // 还有约束未处理
                        if (minSqrD > distDs(j))               // 更新最小距离
                        {
                            bdMinId = j;
                            minSqrD = distDs(j);                   // 更新最小距离
                        }
                    }
                }
                
                minSqrR = INFINITY;                             // 重置点云最小距离
                for (int j = 0; j < N; ++j)                    // 查找剩余点云约束
                {
                    if (pcFlags(j))                             // 未使用的点云约束
                    {
                        // 检查当前约束是否已经分离该点云
                        if (forwardH.block<1, 3>(nH, 0).dot(forwardPC.col(j)) + forwardH(nH, 3) > -epsilon)
                        {
                            pcFlags(j) = 0;                     // 该点云已被分离，标记移除
                        }
                        else
                        {
                            completed = false;                  // 还有点云需要分离
                            if (minSqrR > distRs(j))           // 更新最小距离
                            {
                                pcMinId = j;
                                minSqrR = distRs(j);
                            }
                        }
                    }
                }
                ++nH;                                           // 增加约束计数
            }

            // 第七步：变换约束回世界坐标系
            hPoly.resize(nH, 4);                               // 调整输出矩阵大小
            for (int i = 0; i < nH; ++i)                       // 变换每个约束
            {
                hPoly.block<1, 3>(i, 0) = forwardH.block<1, 3>(i, 0) * forward; // 变换法向量
                hPoly(i, 3) = forwardH(i, 3) - hPoly.block<1, 3>(i, 0).dot(p);  // 变换常数项
            }

            // 第八步：检查是否为最后一次迭代
            if (loop == iterations - 1)                        // 最后一次迭代
            {
                break;                                          // 退出循环
            }

            // 第九步：用当前约束优化椭球参数
            maxVolInsEllipsoid(hPoly, R, p, r);                // 计算最大体积内接椭球
        }

        return true;                                            // 算法成功完成
    }

}

#endif
