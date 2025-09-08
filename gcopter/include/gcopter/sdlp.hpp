/* 
 * Copyright (c) 1990 Michael E. Hohmeyer, 
 *       hohmeyer@icemcfd.com
 * Permission is granted to modify and re-distribute this code in any manner
 * as long as this notice is preserved.  All standard disclaimers apply.
 * 
 * R. Seidel's algorithm for solving LPs (linear programs.)
 */

/* 
 * Copyright (c) 2021 Zhepei Wang,
 *       wangzhepei@live.com
 * 1. Bug fix in "move_to_front" function that "prev[m]" is illegally accessed
 *    while "prev" originally has only m ints. It is fixed by allocating a 
 *    "prev" with m + 1 ints.  
 * 2. Add Eigen interface.
 * 3. Resursive template.
 * Permission is granted to modify and re-distribute this code in any manner
 * as long as this notice is preserved.  All standard disclaimers apply.
 * 
 * Ref: Seidel, R. (1991), "Small-dimensional linear programming and convex 
 *      hulls made easy", Discrete & Computational Geometry 6 (1): 423–434, 
 *      doi:10.1007/BF02574699
 */

/**
 * @file sdlp.hpp
 * @brief 小维度线性规划求解器（Small-Dimensional Linear Programming）
 * 
 * 基于Seidel算法的高效线性规划求解器，专门针对低维度（2D/3D）问题优化。
 * 该算法特别适用于计算几何和轨迹规划中的约束优化问题。
 * 
 * 主要特点：
 * - 递归模板设计，支持任意维度
 * - 针对小维度问题的特殊优化
 * - 处理退化情况和数值稳定性
 * - 支持Eigen矩阵接口
 */

#ifndef SDLP_HPP
#define SDLP_HPP

#include <Eigen/Eigen>
#include <cmath>
#include <random>

/**
 * @namespace sdlp
 * @brief 小维度线性规划命名空间
 * 
 * 包含Seidel线性规划算法的所有实现函数和工具类
 */
namespace sdlp
{
    /// 数值计算精度阈值
    constexpr double eps = 1.0e-12;

    /**
     * @brief 线性规划求解结果状态枚举
     */
    enum
    {
        MINIMUM = 0,    ///< 找到最优解
        INFEASIBLE,     ///< 无可行解
        UNBOUNDED,      ///< 无界解
        AMBIGUOUS,      ///< 解集只有一个顶点（退化情况）
    };

    /**
     * @brief 计算2D向量的点积
     * @param a 第一个2D向量
     * @param b 第二个2D向量
     * @return 点积结果
     */
    inline double dot2(const double a[2],
                       const double b[2])
    {
        return a[0] * b[0] + a[1] * b[1];
    }

    /**
     * @brief 计算2D向量的叉积（外积）
     * @param a 第一个2D向量
     * @param b 第二个2D向量
     * @return 叉积的标量值（z分量）
     * 
     * 在2D中，叉积返回垂直于平面的标量值，用于判断向量的相对方向
     */
    inline double cross2(const double a[2],
                         const double b[2])
    {
        return a[0] * b[1] - a[1] * b[0];
    }

    /**
     * @brief 对2D向量进行单位化
     * @param a 输入向量
     * @param b 输出的单位向量
     * @return true表示输入向量接近零向量，false表示成功单位化
     * 
     * 将向量a归一化为单位向量并存储到b中
     */
    inline bool unit2(const double a[2],
                      double b[2])
    {
        const double mag = std::sqrt(a[0] * a[0] +
                                     a[1] * a[1]);
        if (mag < 2.0 * eps)
        {
            return true;  // 向量太小，无法归一化
        }
        b[0] = a[0] / mag;
        b[1] = a[1] / mag;
        return false;
    }

    /**
     * @brief 对d+1维点进行单位化
     * @tparam d 空间维度
     * @param a 输入输出的(d+1)维齐次坐标点
     * @return true表示点接近原点，false表示成功单位化
     * 
     * 这是齐次坐标系中的点归一化，将点投影到单位球面上
     */
    template <int d>
    inline bool unit(double *a)
    {
        double mag = 0.0;
        // 计算向量的模长平方
        for (int i = 0; i <= d; i++)
        {
            mag += a[i] * a[i];
        }
        // 检查是否为零向量
        if (mag < (d + 1) * eps * eps)
        {
            return true;  // 向量太小，无法归一化
        }
        // 归一化向量
        mag = 1.0 / std::sqrt(mag);
        for (int i = 0; i <= d; i++)
        {
            a[i] *= mag;
        }
        return false;
    }

    /**
     * @brief 无约束线性规划优化
     * @tparam d 空间维度
     * @param n_vec 目标函数的法向量
     * @param d_vec 方向向量
     * @param opt 输出的最优解
     * @return 求解状态（MINIMUM 或 AMBIGUOUS）
     * 
     * 求解无约束的线性规划问题：min n_vec^T * x
     * 其中x在由d_vec定义的方向上自由移动
     */
    template <int d>
    inline int lp_no_con(const double *n_vec,
                         const double *d_vec,
                         double *opt)
    {
        double n_dot_d = 0.0;
        double d_dot_d = 0.0;
        
        // 计算目标向量与方向向量的点积
        for (int i = 0; i <= d; i++)
        {
            n_dot_d += n_vec[i] * d_vec[i];
            d_dot_d += d_vec[i] * d_vec[i];
        }
        
        // 如果方向向量接近零，使用默认处理
        if (d_dot_d < eps * eps)
        {
            n_dot_d = 0.0;
            d_dot_d = 1.0;
        }
        
        // 计算最优点：在方向d_vec上投影目标向量n_vec
        for (int i = 0; i <= d; i++)
        {
            opt[i] = -n_vec[i] +
                     d_vec[i] * n_dot_d / d_dot_d;
        }
        
        // 对最优点进行归一化
        if (unit<d>(opt))
        {
            opt[d] = 1.0;    // 退化情况，设置齐次坐标
            return AMBIGUOUS;
        }
        else
        {
            return MINIMUM;
        }
    }

    /**
     * @brief 将约束移动到列表前端
     * @param i 要移动的约束索引
     * @param next 下一个约束的索引数组
     * @param prev 前一个约束的索引数组
     * @return 原来在i位置的约束索引
     * 
     * 这是一个链表操作，用于约束管理和优化求解过程中的动态重排
     */
    inline int move_to_front(const int i,
                             int *next,
                             int *prev)
    {
        // 如果i已经在前端或就是第一个元素，直接返回
        if (i == 0 || i == next[0])
        {
            return i;
        }
        
        const int previ = prev[i];
        
        // 从当前位置移除约束i
        next[prev[i]] = next[i];
        prev[next[i]] = prev[i];
        
        // 将约束i放到列表前端
        next[i] = next[0];
        prev[i] = 0;
        prev[next[i]] = i;
        next[0] = i;
        
        return previ;
    }

    /**
     * @brief 线性有理函数的最小化
     * @param degen 是否为退化情况
     * @param cw_vec 顺时针边界向量
     * @param ccw_vec 逆时针边界向量  
     * @param n_vec 目标函数法向量
     * @param d_vec 目标函数分母向量
     * @param opt 输出的最优解
     * 
     * 在2D扇形区域内最小化线性有理函数 n_vec^T * x / d_vec^T * x
     * 处理各种边界情况和极点情况
     */
    inline void lp_min_lin_rat(const bool degen,
                               const double cw_vec[2],
                               const double ccw_vec[2],
                               const double n_vec[2],
                               const double d_vec[2],
                               double opt[2])
    {
        // 计算边界点与目标函数向量的点积
        const double d_cw = dot2(cw_vec, d_vec);    // 顺时针边界与分母的点积
        const double d_ccw = dot2(ccw_vec, d_vec);  // 逆时针边界与分母的点积
        const double n_cw = dot2(cw_vec, n_vec);    // 顺时针边界与分子的点积
        const double n_ccw = dot2(ccw_vec, n_vec);  // 逆时针边界与分子的点积
        
        if (degen)
        {
            // 退化情况：直接比较边界点的函数值
            if (n_cw / d_cw < n_ccw / d_ccw)
            {
                opt[0] = cw_vec[0];
                opt[1] = cw_vec[1];
            }
            else
            {
                opt[0] = ccw_vec[0];
                opt[1] = ccw_vec[1];
            }
        }
        // 检查边界是否远离极点（分母不为零）
        else if (std::fabs(d_cw) > 2.0 * eps &&
                 std::fabs(d_ccw) > 2.0 * eps)
        {
            // 有效区域不包含极点
            if (d_cw * d_ccw > 0.0)
            {
                // 分母同号，比较边界点的函数值
                if (n_cw / d_cw < n_ccw / d_ccw)
                {
                    opt[0] = cw_vec[0];
                    opt[1] = cw_vec[1];
                }
                else
                {
                    opt[0] = ccw_vec[0];
                    opt[1] = ccw_vec[1];
                }
            }
            else
            {
                // 有效区域包含极点，最优解在极点处
                if (d_cw > 0.0)
                {
                    // 选择使分母为零的方向（垂直于d_vec）
                    opt[0] = -d_vec[1];
                    opt[1] = d_vec[0];
                }
                else
                {
                    opt[0] = d_vec[1];
                    opt[1] = -d_vec[0];
                }
            }
        }
        else if (std::fabs(d_cw) > 2.0 * eps)
        {
            // 逆时针边界接近极点
            if (n_ccw * d_cw > 0.0)
            {
                // 逆时针边界是正极点，选择顺时针边界
                opt[0] = cw_vec[0];
                opt[1] = cw_vec[1];
            }
            else
            {
                // 逆时针边界是负极点，选择逆时针边界
                opt[0] = ccw_vec[0];
                opt[1] = ccw_vec[1];
            }
        }
        else if (std::fabs(d_ccw) > 2.0 * eps)
        {
            // 顺时针边界接近极点
            if (n_cw * d_ccw > 2.0 * eps)
            {
                // 顺时针边界是正极点，选择逆时针边界
                opt[0] = ccw_vec[0];
                opt[1] = ccw_vec[1];
            }
            else
            {
                // 顺时针边界是负极点，选择顺时针边界
                opt[0] = cw_vec[0];
                opt[1] = cw_vec[1];
            }
        }
        else
        {
            // 两个边界都接近极点，使用叉积判断方向
            if (cross2(d_vec, n_vec) > 0.0)
            {
                opt[0] = cw_vec[0];
                opt[1] = cw_vec[1];
            }
            else
            {
                opt[0] = ccw_vec[0];
                opt[1] = ccw_vec[1];
            }
        }
    }

    /**
     * @brief 楔形（扇形）区域的约束处理
     * @param halves 半平面约束数组，每个元素是2D向量
     * @param m 约束数量
     * @param next 下一个约束索引数组
     * @param prev 前一个约束索引数组
     * @param cw_vec 输出的顺时针边界向量
     * @param ccw_vec 输出的逆时针边界向量
     * @param degen 输出是否为退化情况
     * @return 求解状态
     * 
     * 计算所有半平面约束的交集，形成一个楔形可行域
     * 算法逐步收紧边界，直到找到最小的可行楔形
     */
    inline int wedge(const double (*halves)[2],
                     const int m,
                     int *next,
                     int *prev,
                     double cw_vec[2],
                     double ccw_vec[2],
                     bool *degen)
    {
        int i;
        double d_cw, d_ccw;
        bool offensive;

        *degen = false;
        
        // 寻找第一个有效的半平面约束来初始化边界
        for (i = 0; i != m; i = next[i])
        {
            if (!unit2(halves[i], ccw_vec))  // 将约束向量单位化
            {
                // 设置初始的顺时针和逆时针边界
                cw_vec[0] = ccw_vec[1];     // 顺时针：将向量逆时针旋转90度
                cw_vec[1] = -ccw_vec[0];
                ccw_vec[0] = -cw_vec[0];    // 逆时针：与顺时针相反
                ccw_vec[1] = -cw_vec[1];
                break;
            }
        }
        
        // 如果所有约束都是零向量，则问题无界
        if (i == m)
        {
            return UNBOUNDED;
        }
        
        // 处理剩余的约束，逐步收紧楔形区域
        i = 0;
        while (i != m)
        {
            offensive = false;
            d_cw = dot2(cw_vec, halves[i]);     // 顺时针边界与当前约束的点积
            d_ccw = dot2(ccw_vec, halves[i]);   // 逆时针边界与当前约束的点积
            
            if (d_ccw >= 2.0 * eps)
            {
                // 逆时针边界违反约束，需要调整顺时针边界
                if (d_cw <= -2.0 * eps)
                {
                    cw_vec[0] = halves[i][1];
                    cw_vec[1] = -halves[i][0];
                    unit2(cw_vec, cw_vec);
                    offensive = true;
                }
            }
            else if (d_cw >= 2.0 * eps)
            {
                // 顺时针边界违反约束，需要调整逆时针边界
                if (d_ccw <= -2.0 * eps)
                {
                    ccw_vec[0] = -halves[i][1];
                    ccw_vec[1] = halves[i][0];
                    unit2(ccw_vec, ccw_vec);
                    offensive = true;
                }
            }
            else if (d_ccw <= -2.0 * eps &&
                     d_cw <= -2.0 * eps)
            {
                // 两个边界都违反约束，无可行解
                return INFEASIBLE;
            }
            else if (d_cw <= -2.0 * eps ||
                     d_ccw <= -2.0 * eps ||
                     cross2(cw_vec, halves[i]) < 0.0)
            {
                // 退化情况处理
                if (d_cw <= -2.0 * eps)
                {
                    unit2(ccw_vec, cw_vec);
                }
                else if (d_ccw <= -2.0 * eps)
                {
                    unit2(cw_vec, ccw_vec);
                }
                *degen = true;
                offensive = true;
            }
            
            // 将违反约束的半平面移到前面进行优先处理
            if (offensive)
            {
                i = move_to_front(i, next, prev);
            }
            i = next[i];
            if (*degen)
            {
                break;
            }
        }
        
        // 退化情况下的额外检查
        if (*degen)
        {
            while (i != m)
            {
                d_cw = dot2(cw_vec, halves[i]);
                d_ccw = dot2(ccw_vec, halves[i]);
                if (d_cw < -2.0 * eps)
                {
                    if (d_ccw < -2.0 * eps)
                    {
                        return INFEASIBLE;
                    }
                    else
                    {
                        cw_vec[0] = ccw_vec[0];
                        cw_vec[1] = ccw_vec[1];
                    }
                }
                else if (d_ccw < -2.0 * eps)
                {
                    ccw_vec[0] = cw_vec[0];
                    ccw_vec[1] = cw_vec[1];
                }
                i = next[i];
            }
        }
        return MINIMUM;
    }

    /* return the minimum on the projective line */
    /**
     * @brief 2D线性规划基础情况求解
     * @param halves 半平面约束数组
     * @param m 约束终止标记
     * @param n_vec 目标函数分子向量
     * @param d_vec 目标函数分母向量
     * @param opt 输出的最优解
     * @param next 下一个约束索引的双向链表
     * @param prev 前一个约束索引的双向链表
     * @return 求解状态
     * 
     * 求解2D线性规划问题：min (n_vec^T * x) / (d_vec^T * x)
     * 约束条件：halves[i]^T * x >= 0 for all i
     */
    inline int lp_base_case(const double (*halves)[2], /* halves --- half lines */
                            const int m,               /* m      --- terminal marker */
                            const double n_vec[2],     /* n_vec  --- numerator funciton */
                            const double d_vec[2],     /* d_vec  --- denominator function */
                            double opt[2],             /* opt    --- optimum  */
                            int *next,                 /* next, prev  --- double linked list of indices */
                            int *prev)
    {
        double cw_vec[2], ccw_vec[2];
        bool degen;
        int status;

        // 寻找直线上的可行域（楔形区域）
        status = wedge(halves, m, next, prev, cw_vec, ccw_vec, &degen);

        if (status == INFEASIBLE)
        {
            return status;  // 无可行解
        }
        
        // 平面上无非平凡约束：返回无约束最优解
        if (status == UNBOUNDED)
        {
            return lp_no_con<1>(n_vec, d_vec, opt);
        }

        // 检查分子和分母向量是否平行（叉积接近零）
        if (std::fabs(cross2(n_vec, d_vec)) < 2.0 * eps * eps)
        {
            if (dot2(n_vec, n_vec) < 2.0 * eps * eps ||
                dot2(d_vec, d_vec) > 2.0 * eps * eps)
            {
                // 分子为零或分子分母线性相关的情况
                opt[0] = cw_vec[0];
                opt[1] = cw_vec[1];
                status = AMBIGUOUS;
            }
            else
            {
                // 分子非零且分母为零：在单位圆上最小化线性函数
                if (!degen &&
                    cross2(cw_vec, n_vec) <= 0.0 &&
                    cross2(n_vec, ccw_vec) <= 0.0)
                {
                    // 最优解在可行域内部
                    opt[0] = -n_vec[0];
                    opt[1] = -n_vec[1];
                }
                else if (dot2(n_vec, cw_vec) > dot2(n_vec, ccw_vec))
                {
                    // 最优解在逆时针边界
                    opt[0] = ccw_vec[0];
                    opt[1] = ccw_vec[1];
                }
                else
                {
                    /* optimum is at CW boundary */
                    opt[0] = cw_vec[0];
                    opt[1] = cw_vec[1];
                }
                status = MINIMUM;
            }
        }
        else
        {
            /* niether numerator nor denominator is zero */
            lp_min_lin_rat(degen, cw_vec, ccw_vec, n_vec, d_vec, opt);
            status = MINIMUM;
        }
        return status;
    }

    /**
     * @brief 寻找平面系数中绝对值最大的分量
     * @tparam d 空间维度
     * @param pln 平面系数数组 (d+1维)
     * @param imax 输出最大系数的索引
     * 
     * 在Gauss消元法中，选择绝对值最大的系数作为主元
     * 可以提高数值稳定性，避免小数除法带来的误差放大
     */
    template <int d>
    inline void findimax(const double *pln,
                         int *imax)
    {
        *imax = 0;
        double rmax = std::fabs(pln[0]);
        for (int i = 1; i <= d; i++)
        {
            const double ab = std::fabs(pln[i]);
            if (ab > rmax)
            {
                *imax = i;                    // 更新最大系数的索引
                rmax = ab;                    // 更新最大系数值
            }
        }
    }

    /**
     * @brief 从低维向量恢复高维向量（回代过程）
     * @tparam d 原始空间维度
     * @param equation 消元所用的约束平面系数
     * @param ivar 被消元的变量索引
     * @param low_vector 低维空间的解向量 (d维)
     * @param vector 输出的高维空间解向量 (d+1维)
     * 
     * 这是Gauss消元法的回代步骤：
     * 已知 equation[0]*x[0] + ... + equation[d]*x[d] = 0
     * 以及除x[ivar]外所有变量的值，求解x[ivar]
     */
    template <int d>
    inline void vector_up(const double *equation,
                          const int ivar,
                          const double *low_vector,
                          double *vector)
    {
        vector[ivar] = 0.0;                              // 初始化被消元变量
        
        // 第一步：复制低维向量到高维向量（跳过被消元的变量）
        for (int i = 0; i <= d; i++)
        {
            if (i != ivar)
            {
                const int j = i < ivar ? i : i - 1;      // 低维向量中对应的索引
                vector[i] = low_vector[j];                // 复制已知变量的值
                vector[ivar] -= equation[i] * low_vector[j];  // 累积等式左端项
            }
        }
        
        // 第二步：求解被消元变量的值
        // 由约束 equation^T * vector = 0 得到：
        // vector[ivar] = -sum(equation[i] * vector[i]) / equation[ivar]
        vector[ivar] /= equation[ivar];
    }

    /**
     * @brief 将高维向量投影到低维空间（正交投影）
     * @tparam d 原始空间维度
     * @param elim_eqn 约束平面的法向量 (d+1维)
     * @param ivar 被消元的变量索引
     * @param old_vec 原始高维向量 (d+1维)
     * @param new_vec 输出的低维向量 (d维)
     * 
     * 正交投影过程：
     * 1. 计算原向量在约束平面法向量上的投影长度
     * 2. 从原向量中减去该投影，得到垂直于法向量的分量
     * 3. 去掉第ivar个分量，形成低维向量
     * 
     * 数学公式：new_vec = old_vec - (old_vec·elim_eqn/|elim_eqn|²) * elim_eqn
     */
    template <int d>
    inline void vector_down(const double *elim_eqn,
                            const int ivar,
                            const double *old_vec,
                            double *new_vec)
    {
        double ve = 0.0;  // old_vec与elim_eqn的点积
        double ee = 0.0;  // elim_eqn的模长平方
        
        // 计算投影系数
        for (int i = 0; i <= d; i++)
        {
            ve += old_vec[i] * elim_eqn[i];
            ee += elim_eqn[i] * elim_eqn[i];
        }
        const double fac = ve / ee;  // 投影长度系数
        
        // 执行正交投影并降维
        for (int i = 0; i <= d; i++)
        {
            if (i != ivar)
            {
                new_vec[i < ivar ? i : i - 1] =              // 紧缩索引
                    old_vec[i] - elim_eqn[i] * fac;          // 正交投影
            }
        }
    }

    /**
     * @brief 将高维平面投影到低维空间（消元投影）
     * @tparam d 原始空间维度
     * @param elim_eqn 用于消元的约束平面系数 (d+1维)
     * @param ivar 被消元的变量索引
     * @param old_plane 原始高维平面系数 (d+1维)
     * @param new_plane 输出的低维平面系数 (d维)
     * 
     * 消元过程：
     * 1. 利用约束 elim_eqn^T * x = 0 消除变量 x[ivar]
     * 2. 将原平面投影到不包含 x[ivar] 的(d-1)维子空间
     * 3. new_plane = old_plane - (old_plane[ivar]/elim_eqn[ivar]) * elim_eqn
     */
    template <int d>
    inline void plane_down(const double *elim_eqn,
                           const int ivar,
                           const double *old_plane,
                           double *new_plane)
    {
        const double crit = old_plane[ivar] / elim_eqn[ivar];  // 消元系数
        
        // 执行消元投影，跳过被消元的变量
        for (int i = 0; i <= d; i++)
        {
            if (i != ivar)
            {
                new_plane[i < ivar ? i : i - 1] =             // 紧缩索引
                    old_plane[i] - elim_eqn[i] * crit;        // 消元公式
            }
        }
    }

    /**
     * @brief d维线性分式规划递归求解器（Seidel算法核心）
     * @tparam d 空间维度
     * @param halves 半空间约束数组 (max_size)×(d+1)
     * @param max_size 半空间数组的最大大小
     * @param m 约束链表的终止标记
     * @param n_vec 目标函数分子向量 (d+1维)
     * @param d_vec 目标函数分母向量 (d+1维)
     * @param opt 输出的最优解 (d+1维齐次坐标)
     * @param work 工作空间指针 [(max_size+3)×(d+2)×(d-1)/2个double]
     * @param next 约束索引的下一个指针数组（双向链表）
     * @param prev 约束索引的前一个指针数组（双向链表）
     * @return 求解状态
     * 
     * 算法说明：
     * - 半空间约束形式：halves[i][0]*x[0] + ... + halves[i][d]*x[d] >= 0
     * - 目标函数：minimize dot(x,n_vec) / dot(x,d_vec)
     * - 约束系数应该已归一化
     * - 约束顺序：0 -> next[0] -> next[next[0]] -> ... (随机序列)
     * - 对于标准d维线性规划：n_vec = (c0,c1,...,cd-1,0), d_vec = (0,0,...,0,1)
     * 
     * Seidel算法递归思想：
     * 1. 先求解无约束问题的最优解
     * 2. 逐个检查约束是否违反
     * 3. 若违反，则递归求解在该约束边界上的(d-1)维子问题
     * 4. 期望时间复杂度：O(d! × n)，对小维度d效率很高
     */
    template <int d>
    inline int linfracprog(const double *halves, /* halves  --- half spaces */
                           const int max_size,   /* max_size --- size of halves array */
                           const int m,          /* m       --- terminal marker */
                           const double *n_vec,  /* n_vec   --- numerator vector */
                           const double *d_vec,  /* d_vec   --- denominator vector */
                           double *opt,          /* opt     --- optimum */
                           double *work,         /* work    --- work space (see below) */
                           int *next,            /* next    --- array of indices into halves */
                           int *prev)            /* prev    --- array of indices into halves */
    /*
    **
    ** 半空间约束形式：
    ** halves[i][0]*x[0] + halves[i][1]*x[1] + 
    ** ... + halves[i][d-1]*x[d-1] + halves[i][d]*x[d] >= 0
    **
    ** 约束系数应该已归一化
    ** 半空间应该按随机顺序排列
    ** 约束处理顺序：0, next[0], next[next[0]], ...
    ** 且满足 prev[next[i]] = i
    **
    ** halves: (max_size)×(d+1) 维数组
    **
    ** 已经为以下半空间计算了最优解：
    ** 0, next[0], next[next[0]], ..., prev[0]
    ** 下一个需要测试的平面是索引为0的平面
    **
    ** m 是第一个不在链表上的平面索引，即链表的终止标记
    **
    ** 目标函数为 dot(x,n_vec)/dot(x,d_vec)
    ** 对于标准d维线性规划问题：
    ** n_vec = (x0, x1, x2, ..., xd-1, 0)
    ** d_vec = (0,  0,  0, ...,    0, 1)
    ** halves[0] = (0, 0, ..., 1)
    **
    ** work 指向 (max_size+3)*(d+2)*(d-1)/2 个double的工作空间
    */
    {
        int status, imax;
        double *new_opt, *new_n_vec, *new_d_vec, *new_halves, *new_work;
        const double *plane_i;

        // 检查分母向量是否为零向量
        double val = 0.0;
        for (int j = 0; j <= d; j++)
        {
            val += d_vec[j] * d_vec[j];
        }
        const bool d_vec_zero = (val < (d + 1) * eps * eps);

        // 第一步：求解无约束最优解
        status = lp_no_con<d>(n_vec, d_vec, opt);
        if (m <= 0)
        {
            return status;  // 无约束情况，直接返回
        }

        // 第二步：为下一层递归分配内存空间
        new_opt = work;                                    // (d) 新的最优解
        new_n_vec = new_opt + d;                          // (d) 新的分子向量
        new_d_vec = new_n_vec + d;                        // (d) 新的分母向量
        new_halves = new_d_vec + d;                       // (max_size×d) 新的约束矩阵
        new_work = new_halves + max_size * d;             // 剩余工作空间

        // 第三步：Seidel算法主循环 - 逐个检查约束
        for (int i = 0; i != m; i = next[i])
        {
            // 检查当前最优解是否满足第i个半空间约束
            plane_i = halves + i * (d + 1);               // 获取第i个约束平面
            
            // 计算当前最优解在约束平面上的值：plane_i^T * opt
            val = 0.0;
            for (int j = 0; j <= d; j++)
            {
                val += opt[j] * plane_i[j];
            }
            
            // 如果违反约束（值小于-eps），需要递归求解约束边界上的子问题
            if (val < -(d + 1) * eps)
            {
                // 第四步：找到系数绝对值最大的变量进行消元
                findimax<d>(plane_i, &imax);
                
                // 第五步：消元构造(d-1)维子问题
                if (i != 0)
                {
                    const double fac = 1.0 / plane_i[imax];  // 消元因子
                    
                    // 对之前的所有约束进行投影消元
                    for (int j = 0; j != i; j = next[j])
                    {
                        const double *old_plane = halves + j * (d + 1);
                        const double crit = old_plane[imax] * fac;  // 消元系数
                        double *new_plane = new_halves + j * d;
                        
                        // 执行消元：new_plane = old_plane - crit * plane_i
                        for (int k = 0; k <= d; k++)
                        {
                            const int l = k < imax ? k : k - 1;    // 跳过被消元的变量
                            new_plane[l] = k != imax ? old_plane[k] - plane_i[k] * crit : new_plane[l];
                        }
                    }
                }
                
                // 第六步：将目标函数投影到低维空间
                if (d_vec_zero)
                {
                    // 分母为零的情况：只投影分子向量
                    vector_down<d>(plane_i, imax, n_vec, new_n_vec);
                    for (int j = 0; j < d; j++)
                    {
                        new_d_vec[j] = 0.0;
                    }
                }
                else
                {
                    // 一般情况：同时投影分子和分母向量
                    plane_down<d>(plane_i, imax, n_vec, new_n_vec);
                    plane_down<d>(plane_i, imax, d_vec, new_d_vec);
                }
                
                // 第七步：递归求解(d-1)维子问题
                status = linfracprog<d - 1>(new_halves, max_size, i, new_n_vec,
                                            new_d_vec, new_opt, new_work, next, prev);
                
                // 第八步：回代求解 - 从低维解恢复高维解
                if (status != INFEASIBLE)
                {
                    vector_up<d>(plane_i, imax, new_opt, opt);

                    // 内联的向量归一化代码（相当于调用unit<d>()）
                    double mag = 0.0;
                    for (int j = 0; j <= d; j++)
                    {
                        mag += opt[j] * opt[j];
                    }
                    mag = 1.0 / sqrt(mag);
                    for (int j = 0; j <= d; j++)
                    {
                        opt[j] *= mag;         // 归一化解向量
                    }
                }
                else
                {
                    return status;             // 子问题无解，整个问题无解
                }
                
                // 第九步：动态重排 - 将违反约束的平面移到前面，提高后续效率
                i = move_to_front(i, next, prev);
            }
        }
        
        // 算法结束：所有约束都已满足，返回当前状态
        return status;
    }

    /**
     * @brief 1D线性分式规划的模板特化
     * @param halves 半平面约束数组
     * @param max_size 最大约束数量
     * @param m 当前约束数量
     * @param n_vec 目标函数分子向量
     * @param d_vec 目标函数分母向量
     * @param opt 输出的最优解
     * @param work 工作数组
     * @param next 下一个约束索引数组
     * @param prev 前一个约束索引数组
     * @return 求解状态
     */
    template <>
    inline int linfracprog<1>(const double *halves,
                              const int max_size,
                              const int m,
                              const double *n_vec,
                              const double *d_vec,
                              double *opt,
                              double *work,
                              int *next,
                              int *prev)
    {
        if (m > 0)
        {
            return lp_base_case((const double(*)[2])halves, m,
                                n_vec, d_vec, opt, next, prev);
        }
        else
        {
            return lp_no_con<1>(n_vec, d_vec, opt);
        }
    }

    /**
     * @brief 生成随机排列
     * @param n 排列长度
     * @param p 输出的排列数组
     * 
     * 使用Fisher-Yates洗牌算法生成0到n-1的随机排列
     * 用于随机化约束处理顺序，提高算法性能
     */
    inline void rand_permutation(const int n,
                                 int *p)
    {
        typedef std::uniform_int_distribution<int> rand_int;
        typedef rand_int::param_type rand_range;
        static std::mt19937_64 gen;
        static rand_int rdi(0, 1);
        int j, k;
        for (int i = 0; i < n; i++)
        {
            p[i] = i;
        }
        for (int i = 0; i < n; i++)
        {
            rdi.param(rand_range(0, n - i - 1));
            j = rdi(gen) + i;
            k = p[j];
            p[j] = p[i];
            p[i] = k;
        }
    }

    /**
     * @brief Eigen接口的线性规划求解器
     * @tparam d 决策变量的维度
     * @param c 目标函数系数向量
     * @param A 约束矩阵 (m×d)
     * @param b 约束右端向量 (m×1)
     * @param x 输出的最优解向量 (d×1)
     * @return 最优目标函数值
     * 
     * 求解标准形式的线性规划问题：
     * minimize    c^T * x
     * subject to  A * x <= b
     * 
     * 使用Seidel算法，特别适用于小维度问题
     */
    template <int d>
    inline double linprog(const Eigen::Matrix<double, d, 1> &c,
                          const Eigen::Matrix<double, -1, d> &A,
                          const Eigen::Matrix<double, -1, 1> &b,
                          Eigen::Matrix<double, d, 1> &x)
    /*
    **  标准线性规划问题：min c^T*x, s.t. A*x<=b
    **  其中 dim(x) << dim(b)（低维决策变量，适合Seidel算法）
    */
    {
        int m = b.size() + 1;
        x.setZero();
        
        // 如果没有约束，检查目标函数是否有界
        if (m <= 1)
        {
            return c.cwiseAbs().maxCoeff() > 0.0 ? -INFINITY : 0.0;
        }

        // 准备数据结构
        Eigen::VectorXi perm(m - 1);                    // 随机排列数组
        Eigen::VectorXi next(m);                        // 下一个约束索引
        Eigen::VectorXi prev(m + 1);                    // 前一个约束索引（修复了原始bug）
        Eigen::Matrix<double, d + 1, 1> n_vec;          // 齐次化的目标函数向量
        Eigen::Matrix<double, d + 1, 1> d_vec;          // 齐次化的分母向量
        Eigen::Matrix<double, d + 1, 1> opt;            // 齐次化的最优解
        Eigen::Matrix<double, d + 1, -1, Eigen::ColMajor> halves(d + 1, m);  // 齐次化的约束矩阵
        Eigen::VectorXd work((m + 3) * (d + 2) * (d - 1) / 2);              // 工作数组

        // 构造齐次化约束矩阵
        halves.col(0).setZero();                        // 第一列对应齐次化坐标
        halves(d, 0) = 1.0;                            // 齐次化约束
        halves.topRightCorner(d, m - 1) = -A.transpose();    // -A^T (约束矩阵转置并取负)
        halves.bottomRightCorner(1, m - 1) = b.transpose();  // b^T (右端向量)
        
        // 归一化所有约束向量（linfracprog要求）
        halves.colwise().normalize();
        
        // 设置目标函数向量（齐次化）
        n_vec.head(d) = c;
        n_vec(d) = 0.0;
        d_vec.setZero();
        d_vec(d) = 1.0;

        // 随机化输入约束的处理顺序
        rand_permutation(m - 1, perm.data());
        
        // 构建约束处理的双向链表
        prev(0) = 0;                                   // 第0个位置的前驱
        next(0) = perm(0) + 1;                        // 链接第0个位置到第一个随机约束
        prev(perm(0) + 1) = 0;
        
        // 链接其他约束
        for (int i = 0; i < m - 2; i++)
        {
            next(perm(i) + 1) = perm(i + 1) + 1;
            prev(perm(i + 1) + 1) = perm(i) + 1;
        }
        next(perm(m - 2) + 1) = m;                    // 标记最后一个约束

        // 调用核心的线性分式规划求解器
        int status = sdlp::linfracprog<d>(halves.data(), m, m,
                                          n_vec.data(), d_vec.data(),
                                          opt.data(), work.data(),
                                          next.data(), prev.data());

        // 处理求解结果（linprog和linfracprog的状态定义不同）
        double minimum = INFINITY;
        if (status != sdlp::INFEASIBLE)
        {
            if (opt(d) != 0.0 && status != sdlp::UNBOUNDED)
            {
                x = opt.head(d) / opt(d);              // 从齐次坐标转换为笛卡尔坐标
                minimum = c.dot(x);                    // 计算最优目标函数值
            }

            if (opt(d) == 0.0 || status == sdlp::UNBOUNDED)
            {
                x = opt.head(d);           // 无界情况下的方向向量
                minimum = -INFINITY;       // 目标函数值为负无穷
            }
        }

        return minimum;                    // 返回最优目标函数值
    }

} // namespace sdlp

#endif
