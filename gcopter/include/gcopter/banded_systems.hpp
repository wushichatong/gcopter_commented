
#pragma once

#include <Eigen/Eigen>

#include <cmath>
#include <vector>

/**
 * @namespace minco
 * @brief MINCO轨迹优化算法命名空间
 * 
 * 包含所有与最小控制输入轨迹生成相关的类和函数
 */
namespace minco
{


    /**
     * @class BandedSystem
     * @brief 带状线性系统求解器
     * 
     * 用于高效求解带状线性方程组 Ax=b。
     * A是一个N×N的带状矩阵，具有下带宽lowerBw和上带宽upperBw。
     * 带状LU分解的时间复杂度为O(N)，比一般矩阵的O(N³)要高效得多。
     * 
     * 带状矩阵结构特别适用于样条插值和有限差分等数值计算问题，
     * 其中系数矩阵通常具有稀疏的带状结构。
     */
    class BandedSystem
    {
    public:
        /**
         * @brief 创建带状系统
         * @param n 矩阵的维度 (N×N)
         * @param p 下带宽 (主对角线下方的非零带数)
         * @param q 上带宽 (主对角线上方的非零带数)
         * 
         * 分配内存并初始化带状矩阵存储结构
         */
        inline void create(const int &n, const int &p, const int &q)
        {
            // 防止在销毁前重新创建的情况
            destroy();
            N = n;
            lowerBw = p;
            upperBw = q;
            int actualSize = N * (lowerBw + upperBw + 1);
            ptrData = new double[actualSize];
            std::fill_n(ptrData, actualSize, 0.0);
            return;
        }

        /**
         * @brief 销毁带状系统，释放内存
         */
        inline void destroy()
        {
            if (ptrData != nullptr)
            {
                delete[] ptrData;
                ptrData = nullptr;
            }
            return;
        }

    private:
        int N;           ///< 矩阵维度
        int lowerBw;     ///< 下带宽
        int upperBw;     ///< 上带宽
        /// 数据指针（强制性nullptr初始化）
        double *ptrData = nullptr;

    public:
        /**
         * @brief 重置矩阵为零矩阵
         */
        inline void reset(void)
        {
            std::fill_n(ptrData, N * (lowerBw + upperBw + 1), 0.0);
            return;
        }

        /**
         * @brief 常量访问运算符
         * @param i 行索引
         * @param j 列索引
         * @return 矩阵元素的常量引用
         * 
         * 带状矩阵按照"Matrix Computation"建议的方式存储
         * 存储格式将带状矩阵压缩为矩形数组
         */
        inline const double &operator()(const int &i, const int &j) const
        {
            return ptrData[(i - j + upperBw) * N + j];
        }

        /**
         * @brief 非常量访问运算符
         * @param i 行索引  
         * @param j 列索引
         * @return 矩阵元素的引用
         */
        inline double &operator()(const int &i, const int &j)
        {
            return ptrData[(i - j + upperBw) * N + j];
        }

        /**
         * @brief 执行带状LU分解（就地分解）
         * 
         * 注意：为了效率考虑，不对矩阵"A"应用主元选择！
         * 
         * 算法过程：
         * 1. 对每一列进行高斯消元
         * 2. 利用带状结构减少计算量
         * 3. 将L和U矩阵存储在原矩阵空间中
         * 
         * 时间复杂度：O(N × lowerBw × upperBw)
         */
        inline void factorizeLU()
        {
            int iM, jM;
            double cVl;
            // 对每个主对角线元素进行LU分解
            for (int k = 0; k <= N - 2; k++)
            {
                iM = std::min(k + lowerBw, N - 1);  // 当前列的下边界
                cVl = operator()(k, k);             // 当前主元
                
                // 计算L矩阵的第k列（消元因子）
                for (int i = k + 1; i <= iM; i++)
                {
                    if (operator()(i, k) != 0.0)
                    {
                        operator()(i, k) /= cVl;    // L[i,k] = A[i,k] / A[k,k]
                    }
                }
                
                jM = std::min(k + upperBw, N - 1);  // 当前行的右边界
                
                // 更新剩余子矩阵（高斯消元）
                for (int j = k + 1; j <= jM; j++)
                {
                    cVl = operator()(k, j);         // U[k,j]
                    if (cVl != 0.0)
                    {
                        for (int i = k + 1; i <= iM; i++)
                        {
                            if (operator()(i, k) != 0.0)
                            {
                                // A[i,j] = A[i,j] - L[i,k] * U[k,j]
                                operator()(i, j) -= operator()(i, k) * cVl;
                            }
                        }
                    }
                }
            }
            return;
        }

        /**
         * @brief 求解线性方程组 Ax=b，结果存储在b中
         * @tparam EIGENMAT Eigen矩阵类型
         * @param b 输入的右端向量矩阵，大小为N×m（求解m个向量）
         * 
         * 该函数实现两步求解过程：
         * 1. 前向替换：求解 Ly = b
         * 2. 后向替换：求解 Ux = y
         * 
         * 利用带状矩阵的稀疏结构提高求解效率
         */
        template <typename EIGENMAT>
        inline void solve(EIGENMAT &b) const
        {
            int iM;
            
            // 第一步：前向替换求解 Ly = b
            for (int j = 0; j <= N - 1; j++)
            {
                iM = std::min(j + lowerBw, N - 1);
                for (int i = j + 1; i <= iM; i++)
                {
                    if (operator()(i, j) != 0.0)
                    {
                        // y[i] = b[i] - L[i,j] * y[j]
                        b.row(i) -= operator()(i, j) * b.row(j);
                    }
                }
            }
            
            // 第二步：后向替换求解 Ux = y
            for (int j = N - 1; j >= 0; j--)
            {
                // x[j] = y[j] / U[j,j]
                b.row(j) /= operator()(j, j);
                iM = std::max(0, j - upperBw);
                for (int i = iM; i <= j - 1; i++)
                {
                    if (operator()(i, j) != 0.0)
                    {
                        // x[i] = x[i] - U[i,j] * x[j]
                        b.row(i) -= operator()(i, j) * b.row(j);
                    }
                }
            }
            return;
        }

        /**
         * @brief 求解转置线性方程组 A^T x = b，结果存储在b中
         * @tparam EIGENMAT Eigen矩阵类型
         * @param b 输入的右端向量矩阵，大小为N×m（求解m个向量）
         * 
         * 求解矩阵A的转置与向量的乘积方程。
         * 对于带状LU分解，A^T = (LU)^T = U^T L^T
         * 
         * 求解步骤：
         * 1. 前向替换：求解 U^T y = b  
         * 2. 后向替换：求解 L^T x = y
         */
        template <typename EIGENMAT>
        inline void solveAdj(EIGENMAT &b) const
        {
            int iM;
            
            // 第一步：前向替换求解 U^T y = b
            for (int j = 0; j <= N - 1; j++)
            {
                // y[j] = b[j] / U^T[j,j] = b[j] / U[j,j]
                b.row(j) /= operator()(j, j);
                iM = std::min(j + upperBw, N - 1);
                for (int i = j + 1; i <= iM; i++)
                {
                    if (operator()(j, i) != 0.0)
                    {
                        // y[i] = y[i] - U^T[j,i] * y[j] = y[i] - U[j,i] * y[j]
                        b.row(i) -= operator()(j, i) * b.row(j);
                    }
                }
            }
            
            // 第二步：后向替换求解 L^T x = y
            for (int j = N - 1; j >= 0; j--)
            {
                iM = std::max(0, j - lowerBw);
                for (int i = iM; i <= j - 1; i++)
                {
                    if (operator()(j, i) != 0.0)
                    {
                        // x[i] = x[i] - L^T[j,i] * x[j] = x[i] - L[i,j] * x[j]
                        b.row(i) -= operator()(j, i) * b.row(j);
                    }
                }
            }
            return;
        }
    };

} // namespace minco