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
 * @file geo_utils.hpp
 * @brief 几何工具库：多面体几何处理和计算
 * 
 * 该文件提供了处理3D几何多面体的核心工具函数，主要用于：
 * 1. 凸多面体的内部点查找
 * 2. 多面体重叠检测
 * 3. 顶点枚举和凸包计算
 * 4. 数值精度处理和去重
 * 
 * 多面体表示约定：
 * - 使用半空间表示：每行 [h0, h1, h2, h3] 表示约束 h0*x + h1*y + h2*z + h3 ≤ 0
 * - 法向量指向外部：约束定义了多面体的外表面
 * - 内部点满足所有约束：所有不等式同时成立
 * 
 * 应用场景：
 * - 轨迹规划中的安全走廊计算
 * - 障碍物与安全区域的重叠检测
 * - 凸优化问题的可行域处理
 * - 几何约束的数值稳定性处理
 */

#ifndef GEO_UTILS_HPP
#define GEO_UTILS_HPP

#include "quickhull.hpp"    // QuickHull凸包算法
#include "sdlp.hpp"         // 简单线性规划求解器

#include <Eigen/Eigen>

#include <cfloat>
#include <cstdint>
#include <set>
#include <chrono>

namespace geo_utils
{

    // 多面体表示：每行 [h0, h1, h2, h3] 定义半空间约束
    // h0*x + h1*y + h2*z + h3 ≤ 0
    /**
     * @brief 内部点查找函数：查找凸多面体的内部点
     * 
     * 使用线性规划方法寻找给定凸多面体的内部点。算法原理：
     * 1. 将半空间约束归一化：使法向量为单位向量
     * 2. 构建线性规划：最大化到所有约束面的最小距离
     * 3. 目标函数：max t，约束：norm(h)*x + t ≤ -h3/|h|
     * 4. 如果最优值 t > 0，则找到严格内部点
     * 
     * 数学原理：
     * - 距离归一化：将约束h*x + h3 ≤ 0转换为单位法向量形式
     * - 几何解释：寻找到所有约束面距离最大的点
     * - 等价于求解切比雪夫中心问题的简化版本
     * 
     * @param hPoly 多面体约束矩阵，每行 [h0, h1, h2, h3]
     * @param interior 输出的内部点坐标
     * @return bool 是否成功找到内部点（距离 > 0）
     * 
     * 实现细节：
     * - 法向量归一化防止数值不稳定
     * - 松弛变量表示到约束面的距离
     * - 线性规划求解最大内切球半径
     */
    inline bool findInterior(const Eigen::MatrixX4d &hPoly,
                             Eigen::Vector3d &interior)
    {
        // 获取约束数量
        const int m = hPoly.rows();

        // 构建线性规划问题：[x1, x2, x3, t]，其中t为距离变量
        Eigen::MatrixX4d A(m, 4);           // 约束矩阵
        Eigen::VectorXd b(m);               // 约束右端向量  
        Eigen::Vector4d c, x;               // 目标函数和解向量
        
        // 归一化处理：计算每个约束的法向量模长
        const Eigen::ArrayXd hNorm = hPoly.leftCols<3>().rowwise().norm();
        
        // 构建归一化约束矩阵 A = [h_norm, 1]
        A.leftCols<3>() = hPoly.leftCols<3>().array().colwise() / hNorm;  // 单位法向量
        A.rightCols<1>().setConstant(1.0);                                // 距离变量系数
        
        // 构建右端向量：b = -h3/|h|（归一化距离）
        b = -hPoly.rightCols<1>().array() / hNorm;
        
        // 目标函数：c = [0, 0, 0, -1]^T，最大化距离t
        c.setZero();
        c(3) = -1.0;  // 最大化t等价于最小化-t

        // 调用线性规划求解器：min c^T*x subject to A*x ≤ b
        const double minmaxsd = sdlp::linprog<4>(c, A, b, x);
        
        // 提取空间坐标（前3个分量）
        interior = x.head<3>();

        // 检查结果：minmaxsd < 0 表示 t > 0，即找到严格内部点
        return minmaxsd < 0.0 && !std::isinf(minmaxsd);
    }

    /**
     * @brief 多面体重叠检测：判断两个凸多面体是否相交
     * 
     * 使用分离轴定理的对偶形式进行重叠检测。算法原理：
     * 1. 合并两个多面体的所有约束
     * 2. 寻找同时满足两组约束的内部点
     * 3. 如果存在公共内部点，则两多面体重叠
     * 4. 通过线性规划求解可行性问题
     * 
     * 数学原理：
     * - 两凸集相交 ⟺ 存在点同时属于两个凸集
     * - 转化为可行性问题：find x s.t. Ax ≤ b AND Cx ≤ d
     * - 等价于联合约束系统：[A; C]*x ≤ [b; d]
     * - 使用内部点查找算法判断联合约束的可行性
     * 
     * @param hPoly0 第一个多面体的约束矩阵
     * @param hPoly1 第二个多面体的约束矩阵  
     * @param eps 数值容差，避免边界情况的误判
     * @return bool 是否存在重叠（true=重叠，false=分离）
     * 
     * 应用场景：
     * - 轨迹规划中的碰撞检测
     * - 安全走廊的有效性验证
     * - 多机器人系统的空间冲突检测
     * - 动态障碍物的预测碰撞判断
     */
    inline bool overlap(const Eigen::MatrixX4d &hPoly0,
                        const Eigen::MatrixX4d &hPoly1,
                        const double eps = 1.0e-6)

    {
        // 获取两个多面体的约束数量
        const int m = hPoly0.rows();  // 第一个多面体的约束数
        const int n = hPoly1.rows();  // 第二个多面体的约束数
        
        // 构建联合约束系统：合并两个多面体的所有约束
        Eigen::MatrixX4d A(m + n, 4);     // 联合约束矩阵
        Eigen::Vector4d c, x;             // 目标函数和解向量
        Eigen::VectorXd b(m + n);         // 联合约束右端向量

        // 填充联合约束矩阵：上半部分为第一个多面体
        A.leftCols<3>().topRows(m) = hPoly0.leftCols<3>();     // 第一个多面体的法向量
        A.leftCols<3>().bottomRows(n) = hPoly1.leftCols<3>();  // 第二个多面体的法向量
        A.rightCols<1>().setConstant(1.0);                     // 距离变量列（全1）

        // 填充右端向量：提取距离约束
        b.topRows(m) = -hPoly0.rightCols<1>();    // 第一个多面体的距离约束
        b.bottomRows(n) = -hPoly1.rightCols<1>();  // 第二个多面体的距离约束

        // 目标函数：最大化公共内部距离
        c.setZero();
        c(3) = -1.0;  // 最大化t等价于最小化-t

        // 求解联合可行性问题
        const double minmaxsd = sdlp::linprog<4>(c, A, b, x);

        // 判断重叠：考虑数值容差的严格内部点存在性
        // minmaxsd < -eps 意味着公共内部距离 > eps，确认重叠
        return minmaxsd < -eps && !std::isinf(minmaxsd);
    }

    /**
     * @brief 顶点比较器：用于顶点去重的字典序比较
     * 
     * 实现3D点的严格弱序关系，用于std::set的排序和去重。
     * 比较规则：先比较x坐标，再比较y坐标，最后比较z坐标。
     * 
     * 数学性质：
     * - 反自反性：!(a < a)
     * - 非对称性：a < b ⟹ !(b < a)  
     * - 传递性：a < b ∧ b < c ⟹ a < c
     * 
     * @note 用于消除数值计算中的重复顶点
     */
    struct filterLess
    {
        /**
         * @brief 字典序比较操作符
         * @param l 左操作数（3D点）
         * @param r 右操作数（3D点）
         * @return bool l是否字典序小于r
         */
        inline bool operator()(const Eigen::Vector3d &l,
                               const Eigen::Vector3d &r)
        {
            // 字典序比较：x优先，然后y，最后z
            return l(0) < r(0) ||                              // x坐标比较
                   (l(0) == r(0) &&                           // x相等时比较y
                    (l(1) < r(1) ||                           // y坐标比较
                     (l(1) == r(1) &&                         // y也相等时比较z
                      l(2) < r(2))));                         // z坐标比较
        }
    };

    /**
     * @brief 顶点过滤函数：去除重复顶点和数值噪声
     * 
     * 使用自适应容差进行顶点去重，避免数值精度导致的伪重复。
     * 算法特点：
     * 1. 自适应精度：根据数据量级确定容差
     * 2. 量化舍入：将坐标量化到网格点
     * 3. 集合去重：使用有序集合消除重复
     * 4. 保持原值：输出保持原始精度
     * 
     * 数值稳定性考虑：
     * - 防止浮点误差导致的伪重复顶点
     * - 容差与数据量级成比例，避免过度或不足的精度
     * - 保证几何一致性和拓扑正确性
     * 
     * @param rV 原始顶点矩阵，每列为一个3D顶点
     * @param epsilon 用户指定的容差参数
     * @param fV 输出的过滤后顶点矩阵
     * 
     * 应用场景：
     * - 凸包计算前的预处理
     * - 多面体顶点的清理和优化  
     * - 数值算法的稳定性提升
     */
    inline void filterVs(const Eigen::Matrix3Xd &rV,
                         const double &epsilon,
                         Eigen::Matrix3Xd &fV)
    {
        // 计算数据的动态范围：用于自适应精度控制
        const double mag = std::max(fabs(rV.maxCoeff()), fabs(rV.minCoeff()));
        
        // 确定有效分辨率：在用户容差和机器精度之间取最大值
        const double res = mag * std::max(fabs(epsilon) / mag, DBL_EPSILON);
        
        // 使用有序集合进行去重：基于量化坐标的字典序
        std::set<Eigen::Vector3d, filterLess> filter;
        
        // 初始化输出矩阵
        fV = rV;
        int offset = 0;              // 有效顶点计数器
        Eigen::Vector3d quanti;      // 量化后的坐标
        
        // 遍历所有输入顶点
        for (int i = 0; i < rV.cols(); i++)
        {
            // 坐标量化：舍入到网格点以消除数值噪声
            quanti = (rV.col(i) / res).array().round();
            
            // 检查量化坐标是否已存在
            if (filter.find(quanti) == filter.end())
            {
                // 新顶点：加入集合并保存原始坐标
                filter.insert(quanti);
                fV.col(offset) = rV.col(i);  // 保持原始精度
                offset++;
            }
            // 重复顶点被自动跳过
        }
        
        // 截取有效部分：只保留去重后的顶点
        fV = fV.leftCols(offset).eval();
        return;
    }

    // 多面体表示：每行 [h0, h1, h2, h3] 定义半空间约束
    // h0*x + h1*y + h2*z + h3 ≤ 0
    // 推荐容差：epsilon = 1.0e-6
    /**
     * @brief 顶点枚举函数：计算凸多面体的所有顶点
     * 
     * 使用对偶变换和QuickHull算法枚举凸多面体的顶点。算法流程：
     * 1. 坐标变换：将问题转换为相对于内部点的坐标系
     * 2. 对偶变换：将半空间约束转换为点集
     * 3. 凸包计算：使用QuickHull算法计算对偶点的凸包  
     * 4. 反变换：将对偶凸包的面转换为原始多面体的顶点
     * 5. 坐标还原：转换回原始坐标系
     * 
     * 数学原理（极对偶性）：
     * - 原始多面体：P = {x | Ax + b ≤ 0}
     * - 坐标变换：y = x - x_inner，将内部点移至原点
     * - 对偶变换：每个约束a^T*y ≤ c对应对偶点a/c  
     * - 对偶定理：原始顶点 ↔ 对偶面，原始面 ↔ 对偶顶点
     * - 凸包算法：计算对偶点集的凸包
     * 
     * @param hPoly 多面体的半空间表示
     * @param inner 多面体的内部点（必须严格在内部）
     * @param vPoly 输出的顶点矩阵，每列为一个顶点
     * @param epsilon 数值容差，用于QuickHull和去重
     * 
     * 注意事项：
     * - 内部点必须严格在多面体内部
     * - 算法复杂度取决于QuickHull的性能
     * - 输出顶点已自动去重和数值稳定化
     */
    inline void enumerateVs(const Eigen::MatrixX4d &hPoly,
                            const Eigen::Vector3d &inner,
                            Eigen::Matrix3Xd &vPoly,
                            const double epsilon = 1.0e-6)
    {
        // 步骤1：坐标变换 - 计算相对于内部点的约束距离
        // 对于约束h*x + d ≤ 0和内部点x_inner，计算b = -(d + h*x_inner)
        const Eigen::VectorXd b = -hPoly.rightCols<1>() - hPoly.leftCols<3>() * inner;
        
        // 步骤2：对偶变换 - 将半空间约束转换为对偶点
        // 约束h*y ≤ b对应对偶点h/b（其中y = x - x_inner）
        const Eigen::Matrix<double, 3, -1, Eigen::ColMajor> A =
            (hPoly.leftCols<3>().array().colwise() / b.array()).transpose();

        // 步骤3：凸包计算 - 使用QuickHull算法
        quickhull::QuickHull<double> qh;
        const double qhullEps = std::min(epsilon, quickhull::defaultEps<double>());
        
        // 计算对偶点集的凸包（CCW=false因为QuickHull法向量指向内部）
        const auto cvxHull = qh.getConvexHull(A.data(), A.cols(), false, true, qhullEps);
        const auto &idBuffer = cvxHull.getIndexBuffer();  // 三角面片索引
        const int hNum = idBuffer.size() / 3;             // 面片数量

        // 步骤4：反变换 - 从对偶面计算原始顶点
        Eigen::Matrix3Xd rV(3, hNum);     // 原始顶点集合
        Eigen::Vector3d normal, point, edge0, edge1;
        
        for (int i = 0; i < hNum; i++)
        {
            // 提取三角面片的三个顶点
            point = A.col(idBuffer[3 * i + 1]);        // 面片中心点
            edge0 = point - A.col(idBuffer[3 * i]);    // 第一条边
            edge1 = A.col(idBuffer[3 * i + 2]) - point; // 第二条边
            
            // 计算面法向量（叉积，CW顺序产生外法向量）
            normal = edge0.cross(edge1);
            
            // 对偶变换：面法向量/到原点距离 = 原始顶点在变换坐标系中的位置
            rV.col(i) = normal / normal.dot(point);
        }
        
        // 步骤5：顶点去重和数值稳定化
        filterVs(rV, epsilon, vPoly);
        
        // 步骤6：坐标还原 - 转换回原始坐标系
        vPoly = (vPoly.array().colwise() + inner.array()).eval();
        return;
    }

    // 多面体表示：每行 [h0, h1, h2, h3] 定义半空间约束  
    // h0*x + h1*y + h2*z + h3 ≤ 0
    // 推荐容差：epsilon = 1.0e-6
    /**
     * @brief 顶点枚举函数（自动内部点版本）：计算凸多面体的所有顶点
     * 
     * 这是enumerateVs的便利版本，自动查找内部点然后进行顶点枚举。
     * 适用于不确定内部点位置的情况。
     * 
     * 算法流程：
     * 1. 调用findInterior查找多面体的内部点
     * 2. 如果找到内部点，调用完整版本的enumerateVs
     * 3. 返回操作是否成功完成
     * 
     * @param hPoly 多面体的半空间表示
     * @param vPoly 输出的顶点矩阵，每列为一个顶点  
     * @param epsilon 数值容差，用于内部点查找和顶点枚举
     * @return bool 是否成功计算顶点（多面体非空且算法收敛）
     * 
     * 错误情况：
     * - 多面体为空集（无内部点）
     * - 数值不稳定（极薄或退化多面体）
     * - 约束矛盾（过约束系统）
     * 
     * 使用建议：
     * - 优先使用此版本，除非已知内部点
     * - 对于复杂多面体，可能需要调整epsilon
     * - 返回false时检查输入约束的一致性
     */
    inline bool enumerateVs(const Eigen::MatrixX4d &hPoly,
                            Eigen::Matrix3Xd &vPoly,
                            const double epsilon = 1.0e-6)
    {
        // 第一步：查找多面体的内部点
        Eigen::Vector3d inner;
        if (findInterior(hPoly, inner))
        {
            // 内部点找到：调用完整版本进行顶点枚举
            enumerateVs(hPoly, inner, vPoly, epsilon);
            return true;
        }
        else
        {
            // 内部点未找到：多面体可能为空集或数值问题
            return false;
        }
    }

} // namespace geo_utils

/**
 * @brief 几何工具库总结
 * 
 * 本文件实现了轨迹规划中必需的几何计算工具：
 * 
 * 核心算法：
 * 1. 内部点查找 - 基于线性规划的切比雪夫中心求解
 * 2. 重叠检测 - 基于分离轴定理的可行性判断  
 * 3. 顶点枚举 - 基于极对偶变换的QuickHull算法
 * 4. 数值稳定化 - 自适应精度的去重过滤
 * 
 * 数学基础：
 * - 凸几何学：半空间表示、极对偶性
 * - 线性规划：可行性问题、最优化理论
 * - 计算几何：凸包算法、数值稳定性
 * 
 * 应用领域：
 * - 轨迹规划的安全走廊计算
 * - 多机器人系统的空间冲突检测
 * - 凸优化问题的约束处理
 * - 动态环境中的几何推理
 * 
 * 性能特点：
 * - 数值稳定：自适应精度控制
 * - 计算高效：利用线性规划和凸包算法
 * - 接口简洁：支持多种使用模式
 * - 错误处理：完善的边界情况处理
 */

#endif
