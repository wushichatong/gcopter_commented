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

#ifndef TRAJECTORY_HPP
#define TRAJECTORY_HPP

#include "gcopter/root_finder.hpp"  // 多项式求根器

#include <Eigen/Eigen>

#include <iostream>
#include <cmath>
#include <cfloat>
#include <vector>

/**
 * @class Piece
 * @brief D阶多项式轨迹段类，表示一段连续的3D轨迹
 * @tparam D 多项式阶数
 * 
 * 每个轨迹段由一个D阶多项式表示，形式为：
 * p(t) = c_D * t^D + c_(D-1) * t^(D-1) + ... + c_1 * t + c_0
 * 其中每个系数c_i都是3维向量，分别对应x, y, z轴
 */
template <int D>
class Piece
{
public:
    // 类型定义
    typedef Eigen::Matrix<double, 3, D + 1> CoefficientMat;     // 位置系数矩阵 [3 x (D+1)]
    typedef Eigen::Matrix<double, 3, D> VelCoefficientMat;      // 速度系数矩阵 [3 x D]
    typedef Eigen::Matrix<double, 3, D - 1> AccCoefficientMat;  // 加速度系数矩阵 [3 x (D-1)]

private:
    double duration;           // 轨迹段持续时间
    CoefficientMat coeffMat;   // 多项式系数矩阵，每列对应一个系数项

public:
    /**
     * @brief 默认构造函数
     */
    Piece() = default;


    /**
     * @brief 构造函数
     * @param dur 轨迹段持续时间
     * @param cMat 多项式系数矩阵
     */
    Piece(double dur, const CoefficientMat &cMat)
        : duration(dur), coeffMat(cMat) {}

    /**
     * @brief 获取轨迹维度（固定为3D）
     * @return 维度数（3）
     */
    inline int getDim() const
    {
        return 3;
    }

    /**
     * @brief 获取多项式阶数
     * @return 多项式阶数D
     */
    inline int getDegree() const
    {
        return D;
    }

    /**
     * @brief 获取轨迹段持续时间
     * @return 持续时间（秒）
     */
    inline double getDuration() const
    {
        return duration;
    }

    /**
     * @brief 获取多项式系数矩阵
     * @return 系数矩阵的常量引用
     */
    inline const CoefficientMat &getCoeffMat() const
    {
        return coeffMat;
    }

    /**
     * @brief 计算指定时刻的位置
     * @param t 时间参数 [0, duration]
     * @return 3D位置向量
     * 
     * 使用Horner方法高效计算多项式值：
     * p(t) = c_D * t^D + c_(D-1) * t^(D-1) + ... + c_1 * t + c_0
     */
    inline Eigen::Vector3d getPos(const double &t) const
    {
        Eigen::Vector3d pos(0.0, 0.0, 0.0);
        double tn = 1.0;  // t的n次幂
        // 从高阶到低阶逐项计算
        for (int i = D; i >= 0; i--)
        {
            pos += tn * coeffMat.col(i);
            tn *= t;
        }
        return pos;
    }

    /**
     * @brief 计算指定时刻的速度（位置的一阶导数）
     * @param t 时间参数 [0, duration]
     * @return 3D速度向量
     * 
     * 速度多项式：v(t) = D*c_D*t^(D-1) + (D-1)*c_(D-1)*t^(D-2) + ... + c_1
     */
    inline Eigen::Vector3d getVel(const double &t) const
    {
        Eigen::Vector3d vel(0.0, 0.0, 0.0);
        double tn = 1.0;  // t的n次幂
        int n = 1;        // 导数系数
        for (int i = D - 1; i >= 0; i--)
        {
            vel += n * tn * coeffMat.col(i);
            tn *= t;
            n++;
        }
        return vel;
    }


    /**
     * @brief 计算指定时刻的加速度（位置的二阶导数）
     * @param t 时间参数 [0, duration]
     * @return 3D加速度向量
     * 
     * 加速度多项式：a(t) = D*(D-1)*c_D*t^(D-2) + (D-1)*(D-2)*c_(D-1)*t^(D-3) + ... + 2*c_2
     */
    inline Eigen::Vector3d getAcc(const double &t) const
    {
        Eigen::Vector3d acc(0.0, 0.0, 0.0);
        double tn = 1.0;  // t的n次幂
        int m = 1;        // 第一个导数系数
        int n = 2;        // 第二个导数系数
        for (int i = D - 2; i >= 0; i--)
        {
            acc += m * n * tn * coeffMat.col(i);
            tn *= t;
            m++;
            n++;
        }
        return acc;
    }

    /**
     * @brief 计算指定时刻的加加速度/急动度（位置的三阶导数）
     * @param t 时间参数 [0, duration]
     * @return 3D加加速度向量
     * 
     * 加加速度多项式：j(t) = D*(D-1)*(D-2)*c_D*t^(D-3) + ... + 6*c_3
     */
    inline Eigen::Vector3d getJer(const double &t) const
    {
        Eigen::Vector3d jer(0.0, 0.0, 0.0);
        double tn = 1.0;  // t的n次幂
        int l = 1;        // 第一个导数系数
        int m = 2;        // 第二个导数系数
        int n = 3;        // 第三个导数系数
        for (int i = D - 3; i >= 0; i--)
        {
            jer += l * m * n * tn * coeffMat.col(i);
            tn *= t;
            l++;
            m++;
            n++;
        }
        return jer;
    }

    /**
     * @brief 归一化位置系数矩阵
     * @return 归一化后的位置系数矩阵
     * 
     * 将参数时间从[0,1]映射到[0,duration]的系数变换
     */
    inline CoefficientMat normalizePosCoeffMat() const
    {
        CoefficientMat nPosCoeffsMat;
        double t = 1.0;
        for (int i = D; i >= 0; i--)
        {
            nPosCoeffsMat.col(i) = coeffMat.col(i) * t;
            t *= duration;  // t = duration^(D-i)
        }
        return nPosCoeffsMat;
    }

    /**
     * @brief 归一化速度系数矩阵
     * @return 归一化后的速度系数矩阵
     */
    inline VelCoefficientMat normalizeVelCoeffMat() const
    {
        VelCoefficientMat nVelCoeffMat;
        int n = 1;
        double t = duration;  // duration的幂次
        for (int i = D - 1; i >= 0; i--)
        {
            nVelCoeffMat.col(i) = n * coeffMat.col(i) * t;
            t *= duration;
            n++;
        }
        return nVelCoeffMat;
    }

    /**
     * @brief 归一化加速度系数矩阵
     * @return 归一化后的加速度系数矩阵
     */
    inline AccCoefficientMat normalizeAccCoeffMat() const
    {
        AccCoefficientMat nAccCoeffMat;
        int n = 2;        // 第二个导数系数
        int m = 1;        // 第一个导数系数
        double t = duration * duration;  // duration的平方开始
        for (int i = D - 2; i >= 0; i--)
        {
            nAccCoeffMat.col(i) = n * m * coeffMat.col(i) * t;
            n++;
            m++;
            t *= duration;
        }
        return nAccCoeffMat;
    }

    /**
     * @brief 计算轨迹段内的最大速度模长
     * @return 最大速度值
     * 
     * 通过求解速度模长平方的导数为零的点来找到极值
     */
    inline double getMaxVelRate() const
    {
        VelCoefficientMat nVelCoeffMat = normalizeVelCoeffMat();
        
        // 计算速度模长的平方：|v(t)|^2 = vx^2 + vy^2 + vz^2
        Eigen::VectorXd coeff = RootFinder::polySqr(nVelCoeffMat.row(0)) +
                                RootFinder::polySqr(nVelCoeffMat.row(1)) +
                                RootFinder::polySqr(nVelCoeffMat.row(2));
        int N = coeff.size();
        int n = N - 1;
        
        // 计算导数系数（寻找极值点）
        for (int i = 0; i < N; i++)
        {
            coeff(i) *= n;
            n--;
        }
        
        // 如果导数系数太小，说明速度基本恒定
        if (coeff.head(N - 1).squaredNorm() < DBL_EPSILON)
        {
            return getVel(0.0).norm();
        }
        else
        {
            // 使用数值方法在扩展区间内寻找极值
            double l = -0.0625;   // 左边界（略小于0）
            double r = 1.0625;    // 右边界（略大于1）
            
            // 调整左边界，避免在边界上导数为零的情况
            while (fabs(RootFinder::polyVal(coeff.head(N - 1), l)) < DBL_EPSILON)
            {
                l = 0.5 * l;
            }
            // 调整右边界，避免在边界上导数为零的情况
            while (fabs(RootFinder::polyVal(coeff.head(N - 1), r)) < DBL_EPSILON)
            {
                r = 0.5 * (r + 1.0);
            }
            
            // 求解多项式的根（极值点）
            std::set<double> candidates = RootFinder::solvePolynomial(coeff.head(N - 1), l, r,
                                                                      FLT_EPSILON / duration);
            candidates.insert(0.0);  // 添加区间端点
            candidates.insert(1.0);
            
            // 在所有候选点中找到最大速度值
            double maxVelRateSqr = -INFINITY;
            double tempNormSqr;
            for (std::set<double>::const_iterator it = candidates.begin();
                 it != candidates.end();
                 it++)
            {
                if (0.0 <= *it && 1.0 >= *it)  // 确保在有效时间区间内
                {
                    tempNormSqr = getVel((*it) * duration).squaredNorm();
                    maxVelRateSqr = maxVelRateSqr < tempNormSqr ? tempNormSqr : maxVelRateSqr;
                }
            }
            return sqrt(maxVelRateSqr);  // 返回最大速度模长
        }
    }

    /**
     * @brief 计算轨迹段内的最大加速度模长
     * @return 最大加速度值
     * 
     * 类似于getMaxVelRate，通过求解加速度模长平方的导数为零的点来找到极值
     */
    inline double getMaxAccRate() const
    {
        AccCoefficientMat nAccCoeffMat = normalizeAccCoeffMat();
        
        // 计算加速度模长的平方：|a(t)|^2 = ax^2 + ay^2 + az^2
        Eigen::VectorXd coeff = RootFinder::polySqr(nAccCoeffMat.row(0)) +
                                RootFinder::polySqr(nAccCoeffMat.row(1)) +
                                RootFinder::polySqr(nAccCoeffMat.row(2));
        int N = coeff.size();
        int n = N - 1;
        
        // 计算导数系数（寻找极值点）
        for (int i = 0; i < N; i++)
        {
            coeff(i) *= n;
            n--;
        }
        
        // 如果导数系数太小，说明加速度基本恒定
        if (coeff.head(N - 1).squaredNorm() < DBL_EPSILON)
        {
            return getAcc(0.0).norm();
        }
        else
        {
            // 使用数值方法在扩展区间内寻找极值
            double l = -0.0625;
            double r = 1.0625;
            while (fabs(RootFinder::polyVal(coeff.head(N - 1), l)) < DBL_EPSILON)
            {
                l = 0.5 * l;
            }
            while (fabs(RootFinder::polyVal(coeff.head(N - 1), r)) < DBL_EPSILON)
            {
                r = 0.5 * (r + 1.0);
            }
            
            // 求解多项式的根（极值点）
            std::set<double> candidates = RootFinder::solvePolynomial(coeff.head(N - 1), l, r,
                                                                      FLT_EPSILON / duration);
            candidates.insert(0.0);  // 添加区间端点
            candidates.insert(1.0);
            
            // 在所有候选点中找到最大加速度值
            double maxAccRateSqr = -INFINITY;
            double tempNormSqr;
            for (std::set<double>::const_iterator it = candidates.begin();
                 it != candidates.end();
                 it++)
            {
                if (0.0 <= *it && 1.0 >= *it)  // 确保在有效时间区间内
                {
                    tempNormSqr = getAcc((*it) * duration).squaredNorm();
                    maxAccRateSqr = maxAccRateSqr < tempNormSqr ? tempNormSqr : maxAccRateSqr;
                }
            }
            return sqrt(maxAccRateSqr);  // 返回最大加速度模长
        }
    }

    /**
     * @brief 检查轨迹段是否满足最大速度约束
     * @param maxVelRate 最大允许速度
     * @return true表示满足约束，false表示违反约束
     */
    inline bool checkMaxVelRate(const double &maxVelRate) const
    {
        double sqrMaxVelRate = maxVelRate * maxVelRate;
        // 检查起点和终点的速度是否超限
        if (getVel(0.0).squaredNorm() >= sqrMaxVelRate ||
            getVel(duration).squaredNorm() >= sqrMaxVelRate)
        {
            return false;
        }
        else
        {
            // 使用多项式根计数方法检查整个区间
            VelCoefficientMat nVelCoeffMat = normalizeVelCoeffMat();
            Eigen::VectorXd coeff = RootFinder::polySqr(nVelCoeffMat.row(0)) +
                                    RootFinder::polySqr(nVelCoeffMat.row(1)) +
                                    RootFinder::polySqr(nVelCoeffMat.row(2));
            double t2 = duration * duration;
            coeff.tail<1>()(0) -= sqrMaxVelRate * t2;  // 减去约束值
            return RootFinder::countRoots(coeff, 0.0, 1.0) == 0;  // 无根表示不违反约束
        }
    }

    /**
     * @brief 检查轨迹段是否满足最大加速度约束
     * @param maxAccRate 最大允许加速度
     * @return true表示满足约束，false表示违反约束
     */
    inline bool checkMaxAccRate(const double &maxAccRate) const
    {
        double sqrMaxAccRate = maxAccRate * maxAccRate;
        // 检查起点和终点的加速度是否超限
        if (getAcc(0.0).squaredNorm() >= sqrMaxAccRate ||
            getAcc(duration).squaredNorm() >= sqrMaxAccRate)
        {
            return false;
        }
        else
        {
            // 使用多项式根计数方法检查整个区间
            AccCoefficientMat nAccCoeffMat = normalizeAccCoeffMat();
            Eigen::VectorXd coeff = RootFinder::polySqr(nAccCoeffMat.row(0)) +
                                    RootFinder::polySqr(nAccCoeffMat.row(1)) +
                                    RootFinder::polySqr(nAccCoeffMat.row(2));
            double t2 = duration * duration;
            double t4 = t2 * t2;
            coeff.tail<1>()(0) -= sqrMaxAccRate * t4;  // 减去约束值
            return RootFinder::countRoots(coeff, 0.0, 1.0) == 0;  // 无根表示不违反约束
        }
    }
};

/**
 * @class Trajectory
 * @brief D阶多项式轨迹类，由多个轨迹段组成的完整轨迹
 * @tparam D 多项式阶数
 * 
 * 轨迹由若干个连续的轨迹段组成，每个段都是一个D阶多项式。
 * 段与段之间在连接点处保持一定阶数的连续性。
 */
template <int D>
class Trajectory
{
private:
    typedef std::vector<Piece<D>> Pieces;  // 轨迹段容器类型
    Pieces pieces;  // 轨迹段列表

public:
    /**
     * @brief 默认构造函数
     */
    Trajectory() = default;

    /**
     * @brief 构造函数
     * @param durs 各轨迹段的持续时间列表
     * @param cMats 各轨迹段的系数矩阵列表
     */
    Trajectory(const std::vector<double> &durs,
               const std::vector<typename Piece<D>::CoefficientMat> &cMats)
    {
        int N = std::min(durs.size(), cMats.size());
        pieces.reserve(N);
        for (int i = 0; i < N; i++)
        {
            pieces.emplace_back(durs[i], cMats[i]);
        }
    }

    /**
     * @brief 获取轨迹段数量
     * @return 轨迹段的数量
     */
    inline int getPieceNum() const
    {
        return pieces.size();
    }

    /**
     * @brief 获取各轨迹段的持续时间
     * @return 包含所有段持续时间的向量
     */
    inline Eigen::VectorXd getDurations() const
    {
        int N = getPieceNum();
        Eigen::VectorXd durations(N);
        for (int i = 0; i < N; i++)
        {
            durations(i) = pieces[i].getDuration();
        }
        return durations;
    }

    /**
     * @brief 获取轨迹总持续时间
     * @return 所有段持续时间的总和
     */
    inline double getTotalDuration() const
    {
        int N = getPieceNum();
        double totalDuration = 0.0;
        for (int i = 0; i < N; i++)
        {
            totalDuration += pieces[i].getDuration();
        }
        return totalDuration;
    }

    /**
     * @brief 获取轨迹上的关键位置点
     * @return 包含各段起点和最后一段终点的位置矩阵 [3 x (N+1)]
     * 
     * 返回轨迹上的离散位置点，包括每段的起点和整条轨迹的终点
     */
    inline Eigen::Matrix3Xd getPositions() const
    {
        int N = getPieceNum();
        Eigen::Matrix3Xd positions(3, N + 1);
        
        // 获取每段的起点（t=0时刻的位置）
        for (int i = 0; i < N; i++)
        {
            positions.col(i) = pieces[i].getCoeffMat().col(D);  // 常数项即为起点位置
        }
        // 获取最后一段的终点
        positions.col(N) = pieces[N - 1].getPos(pieces[N - 1].getDuration());
        return positions;
    }

    /**
     * @brief 常量索引操作符，获取指定轨迹段
     * @param i 轨迹段索引
     * @return 轨迹段的常量引用
     */
    inline const Piece<D> &operator[](int i) const
    {
        return pieces[i];
    }

    /**
     * @brief 索引操作符，获取指定轨迹段
     * @param i 轨迹段索引
     * @return 轨迹段的引用
     */
    inline Piece<D> &operator[](int i)
    {
        return pieces[i];
    }

    /**
     * @brief 清空轨迹，移除所有轨迹段
     */
    inline void clear(void)
    {
        pieces.clear();
        return;
    }

    /**
     * @brief 获取常量迭代器起始位置
     * @return 指向第一个轨迹段的常量迭代器
     */
    inline typename Pieces::const_iterator begin() const
    {
        return pieces.begin();
    }

    /**
     * @brief 获取常量迭代器结束位置
     * @return 指向最后一个轨迹段之后的常量迭代器
     */
    inline typename Pieces::const_iterator end() const
    {
        return pieces.end();
    }

    /**
     * @brief 获取迭代器起始位置
     * @return 指向第一个轨迹段的迭代器
     */
    inline typename Pieces::iterator begin()
    {
        return pieces.begin();
    }

    /**
     * @brief 获取迭代器结束位置
     * @return 指向最后一个轨迹段之后的迭代器
     */
    inline typename Pieces::iterator end()
    {
        return pieces.end();
    }

    /**
     * @brief 预分配轨迹段容器空间
     * @param n 预分配的轨迹段数量
     */
    inline void reserve(const int &n)
    {
        pieces.reserve(n);
        return;
    }

    inline void emplace_back(const Piece<D> &piece)
    {
        pieces.emplace_back(piece);
        return;
    }

    /**
     * @brief 添加一个新的轨迹段
     * @param dur 轨迹段持续时间
     * @param cMat 轨迹段系数矩阵
     */
    inline void emplace_back(const double &dur,
                             const typename Piece<D>::CoefficientMat &cMat)
    {
        pieces.emplace_back(dur, cMat);
        return;
    }

    /**
     * @brief 将另一个轨迹追加到当前轨迹后面
     * @param traj 要追加的轨迹
     */
    inline void append(const Trajectory<D> &traj)
    {
        pieces.insert(pieces.end(), traj.begin(), traj.end());
        return;
    }

    /**
     * @brief 根据全局时间定位对应的轨迹段索引
     * @param t 全局时间（输入输出参数，输出时为段内局部时间）
     * @return 轨迹段索引
     * 
     * 该函数将全局时间转换为段索引和段内局部时间
     */
    inline int locatePieceIdx(double &t) const
    {
        int N = getPieceNum();
        int idx;
        double dur;
        
        // 遍历轨迹段，找到时间t所在的段
        for (idx = 0;
             idx < N &&
             t > (dur = pieces[idx].getDuration());
             idx++)
        {
            t -= dur;  // 减去已经过的时间
        }
        
        // 如果超出最后一段，则定位到最后一段的末尾
        if (idx == N)
        {
            idx--;
            t += pieces[idx].getDuration();
        }
        return idx;
    }

    /**
     * @brief 根据全局时间获取位置
     * @param t 全局时间
     * @return 3D位置向量
     */
    inline Eigen::Vector3d getPos(double t) const
    {
        int pieceIdx = locatePieceIdx(t);
        return pieces[pieceIdx].getPos(t);
    }

    /**
     * @brief 根据全局时间获取速度
     * @param t 全局时间
     * @return 3D速度向量
     */
    inline Eigen::Vector3d getVel(double t) const
    {
        int pieceIdx = locatePieceIdx(t);
        return pieces[pieceIdx].getVel(t);
    }

    /**
     * @brief 根据全局时间获取加速度
     * @param t 全局时间
     * @return 3D加速度向量
     */
    inline Eigen::Vector3d getAcc(double t) const
    {
        int pieceIdx = locatePieceIdx(t);
        return pieces[pieceIdx].getAcc(t);
    }

    /**
     * @brief 根据全局时间获取加加速度
     * @param t 全局时间
     * @return 3D加加速度向量
     */
    inline Eigen::Vector3d getJer(double t) const
    {
        int pieceIdx = locatePieceIdx(t);
        return pieces[pieceIdx].getJer(t);
    }

    /**
     * @brief 获取连接点处的位置
     * @param juncIdx 连接点索引
     * @return 3D位置向量
     * 
     * 连接点是轨迹段之间的连接处，索引从0到段数N
     */
    inline Eigen::Vector3d getJuncPos(int juncIdx) const
    {
        if (juncIdx != getPieceNum())
        {
            // 如果不是最后一个连接点，使用段起始点（最高阶系数）
            return pieces[juncIdx].getCoeffMat().col(D);
        }
        else
        {
            // 如果是最后一个连接点，使用最后一段的终点
            return pieces[juncIdx - 1].getPos(pieces[juncIdx - 1].getDuration());
        }
    }

    /**
     * @brief 获取连接点处的速度
     * @param juncIdx 连接点索引
     * @return 3D速度向量
     */
    inline Eigen::Vector3d getJuncVel(int juncIdx) const
    {
        if (juncIdx != getPieceNum())
        {
            // 速度为一阶导数的最高阶系数
            return pieces[juncIdx].getCoeffMat().col(D - 1);
        }
        else
        {
            // 最后一个连接点使用最后一段的终点速度
            return pieces[juncIdx - 1].getVel(pieces[juncIdx - 1].getDuration());
        }
    }

    /**
     * @brief 获取连接点处的加速度
     * @param juncIdx 连接点索引
     * @return 3D加速度向量
     */
    inline Eigen::Vector3d getJuncAcc(int juncIdx) const
    {
        if (juncIdx != getPieceNum())
        {
            // 加速度为二阶导数的最高阶系数乘以2
            return pieces[juncIdx].getCoeffMat().col(D - 2) * 2.0;
        }
        else
        {
            // 最后一个连接点使用最后一段的终点加速度
            return pieces[juncIdx - 1].getAcc(pieces[juncIdx - 1].getDuration());
        }
    }

    /**
     * @brief 获取整个轨迹的最大速度范数
     * @return 最大速度范数
     * 
     * 遍历所有轨迹段，找到最大的速度模长
     */
    inline double getMaxVelRate() const
    {
        int N = getPieceNum();
        double maxVelRate = -INFINITY;
        double tempNorm;
        for (int i = 0; i < N; i++)
        {
            tempNorm = pieces[i].getMaxVelRate();
            maxVelRate = maxVelRate < tempNorm ? tempNorm : maxVelRate;
        }
        return maxVelRate;
    }

    /**
     * @brief 获取整个轨迹的最大加速度范数
     * @return 最大加速度范数
     * 
     * 遍历所有轨迹段，找到最大的加速度模长
     */
    inline double getMaxAccRate() const
    {
        int N = getPieceNum();
        double maxAccRate = -INFINITY;
        double tempNorm;
        for (int i = 0; i < N; i++)
        {
            tempNorm = pieces[i].getMaxAccRate();
            maxAccRate = maxAccRate < tempNorm ? tempNorm : maxAccRate;
        }
        return maxAccRate;
    }

    /**
     * @brief 检查整个轨迹是否满足最大速度约束
     * @param maxVelRate 最大速度限制
     * @return 是否满足约束
     * 
     * 遍历所有轨迹段，检查每段是否都满足速度约束
     */
    inline bool checkMaxVelRate(const double &maxVelRate) const
    {
        int N = getPieceNum();
        bool feasible = true;
        for (int i = 0; i < N && feasible; i++)
        {
            feasible = feasible && pieces[i].checkMaxVelRate(maxVelRate);
        }
        return feasible;
    }

    /**
     * @brief 检查整个轨迹是否满足最大加速度约束
     * @param maxAccRate 最大加速度限制
     * @return 是否满足约束
     * 
     * 遍历所有轨迹段，检查每段是否都满足加速度约束
     */
    inline bool checkMaxAccRate(const double &maxAccRate) const
    {
        int N = getPieceNum();
        bool feasible = true;
        for (int i = 0; i < N && feasible; i++)
        {
            feasible = feasible && pieces[i].checkMaxAccRate(maxAccRate);
        }
        return feasible;
    }
};

#endif