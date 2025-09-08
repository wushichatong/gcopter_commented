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

#ifndef SFC_GEN_HPP
#define SFC_GEN_HPP

#include "geo_utils.hpp"  // 几何工具函数
#include "firi.hpp"       // 快速增量多面体识别

// OMPL (Open Motion Planning Library) 相关头文件
#include <ompl/util/Console.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/rrt/InformedRRTstar.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/DiscreteMotionValidator.h>

#include <deque>
#include <memory>
#include <Eigen/Eigen>

/**
 * @namespace sfc_gen
 * @brief 安全飞行通道生成器命名空间
 * 
 * 主要功能包括：
 * 1. 基于RRT*的路径规划
 * 2. 安全飞行通道（SFC）生成
 * 3. 通道优化和简化
 */
namespace sfc_gen
{

    /**
     * @brief 使用Informed RRT*算法进行路径规划
     * @tparam Map 地图类型，需要支持query()方法进行碰撞检测
     * @param s 起点坐标
     * @param g 终点坐标
     * @param lb 搜索空间下界
     * @param hb 搜索空间上界
     * @param mapPtr 地图指针，用于碰撞检测
     * @param timeout 规划超时时间（秒）
     * @param p 输出的路径点序列
     * @return 路径代价，失败时返回INFINITY
     */
    template <typename Map>
    inline double planPath(const Eigen::Vector3d &s,
                           const Eigen::Vector3d &g,
                           const Eigen::Vector3d &lb,
                           const Eigen::Vector3d &hb,
                           const Map *mapPtr,
                           const double &timeout,
                           std::vector<Eigen::Vector3d> &p)
    {
        // 创建3D实向量状态空间
        auto space(std::make_shared<ompl::base::RealVectorStateSpace>(3));

        // 设置搜索空间边界
        ompl::base::RealVectorBounds bounds(3);
        bounds.setLow(0, 0.0);
        bounds.setHigh(0, hb(0) - lb(0));  // X轴范围
        bounds.setLow(1, 0.0);
        bounds.setHigh(1, hb(1) - lb(1));  // Y轴范围
        bounds.setLow(2, 0.0);
        bounds.setHigh(2, hb(2) - lb(2));  // Z轴范围
        space->setBounds(bounds);

        // 创建空间信息对象
        auto si(std::make_shared<ompl::base::SpaceInformation>(space));

        // 设置状态有效性检查器（碰撞检测）
        si->setStateValidityChecker(
            [&](const ompl::base::State *state)
            {
                const auto *pos = state->as<ompl::base::RealVectorStateSpace::StateType>();
                // 将相对坐标转换为绝对坐标
                const Eigen::Vector3d position(lb(0) + (*pos)[0],
                                               lb(1) + (*pos)[1],
                                               lb(2) + (*pos)[2]);
                // 查询该位置是否无碰撞（0表示自由空间）
                return mapPtr->query(position) == 0;
            });
        si->setup();

        // 禁用OMPL日志输出
        ompl::msg::setLogLevel(ompl::msg::LOG_NONE);

        // 设置起点和终点（转换为相对坐标）
        ompl::base::ScopedState<> start(space), goal(space);
        start[0] = s(0) - lb(0);
        start[1] = s(1) - lb(1);
        start[2] = s(2) - lb(2);
        goal[0] = g(0) - lb(0);
        goal[1] = g(1) - lb(1);
        goal[2] = g(2) - lb(2);

        // 创建规划问题定义
        auto pdef(std::make_shared<ompl::base::ProblemDefinition>(si));
        pdef->setStartAndGoalStates(start, goal);
        // 设置优化目标为最短路径
        pdef->setOptimizationObjective(std::make_shared<ompl::base::PathLengthOptimizationObjective>(si));
        
        // 创建Informed RRT*规划器
        auto planner(std::make_shared<ompl::geometric::InformedRRTstar>(si));
        planner->setProblemDefinition(pdef);
        planner->setup();

        // 执行路径规划
        ompl::base::PlannerStatus solved;
        solved = planner->ompl::base::Planner::solve(timeout);

        double cost = INFINITY;

        if (solved)  // 如果成功找到路径
        {
            p.clear();  // 清空输出路径
            // 获取几何路径
            const ompl::geometric::PathGeometric path_ =
                ompl::geometric::PathGeometric(
                    dynamic_cast<const ompl::geometric::PathGeometric &>(*pdef->getSolutionPath()));
            
            // 将路径点转换为绝对坐标并存储
            for (size_t i = 0; i < path_.getStateCount(); i++)
            {
                const auto state = path_.getState(i)->as<ompl::base::RealVectorStateSpace::StateType>()->values;
                p.emplace_back(lb(0) + state[0], lb(1) + state[1], lb(2) + state[2]);
            }
            // 获取路径代价
            cost = pdef->getSolutionPath()->cost(pdef->getOptimizationObjective()).value();
        }

        return cost;
    }

    /**
     * @brief 为给定路径生成凸包覆盖（安全飞行通道）
     * @param path 输入的路径点序列
     * @param points 环境中的障碍物点云
     * @param lowCorner 环境下界
     * @param highCorner 环境上界
     * @param progress 沿路径的步进距离
     * @param range 通道的扩展范围
     * @param hpolys 输出的超平面多面体序列（每个多面体表示一个安全通道段）
     * @param eps 数值精度容忍度
     */
    inline void convexCover(const std::vector<Eigen::Vector3d> &path,
                            const std::vector<Eigen::Vector3d> &points,
                            const Eigen::Vector3d &lowCorner,
                            const Eigen::Vector3d &highCorner,
                            const double &progress,
                            const double &range,
                            std::vector<Eigen::MatrixX4d> &hpolys,
                            const double eps = 1.0e-6)
    {
        hpolys.clear();  // 清空输出多面体列表
        const int n = path.size();
        
        // 初始化边界约束矩阵（6个面：±X, ±Y, ±Z）
        Eigen::Matrix<double, 6, 4> bd = Eigen::Matrix<double, 6, 4>::Zero();
        bd(0, 0) = 1.0;   // +X面
        bd(1, 0) = -1.0;  // -X面
        bd(2, 1) = 1.0;   // +Y面
        bd(3, 1) = -1.0;  // -Y面
        bd(4, 2) = 1.0;   // +Z面
        bd(5, 2) = -1.0;  // -Z面

        Eigen::MatrixX4d hp, gap;  // 超平面多面体和间隙多面体
        Eigen::Vector3d a, b = path[0];  // 当前段的起点和终点
        std::vector<Eigen::Vector3d> valid_pc;  // 有效点云
        std::vector<Eigen::Vector3d> bs;        // 路径段终点列表
        valid_pc.reserve(points.size());
        
        // 沿路径分段生成安全通道
        for (int i = 1; i < n;)
        {
            a = b;  // 当前段起点
            
            // 确定当前段终点
            if ((a - path[i]).norm() > progress)
            {
                // 如果距离超过步进距离，则按步进距离计算终点
                b = (path[i] - a).normalized() * progress + a;
            }
            else
            {
                // 否则直接使用路径点作为终点
                b = path[i];
                i++;
            }
            }
            bs.emplace_back(b);  // 记录段终点

            // 根据当前路径段设置边界约束
            // 每个方向的边界 = max/min(段起终点坐标) ± range，但不超出环境边界
            bd(0, 3) = -std::min(std::max(a(0), b(0)) + range, highCorner(0));  // +X边界
            bd(1, 3) = +std::max(std::min(a(0), b(0)) - range, lowCorner(0));   // -X边界
            bd(2, 3) = -std::min(std::max(a(1), b(1)) + range, highCorner(1));  // +Y边界
            bd(3, 3) = +std::max(std::min(a(1), b(1)) - range, lowCorner(1));   // -Y边界
            bd(4, 3) = -std::min(std::max(a(2), b(2)) + range, highCorner(2));  // +Z边界
            bd(5, 3) = +std::max(std::min(a(2), b(2)) - range, lowCorner(2));   // -Z边界

            // 筛选出在当前边界内的障碍物点
            valid_pc.clear();
            for (const Eigen::Vector3d &p : points)
            {
                // 检查点是否在边界内（所有约束都满足）
                if ((bd.leftCols<3>() * p + bd.rightCols<1>()).maxCoeff() < 0.0)
                {
                    valid_pc.emplace_back(p);
                }
            }
            
            // 将有效点云转换为Eigen矩阵格式
            Eigen::Map<const Eigen::Matrix<double, 3, -1, Eigen::ColMajor>> pc(valid_pc[0].data(), 3, valid_pc.size());

            // 使用FIRI算法生成安全通道的超平面表示
            firi::firi(bd, pc, a, b, hp);

            // 处理相邻通道之间的连接性
            if (hpolys.size() != 0)
            {
                const Eigen::Vector4d ah(a(0), a(1), a(2), 1.0);  // 齐次坐标形式的起点
                // 检查当前通道与前一个通道的重叠度
                if (3 <= ((hp * ah).array() > -eps).cast<int>().sum() +
                             ((hpolys.back() * ah).array() > -eps).cast<int>().sum())
                {
                    // 如果重叠度不足，插入间隙通道
                    firi::firi(bd, pc, a, a, gap, 1);
                    hpolys.emplace_back(gap);
                }
            }

            hpolys.emplace_back(hp);  // 添加当前通道
        }


    /**
     * @brief 对安全飞行通道序列进行shortcut优化
     * @param hpolys 输入输出的超平面多面体序列
     * 
     * 该函数通过检测通道间的重叠性，移除冗余的中间通道，
     * 从而简化通道序列并提高飞行效率
     */
    inline void shortCut(std::vector<Eigen::MatrixX4d> &hpolys)
    {
        std::vector<Eigen::MatrixX4d> htemp = hpolys;  // 备份原始通道序列
        
        // 特殊情况：如果只有一个通道，复制一份以便处理
        if (htemp.size() == 1)
        {
            Eigen::MatrixX4d headPoly = htemp.front();
            htemp.insert(htemp.begin(), headPoly);
        }
        hpolys.clear();  // 清空输出序列

        int M = htemp.size();
        Eigen::MatrixX4d hPoly;
        bool overlap;
        std::deque<int> idices;  // 存储保留的通道索引
        idices.push_front(M - 1);  // 从最后一个通道开始
        
        // 反向遍历通道序列，寻找可以直接连接的通道
        for (int i = M - 1; i >= 0; i--)
        {
            for (int j = 0; j < i; j++)
            {
                if (j < i - 1)
                {
                    // 检查非相邻通道是否重叠
                    overlap = geo_utils::overlap(htemp[i], htemp[j], 0.01);
                }
                else
                {
                    // 相邻通道默认重叠
                    overlap = true;
                }
                
                if (overlap)
                {
                    // 如果重叠，则可以跳过中间的通道
                    idices.push_front(j);
                    i = j + 1;  // 跳转到找到的通道
                    break;
                }
            }
        }
        
        // 按找到的索引重建通道序列
        for (const auto &ele : idices)
        {
            hpolys.push_back(htemp[ele]);
        }
    }

}

#endif
