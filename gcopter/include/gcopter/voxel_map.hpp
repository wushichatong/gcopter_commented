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

#ifndef VOXEL_MAP_HPP
#define VOXEL_MAP_HPP

#include "voxel_dilater.hpp"  // 体素膨胀器
#include <memory>
#include <vector>
#include <Eigen/Eigen>

using namespace std;

/**
 * @namespace voxel_map
 * @brief 体素地图命名空间，包含3D占用网格地图的实现
 */
namespace voxel_map
{
    // 体素状态常量定义
    constexpr uint8_t Unoccupied = 0;  // 未占用状态
    constexpr uint8_t Occupied = 1;    // 被占用状态  
    constexpr uint8_t Dilated = 2;     // 膨胀状态（障碍物安全边界）

    /**
     * @class VoxelMap
     * @brief 3D体素地图类，用于表示和操作三维占用网格地图
     * 
     * 主要功能包括：
     * - 存储3D空间的占用状态
     * - 障碍物膨胀处理
     * - 空间查询和坐标转换
     * - 表面点提取
     */
    class VoxelMap
    {

    public:
        /**
         * @brief 默认构造函数
         */
        VoxelMap() = default;
        
        /**
         * @brief 构造函数，初始化体素地图
         * @param size 地图尺寸（以体素数量表示）[nx, ny, nz]
         * @param origin 地图原点在世界坐标系中的位置
         * @param voxScale 单个体素的边长（米）
         */
        VoxelMap(const Eigen::Vector3i &size,
                 const Eigen::Vector3d &origin,
                 const double &voxScale)
            : mapSize(size),           // 地图尺寸
              o(origin),               // 原点位置
              scale(voxScale),         // 体素尺度
              voxNum(mapSize.prod()),  // 体素总数
              step(1, mapSize(0), mapSize(1) * mapSize(0)),  // 索引步长[1, nx, nx*ny]
              oc(o + Eigen::Vector3d::Constant(0.5 * scale)),  // 体素中心偏移
              bounds((mapSize.array() - 1) * step.array()),    // 边界索引
              stepScale(step.cast<double>().cwiseInverse() * scale),  // 步长尺度转换
              voxels(voxNum, Unoccupied) {}  // 初始化所有体素为未占用状态

    private:
        Eigen::Vector3i mapSize;    // 地图尺寸[nx, ny, nz]
        Eigen::Vector3d o;          // 地图原点坐标
        double scale;               // 体素边长（米）
        int voxNum;                 // 体素总数
        Eigen::Vector3i step;       // 3D索引到1D索引的步长
        Eigen::Vector3d oc;         // 体素中心偏移量
        Eigen::Vector3i bounds;     // 边界索引限制
        Eigen::Vector3d stepScale;  // 索引到坐标的转换比例
        std::vector<uint8_t> voxels;  // 体素数据存储（0=空闲，1=占用，2=膨胀）
        std::vector<Eigen::Vector3i> surf;  // 表面体素索引列表

    public:
        /**
         * @brief 获取地图尺寸
         * @return 地图尺寸向量[nx, ny, nz]
         */
        inline Eigen::Vector3i getSize(void) const
        {
            return mapSize;
        }

        /**
         * @brief 获取体素尺度
         * @return 单个体素的边长（米）
         */
        inline double getScale(void) const
        {
            return scale;
        }

        /**
         * @brief 获取地图原点坐标
         * @return 地图原点在世界坐标系中的位置
         */
        inline Eigen::Vector3d getOrigin(void) const
        {
            return o;
        }

        /**
         * @brief 获取地图对角点坐标
         * @return 地图最大角点在世界坐标系中的位置
         */
        inline Eigen::Vector3d getCorner(void) const
        {
            return mapSize.cast<double>() * scale + o;
        }

        /**
         * @brief 获取体素数据的常量引用
         * @return 包含所有体素状态的向量
         */
        inline const std::vector<uint8_t> &getVoxels(void) const
        {
            return voxels;
        }

        /**
         * @brief 根据世界坐标设置体素为占用状态
         * @param pos 世界坐标系中的3D位置
         */
        inline void setOccupied(const Eigen::Vector3d &pos)
        {
            // 将世界坐标转换为体素索引
            const Eigen::Vector3i id = ((pos - o) / scale).cast<int>();
            // 检查索引是否在有效范围内
            if (id(0) >= 0 && id(1) >= 0 && id(2) >= 0 &&
                id(0) < mapSize(0) && id(1) < mapSize(1) && id(2) < mapSize(2))
            if (id(0) >= 0 && id(1) >= 0 && id(2) >= 0 &&
                id(0) < mapSize(0) && id(1) < mapSize(1) && id(2) < mapSize(2))
            {
                // 将3D索引转换为1D索引并设置为占用状态
                voxels[id.dot(step)] = Occupied;
            }
        }

        /**
         * @brief 根据体素索引设置体素为占用状态
         * @param id 体素的3D索引[ix, iy, iz]
         */
        inline void setOccupied(const Eigen::Vector3i &id)
        {
            // 检查索引是否在有效范围内
            if (id(0) >= 0 && id(1) >= 0 && id(2) >= 0 &&
                id(0) < mapSize(0) && id(1) < mapSize(1) && id(2) < mapSize(2))
            {
                // 将3D索引转换为1D索引并设置为占用状态
                voxels[id.dot(step)] = Occupied;
            }
        }

        /**
         * @brief 对占用体素进行膨胀操作，生成安全边界
         * @param r 膨胀半径（以体素数量为单位）
         * 
         * 膨胀操作用于为障碍物添加安全边界，确保路径规划时保持足够的安全距离
         */
        inline void dilate(const int &r)
        {
            if (r <= 0)  // 如果膨胀半径无效则直接返回
            {
                return;
            }
            else
            {
                std::vector<Eigen::Vector3i> lvec, cvec;  // 上一层和当前层的体素列表
                lvec.reserve(voxNum);
                cvec.reserve(voxNum);
                int i, j, k, idx;
                bool check;
                
                // 第一轮膨胀：找到所有被占用体素的邻居
                for (int x = 0; x <= bounds(0); x++)
                {
                    for (int y = 0; y <= bounds(1); y += step(1))
                    {
                        for (int z = 0; z <= bounds(2); z += step(2))
                        {
                            if (voxels[x + y + z] == Occupied)  // 如果当前体素被占用
                            {
                                // 使用体素膨胀宏对其邻居进行膨胀
                                VOXEL_DILATER(i, j, k,
                                              x, y, z,
                                              step(1), step(2),
                                              bounds(0), bounds(1), bounds(2),
                                              check, voxels, idx, Dilated, cvec)
                            }
                        }
                    }
                }

                // 多轮膨胀：逐层向外扩展膨胀区域
                for (int loop = 1; loop < r; loop++)
                {
                    std::swap(cvec, lvec);  // 交换当前层和上一层
                    for (const Eigen::Vector3i &id : lvec)  // 对上一层的每个体素
                    {
                        // 继续膨胀其邻居体素
                        VOXEL_DILATER(i, j, k,
                                      id(0), id(1), id(2),
                                      step(1), step(2),
                                      bounds(0), bounds(1), bounds(2),
                                      check, voxels, idx, Dilated, cvec)
                    }
                    lvec.clear();  // 清空上一层列表
                }

                surf = cvec;  // 保存最外层表面体素
            }
        }

        /**
         * @brief 获取指定区域内的表面点
         * @param center 区域中心的体素索引
         * @param halfWidth 区域半宽（以体素数量为单位）
         * @param points 输出的表面点坐标列表
         */
        inline void getSurfInBox(const Eigen::Vector3i &center,
                                 const int &halfWidth,
                                 std::vector<Eigen::Vector3d> &points) const
        {
            for (const Eigen::Vector3i &id : surf)  // 遍历所有表面体素
            {
                // 检查体素是否在指定区域内
                if (std::abs(id(0) - center(0)) <= halfWidth &&
                    std::abs(id(1) / step(1) - center(1)) <= halfWidth &&
                    std::abs(id(2) / step(2) - center(2)) <= halfWidth)
                {
                    // 将体素索引转换为世界坐标并添加到点列表
                    points.push_back(id.cast<double>().cwiseProduct(stepScale) + oc);
                }
            }

            return;
        }

        /**
         * @brief 获取所有表面点的世界坐标
         * @param points 输出的表面点坐标列表
         */
        inline void getSurf(std::vector<Eigen::Vector3d> &points) const
        {
            points.reserve(surf.size());  // 预分配空间
            for (const Eigen::Vector3i &id : surf)  // 遍历所有表面体素
            {
                // 将体素索引转换为世界坐标
                points.push_back(id.cast<double>().cwiseProduct(stepScale) + oc);
            }
            return;
        }

        /**
         * @brief 查询指定世界坐标位置的占用状态
         * @param pos 世界坐标系中的3D位置
         * @return true表示被占用或超出边界，false表示空闲
         */
        inline bool query(const Eigen::Vector3d &pos) const
        {
            // 将世界坐标转换为体素索引
            const Eigen::Vector3i id = ((pos - o) / scale).cast<int>();
            // 检查索引是否在有效范围内
            if (id(0) >= 0 && id(1) >= 0 && id(2) >= 0 &&
                id(0) < mapSize(0) && id(1) < mapSize(1) && id(2) < mapSize(2))
            {
                return voxels[id.dot(step)];  // 返回体素状态
            }
            else
            {
                return true;  // 超出边界的位置视为被占用
            }
        }

        /**
         * @brief 查询指定体素索引的占用状态
         * @param id 体素的3D索引[ix, iy, iz]
         * @return true表示被占用或超出边界，false表示空闲
         */
        inline bool query(const Eigen::Vector3i &id) const
        {
            // 检查索引是否在有效范围内
            if (id(0) >= 0 && id(1) >= 0 && id(2) >= 0 &&
                id(0) < mapSize(0) && id(1) < mapSize(1) && id(2) < mapSize(2))
            {
                return voxels[id.dot(step)];  // 返回体素状态
            }
            else
            {
                return true;  // 超出边界的位置视为被占用
            }
        }

        /**
         * @brief 将体素索引转换为世界坐标
         * @param id 体素的3D索引[ix, iy, iz]
         * @return 对应的世界坐标（体素中心点）
         */
        inline Eigen::Vector3d posI2D(const Eigen::Vector3i &id) const
        {
            return id.cast<double>() * scale + oc;
        }

        /**
         * @brief 将世界坐标转换为体素索引
         * @param pos 世界坐标系中的3D位置
         * @return 对应的体素索引[ix, iy, iz]
         */
        inline Eigen::Vector3i posD2I(const Eigen::Vector3d &pos) const
        {
            return ((pos - o) / scale).cast<int>();
        }
    };
}

#endif
