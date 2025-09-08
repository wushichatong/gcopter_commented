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
 * @file minco.hpp
 * @brief MINCO (Minimum Control Effort) 轨迹优化库
 * 
 * 基于最小控制输入的轨迹优化算法实现，用于生成平滑、连续且满足动力学约束的轨迹。
 * MINCO通过最小化高阶导数（如加加速度）来生成时间最优且控制能量最小的轨迹。
 * 
 * 主要特点：
 * - 基于多项式样条的轨迹表示
 * - 支持多种边界条件约束
 * - 高效的带状线性系统求解器
 * - 时间最优化和路径平滑化
 * - 支持多机器人编队轨迹生成
 * 
 * 算法原理：
 * - 将轨迹规划问题转化为二次规划问题
 * - 通过最小化控制输入的积分来获得平滑轨迹
 * - 使用样条函数保证轨迹在连接点的连续性
 */

#ifndef MINCO_HPP
#define MINCO_HPP



#include "gcopter/minco_s2nu.hpp"
#include "gcopter/minco_s3nu.hpp"
#include "gcopter/minco_s4nu.hpp"
#include "gcopter/trajectory.hpp"



/**
 * @namespace minco
 * @brief MINCO轨迹优化算法命名空间
 * 
 * 包含所有与最小控制输入轨迹生成相关的类和函数
 */
namespace minco
{





}

#endif
