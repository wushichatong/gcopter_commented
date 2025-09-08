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
 * @file flatness.hpp
 * @brief 多旋翼无人机平坦性映射(Differential Flatness Map)实现
 * 
 * 平坦性映射是现代无人机轨迹规划的核心理论基础，它建立了轨迹的几何表示
 * (位置、速度、加速度、加加速度)与实际控制输入(推力、姿态、角速度)之间的数学关系。
 * 
 * 理论基础：
 * - 微分平坦性：系统状态和控制输入可由平坦输出及其导数完全确定
 * - 平坦输出：无人机的位置坐标(x, y, z)和偏航角ψ
 * - 正向映射：轨迹导数 → 控制输入
 * - 反向映射：控制输入梯度 → 轨迹导数梯度(用于优化)
 * 
 * 应用场景：
 * 1. 轨迹规划：将几何轨迹转换为可执行的控制指令
 * 2. 轨迹优化：通过反向传播计算梯度进行优化
 * 3. 动力学约束：考虑推力限制、空气阻力等实际因素
 * 4. 姿态控制：自动计算所需的四元数姿态和角速度
 * 
 * 参考文献：https://github.com/ZJU-FAST-Lab/GCOPTER/blob/main/misc/flatness.pdf
 */

#ifndef FLATNESS_HPP
#define FLATNESS_HPP

#include <Eigen/Eigen>

#include <cmath>

namespace flatness
{
    /**
     * @brief 多旋翼无人机平坦性映射类
     * 
     * 该类实现了多旋翼无人机的微分平坦性映射，包括：
     * 1. 正向映射：从轨迹导数计算控制输入
     * 2. 反向映射：从控制输入梯度计算轨迹导数梯度
     * 3. 空气阻力建模：考虑水平、垂直和寄生阻力
     * 4. 数值稳定性：通过平滑因子避免奇异性
     */
    class FlatnessMap  // See https://github.com/ZJU-FAST-Lab/GCOPTER/blob/main/misc/flatness.pdf
    {
    public:
        /**
         * @brief 初始化平坦性映射参数
         * @param vehicle_mass 无人机质量 [kg]
         * @param gravitational_acceleration 重力加速度 [m/s²]，通常为9.81
         * @param horitonral_drag_coeff 水平阻力系数 [N⋅s/m]
         * @param vertical_drag_coeff 垂直阻力系数 [N⋅s/m]
         * @param parasitic_drag_coeff 寄生阻力系数 [N⋅s²/m²]
         * @param speed_smooth_factor 速度平滑因子，避免除零奇异性
         * 
         * 空气阻力建模：
         * - 水平阻力：F_h = d_h × v_horizontal
         * - 垂直阻力：F_v = d_v × v_vertical  
         * - 寄生阻力：F_p = c_p × |v| × v
         * 
         * 平滑因子作用：在计算 |v| 时使用 √(v² + ε) 避免零速度奇异性
         */
        inline void reset(const double &vehicle_mass,
                          const double &gravitational_acceleration,
                          const double &horitonral_drag_coeff,
                          const double &vertical_drag_coeff,
                          const double &parasitic_drag_coeff,
                          const double &speed_smooth_factor)
        {
            mass = vehicle_mass;                              // 储存无人机质量
            grav = gravitational_acceleration;               // 储存重力加速度
            dh = horitonral_drag_coeff;                      // 储存水平阻力系数
            dv = vertical_drag_coeff;                        // 储存垂直阻力系数
            cp = parasitic_drag_coeff;                       // 储存寄生阻力系数
            veps = speed_smooth_factor;                      // 储存速度平滑因子

            return;
        }

        /**
         * @brief 正向平坦性映射：从轨迹导数计算控制输入
         * @param vel 速度向量 [m/s]
         * @param acc 加速度向量 [m/s²]
         * @param jer 加加速度向量 [m/s³]
         * @param psi 偏航角 [rad]
         * @param dpsi 偏航角速度 [rad/s]
         * @param thr 输出：总推力 [N]
         * @param quat 输出：姿态四元数 [w,x,y,z]
         * @param omg 输出：角速度向量 [rad/s]
         * 
         * 正向映射算法流程：
         * 1. 计算考虑阻力的虚拟加速度
         * 2. 归一化得到机体z轴方向
         * 3. 计算推力大小
         * 4. 通过z轴和偏航角计算姿态四元数
         * 5. 计算角速度以实现轨迹跟踪
         * 
         * 核心数学关系：
         * - 虚拟加速度：a_u = a + (d_h/m)×w + [0,0,g]
         * - 机体z轴：z = a_u / |a_u|
         * - 推力：T = (ma + F_drag) · z
         * - 姿态：通过z轴方向和偏航角确定
         */
        inline void forward(const Eigen::Vector3d &vel,
                            const Eigen::Vector3d &acc,
                            const Eigen::Vector3d &jer,
                            const double &psi,
                            const double &dpsi,
                            double &thr,
                            Eigen::Vector4d &quat,
                            Eigen::Vector3d &omg)
        {
            double w0, w1, w2, dw0, dw1, dw2;              // 阻力相关速度项及其导数

            // 第一步：提取输入变量并计算基础项
            v0 = vel(0);                                    // x方向速度
            v1 = vel(1);                                    // y方向速度
            v2 = vel(2);                                    // z方向速度
            a0 = acc(0);                                    // x方向加速度
            a1 = acc(1);                                    // y方向加速度
            a2 = acc(2);                                    // z方向加速度
            
            // 第二步：计算寄生阻力相关项
            cp_term = sqrt(v0 * v0 + v1 * v1 + v2 * v2 + veps); // 平滑化速度模长
            w_term = 1.0 + cp * cp_term;                    // 阻力权重因子
            w0 = w_term * v0;                               // 考虑寄生阻力的等效速度
            w1 = w_term * v1;
            w2 = w_term * v2;
            
            // 第三步：计算虚拟加速度(考虑水平阻力)
            dh_over_m = dh / mass;                          // 水平阻力系数/质量
            zu0 = a0 + dh_over_m * w0;                      // 虚拟加速度x分量
            zu1 = a1 + dh_over_m * w1;                      // 虚拟加速度y分量
            zu2 = a2 + dh_over_m * w2 + grav;               // 虚拟加速度z分量(包含重力)
            
            // 第四步：计算虚拟加速度的几何性质
            zu_sqr0 = zu0 * zu0;                            // 虚拟加速度各分量的平方
            zu_sqr1 = zu1 * zu1;
            zu_sqr2 = zu2 * zu2;
            zu01 = zu0 * zu1;                               // 虚拟加速度分量的交叉项
            zu12 = zu1 * zu2;
            zu02 = zu0 * zu2;
            zu_sqr_norm = zu_sqr0 + zu_sqr1 + zu_sqr2;      // 虚拟加速度模长的平方
            zu_norm = sqrt(zu_sqr_norm);                    // 虚拟加速度模长
            
            // 第五步：计算归一化的机体z轴方向
            z0 = zu0 / zu_norm;                             // 机体z轴在世界坐标系的表示
            z1 = zu1 / zu_norm;
            z2 = zu2 / zu_norm;
            
            // 第六步：计算z轴导数的投影矩阵
            // ng = (I - z⊗z) / |zu|，用于计算z轴导数
            ng_den = zu_sqr_norm * zu_norm;                 // 投影矩阵分母
            ng00 = (zu_sqr1 + zu_sqr2) / ng_den;            // 投影矩阵元素
            ng01 = -zu01 / ng_den;
            ng02 = -zu02 / ng_den;
            ng11 = (zu_sqr0 + zu_sqr2) / ng_den;
            ng12 = -zu12 / ng_den;
            ng22 = (zu_sqr0 + zu_sqr1) / ng_den;
            
            // 第七步：计算阻力项的时间导数
            v_dot_a = v0 * a0 + v1 * a1 + v2 * a2;          // 速度与加速度的点积
            dw_term = cp * v_dot_a / cp_term;               // 寄生阻力时间导数的权重
            dw0 = w_term * a0 + dw_term * v0;               // 等效速度的时间导数
            dw1 = w_term * a1 + dw_term * v1;
            dw2 = w_term * a2 + dw_term * v2;
            
            // 第八步：计算虚拟加速度的时间导数
            dz_term0 = jer(0) + dh_over_m * dw0;            // 虚拟加速度导数的各分量
            dz_term1 = jer(1) + dh_over_m * dw1;
            dz_term2 = jer(2) + dh_over_m * dw2;
            
            // 第九步：计算机体z轴的时间导数
            dz0 = ng00 * dz_term0 + ng01 * dz_term1 + ng02 * dz_term2; // z轴导数
            dz1 = ng01 * dz_term0 + ng11 * dz_term1 + ng12 * dz_term2;
            dz2 = ng02 * dz_term0 + ng12 * dz_term1 + ng22 * dz_term2;
            
            // 第十步：计算总推力
            // 推力 = (质量×加速度 + 垂直阻力) 在机体z轴方向的投影
            f_term0 = mass * a0 + dv * w0;                  // 推力计算的x分量
            f_term1 = mass * a1 + dv * w1;                  // 推力计算的y分量
            f_term2 = mass * (a2 + grav) + dv * w2;         // 推力计算的z分量(包含重力)
            thr = z0 * f_term0 + z1 * f_term1 + z2 * f_term2; // 总推力(沿机体z轴)
            
            // 第十一步：计算姿态四元数
            // 通过机体z轴和偏航角计算姿态
            tilt_den = sqrt(2.0 * (1.0 + z2));              // 倾斜四元数的分母
            tilt0 = 0.5 * tilt_den;                         // 倾斜四元数的w分量
            tilt1 = -z1 / tilt_den;                         // 倾斜四元数的x分量
            tilt2 = z0 / tilt_den;                          // 倾斜四元数的y分量
            
            // 结合偏航角计算最终四元数
            c_half_psi = cos(0.5 * psi);                    // cos(ψ/2)
            s_half_psi = sin(0.5 * psi);                    // sin(ψ/2)
            quat(0) = tilt0 * c_half_psi;                   // 四元数w分量
            quat(1) = tilt1 * c_half_psi + tilt2 * s_half_psi; // 四元数x分量
            quat(2) = tilt2 * c_half_psi - tilt1 * s_half_psi; // 四元数y分量
            quat(3) = tilt0 * s_half_psi;                   // 四元数z分量
            
            // 第十二步：计算角速度
            // 角速度用于实现轨迹跟踪和偏航控制
            c_psi = cos(psi);                               // cos(ψ)
            s_psi = sin(psi);                               // sin(ψ)
            omg_den = z2 + 1.0;                             // 角速度计算的分母
            omg_term = dz2 / omg_den;                       // 角速度计算的中间项
            
            // 机体角速度计算
            omg(0) = dz0 * s_psi - dz1 * c_psi -            // x轴角速度(roll rate)
                     (z0 * s_psi - z1 * c_psi) * omg_term;
            omg(1) = dz0 * c_psi + dz1 * s_psi -            // y轴角速度(pitch rate)
                     (z0 * c_psi + z1 * s_psi) * omg_term;
            omg(2) = (z1 * dz0 - z0 * dz1) / omg_den + dpsi; // z轴角速度(yaw rate)

            return;
        }

        /**
         * @brief 反向平坦性映射：从控制输入梯度计算轨迹导数梯度
         * @param pos_grad 位置梯度
         * @param vel_grad 速度梯度  
         * @param thr_grad 推力梯度
         * @param quat_grad 姿态四元数梯度
         * @param omg_grad 角速度梯度
         * @param pos_total_grad 输出：位置总梯度
         * @param vel_total_grad 输出：速度总梯度
         * @param acc_total_grad 输出：加速度总梯度
         * @param jer_total_grad 输出：加加速度总梯度
         * @param psi_total_grad 输出：偏航角总梯度
         * @param dpsi_total_grad 输出：偏航角速度总梯度
         * 
         * 反向映射用于轨迹优化中的梯度计算，通过自动微分的反向传播
         * 将控制输入的梯度传播到轨迹参数，用于基于梯度的优化算法。
         * 
         * 算法实现：
         * 1. 使用自动微分技术计算雅可比矩阵
         * 2. 按照计算图的逆序进行梯度传播
         * 3. 考虑所有中间变量的梯度贡献
         * 4. 最终得到轨迹导数相对于目标函数的梯度
         * 
         * 注意：该函数必须在调用forward()之后使用，因为需要前向传播的中间结果
         */
        inline void backward(const Eigen::Vector3d &pos_grad,
                             const Eigen::Vector3d &vel_grad,
                             const double &thr_grad,
                             const Eigen::Vector4d &quat_grad,
                             const Eigen::Vector3d &omg_grad,
                             Eigen::Vector3d &pos_total_grad,
                             Eigen::Vector3d &vel_total_grad,
                             Eigen::Vector3d &acc_total_grad,
                             Eigen::Vector3d &jer_total_grad,
                             double &psi_total_grad,
                             double &dpsi_total_grad) const
        {
            // 反向传播的临时变量，用于存储各中间量的梯度
            double w0b, w1b, w2b, dw0b, dw1b, dw2b;          // 阻力速度项梯度
            double z0b, z1b, z2b, dz0b, dz1b, dz2b;          // 机体z轴及其导数梯度
            double v_sqr_normb, cp_termb, w_termb;           // 速度相关项梯度

            // 注意：以下代码实现自动微分的反向传播
            // 按照forward()函数计算的逆序，传播梯度到输入变量
            // 这是一个复杂的链式法则应用，每一步都对应forward()中的一个计算步骤
            
            /* 反向传播代码：由自动微分工具生成，按计算图逆序传播梯度 */
            double zu_sqr_normb, zu_normb, zu0b, zu1b, zu2b;
            double zu_sqr0b, zu_sqr1b, zu_sqr2b, zu01b, zu12b, zu02b;
            double ng00b, ng01b, ng02b, ng11b, ng12b, ng22b, ng_denb;
            double dz_term0b, dz_term1b, dz_term2b, f_term0b, f_term1b, f_term2b;
            double tilt_denb, tilt0b, tilt1b, tilt2b, head0b, head3b;
            double cpsib, spsib, omg_denb, omg_termb;
            double tempb, tilt_den_sqr;

            tilt0b = s_half_psi * (quat_grad(3)) + c_half_psi * (quat_grad(0));
            head3b = tilt0 * (quat_grad(3)) + tilt2 * (quat_grad(1)) - tilt1 * (quat_grad(2));
            tilt2b = c_half_psi * (quat_grad(2)) + s_half_psi * (quat_grad(1));
            head0b = tilt2 * (quat_grad(2)) + tilt1 * (quat_grad(1)) + tilt0 * (quat_grad(0));
            tilt1b = c_half_psi * (quat_grad(1)) - s_half_psi * (quat_grad(2));
            tilt_den_sqr = tilt_den * tilt_den;
            tilt_denb = (z1 * tilt1b - z0 * tilt2b) / tilt_den_sqr + 0.5 * tilt0b;
            omg_termb = -((z0 * c_psi + z1 * s_psi) * (omg_grad(1))) -
                        (z0 * s_psi - z1 * c_psi) * (omg_grad(0));
            tempb = omg_grad(2) / omg_den;
            dpsi_total_grad = omg_grad(2);
            z1b = dz0 * tempb;
            dz0b = z1 * tempb + c_psi * (omg_grad(1)) + s_psi * (omg_grad(0));
            z0b = -(dz1 * tempb);
            dz1b = s_psi * (omg_grad(1)) - z0 * tempb - c_psi * (omg_grad(0));
            omg_denb = -((z1 * dz0 - z0 * dz1) * tempb / omg_den) -
                       dz2 * omg_termb / (omg_den * omg_den);
            tempb = -(omg_term * (omg_grad(1)));
            cpsib = dz0 * (omg_grad(1)) + z0 * tempb;
            spsib = dz1 * (omg_grad(1)) + z1 * tempb;
            z0b += c_psi * tempb;
            z1b += s_psi * tempb;
            tempb = -(omg_term * (omg_grad(0)));
            spsib += dz0 * (omg_grad(0)) + z0 * tempb;
            cpsib += -dz1 * (omg_grad(0)) - z1 * tempb;
            z0b += s_psi * tempb + tilt2b / tilt_den + f_term0 * (thr_grad);
            z1b += -c_psi * tempb - tilt1b / tilt_den + f_term1 * (thr_grad);
            dz2b = omg_termb / omg_den;
            z2b = omg_denb + tilt_denb / tilt_den + f_term2 * (thr_grad);
            psi_total_grad = c_psi * spsib + 0.5 * c_half_psi * head3b -
                             s_psi * cpsib - 0.5 * s_half_psi * head0b;
            f_term0b = z0 * (thr_grad);
            f_term1b = z1 * (thr_grad);
            f_term2b = z2 * (thr_grad);
            ng02b = dz_term0 * dz2b + dz_term2 * dz0b;
            dz_term0b = ng02 * dz2b + ng01 * dz1b + ng00 * dz0b;
            ng12b = dz_term1 * dz2b + dz_term2 * dz1b;
            dz_term1b = ng12 * dz2b + ng11 * dz1b + ng01 * dz0b;
            ng22b = dz_term2 * dz2b;
            dz_term2b = ng22 * dz2b + ng12 * dz1b + ng02 * dz0b;
            ng01b = dz_term0 * dz1b + dz_term1 * dz0b;
            ng11b = dz_term1 * dz1b;
            ng00b = dz_term0 * dz0b;
            jer_total_grad(2) = dz_term2b;
            dw2b = dh_over_m * dz_term2b;
            jer_total_grad(1) = dz_term1b;
            dw1b = dh_over_m * dz_term1b;
            jer_total_grad(0) = dz_term0b;
            dw0b = dh_over_m * dz_term0b;
            tempb = cp * (v2 * dw2b + v1 * dw1b + v0 * dw0b) / cp_term;
            acc_total_grad(2) = mass * f_term2b + w_term * dw2b + v2 * tempb;
            acc_total_grad(1) = mass * f_term1b + w_term * dw1b + v1 * tempb;
            acc_total_grad(0) = mass * f_term0b + w_term * dw0b + v0 * tempb;
            vel_total_grad(2) = dw_term * dw2b + a2 * tempb;
            vel_total_grad(1) = dw_term * dw1b + a1 * tempb;
            vel_total_grad(0) = dw_term * dw0b + a0 * tempb;
            cp_termb = -(v_dot_a * tempb / cp_term);
            tempb = ng22b / ng_den;
            zu_sqr0b = tempb;
            zu_sqr1b = tempb;
            ng_denb = -((zu_sqr0 + zu_sqr1) * tempb / ng_den);
            zu12b = -(ng12b / ng_den);
            tempb = ng11b / ng_den;
            ng_denb += zu12 * ng12b / (ng_den * ng_den) -
                       (zu_sqr0 + zu_sqr2) * tempb / ng_den;
            zu_sqr0b += tempb;
            zu_sqr2b = tempb;
            zu02b = -(ng02b / ng_den);
            zu01b = -(ng01b / ng_den);
            tempb = ng00b / ng_den;
            ng_denb += zu02 * ng02b / (ng_den * ng_den) +
                       zu01 * ng01b / (ng_den * ng_den) -
                       (zu_sqr1 + zu_sqr2) * tempb / ng_den;
            zu_normb = zu_sqr_norm * ng_denb -
                       (zu2 * z2b + zu1 * z1b + zu0 * z0b) / zu_sqr_norm;
            zu_sqr_normb = zu_norm * ng_denb + zu_normb / (2.0 * zu_norm);
            tempb += zu_sqr_normb;
            zu_sqr1b += tempb;
            zu_sqr2b += tempb;
            zu2b = z2b / zu_norm + zu0 * zu02b + zu1 * zu12b + 2 * zu2 * zu_sqr2b;
            w2b = dv * f_term2b + dh_over_m * zu2b;
            zu1b = z1b / zu_norm + zu2 * zu12b + zu0 * zu01b + 2 * zu1 * zu_sqr1b;
            w1b = dv * f_term1b + dh_over_m * zu1b;
            zu_sqr0b += zu_sqr_normb;
            zu0b = z0b / zu_norm + zu2 * zu02b + zu1 * zu01b + 2 * zu0 * zu_sqr0b;
            w0b = dv * f_term0b + dh_over_m * zu0b;
            w_termb = a2 * dw2b + a1 * dw1b + a0 * dw0b +
                      v2 * w2b + v1 * w1b + v0 * w0b;
            acc_total_grad(2) += zu2b;
            acc_total_grad(1) += zu1b;
            acc_total_grad(0) += zu0b;
            cp_termb += cp * w_termb;
            v_sqr_normb = cp_termb / (2.0 * cp_term);
            // 最终梯度汇总：将所有贡献累加到输出梯度
            vel_total_grad(2) += w_term * w2b + 2 * v2 * v_sqr_normb + vel_grad(2); // z方向速度总梯度
            vel_total_grad(1) += w_term * w1b + 2 * v1 * v_sqr_normb + vel_grad(1); // y方向速度总梯度
            vel_total_grad(0) += w_term * w0b + 2 * v0 * v_sqr_normb + vel_grad(0); // x方向速度总梯度
            
            // 位置梯度直接传递(位置不参与平坦性映射计算)
            pos_total_grad(2) = pos_grad(2);                // z方向位置梯度
            pos_total_grad(1) = pos_grad(1);                // y方向位置梯度
            pos_total_grad(0) = pos_grad(0);                // x方向位置梯度

            return;
        }

    private:
        // === 无人机物理参数 ===
        double mass, grav, dh, dv, cp, veps;              // 质量、重力、阻力系数、平滑因子

        // === 输入轨迹导数 ===
        double v0, v1, v2, a0, a1, a2, v_dot_a;           // 速度、加速度分量及其点积
        
        // === 机体坐标系z轴及其导数 ===
        double z0, z1, z2, dz0, dz1, dz2;                 // 归一化z轴方向及其时间导数
        
        // === 阻力计算相关 ===
        double cp_term, w_term, dh_over_m;                // 寄生阻力项、权重因子、水平阻力/质量
        
        // === 虚拟加速度相关 ===
        double zu_sqr_norm, zu_norm, zu0, zu1, zu2;       // 虚拟加速度模长及各分量
        double zu_sqr0, zu_sqr1, zu_sqr2, zu01, zu12, zu02; // 虚拟加速度分量平方和交叉项
        
        // === 投影矩阵元素 ===
        double ng00, ng01, ng02, ng11, ng12, ng22, ng_den; // z轴导数投影矩阵及分母
        
        // === 中间计算项 ===
        double dw_term, dz_term0, dz_term1, dz_term2;     // 阻力导数权重、虚拟加速度导数分量
        double f_term0, f_term1, f_term2;                 // 推力计算的各分量
        
        // === 姿态计算相关 ===
        double tilt_den, tilt0, tilt1, tilt2;             // 倾斜四元数相关
        double c_half_psi, s_half_psi;                     // 半偏航角的三角函数
        
        // === 角速度计算相关 ===
        double c_psi, s_psi, omg_den, omg_term;           // 偏航角三角函数、角速度计算项
    };
}

#endif
