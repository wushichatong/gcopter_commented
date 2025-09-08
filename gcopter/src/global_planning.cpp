// GCOPTER全局路径规划节点头文件
#include "misc/visualizer.hpp"      // 可视化工具
#include "gcopter/trajectory.hpp"   // 轨迹类
#include "gcopter/gcopter.hpp"      // GCOPTER优化器
#include "gcopter/firi.hpp"         // 有限冲激响应
#include "gcopter/flatness.hpp"     // 平坦性映射
#include "gcopter/voxel_map.hpp"    // 体素地图
#include "gcopter/sfc_gen.hpp"      // 安全飞行通道生成

// ROS相关头文件
#include <ros/ros.h>
#include <ros/console.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>

// 标准库头文件
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <random>

/**
 * @brief 配置结构体，存储从ROS参数服务器读取的所有参数
 */
struct Config
{
    // ROS话题参数
    std::string mapTopic;       // 地图点云话题名称
    std::string targetTopic;    // 目标点话题名称
    
    // 地图相关参数
    double dilateRadius;        // 障碍物膨胀半径(m)
    double voxelWidth;          // 体素网格宽度(m)
    std::vector<double> mapBound;  // 地图边界[xmin,xmax,ymin,ymax,zmin,zmax]
    
    // 路径规划参数
    double timeoutRRT;          // RRT规划超时时间(s)
    
    // 无人机动力学约束参数
    double maxVelMag;           // 最大速度模长(m/s)
    double maxBdrMag;           // 最大体速率模长(rad/s)
    double maxTiltAngle;        // 最大倾斜角度(rad)
    double minThrust;           // 最小推力(N)
    double maxThrust;           // 最大推力(N)
    
    // 无人机物理参数
    double vehicleMass;         // 无人机质量(kg)
    double gravAcc;             // 重力加速度(m/s²)
    double horizDrag;           // 水平阻力系数
    double vertDrag;            // 垂直阻力系数
    double parasDrag;           // 寄生阻力系数
    double speedEps;            // 速度平滑因子
    
    // 轨迹优化参数
    double weightT;             // 时间权重
    std::vector<double> chiVec; // 惩罚权重向量[位置,速度,角速度,倾角,推力]
    double smoothingEps;        // 平滑参数
    int integralIntervs;        // 积分区间数
    double relCostTol;          // 相对代价容忍度

    /**
     * @brief 构造函数，从ROS参数服务器读取配置参数
     * @param nh_priv 私有节点句柄
     */
    Config(const ros::NodeHandle &nh_priv)
    {
        // 从ROS参数服务器读取各项配置参数
        nh_priv.getParam("MapTopic", mapTopic);
        nh_priv.getParam("TargetTopic", targetTopic);
        nh_priv.getParam("DilateRadius", dilateRadius);
        nh_priv.getParam("VoxelWidth", voxelWidth);
        nh_priv.getParam("MapBound", mapBound);
        nh_priv.getParam("TimeoutRRT", timeoutRRT);
        nh_priv.getParam("MaxVelMag", maxVelMag);
        nh_priv.getParam("MaxBdrMag", maxBdrMag);
        nh_priv.getParam("MaxTiltAngle", maxTiltAngle);
        nh_priv.getParam("MinThrust", minThrust);
        nh_priv.getParam("MaxThrust", maxThrust);
        nh_priv.getParam("VehicleMass", vehicleMass);
        nh_priv.getParam("GravAcc", gravAcc);
        nh_priv.getParam("HorizDrag", horizDrag);
        nh_priv.getParam("VertDrag", vertDrag);
        nh_priv.getParam("ParasDrag", parasDrag);
        nh_priv.getParam("SpeedEps", speedEps);
        nh_priv.getParam("WeightT", weightT);
        nh_priv.getParam("ChiVec", chiVec);
        nh_priv.getParam("SmoothingEps", smoothingEps);
        nh_priv.getParam("IntegralIntervs", integralIntervs);
        nh_priv.getParam("RelCostTol", relCostTol);
    }
};

/**
 * @brief 全局路径规划器类
 * 主要功能包括：接收地图和目标点，进行路径规划，轨迹优化，可视化
 */
class GlobalPlanner
{
private:
    Config config;              // 配置参数

    ros::NodeHandle nh;         // ROS节点句柄
    ros::Subscriber mapSub;     // 地图订阅器
    ros::Subscriber targetSub;  // 目标点订阅器

    bool mapInitialized;        // 地图是否已初始化标志
    voxel_map::VoxelMap voxelMap;  // 体素地图对象
    Visualizer visualizer;      // 可视化工具
    std::vector<Eigen::Vector3d> startGoal;  // 起点和终点向量

    Trajectory<5> traj;         // 5阶多项式轨迹对象
    double trajStamp;           // 轨迹时间戳

public:
    /**
     * @brief 全局规划器构造函数
     * @param conf 配置参数
     * @param nh_ ROS节点句柄
     */
    GlobalPlanner(const Config &conf,
                  ros::NodeHandle &nh_)
        : config(conf),
          nh(nh_),
          mapInitialized(false),
          visualizer(nh)
    {
        // 根据地图边界和体素宽度计算体素地图尺寸
        const Eigen::Vector3i xyz((config.mapBound[1] - config.mapBound[0]) / config.voxelWidth,
                                  (config.mapBound[3] - config.mapBound[2]) / config.voxelWidth,
                                  (config.mapBound[5] - config.mapBound[4]) / config.voxelWidth);

        // 设置体素地图原点偏移
        const Eigen::Vector3d offset(config.mapBound[0], config.mapBound[2], config.mapBound[4]);

        // 初始化体素地图
        voxelMap = voxel_map::VoxelMap(xyz, offset, config.voxelWidth);

        // 订阅地图点云话题
        mapSub = nh.subscribe(config.mapTopic, 1, &GlobalPlanner::mapCallBack, this,
                              ros::TransportHints().tcpNoDelay());

        // 订阅目标点话题
        targetSub = nh.subscribe(config.targetTopic, 1, &GlobalPlanner::targetCallBack, this,
                                 ros::TransportHints().tcpNoDelay());
    }

    /**
     * @brief 地图点云回调函数
     * 将接收到的点云数据转换为体素地图，并进行障碍物膨胀
     * @param msg 点云消息指针
     */
    inline void mapCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        if (!mapInitialized)
        {
            size_t cur = 0;
            const size_t total = msg->data.size() / msg->point_step;  // 计算点云总数
            float *fdata = (float *)(&msg->data[0]);  // 将数据转换为float指针
            
            // 遍历点云中的每个点
            for (size_t i = 0; i < total; i++)
            {
                cur = msg->point_step / sizeof(float) * i;

                // 检查点坐标是否有效（不是NaN或无穷大）
                if (std::isnan(fdata[cur + 0]) || std::isinf(fdata[cur + 0]) ||
                    std::isnan(fdata[cur + 1]) || std::isinf(fdata[cur + 1]) ||
                    std::isnan(fdata[cur + 2]) || std::isinf(fdata[cur + 2]))
                {
                    continue;
                }
                // 将有效点标记为被占用
                voxelMap.setOccupied(Eigen::Vector3d(fdata[cur + 0],
                                                     fdata[cur + 1],
                                                     fdata[cur + 2]));
            }

            // 对障碍物进行膨胀处理，增加安全边界
            voxelMap.dilate(std::ceil(config.dilateRadius / voxelMap.getScale()));

            mapInitialized = true;  // 标记地图初始化完成
        }
    }

    /**
     * @brief 主要规划函数
     * 执行路径规划、安全飞行通道生成、轨迹优化等核心流程
     */
    inline void plan()
    {
        if (startGoal.size() == 2)  // 确保有起点和终点
        {
            std::vector<Eigen::Vector3d> route;  // 路径点序列
            
            // 使用SFC生成器进行路径规划
            sfc_gen::planPath<voxel_map::VoxelMap>(startGoal[0],  // 起点
                                                   startGoal[1],  // 终点
                                                   voxelMap.getOrigin(),
                                                   voxelMap.getCorner(),
                                                   &voxelMap, 0.01,
                                                   route);
            
            std::vector<Eigen::MatrixX4d> hPolys;  // 超平面多面体序列
            std::vector<Eigen::Vector3d> pc;       // 点云
            voxelMap.getSurf(pc);  // 获取地图表面点

            // 生成凸包覆盖（安全飞行通道）
            sfc_gen::convexCover(route,
                                 pc,
                                 voxelMap.getOrigin(),
                                 voxelMap.getCorner(),
                                 7.0,  // 最大距离
                                 3.0,  // 最小距离
                                 hPolys);
            sfc_gen::shortCut(hPolys);  // 对通道进行shortcut优化

            if (route.size() > 1)
            {
                // 可视化多面体安全通道
                visualizer.visualizePolytope(hPolys);

                // 设置初始和终止状态（位置、速度、加速度）
                Eigen::Matrix3d iniState;
                Eigen::Matrix3d finState;
                iniState << route.front(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
                finState << route.back(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

                gcopter::GCOPTER_PolytopeSFC gcopter;  // GCOPTER优化器实例

                // 约束参数定义：
                // magnitudeBounds = [v_max, omg_max, theta_max, thrust_min, thrust_max]^T
                // penaltyWeights = [pos_weight, vel_weight, omg_weight, theta_weight, thrust_weight]^T
                // physicalParams = [vehicle_mass, gravitational_acceleration, horitonral_drag_coeff,
                //                   vertical_drag_coeff, parasitic_drag_coeff, speed_smooth_factor]^T
                
                // 初始化约束参数
                Eigen::VectorXd magnitudeBounds(5);  // 幅值边界约束
                Eigen::VectorXd penaltyWeights(5);   // 惩罚权重
                Eigen::VectorXd physicalParams(6);   // 物理参数
                
                // 设置幅值边界约束
                magnitudeBounds(0) = config.maxVelMag;     // 最大速度
                magnitudeBounds(1) = config.maxBdrMag;     // 最大体速率
                magnitudeBounds(2) = config.maxTiltAngle;  // 最大倾斜角
                magnitudeBounds(3) = config.minThrust;     // 最小推力
                magnitudeBounds(4) = config.maxThrust;     // 最大推力
                
                // 设置惩罚权重
                penaltyWeights(0) = (config.chiVec)[0];    // 位置权重
                penaltyWeights(1) = (config.chiVec)[1];    // 速度权重
                penaltyWeights(2) = (config.chiVec)[2];    // 角速度权重
                penaltyWeights(3) = (config.chiVec)[3];    // 倾角权重
                penaltyWeights(4) = (config.chiVec)[4];    // 推力权重
                
                // 设置物理参数
                physicalParams(0) = config.vehicleMass;    // 无人机质量
                physicalParams(1) = config.gravAcc;        // 重力加速度
                physicalParams(2) = config.horizDrag;      // 水平阻力系数
                physicalParams(3) = config.vertDrag;       // 垂直阻力系数
                physicalParams(4) = config.parasDrag;      // 寄生阻力系数
                physicalParams(5) = config.speedEps;       // 速度平滑因子
                
                const int quadratureRes = config.integralIntervs;  // 积分区间数

                traj.clear();  // 清空之前的轨迹

                // 设置GCOPTER优化器参数
                if (!gcopter.setup(config.weightT,      // 时间权重
                                   iniState, finState,  // 初始和终止状态
                                   hPolys, INFINITY,    // 安全通道和时间上界
                                   config.smoothingEps, // 平滑参数
                                   quadratureRes,       // 积分区间数
                                   magnitudeBounds,     // 幅值边界
                                   penaltyWeights,      // 惩罚权重
                                   physicalParams))     // 物理参数
                {
                    return;  // 设置失败则退出
                }

                // 执行轨迹优化
                if (std::isinf(gcopter.optimize(traj, config.relCostTol)))
                {
                    return;  // 优化失败则退出
                }

                // 如果生成了有效轨迹
                if (traj.getPieceNum() > 0)
                {
                    trajStamp = ros::Time::now().toSec();  // 记录轨迹时间戳
                    visualizer.visualize(traj, route);     // 可视化轨迹和路径
                }
            }
        }
    }

    /**
     * @brief 目标点回调函数
     * 接收目标点位置，检查可行性，更新起终点并触发路径规划
     * @param msg 位姿消息指针
     */
    inline void targetCallBack(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        if (mapInitialized)  // 确保地图已初始化
        {
            if (startGoal.size() >= 2)  // 如果已有两个点，则清空重新开始
            {
                startGoal.clear();
            }
            
            // 根据消息中的orientation.z计算目标点的z坐标
            // z坐标在地图范围内按比例分布
            const double zGoal = config.mapBound[4] + config.dilateRadius +
                                 fabs(msg->pose.orientation.z) *
                                     (config.mapBound[5] - config.mapBound[4] - 2 * config.dilateRadius);
            
            // 构造目标点3D坐标
            const Eigen::Vector3d goal(msg->pose.position.x, msg->pose.position.y, zGoal);
            
            // 检查目标点是否在自由空间中（0表示自由，非0表示被占用）
            if (voxelMap.query(goal) == 0)
            {
                // 可视化起点/终点
                visualizer.visualizeStartGoal(goal, 0.5, startGoal.size());
                startGoal.emplace_back(goal);  // 添加到起终点列表
            }
            else
            {
                ROS_WARN("Infeasible Position Selected !!!\n");  // 目标点不可行
            }

            plan();  // 触发路径规划
        }
        return;
    }

    /**
     * @brief 周期性处理函数
     * 计算当前轨迹时刻的动力学状态，发布控制量并进行可视化
     */
    inline void process()
    {
        // 组装物理参数向量
        Eigen::VectorXd physicalParams(6);
        physicalParams(0) = config.vehicleMass;  // 无人机质量
        physicalParams(1) = config.gravAcc;     // 重力加速度
        physicalParams(2) = config.horizDrag;   // 水平阻力系数
        physicalParams(3) = config.vertDrag;    // 垂直阻力系数
        physicalParams(4) = config.parasDrag;   // 寄生阻力系数
        physicalParams(5) = config.speedEps;    // 速度平滑因子

        // 初始化平坦性映射器
        flatness::FlatnessMap flatmap;
        flatmap.reset(physicalParams(0), physicalParams(1), physicalParams(2),
                      physicalParams(3), physicalParams(4), physicalParams(5));

        if (traj.getPieceNum() > 0)  // 如果存在有效轨迹
        {
            // 计算当前时间相对于轨迹起始时间的偏移
            const double delta = ros::Time::now().toSec() - trajStamp;
            
            // 检查是否在轨迹有效时间范围内
            if (delta > 0.0 && delta < traj.getTotalDuration())
            {
                // 动力学输出变量
                double thr;                // 推力
                Eigen::Vector4d quat;      // 姿态四元数
                Eigen::Vector3d omg;       // 机体系角速度

                // 通过平坦性映射将轨迹导数转换为控制量
                flatmap.forward(traj.getVel(delta),  // 速度
                                traj.getAcc(delta),  // 加速度
                                traj.getJer(delta),  // 加加速度(jerk)
                                0.0, 0.0,            // 偏航角和偏航角速度(设为0)
                                thr, quat, omg);     // 输出：推力、四元数、角速度

                // 计算各种状态量
                double speed = traj.getVel(delta).norm();                          // 速度模长
                double bodyratemag = omg.norm();                                   // 体速率模长
                double tiltangle = acos(1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2)));  // 倾斜角

                // 封装ROS消息
                std_msgs::Float64 speedMsg, thrMsg, tiltMsg, bdrMsg;
                speedMsg.data = speed;
                thrMsg.data = thr;
                tiltMsg.data = tiltangle;
                bdrMsg.data = bodyratemag;

                // 发布各种状态量
                visualizer.speedPub.publish(speedMsg);   // 发布速度
                visualizer.thrPub.publish(thrMsg);       // 发布推力
                visualizer.tiltPub.publish(tiltMsg);     // 发布倾斜角
                visualizer.bdrPub.publish(bdrMsg);       // 发布体速率

                // 可视化当前位置（用球体表示）
                visualizer.visualizeSphere(traj.getPos(delta),  // 当前位置
                                           config.dilateRadius); // 球体半径
            }
        }
    }
};

/**
 * @brief 主函数
 * 初始化ROS节点，创建全局规划器，启动主循环
 */
int main(int argc, char **argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "global_planning_node");
    ros::NodeHandle nh_;

    // 创建全局规划器实例，传入配置参数和节点句柄
    GlobalPlanner global_planner(Config(ros::NodeHandle("~")), nh_);

    // 设置主循环频率为1000Hz
    ros::Rate lr(1000);
    
    // 主循环
    while (ros::ok())
    {
        global_planner.process();  // 处理轨迹跟踪和可视化
        ros::spinOnce();           // 处理ROS回调
        lr.sleep();                // 按设定频率休眠
    }

    return 0;
}
