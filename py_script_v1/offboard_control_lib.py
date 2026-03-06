#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OffboardControl + Vehicle 封装示例（支持调试输出与节流打印）
OffboardControl + Vehicle example (with debug print and throttled logging)

功能说明：
- OffboardControl: 提供底层飞控命令接口（ROS2 Publisher/Subscriber）
- Vehicle: 封装生命周期与高层调用（隐藏ROS2特性，方便上层调用）

Functionality:
- OffboardControl: Low-level PX4 offboard control interface via ROS2 publishers/subscribers.
- Vehicle: High-level wrapper managing node lifecycle and providing a simple Python API.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
    GotoSetpoint,
    VehicleAttitude,
    VehicleAngularVelocity
)

from sensor_msgs.msg import LaserScan
import time
import math
import threading
from rclpy.executors import MultiThreadedExecutor
from px4_msgs.msg import VehicleAttitudeSetpoint
import numpy as np  # 用于 NaN
from scipy.spatial import cKDTree

# ---------------- 常量定义 | Constants ----------------
DISTANCE_TOLERANCE = 0.1  # 位置误差容忍度 (米) | Position error tolerance (meters)
YAW_TOLERANCE = 0.1       # 航向角误差容忍度 (弧度) | Yaw error tolerance (radians)
namespace = ''  # 可根据实例设置命名空间，例如 "/px4_1" | Namespace can be set per instance, e.g., "/px4_1"


class OffboardControl(Node):
    """
    ROS2 节点：实现 PX4 离板模式控制（基于位置）
    ROS2 Node for PX4 offboard control (position-based)
    """

    def __init__(self):
        super().__init__('offboard_control_center')
        self.get_logger().info("🚀 [INIT] Initializing OffboardControl node...")

        # 配置 QoS：最佳努力传输 + 瞬时本地持久化（适用于 PX4 快速状态流）
        # 核心原则：Subscriber 的 QoS 要求不能比 Publisher 更“严格”。
        # reliability=RELIABLE 会显著增加 CPU、内存和网络开销。
        # Configure QoS: Best-effort reliability + transient local durability (suitable for PX4 fast streams)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ---------------- 发布者 | Publishers ----------------
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, f'{namespace}/fmu/in/offboard_control_mode', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, f'{namespace}/fmu/in/vehicle_command', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, f'{namespace}/fmu/in/trajectory_setpoint', qos_profile)
        self.goto_setpoint_publisher = self.create_publisher(
            GotoSetpoint, f'{namespace}/fmu/in/goto_setpoint', qos_profile)
        
        self.vehicle_attitude_setpoint_publisher = self.create_publisher(VehicleAttitudeSetpoint
          , f'{namespace}/fmu/in/vehicle_attitude_setpoint_v1', qos_profile)

        self.get_logger().info(f"[PUB] Created publishers under namespace: '{namespace or 'default'}'")

        # ---------------- 订阅者 | Subscribers ----------------
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, f'{namespace}/fmu/out/vehicle_local_position_v1',
            self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, f'{namespace}/fmu/out/vehicle_status',
            self.vehicle_status_callback, qos_profile)
        self.vehicle_attitude_subscriber = self.create_subscription(
            VehicleAttitude, f'{namespace}/fmu/out/vehicle_attitude',
            self.vehicle_attitude_callback, qos_profile)
        self.lidar_scan_subscriber = self.create_subscription(
            LaserScan, '/lidar_scan',
            self.lidar_scan_callback, qos_profile)
        self.angular_velocity_sub = self.create_subscription(
            VehicleAngularVelocity,
            '/fmu/out/vehicle_angular_velocity',
            self.vehicle_angular_velocity_callback,
            qos_profile
        )


        self.get_logger().info("[SUB] Subscribed to local_position/status/attitude/lidar_scan")

        # ---------------- 状态变量 | State Variables ----------------
        self.offboard_setpoint_counter = 0          # 心跳计数器 | Heartbeat counter
        self.takeoff_height = 2.0                   # 默认起飞高度 | Default takeoff altitude
        self.home_position = [0.0, 0.0, 0.0]        # Home 点（ENU 坐标系）| Home position in ENU
        self.vehicle_local_position_enu = VehicleLocalPosition()  # 存储转换后的 ENU 位置 | Store converted ENU position
        self.vehicle_local_position_ned = VehicleLocalPosition() 
        self.vehicle_local_position_received = False  # 是否收到有效位置 | Whether valid position received
        self.vehicle_status = VehicleStatus()         # 当前飞行器状态 | Current vehicle status
        self.vehicle_attitude = None                # 姿态四元数缓存 | Cached attitude quaternion
        self.lidar_scan = None                      # 激光雷达扫描缓存 | Cached lidar scan
        self.dwa_active = False                     # DWA 模式标志 | DWA active flag
        self.control_mode = 'position'
        self.yaw_rate_frd = 0.0

        # 标志位与线程锁 | Flags & Thread Locks
        self.is_takeoff_complete = False
        self.target_reached = False
        self.lock = threading.Lock()

        # 节流日志时间记录表 | Throttled log timestamp dict
        self._last_log = {}
        self.target = (0.0, 0.0, 0.0, 0.0)  # (x, y, z, yaw) in ENU

        self.get_logger().info("✅ [INIT] OffboardControl initialized successfully!")


    # ---------------- 工具函数：节流日志 | Utility: Throttled Logging ----------------
    def throttle_log(self, interval_sec: float, msg: str, level: str = "info", tag: str = "default"):
        """
        节流打印函数：避免高频日志刷屏（如位置反馈、心跳）
        Throttled logging: prevent console flooding from high-frequency logs (e.g., position, heartbeat)
        
        参数 | Args:
        - interval_sec: 最小打印间隔（秒）| Minimum interval between logs (seconds)
        - msg: 日志内容 | Log message
        - level: 日志级别 ("info", "warn", "error") | Log level
        - tag: 日志标签，用于区分不同来源 | Log tag for source differentiation
        """
        now = time.time()
        if tag not in self._last_log or (now - self._last_log[tag]) > interval_sec:
            if level == "info":
                self.get_logger().info(msg)
            elif level == "warn":
                self.get_logger().warning(msg)
            elif level == "error":
                self.get_logger().error(msg)
            self._last_log[tag] = now


    # ---------------- 心跳机制 | Heartbeat Mechanism ----------------
    def heartbeat_thread_start(self):
        """启动心跳线程：以固定频率发布控制模式与轨迹点，维持 Offboard 模式"""
        """Start heartbeat thread: publish control mode & setpoints at fixed rate to maintain offboard mode"""
        self.stop_heartbeat = threading.Event()
        # 位置环与速度环运行频率为 50Hz,位置姿态控制超过没有意义。对于小型飞控，机载通信频率过高，会报cpu load too high错误，导致无法解锁
        self.heartbeat_hz = 20
        self.heartbeat_thread = threading.Thread(target=self.heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        self.get_logger().info(f"🔁 [HEARTBEAT] Started heartbeat thread at {self.heartbeat_hz} Hz")


    def heartbeat_loop(self):
        """心跳主循环：每 1/20 秒发送一次控制信号"""
        """Heartbeat main loop: send control signal every 1/20 second"""
        rate = 1.0 / float(self.heartbeat_hz)  #TODO 使用ros2时间相关API，如rate,实现更精确控制
        self.get_logger().debug("[HEARTBEAT] Entering heartbeat loop...")
        while not self.stop_heartbeat.is_set() and rclpy.ok():
            try:
                self.publish_offboard_control_heartbeat_signal(self.control_mode)
                self.publish_current_setpoint()
                self.offboard_setpoint_counter += 1
                # 节流打印心跳计数（每5秒一次）
                self.throttle_log(5.0, f"[HEARTBEAT] Published setpoint #{self.offboard_setpoint_counter}", tag="heartbeat")
            except Exception as e:
                self.get_logger().error(f"[HEARTBEAT] Exception in loop: {e}")
            time.sleep(rate)
        self.get_logger().info("⏹️ [HEARTBEAT] Heartbeat thread exiting")


    def publish_current_setpoint(self):
        """发布当前目标点（线程安全），根据控制模式自适应"""
        """Publish current target setpoint (thread-safe), adapt based on control mode"""
        
        mode = self.control_mode
        target = self.target
        if mode == 'position':
            x, y, z, yaw = target
            self.publish_trajectory_setpoint(position=[x, y, z], yaw=yaw)
        elif mode == 'velocity':
            vx, vy, vz, yawspeed = target
            self.publish_trajectory_setpoint(velocity=[vx, vy, vz], yawspeed=yawspeed)
        elif mode == 'attitude':
            roll, pitch, yaw, thrust = target
            q_d = self.euler_to_quaternion(roll, pitch, yaw)
            thrust_body = [0.0, 0.0, -thrust]  # 多旋翼假设，负 z 向上
            self.publish_attitude_setpoint(q_d, thrust_body, yaw_sp_move_rate=None)  # 或设置 yaw_sp_move_rate 如果需要
            self.publish_attitude_setpoint(roll, pitch, yaw, thrust)
        else:
            self.get_logger().warning(f"[SETPOINT] Unsupported control mode: {mode}")


    def set_control_mode(self, mode: str):
        """设置控制模式：'position', 'velocity', 'attitude'"""
        if mode in ['position', 'velocity', 'attitude']:
            self.control_mode = mode
            self.get_logger().info(f"🔄 [MODE] Switched to {mode} control mode")
        else:
            self.get_logger().error(f"❌ [MODE] Invalid control mode: {mode}")


    def update_position_setpoint(self, x: float, y: float, z: float, yaw: float):
        """更新位置目标点（线程安全）"""
        
        if self.control_mode != 'position':
                self.set_control_mode('position')
        old_target = self.target
        self.target = (float(x), float(y), float(z), float(yaw))
        self.get_logger().debug(f"🎯 [POSITION] Updated from {old_target} → {self.target} (ENU)")


    def update_velocity_setpoint(self, vx: float, vy: float, vz: float, yawspeed: float):
        """更新速度目标点（线程安全）"""
        
        if self.control_mode != 'velocity':
            self.set_control_mode('velocity')
        old_target = self.target
        self.target = (float(vx), float(vy), float(vz), float(yawspeed))
        self.get_logger().info(f"🎯 [VELOCITY] Updated from  {old_target} → {self.target} (ENU)")


    def update_attitude_setpoint(self, roll: float, pitch: float, yaw: float, thrust: float):
        """更新姿态目标点（线程安全），使用欧拉角和推力"""
        with self.lock:
            if self.control_mode != 'attitude':
                self.set_control_mode('attitude')
            old_target = self.target
            self.target = (float(roll), float(pitch), float(yaw), float(thrust))
        self.get_logger().debug(f"🎯 [ATTITUDE] Updated from {old_target} → {self.target}")

    
    def request_vehicle_command(self, command, param1=0.0, param2=0.0):
        """Send a vehicle command request."""
        '''non-blocking'''
        request = VehicleCommandSrv.Request()
        msg = VehicleCommand()
        # Ensure the parameters are floats
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        request.request = msg
        self.service_done = False
        self.service_result = None
        future = self.vehicle_command_client.call_async(request)
        #print(f'future:{future}')
        #不等待服务返回\服务返回后，执行 response_callback(future).适合：不需要额外参数传回调
        future.add_done_callback(partial(self.response_callback))
        self.get_logger().info('Command sent (non-blocking)')

    def response_callback(self, future):
            """Handle the response from the vehicle command service."""
            try:
                response = future.result()
                #print(f'response:{response}')
                reply = response.reply
                self.service_result = response.reply.result
                if self.service_result == reply.VEHICLE_CMD_RESULT_ACCEPTED:
                    self.get_logger().info('Command accepted')
                elif self.service_result == reply.VEHICLE_CMD_RESULT_TEMPORARILY_REJECTED:
                    self.get_logger().warning('Command temporarily rejected')
                elif self.service_result == reply.VEHICLE_CMD_RESULT_DENIED:
                    self.get_logger().warning('Command denied')
                elif self.service_result == reply.VEHICLE_CMD_RESULT_UNSUPPORTED:
                    self.get_logger().warning('Command unsupported')
                elif self.service_result == reply.VEHICLE_CMD_RESULT_FAILED:
                    self.get_logger().warning('Command failed')
                elif self.service_result == reply.VEHICLE_CMD_RESULT_IN_PROGRESS:
                    self.get_logger().warning('Command in progress')
                elif self.service_result == reply.VEHICLE_CMD_RESULT_CANCELLED:
                    self.get_logger().warning('Command cancelled')
                else:
                    self.get_logger().warning('Command reply unknown')
                self.service_done = True

            except Exception as e:
                self.get_logger().error(f'Service call failed: {e}')

    async def request_vehicle_command_blocking(self, command, param1=0.0, param2=0.0, timeout_sec=5.0):
        """
        Send vehicle command and BLOCK (await) until response is received or timeout.

        This function:
        - Sends a command via call_async()
        - Awaits the service response using asyncio.wait_for()
        - Processes the result immediately (no callback needed)
        """
        # -------------------------
        # 1. 构造服务请求
        # -------------------------
        request = VehicleCommandSrv.Request()
        msg = VehicleCommand()

        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        request.request = msg

        self.get_logger().info(f"Sending command {command} (blocking for response)...")

        # -------------------------
        # 2. 异步发送并等待响应
        # -------------------------
        future = self.vehicle_command_client.call_async(request)

        try:
            # 👇 阻塞等待（await）
            response = await asyncio.wait_for(future, timeout=timeout_sec)

        except asyncio.TimeoutError:
            self.get_logger().error(f"Command {command} timed out after {timeout_sec} seconds")
            return None

        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            return None

        # -------------------------
        # 3. 服务响应处理
        # -------------------------
        reply = response.reply
        result = reply.result
        self.service_result = result

        if result == reply.VEHICLE_CMD_RESULT_ACCEPTED:
            self.get_logger().info("Command accepted")
        elif result == reply.VEHICLE_CMD_RESULT_TEMPORARILY_REJECTED:
            self.get_logger().warning("Command temporarily rejected")
        elif result == reply.VEHICLE_CMD_RESULT_DENIED:
            self.get_logger().warning("Command denied")
        elif result == reply.VEHICLE_CMD_RESULT_UNSUPPORTED:
            self.get_logger().warning("Command unsupported")
        elif result == reply.VEHICLE_CMD_RESULT_FAILED:
            self.get_logger().warning("Command failed")
        elif result == reply.VEHICLE_CMD_RESULT_IN_PROGRESS:
            self.get_logger().warning("Command in progress")
        elif result == reply.VEHICLE_CMD_RESULT_CANCELLED:
            self.get_logger().warning("Command cancelled")
        else:
            self.get_logger().warning("Command reply unknown")

        return result


    # ---------------- 坐标系转换 | Coordinate Conversion ----------------
    def ned_to_enu(self, x_ned, y_ned, z_ned):
        """将 NED 坐标转为 ENU 坐标系"""
        """Convert NED coordinates to ENU"""
        return y_ned, x_ned, -z_ned

    def enu_to_ned(self, x_enu, y_enu, z_enu):
        """将 ENU 坐标转为 NED 坐标系"""
        """Convert ENU coordinates to NED"""
        return y_enu, x_enu, -z_enu

    def normalize_yaw(self, yaw_diff: float) -> float:
        """将航向角差归一化到 [-π, π] 并返回绝对值"""
        """Normalize yaw difference to [-π, π] and return absolute value"""
        while yaw_diff > math.pi:
            yaw_diff -= 2 * math.pi
        while yaw_diff < -math.pi:
            yaw_diff += 2 * math.pi
        return abs(yaw_diff)
    
    def euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> list[float]:
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return [float(w), float(x), float(y), float(z)]  # 正则化可选


    # ---------------- ROS2 回调函数 | ROS2 Callbacks ----------------
    def vehicle_local_position_callback(self, msg: VehicleLocalPosition):
        """
        位置回调：接收 NED 坐标，转换为 ENU 并存储
        Position callback: receive NED, convert to ENU, and store
        """
        try:
            x_enu, y_enu, z_enu = self.ned_to_enu(msg.x, msg.y, msg.z)
            heading_enu = -msg.heading + math.radians(90)  # NED heading → ENU heading
            with self.lock:
                self.vehicle_local_position_enu.x = x_enu
                self.vehicle_local_position_enu.y = y_enu
                self.vehicle_local_position_enu.z = z_enu
                self.vehicle_local_position_enu.heading = heading_enu
                self.vehicle_local_position_received = True
            self.throttle_log(
                2.0,
                f"[POSITION] ENU=({x_enu:.2f}, {y_enu:.2f}, {z_enu:.2f}), "
                f"heading={math.degrees(heading_enu):.1f}°",
                tag="position"
            )
            self.vehicle_local_position_ned.vx = msg.vx
            self.vehicle_local_position_ned.vy = msg.vy
            self.vehicle_local_position_ned.vz = msg.vz
            #self.vehicle_local_position_ned.

        except Exception as e:
            self.get_logger().error(f"[POSITION] Callback error: {e}")

        
    def vehicle_angular_velocity_callback(self, msg):
            """
            Callback for VehicleAngularVelocity.
            Stores the yaw rate (body Z-axis angular velocity in FRD frame).
            
            In PX4's FRD (Forward-Right-Down) body frame:
            - xyz[0] = roll rate  (around X, forward)
            - xyz[1] = pitch rate (around Y, right)
            - xyz[2] = yaw rate   (around Z, down) → 正值表示顺时针偏航（向右转）
            """
            if len(msg.xyz) >= 3:
                self.yaw_rate_frd = float(msg.xyz[2])  # yaw_rate in rad/s
            else:
                self.get_logger().warn("Received invalid VehicleAngularVelocity message (xyz too short)")




    def vehicle_status_callback(self, msg: VehicleStatus):
        """状态回调：更新飞行器导航与解锁状态"""
        """Status callback: update navigation and arming state"""
        with self.lock:
            old_nav = getattr(self.vehicle_status, 'nav_state', 'N/A')
            old_arm = getattr(self.vehicle_status, 'arming_state', 'N/A')
            self.vehicle_status = msg
        self.throttle_log(
            5.0,
            f"[STATUS] nav_state={msg.nav_state} (was {old_nav}), "
            f"arming_state={msg.arming_state} (was {old_arm})",
            tag="status"
        )


    def vehicle_attitude_callback(self, msg: VehicleAttitude):
        """姿态回调：缓存飞行器姿态四元数"""
        
        self.vehicle_attitude = msg
        #self.throttle_log(1.0, f"vehicle_attitude updated",tag="vehicle_attitude")

    def lidar_scan_callback(self, msg: LaserScan):
        """激光雷达回调：缓存扫描数据"""
        with self.lock:
            self.lidar_scan = msg

    def process_radar_data(self, distance: float, a: float) -> tuple[float, bool]:
        """
        处理2D雷达数据（FLU坐标系）：
        1. 剔除离群点（使用中值滤波和Z分数阈值）。
        2. 计算正前方走廊（宽度a）内最近障碍物距离（沿x轴）。
        3. 识别所有可通过的环形间隙，并输出[theta_starting, theta_ending]（rad）。
        4. 判断如果沿雷达x轴前进distance米，是否会遇到障碍物（has_passable_area = True if not encounter）。

        参数:
        - distance: 前方检查距离 (m)
        - a: 无人机正方形机架边长 (m)

        返回:
        - min_front_dist: 正前方走廊内最近障碍物距离 (m)，若无则inf
        - has_passable_area: 沿x轴前进distance米是否无障碍 (bool)
        """
        from scipy.signal import medfilt

        with self.lock:
            scan = self.lidar_scan
        if scan is None:
            self.get_logger().warning("[RADAR] No lidar scan available")
            return float('inf'), False

        ranges = np.array(scan.ranges, dtype=float)
        num_readings = len(ranges)

        # ========== 步骤1: 剔除离群点 ==========
        finite_mask = np.isfinite(ranges) & (ranges > 0.0)
        filtered_ranges = ranges.copy()
        if np.sum(finite_mask) > 5:
            filtered_ranges[finite_mask] = medfilt(ranges[finite_mask], kernel_size=5)

        diffs = np.abs(ranges[finite_mask] - filtered_ranges[finite_mask])
        if len(diffs) > 0:
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)
            z_scores = (diffs - mean_diff) / (std_diff + 1e-6)
            outlier_mask = z_scores > 3.0
            outlier_indices = np.where(finite_mask)[0][outlier_mask]
            filtered_ranges[outlier_indices] = np.nan

        valid_mask = np.isfinite(filtered_ranges) & (filtered_ranges > 0.0)
        self.get_logger().debug(f"[RADAR] Valid points: {np.sum(valid_mask)}/{num_readings}")

        # 角度数组
        angles = scan.angle_min + np.arange(num_readings) * scan.angle_increment

        # ========== 步骤2 & 4: 计算正前方min_dist并检查直线路径 ==========
        # 转换为笛卡尔坐标（所有有效点）
        obs_angles = angles[valid_mask]
        obs_ranges = filtered_ranges[valid_mask]
        obs_x = obs_ranges * np.cos(obs_angles)
        obs_y = obs_ranges * np.sin(obs_angles)

        # 正前方走廊掩码：|y| < a/2, x > 0
        corridor_mask = (np.abs(obs_y) < a / 2) & (obs_x > 0.0)
        corridor_x = obs_x[corridor_mask]

        if len(corridor_x) > 0:
            min_front_dist = float(np.min(corridor_x))
            # 检查是否在distance内遇到障碍
            has_passable_area = min_front_dist > distance
        else:
            min_front_dist = float('inf')
            has_passable_area = True

        self.get_logger().info(f"[RADAR] Front obstacle distance: {min_front_dist:.2f}m")
        result_str = "✅ No" if has_passable_area else "❌ Yes"
        self.get_logger().info(f"[RADAR] Will encounter obstacle within {distance:.2f}m along x-axis: {result_str}")

        # ========== 步骤3: 环形间隙检查 ==========
        # 仅考虑distance内的障碍
        distance_mask = valid_mask & (filtered_ranges <= distance)
        obs_indices = np.where(distance_mask)[0]
        obs_angles = angles[obs_indices]
        obs_ranges = filtered_ranges[obs_indices]

        # 按角度排序
        angle_sort_idx = np.argsort(obs_angles)
        sorted_angles = obs_angles[angle_sort_idx]
        sorted_ranges = obs_ranges[angle_sort_idx]

        passable_gaps = []

        # 特殊情况：无障碍
        if len(sorted_angles) == 0:
            effective_d = distance - a / 2
            fov = scan.angle_max - scan.angle_min
            if effective_d > 0 and fov >= a / effective_d:
                passable_gaps.append((scan.angle_min, scan.angle_max))

        # 非环绕间隙
        if len(sorted_angles) >= 2:
            angle_diffs = np.diff(sorted_angles)
            for i in range(len(angle_diffs)):
                delta_theta = angle_diffs[i]
                r1 = sorted_ranges[i]
                r2 = sorted_ranges[i + 1]
                effective_d = min(r1, r2) - a / 2
                if effective_d > 0 and delta_theta >= a / effective_d:
                    passable_gaps.append((sorted_angles[i], sorted_angles[i + 1]))

        # 环绕间隙
        if len(sorted_angles) >= 2:  # 至少2点才能环绕
            wrap_delta = sorted_angles[0] + 2 * math.pi - sorted_angles[-1]
            if wrap_delta > 0:
                r1 = sorted_ranges[-1]
                r2 = sorted_ranges[0]
                effective_d = min(r1, r2) - a / 2
                if effective_d > 0 and wrap_delta >= a / effective_d:
                    passable_gaps.append((sorted_angles[-1], sorted_angles[0] + 2 * math.pi))

        # 输出可通过区域
        if passable_gaps:
            gaps_str = ', '.join([f"[{s:.2f}, {e:.2f}] rad ({math.degrees(s):.1f}°, {math.degrees(e):.1f}°)" for s, e in passable_gaps])
            self.get_logger().info(f"[RADAR] Passable regions in FLU: {gaps_str}")
        else:
            self.get_logger().info("[RADAR] No passable regions found")

        return min_front_dist, has_passable_area
  
    def quaternion_to_dcm(self, q: list[float]) -> np.ndarray:
        """将四元数转换为方向余弦矩阵（FRD -> NED）。"""
        w, x, y, z = q
        return np.array([
            [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2],
        ])

    def is_obstacle_ahead(self, safe_dist: float, front_angle=60) -> bool:
        """判断机体前方是否存在障碍物（雷达前向 ±front_angle°）。"""
        scan = self.lidar_scan
        if scan is None:
            return False

        front_angle_radians = math.radians(front_angle)
        min_dist = float('inf')
        for i, r in enumerate(scan.ranges):
            theta = scan.angle_min + i * scan.angle_increment
            if abs(theta) <= front_angle_radians and math.isfinite(r) and r > 0.0:
                min_dist = min(min_dist, r)

        self.throttle_log(0.5, f"[DWA] Front min distance: {min_dist:.2f} m", tag="dwa_front")
        return min_dist < safe_dist

    def compute_dwa(self, target_enu: list[float]) -> tuple[float, float, float]:
        """计算 DWA 最优速度（FLU 局部坐标），返回 (vx, vy, w)。"""
        with self.lock:
            pos_ok = self.vehicle_local_position_received
            pos = self.vehicle_local_position_enu
            att = self.vehicle_attitude
            scan = self.lidar_scan

        if not pos_ok or att is None or scan is None:
            self.throttle_log(1.0, "[DWA] Missing position/attitude/scan, output zero velocity", level="warning", tag="dwa_missing")
            return 0.0, 0.0, 0.0

        cur_ned = np.array(self.enu_to_ned(pos.x, pos.y, pos.z), dtype=float)
        tgt_ned = np.array(self.enu_to_ned(target_enu[0], target_enu[1], target_enu[2]), dtype=float)
        rel_ned = tgt_ned - cur_ned
        dcm = self.quaternion_to_dcm(att.q)
        rel_frd = dcm.T @ rel_ned
        rel_flu = np.array([rel_frd[0], -rel_frd[1], -rel_frd[2]], dtype=float)
        goal_heading = math.atan2(rel_flu[1], rel_flu[0])
        goal_x = rel_flu[0]
        goal_y = rel_flu[1]

        obs = []
        for i, r in enumerate(scan.ranges):
            if math.isfinite(r) and r > 0.0:
                theta = scan.angle_min + i * scan.angle_increment
                obs.append((r * math.cos(theta), r * math.sin(theta)))

        obs_points = np.array(obs) if obs else np.empty((0, 2))
        obs_tree = cKDTree(obs_points) if len(obs_points) > 0 else None

        max_speed = 0.4
        max_w = math.pi / 2.0
        v_samples = 10
        w_samples = 20
        dt = 0.5
        predict_time = 2.0
        steps = int(predict_time / dt)
        robot_radius = 0.5
        alpha, beta, gamma = 0.2, 0.2, 0.3

        best_score = -float('inf')
        best_vx = best_vy = best_w = 0.0
  
    # 当前速度参与（ENU->NED->FRD->FLU）
        vel_ned = np.array([self.vehicle_local_position_ned.vx,
                            self.vehicle_local_position_ned.vy,
                            self.vehicle_local_position_ned.vz])
        vel_frd = dcm.T @ vel_ned
        vel_flu = np.array([vel_frd[0], -vel_frd[1], -vel_frd[2]])
        w_now = -self.yaw_rate_frd  # 转换到FLU


       # 动态窗口（加速度约束）
        a_max = 1.0  # 最大线加速度
        alpha_max = math.pi / 4.0  # 最大角加速度
        vx_min = max(-max_speed, vel_flu[0] - a_max * dt)
        vx_max = min(max_speed, vel_flu[0] + a_max * dt)
        vy_min = max(-max_speed, vel_flu[1] - a_max * dt)
        vy_max = min(max_speed, vel_flu[1] + a_max * dt)
        w_min = max(-max_w, w_now - alpha_max * dt)
        w_max = min(max_w, w_now + alpha_max * dt)
        #feasible = True

        for vx in np.linspace(vx_min/6, vx_max, v_samples):
            for vy in np.linspace(vy_min/3, vy_max, v_samples):
                speed = math.hypot(vx, vy)
                if speed > max_speed + 1e-6:
                    continue
                for w in np.linspace(w_min, w_max, w_samples):
                    x = y = th = 0.0
                    collided = False
                    min_dist = float('inf')
                    for _ in range(steps):
                        # dvx = vx - vel_flu[0]
                        # dvy = vy - vel_flu[1]
                        # # 当前加速度
                        # ax = dvx / dt
                        # ay = dvy / dt
                        # if abs(ax) > a_max + 1e-6 or abs(ay) > a_max + 1e-6:
                        #   feasible = False
                        #   break
                        dx = vx * math.cos(th) - vy * math.sin(th)
                        dy = vx * math.sin(th) + vy * math.cos(th)
                        x += dx * dt
                        y += dy * dt
                        th += w * dt

                        if obs_tree is not None:
                            dist, _ = obs_tree.query([x, y])
                            min_dist = min(min_dist, dist)
                            if dist <= robot_radius:
                                collided = True
                                break

                    if collided:
                        continue

                    if obs_tree is None:
                        min_dist = max_speed * predict_time

                    #heading_score = (math.pi - abs(th - goal_heading)) / math.pi
                    angle_diff = math.atan2(math.sin(th - goal_heading), math.cos(th - goal_heading))
                    heading_score = (math.pi - abs(angle_diff)) / math.pi
                    vel_score = speed / max_speed
                    clear_score = min_dist / (max_speed * predict_time + 1e-6)
                    score = alpha * heading_score + beta * vel_score + gamma * clear_score
                    goal_dist = math.hypot(x - goal_x, y - goal_y)
                    goal_score = 1.0 / (goal_dist + 0.1)
                    score += 0.3 * goal_score

                    if score > best_score:
                        best_score = score
                        best_vx, best_vy, best_w = vx, vy, w

        if best_score == -float('inf') or (best_vx == 0.0 and best_vy == 0.0):
            self.throttle_log(1.0, "[DWA] No valid trajectory, rotate recovery", level="warning", tag="dwa_recovery")
            if obs_tree is not None:
                _, nearest_idx = obs_tree.query([0.0, 0.0])
                nearest_theta = math.atan2(obs_points[nearest_idx][1], obs_points[nearest_idx][0])
                best_w = math.copysign(max_w / 2.0, nearest_theta)

        self.throttle_log(0.5, f"[DWA] best vx={best_vx:.2f}, vy={best_vy:.2f}, w={best_w:.2f}", tag="dwa_best")
        return best_vx, best_vy, best_w


    # ---------------- 飞行器命令 | Vehicle Commands ----------------
    def arm(self):
        """发送解锁命令，并记录 Home 点"""
        """Send arm command and record home position"""
        self.get_logger().info("🔓 Sending ARM command...")
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info("✅ Arm command sent")

        if not self.vehicle_local_position_received:
            self.get_logger().warning("⚠️ Waiting for first local position message...")

        # 等待有效位置
        while not self.vehicle_local_position_received and rclpy.ok():
            time.sleep(0.5)

        with self.lock:
            self.home_position = [
                self.vehicle_local_position_enu.x,
                self.vehicle_local_position_enu.y,
                self.vehicle_local_position_enu.z,
            ]
        self.get_logger().info(f"🏠 Home position recorded: {self.home_position} (ENU)")


    def disarm(self):
        """发送上锁命令"""
        """Send disarm command"""
        self.get_logger().info("🔒 Sending DISARM command...")
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info("✅ Disarm command sent")


    def arm_srv(self):
        self.get_logger().info('Requesting arm')
        self.request_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        while not self.service_done:
            time.sleep(0.05)
            self.get_logger().info('waiting for arm service to be done')
        self.get_logger().info('arm service has done')
        if self.service_result == 0:
                self.get_logger().info('Vehicle is armed')
                self.state = 'armed'
        # record takeoff position and RTL position
        with self.lock:
            self.home_position = [
                self.vehicle_local_position_enu.x,
                self.vehicle_local_position_enu.y,
                self.vehicle_local_position_enu.z,
            ]
        self.get_logger().info(f"🏠 Home position recorded: {self.home_position} (ENU)")


    def disarm_srv(self):
        self.get_logger().info('Requesting disarm')
        self.request_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)




    def engage_offboard_mode(self, prewarm_count=10, prewarm_timeout=5.0):
        """
        进入 Offboard 模式前，需先预热（发送至少若干个控制点）
        Must pre-warm by sending several setpoints before engaging offboard mode
        """
        self.get_logger().info(f"🔄 Engaging OFFBOARD mode (prewarm: {prewarm_count} msgs or {prewarm_timeout}s)")

        start = time.time()
        while self.offboard_setpoint_counter < prewarm_count and (time.time() - start) < prewarm_timeout and rclpy.ok():
            time.sleep(0.05)

        if self.offboard_setpoint_counter < prewarm_count:
            self.get_logger().warning(
                f"⚠️ Prewarm insufficient: only {self.offboard_setpoint_counter}/{prewarm_count} setpoints sent"
            )
        else:
            self.get_logger().info(f"✅ Prewarm complete: {self.offboard_setpoint_counter} setpoints sent")

        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("✅ Switched to OFFBOARD mode!")


    def engage_offboard_mode_srv(self, prewarm_count=10, prewarm_timeout=5.0):
        """
        进入 Offboard 模式前，需先预热（发送至少若干个控制点）
        Must pre-warm by sending several setpoints before engaging offboard mode
        """
        self.get_logger().info(f"🔄 Engaging OFFBOARD mode (prewarm: {prewarm_count} msgs or {prewarm_timeout}s)")

        start = time.time()
        while self.offboard_setpoint_counter < prewarm_count and (time.time() - start) < prewarm_timeout and rclpy.ok():
            time.sleep(0.05)

        if self.offboard_setpoint_counter < prewarm_count:
            self.get_logger().warning(
                f"⚠️ Prewarm insufficient: only {self.offboard_setpoint_counter}/{prewarm_count} setpoints sent"
            )
        else:
            self.get_logger().info(f"✅ Prewarm complete: {self.offboard_setpoint_counter} setpoints sent")
        self.get_logger().info('Requesting switch to Offboard mode')
        self.request_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1, 6)



    def hover(self, duration: float, timeout: float = None) -> bool:
        """
        阻塞式悬停：保持当前位置和航向，持续指定时长
        Blocking hover: maintain current position and heading for the specified duration
        
        参数 | Args:
        - duration: 悬停时长（秒）| Hover duration (seconds)
        - timeout: 可选超时（秒），默认等于 duration + 10 | Optional timeout (seconds), defaults to duration + 10
        
        返回 | Returns:
        - bool: 是否成功完成悬停 | True if hover completed successfully
        """
        if duration <= 0:
            self.get_logger().error("❌ Hover duration must be positive!")
            return False

        # 默认超时为 duration + 缓冲
        if timeout is None:
            timeout = duration + 10.0

        # 获取当前 ENU 位置和航向
        with self.lock:
            if not self.vehicle_local_position_received:
                self.get_logger().warning("⚠️ No valid position received; cannot hover.")
                return False
            cx = self.vehicle_local_position_enu.x
            cy = self.vehicle_local_position_enu.y
            cz = self.vehicle_local_position_enu.z
            ch = self.vehicle_local_position_enu.heading

        # 更新 setpoint 为当前位置（position 模式）
        self.update_position_setpoint(cx, cy, cz, ch)
        self.get_logger().info(f"🛸 Starting hover at ENU ({cx:.2f}, {cy:.2f}, {cz:.2f}), yaw={math.degrees(ch):.1f}° for {duration:.1f}s")

        start = time.time()
        last_log = start
        while rclpy.ok() and (time.time() - start) < timeout:
            elapsed = time.time() - start
            if elapsed >= duration:
                self.get_logger().info("✅ Hover duration completed!")
                return True

            # 节流日志剩余时间
            if time.time() - last_log >= 1.0:
                self.throttle_log(
                    1.0,
                    f"[HOVER] Elapsed: {elapsed:.1f}/{duration:.1f}s",
                    tag="hover"
                )
                last_log = time.time()

            time.sleep(0.1)

        self.get_logger().warning("⚠️ Hover timed out!")
        return False
    
    def land(self, latitude: float = np.nan, longitude: float = np.nan, altitude: float = 0.0, yaw: float = np.nan, abort_alt: float = 0.0, land_mode: int = 0, timeout: float = 60.0) -> bool:
        """
        阻塞式降落：发送 MAV_CMD_NAV_LAND 命令，并在指定位置降落
        Blocking land: send MAV_CMD_NAV_LAND command to land at specified location
        如果在 Offboard 模式下，commander 会覆盖外部设定点，确保切换到内部控制
        
        参数 | Args:
        - latitude: 纬度（NaN 使用当前位置）| Latitude (NaN for current position)
        - longitude: 经度（NaN 使用当前位置）| Longitude (NaN for current position)
        - altitude: 降落高度（地面高度）| Landing altitude (ground level)
        - yaw: 期望偏航角（NaN 使用系统默认）| Desired yaw angle (NaN for system default)
        - abort_alt: 中止高度（0=默认）| Abort altitude (0=undefined/system default)
        - land_mode: 降落模式（0=正常，参考 PRECISION_LAND_MODE）| Land mode (e.g., PRECISION_LAND_MODE)
        - timeout: 超时时间（秒）| Timeout (seconds)
        
        返回 | Returns:
        - bool: 是否成功降落 | True if landed successfully
        """
        self.get_logger().info(f"🛬 Sending LAND command at lat={latitude}, lon={longitude}, alt={altitude:.2f}m, yaw={yaw if not np.isnan(yaw) else 'default'}")

        # 发送 VehicleCommand: MAV_CMD_NAV_LAND (21)
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_NAV_LAND,
            param1=float(abort_alt),
            param2=float(land_mode),
            param3=0.0,  # Empty
            param4=float(yaw),
            param5=float(latitude),
            param6=float(longitude),
            param7=float(altitude)
        )

        # 阻塞等待降落完成：检查高度 ≈ 0 或状态为 landed
        start = time.time()
        last_log = start
        while rclpy.ok() and time.time() - start < timeout:
            with self.lock:
                cz = self.vehicle_local_position_enu.z
                nav_state = self.vehicle_status.nav_state
            remaining_time = timeout - (time.time() - start)

            # 节流日志
            if time.time() - last_log >= 1.0:
                self.throttle_log(
                    1.0,
                    f"[LAND] Altitude: {cz:.2f}m, nav_state={nav_state}, remaining time: {remaining_time:.1f}s",
                    tag="land"
                )
                last_log = time.time()

            # 检查是否降落：高度 < 0.1m 或 nav_state 表示 landed (VehicleStatus.NAVIGATION_STATE_AUTO_LAND 或 arming_state disarmed)
            if cz < 0.1 or nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND or self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                self.get_logger().info("✅ Landing complete!")
                return True

            time.sleep(0.1)

        self.get_logger().warning("⚠️ Land timed out!")
        return False


    # ---------------- 发布辅助函数 | Publish Helpers ----------------
    def publish_offboard_control_heartbeat_signal(self, control_mode='position'):
        """发布 Offboard 控制模式信号（维持模式激活）"""
        """Publish offboard control mode signal (to keep mode active)"""
        msg = OffboardControlMode()
        msg.position = (control_mode == 'position')
        msg.velocity = (control_mode == 'velocity')
        msg.acceleration = (control_mode == 'acceleration')  # 如果支持
        msg.attitude = (control_mode == 'attitude')
        msg.body_rate = (control_mode == 'body_rate')  # 如果支持
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)


    def publish_trajectory_setpoint(
        self,
        position: list[float] = None,
        velocity: list[float] = None,
        acceleration: list[float] = None,
        jerk: list[float] = None,
        yaw: float = None,
        yawspeed: float = None
    ):
            """发布轨迹点（ENU 输入 → NED 发布），自适应位置、速度、加速度等控制"""
            """Publish trajectory setpoint (ENU input → NED published), adaptive to position, velocity, acceleration, etc."""
            msg = TrajectorySetpoint()
            nan3 = [np.nan] * 3  # 默认 NaN 数组，表示不控制

            # 处理位置
            if position is not None:
                x_enu, y_enu, z_enu = position
                x_ned, y_ned, z_ned = self.enu_to_ned(x_enu, y_enu, z_enu)
                msg.position = [float(x_ned), float(y_ned), float(z_ned)]
            else:
                msg.position = nan3

            # 处理速度
            if velocity is not None:
                vx_enu, vy_enu, vz_enu = velocity
                vx_ned, vy_ned, vz_ned = self.enu_to_ned(vx_enu, vy_enu, vz_enu)  # 速度转换类似位置（方向变换）
                msg.velocity = [float(vx_ned), float(vy_ned), float(vz_ned)]
            else:
                msg.velocity = nan3

            # 处理加速度
            if acceleration is not None:
                ax_enu, ay_enu, az_enu = acceleration
                ax_ned, ay_ned, az_ned = self.enu_to_ned(ax_enu, ay_enu, az_enu)
                msg.acceleration = [float(ax_ned), float(ay_ned), float(az_ned)]
            else:
                msg.acceleration = nan3

            # 处理 jerk（仅用于日志，不影响控制）
            if jerk is not None:
                jx_enu, jy_enu, jz_enu = jerk
                jx_ned, jy_ned, jz_ned = self.enu_to_ned(jx_enu, jy_enu, jz_enu)
                msg.jerk = [float(jx_ned), float(jy_ned), float(jz_ned)]
            else:
                msg.jerk = nan3

            # 处理 yaw
            if yaw is not None:
                msg.yaw = float(-yaw + math.radians(90))  # ENU yaw → NED yaw
            else:
                msg.yaw = np.nan

            # 处理 yawspeed
            if yawspeed is not None:
                msg.yawspeed = -float(yawspeed)  # yawspeed 不需要转换（角速度标量）
            else:
                msg.yawspeed = np.nan

            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_setpoint_publisher.publish(msg)

            # 调试日志：仅记录非 NaN 值
            log_str = "[PUB] TrajectorySetpoint NED: "
            if position is not None:
                log_str += f"pos=({msg.position[0]:.2f},{msg.position[1]:.2f},{msg.position[2]:.2f}), "
            if velocity is not None:
                log_str += f"vel=({msg.velocity[0]:.2f},{msg.velocity[1]:.2f},{msg.velocity[2]:.2f}), "
            if acceleration is not None:
                log_str += f"acc=({msg.acceleration[0]:.2f},{msg.acceleration[1]:.2f},{msg.acceleration[2]:.2f}), "
            if jerk is not None:
                log_str += f"jerk=({msg.jerk[0]:.2f},{msg.jerk[1]:.2f},{msg.jerk[2]:.2f}), "
            if yaw is not None:
                log_str += f"yaw={msg.yaw:.2f}, "
            if yawspeed is not None:
                log_str += f"yawspeed={msg.yawspeed:.2f}"
            self.get_logger().debug(log_str)

    def publish_attitude_setpoint(
        self,
        q_d: list[float],
        thrust_body: list[float],
        yaw_sp_move_rate: float = None
    ):
            """发布姿态设定点（基于四元数和推力）"""
            """Publish attitude setpoint (based on quaternion and thrust)"""
            msg = VehicleAttitudeSetpoint()
            msg.q_d = [float(q) for q in q_d]  # 期望四元数
            msg.thrust_body = [float(t) for t in thrust_body]  # 体坐标系下归一化推力 [-1,1]
            
            # 处理 yaw_sp_move_rate
            if yaw_sp_move_rate is not None:
                msg.yaw_sp_move_rate = float(yaw_sp_move_rate)
            else:
                msg.yaw_sp_move_rate = np.nan  # 默认不控制

            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.vehicle_attitude_setpoint_publisher.publish(msg)  

            log_str = f"[PUB] VehicleAttitudeSetpoint: q_d={msg.q_d}, thrust_body={msg.thrust_body}"
            if yaw_sp_move_rate is not None:
                log_str += f", yaw_sp_move_rate={msg.yaw_sp_move_rate:.2f}"
            self.get_logger().debug(log_str)

    def publish_vehicle_command(self, command, **params):
            msg = VehicleCommand()
            msg.command = command
            msg.param1 = params.get("param1", 0.0)
            msg.param2 = params.get("param2", 0.0)
            msg.param3 = params.get("param3", 0.0)
            msg.param4 = params.get("param4", 0.0)
            msg.param5 = params.get("param5", 0.0)
            msg.param6 = params.get("param6", 0.0)
            msg.param7 = params.get("param7", 0.0)
            # 假设 namespace="/px4_1"
            try:
                sys_id = int(namespace.strip('/px4_')) + 1 
            except:
                sys_id = 1
            msg.target_system = sys_id
            msg.target_component = 1
            msg.source_system = 1
            msg.source_component = 1
            msg.from_external = True
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            try:
                self.vehicle_command_publisher.publish(msg)
            except Exception as e:
                self.get_logger().error(f"publish_vehicle_command error: {e}")


    # ---------------- 高层飞行控制 | High-Level Motion Control ----------------
    def takeoff(self, takeoff_height=2.0, timeout=200.0) -> bool:
        """阻塞式起飞：从 Home 点垂直上升至指定高度"""
        '''使用 setpoint 来模拟起飞路径。'''
        """Blocking takeoff: ascend vertically from home to specified altitude"""
        if takeoff_height <= 0:
            self.get_logger().error("❌ Takeoff height must be positive!")
            return False

        with self.lock:
            home_z = self.home_position[2]
        target_alt = home_z + takeoff_height
        current_heading = self.vehicle_local_position_enu.heading if self.vehicle_local_position_received else 0.0
        self.update_position_setpoint(self.home_position[0], self.home_position[1], target_alt, current_heading)

        self.get_logger().info(f"🛫 Starting takeoff to {target_alt:.2f} m (from {home_z:.2f} m)")

        start = time.time()
        while rclpy.ok() and time.time() - start < timeout:
            with self.lock:
                current_z = self.vehicle_local_position_enu.z
            remaining = target_alt - current_z
            self.throttle_log(
                1.0,
                f"[TAKEOFF] Altitude: {current_z:.2f}/{target_alt:.2f} m, Δ={remaining:.2f} m",
                tag="takeoff"
            )
            if remaining <= 0.1:
                self.get_logger().info("✅ Takeoff complete!")
                return True
            time.sleep(0.1)

        self.get_logger().warning("⚠️ Takeoff timed out!")
        return False
    
    def simulated_land(self, descent_rate: float = -0.5, ground_tolerance: float = 0.1, timeout: float = 60.0) -> bool:
        """
        阻塞式模拟降落：使用 setpoint 逐步降低高度，实现平稳降落（需在 Offboard 模式下）
        Blocking simulated land: use setpoints to gradually decrease altitude for smooth landing (requires Offboard mode)
        
        参数 | Args:
        - descent_rate: 下降速度（m/s），负值表示向下 | Descent rate (m/s), negative for downward
        - ground_tolerance: 地面容忍高度（m） | Ground tolerance altitude (m)
        - timeout: 超时时间（秒） | Timeout (seconds)
        
        返回 | Returns:
        - bool: 是否成功降落 | True if landed successfully
        """
        if descent_rate >= 0:
            self.get_logger().error("❌ Descent rate must be negative for landing!")
            return False

        # 获取当前 ENU 位置和航向（假设在 position/velocity 模式）
        with self.lock:
            if not self.vehicle_local_position_received:
                self.get_logger().warning("⚠️ No valid position received; cannot land.")
                return False
            cx = self.vehicle_local_position_enu.x
            cy = self.vehicle_local_position_enu.y
            cz = self.vehicle_local_position_enu.z
            ch = self.vehicle_local_position_enu.heading

        # 切换到 velocity 模式以控制下降速度（更平稳）
        self.set_control_mode('velocity')
        self.get_logger().info(f"🛬 Starting simulated land from altitude {cz:.2f}m with descent rate {descent_rate:.2f} m/s")

        start = time.time()
        last_log = start
        target_z = 0.0  # 目标地面高度
        while rclpy.ok() and time.time() - start < timeout:
            with self.lock:
                current_z = self.vehicle_local_position_enu.z
                nav_state = self.vehicle_status.nav_state

            # 计算剩余距离并更新 velocity setpoint（水平速度 0，垂直 descent_rate，yawspeed 0）
            remaining_dist = current_z - target_z
            vz = max(descent_rate, -remaining_dist * 2.0)  # 接近地面时减速（简单线性减速）
            self.update_velocity_setpoint(0.0, 0.0, vz, 0.0)  # vx=0, vy=0, vz=下降, yawspeed=0

            # 节流日志
            if time.time() - last_log >= 1.0:
                self.throttle_log(
                    1.0,
                    f"[SIM_LAND] Altitude: {current_z:.2f}m, vz={vz:.2f} m/s, remaining: {remaining_dist:.2f}m",
                    tag="sim_land"
                )
                last_log = time.time()

            # 检查是否到达地面
            if current_z <= ground_tolerance:
                # 停止下降，发送零速度
                self.update_velocity_setpoint(0.0, 0.0, 0.0, 0.0)
                self.get_logger().info("✅ Simulated landing complete! Altitude near ground.")
                return True

            time.sleep(0.05)  # 高频循环以确保平稳

        self.get_logger().warning("⚠️ Simulated land timed out!")
        # 超时后恢复悬停
        self.update_velocity_setpoint(0.0, 0.0, 0.0, 0.0)
        return False


    def fly_to_trajectory_setpoint_dwa(self, x, y, z, yaw, timeout=600.0) -> bool:
        """阻塞式 DWA 避障飞行到目标点（ENU 坐标）。"""
        self.get_logger().info(
            f"🧭 [DWA] Fly to target with avoidance: ({x:.2f}, {y:.2f}, {z:.2f}), yaw={math.degrees(yaw):.1f}°"
        )
        
        target_enu = [float(x), float(y), float(z)]

        start = time.time()
        while rclpy.ok() and time.time() - start < timeout:
            with self.lock:
                if not self.vehicle_local_position_received:
                    pos_ok = False
                else:
                    pos_ok = True
                    cx = self.vehicle_local_position_enu.x
                    cy = self.vehicle_local_position_enu.y
                    cz = self.vehicle_local_position_enu.z
                    ch = self.vehicle_local_position_enu.heading

            if not pos_ok:
                time.sleep(0.1)
                continue

            dist = math.sqrt((cx - x) ** 2 + (cy - y) ** 2 + (cz - z) ** 2)
            yaw_diff = self.normalize_yaw(ch - yaw)
            self.throttle_log(
                1.0,
                f"[DWA] Remaining distance: {dist:.2f} m, yaw diff: {yaw_diff:.2f} rad",
                tag="flyto_dwa"
            )
            if dist < DISTANCE_TOLERANCE and yaw_diff < YAW_TOLERANCE:
                self.update_position_setpoint(x, y, z, yaw)
                self.get_logger().info(f"✅ [DWA] Target{self.target} reached!")
                return True
            safe_dist = 1

            obstacle_ahead = self.is_obstacle_ahead(safe_dist, front_angle=80)
            if (not self.dwa_active) and obstacle_ahead:
                self.dwa_active = True
                self.get_logger().info("🚧 [DWA] Entering local avoidance mode")
            elif self.dwa_active and (not self.is_obstacle_ahead(safe_dist * 1.05, front_angle=60)):
                self.dwa_active = False
                self.get_logger().info("✅ [DWA] Exiting local avoidance mode")

            if not self.dwa_active:
                self.update_position_setpoint(x, y, z, yaw)
            else:
                dwa_start = self.get_clock().now()  # 使用 ROS2 时间戳（更精确）
                vx_flu, vy_flu, w_flu = self.compute_dwa(target_enu)
                dwa_duration = (self.get_clock().now() - dwa_start).nanoseconds / 1e9  # 转换为秒
                # 使用 ROS2 日志系统（带时间戳和节点名）
                self.get_logger().info(
                    f"[DWA] Computed in {dwa_duration:.4f}s | "
                    f"Output: vx_flu={vx_flu:.2f}, vy_flu={vy_flu:.2f}, w_flu={w_flu:.2f} (FLU)"
                )
                w_frd = -w_flu
                body_vel_frd = np.array([vx_flu, -vy_flu, 0.0], dtype=float)
                att = self.vehicle_attitude
                if att is None:
                    self.update_velocity_setpoint(0.0, 0.0, 0.0, 0.0)
                else:
                    dcm = self.quaternion_to_dcm(att.q)
                    ned_vel = dcm @ body_vel_frd
                    vx_enu, vy_enu, vz_enu = self.ned_to_enu(ned_vel[0], ned_vel[1], ned_vel[2])
                    yaw_rate_enu= -w_frd 
                    self.update_velocity_setpoint(vx_enu, vy_enu, vz_enu, yaw_rate_enu)

            time.sleep(0.1)

        self.get_logger().warning("⚠️ [DWA] fly_to_trajectory_setpoint_dwa timed out!")
        return False


    def fly_to_trajectory_setpoint(self, x, y, z, yaw, timeout=60.0) -> bool:
        """阻塞式飞往目标点（ENU 坐标）"""
        """Blocking flight to target point (in ENU coordinates)"""
        self.update_position_setpoint(x, y, z, yaw)
        self.get_logger().info(f"✈️ Flying to ENU target: ({x:.2f}, {y:.2f}, {z:.2f}), yaw={math.degrees(yaw):.1f}°")

        start = time.time()
        while rclpy.ok() and time.time() - start < timeout:
            with self.lock:
                cx = self.vehicle_local_position_enu.x
                cy = self.vehicle_local_position_enu.y
                cz = self.vehicle_local_position_enu.z
                ch = self.vehicle_local_position_enu.heading
            dist = math.sqrt((cx - x) ** 2 + (cy - y) ** 2 + (cz - z) ** 2)
            yaw_diff = self.normalize_yaw(ch - yaw)
            self.throttle_log(
                1.0,
                f"[FLYTO] Remaining distance: {dist:.2f} m, yaw diff: {yaw_diff:.2f} rad ({math.degrees(yaw_diff):.1f}°)",
                tag="flyto"
            )
            if dist < DISTANCE_TOLERANCE and yaw_diff < YAW_TOLERANCE:
                self.get_logger().info(f"✅ Target {self.target} reached!")
                return True
            time.sleep(0.1)

        self.get_logger().warning("⚠️ fly_to_trajectory_setpoint timed out!")
        return False


# ---------------- 高层封装类：Vehicle | High-Level Wrapper: Vehicle ----------------
class Vehicle:
    """
    高层封装类：隐藏 ROS2 生命周期管理，提供简洁的 Python 控制接口
    High-level wrapper: hides ROS2 lifecycle details, provides clean Python API
    """

    def __init__(self):
        print("🌍 Initializing ROS2...")
        if not rclpy.ok():  # 关键修复：检查是否已初始化
           rclpy.init()
        self.drone = OffboardControl()
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.drone)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()
        self.drone.get_logger().info("🌀 Vehicle node spinning in background thread")
        self.drone.heartbeat_thread_start()
        self.drone.engage_offboard_mode()

    def close(self):
        """关闭所有线程与 ROS2 资源"""
        """Shut down all threads and ROS2 resources"""
        self.drone.get_logger().info("🛑 Shutting down Vehicle...")
        self.drone.stop_heartbeat.set()
        if self.drone.heartbeat_thread.is_alive():
            self.drone.heartbeat_thread.join(timeout=3.0)
        rclpy.shutdown()
        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=3.0)
        self.drone.destroy_node()
        self.executor.shutdown()
        print("✅ Vehicle shutdown complete!")


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()