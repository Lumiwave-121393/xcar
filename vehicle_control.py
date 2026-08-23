"""
vehicle_control.py —— 车辆高层控制模块
=======================================

基于 serial_comm.py 的串口通信能力，提供车辆控制的高层接口：

  1. 舵机/电机控制（封装底层协议发送）
  2. 视觉结果到控制量的映射（车道偏移→舵机，障碍物距离→速度）
  3. PID 转向控制（预留）
  4. 紧急停车

使用方式:
    from vehicle_control import VehicleController

    ctrl = VehicleController(port='/dev/ttyUSB0', baudrate=115200)
    ctrl.open()
    ctrl.start_heartbeat()
    ctrl.send_control(1500, 300)
"""

import logging
from serial_comm import SerialComm, list_available_ports
from serial_comm import (
    FRAME_HEADER, CMD_CONTROL, CMD_HEARTBEAT, CMD_CONFIG
)

logger = logging.getLogger("VehicleControl")


# ====================================================================
# 控制常量
# ====================================================================

SERVO_MIN = 1200
SERVO_MID = 1500
SERVO_MAX = 1800

MOTOR_MAX_FORWARD  = 2000
MOTOR_MAX_BACKWARD = -2000
MOTOR_STOP         = 0


def clamp(value, low, high):
    """将 value 限制在 [low, high] 区间内"""
    return max(low, min(high, value))


# ====================================================================
# VehicleController —— 车辆控制类
# ====================================================================

class VehicleController:
    """
    车辆高层控制器

    内部持有一个 SerialComm 实例处理所有串口通信。
    """

    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, timeout=0.05):
        self._comm = SerialComm()
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout

        self._servo_value = SERVO_MID
        self._motor_value = MOTOR_STOP

    # ================================================================
    # 生命周期
    # ================================================================

    def open(self) -> bool:
        return self._comm.open(self._port, self._baudrate, self._timeout)

    def close(self):
        self._comm.stop_heartbeat()
        self._comm.close()

    def is_connected(self) -> bool:
        return self._comm.is_connected()

    def start_heartbeat(self):
        self._comm.start_heartbeat()

    def stop_heartbeat(self):
        self._comm.stop_heartbeat()

    # ================================================================
    # 核心控制接口
    # ================================================================

    def send_control(self, servo: int, motor: int):
        """
        发送舵机+电机控制指令

        servo: 舵机脉宽 [500, 2500] us
        motor: 电机速度 [-1000, 1000]
        """
        servo = clamp(servo, SERVO_MIN, SERVO_MAX)
        motor = clamp(motor, MOTOR_MAX_BACKWARD, MOTOR_MAX_FORWARD)
        self._servo_value = servo
        self._motor_value = motor
        self._comm.send_control(servo, motor)

    def emergency_stop(self):
        """紧急停车: 舵机回中 + 电机停转"""
        self.send_control(SERVO_MID, MOTOR_STOP)
        logger.info("紧急停车!")

    # ================================================================
    # 视觉结果 → 控制量 映射
    # ================================================================

    def steer_by_lane_offset(self, offset_pixels: float,
                              image_width: int,
                              max_steer_pwm: int = 500) -> int:
        """
        车道线偏移 → 舵机 PWM 值

        offset_pixels: 车道中心 vs 画面中心的水平偏移
                       正=偏右，负=偏左
        image_width:   图像宽度（像素）
        max_steer_pwm: 最大转向偏移量（默认 ±500us）
        """
        normalized = offset_pixels / (image_width / 2.0)
        normalized = clamp(normalized, -1.0, 1.0)
        delta = int(normalized * max_steer_pwm)
        servo = SERVO_MID + delta
        return clamp(servo, SERVO_MIN, SERVO_MAX)

    def speed_by_distance(self, distance: float,
                           min_dist: float = 0.3,
                           max_dist: float = 5.0,
                           min_speed: int = -200,
                           max_speed: int = 1000) -> int:
        """
        障碍物距离 → 电机速度

        距离 < min_dist → 倒车
        距离 > max_dist → 全速
        中间线性插值
        """
        if distance < 0:
            return int(max_speed * 0.3)
        if distance <= min_dist:
            return min_speed
        if distance >= max_dist:
            return max_speed
        ratio = (distance - min_dist) / (max_dist - min_dist)
        speed = min_speed + int(ratio * (max_speed - min_speed))
        return clamp(speed, MOTOR_MAX_BACKWARD, MOTOR_MAX_FORWARD)

    # ================================================================
    # PID 控制（预留）
    # ================================================================

    def steer_pid(self, error: float, dt: float = 0.0) -> int:
        """P 控制转向"""
        kp = 500
        delta = int(error * kp)
        servo = SERVO_MID + delta
        return clamp(servo, SERVO_MIN, SERVO_MAX)

    # ================================================================
    # 属性
    # ================================================================

    @property
    def servo(self) -> int:
        return self._servo_value

    @property
    def motor(self) -> int:
        return self._motor_value
