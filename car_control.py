"""
car_control.py —— 兼容重导出模块
================================

本文件仅为兼容旧代码而保留，将所有功能委托给新模块：
  - serial_comm.py   → 串口通信底层
  - vehicle_control.py → 车辆控制高层

所有旧代码中的 import 无需修改，继续可用。
"""

# ---- 重导出核心类 ----
from vehicle_control import VehicleController as CarController

# ---- 重导出常量 ----
from vehicle_control import (
    SERVO_MIN, SERVO_MID, SERVO_MAX,
    MOTOR_MAX_FORWARD, MOTOR_MAX_BACKWARD, MOTOR_STOP,
)

# ---- 重导出工具函数 ----
from serial_comm import list_available_ports
