"""
obstacle_detection —— 目标检测与避障包
======================================

在语义分割循迹基础上，集成 YOLO 目标检测，实现：

  1. 行人检测 —— 检测移动行人，停车等待或绕行
  2. 车辆检测 —— 检测路边停靠车辆，微调车道绕行
  3. 路牌检测 —— 停车 → OCR 识别 → LLM 决策（直行/岔路）

主要类:
    ObstacleDetector  : YOLO 目标检测封装
    ObjectTracker     : IoU 多目标追踪器
    AvoidancePlanner  : 避障决策逻辑
    SignHandler       : 路牌检测→OCR→LLM 流水线
    NavigationController : 总控循环（循迹 + 检测 + 避障）

使用方式:
    from obstacle_detection import NavigationController

    nav = NavigationController()
    nav.run_camera_mode()       # 摄像头模式
    nav.run_shm_mode()          # 共享内存模式
"""

import logging

logger = logging.getLogger("obstacle_detection")

# ── 无硬件依赖的模块（始终可导入） ──

from . import config
from .ocr_client import OCRClient
from .llm_client import LLMClient
from .object_tracker import ObjectTracker
from .avoidance import AvoidancePlanner
from .sign_handler import SignHandler

# ── 硬件依赖的模块（仅在目标设备上可导入） ──

_import_errors = {}

try:
    from .detector import ObstacleDetector
except Exception as e:
    _import_errors["ObstacleDetector"] = str(e)
    ObstacleDetector = None

try:
    from .navigation import NavigationController
except Exception as e:
    _import_errors["NavigationController"] = str(e)
    NavigationController = None


def check_hardware():
    """检查硬件依赖是否满足，返回缺失项列表"""
    errors = dict(_import_errors)
    if ObstacleDetector is None:
        errors["ObstacleDetector"] = _import_errors.get(
            "ObstacleDetector", "需要 Rockchip NPU (RKNNLite)"
        )
    if NavigationController is None:
        errors["NavigationController"] = _import_errors.get(
            "NavigationController", "需要 pyserial 和车辆控制依赖"
        )
    return errors


def is_hardware_ready():
    """硬件依赖是否全部就绪"""
    return ObstacleDetector is not None and NavigationController is not None


__all__ = [
    "ObstacleDetector",
    "ObjectTracker",
    "AvoidancePlanner",
    "SignHandler",
    "OCRClient",
    "LLMClient",
    "NavigationController",
    "config",
    "check_hardware",
    "is_hardware_ready",
]
