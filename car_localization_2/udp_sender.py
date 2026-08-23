"""
udp_sender.py —— AprilTag 定位数据 UDP 发送模块
================================================

从 AprilTag 检测循环中获取小车位姿，通过 UDP 发送给目标设备（如香橙派）。

数据格式（JSON，严格参照 tag_position_sender.py）：
  {
      "type": "robot_position",
      "pos": [x, y, z],          // 三维坐标，单位：米
      "euler": [pitch, yaw, roll] // 欧拉角（度），第二个值是偏航角
      "ts": 1234567890.123        // 发送端时间戳（秒）
  }

使用方式：
  from udp_sender import UdpPositionSender, get_sender

  sender = get_sender()
  sender.configure(target_ip="10.198.22.84", target_port=9005)
  sender.send(x_mm, y_mm, yaw_deg)
  sender.set_enabled(True / False)
"""

import socket
import json
import time
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ---- 默认配置（可在 config.py 中覆盖）----
DEFAULT_TARGET_IP = "10.198.22.84"
DEFAULT_TARGET_PORT = 9005


class UdpPositionSender:
    """UDP 定位数据发送器（单例模式）"""

    def __init__(self):
        self._sock: Optional[socket.socket] = None
        self._target_ip = DEFAULT_TARGET_IP
        self._target_port = DEFAULT_TARGET_PORT
        self._enabled = True
        self._send_count = 0
        self._last_sent: Optional[Tuple[float, float, float, float]] = None

    # ── 配置 ──────────────────────────────────────────────

    def configure(self, target_ip: str = None, target_port: int = None):
        """配置目标 IP 和端口"""
        if target_ip is not None:
            self._target_ip = target_ip
        if target_port is not None:
            self._target_port = target_port
        logger.info(f"UdpPositionSender 目标: {self._target_ip}:{self._target_port}")

    def set_enabled(self, enabled: bool):
        """开关发送功能"""
        self._enabled = enabled
        state = "启用" if enabled else "禁用"
        logger.info(f"UDP 发送已{state}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def target(self) -> str:
        return f"{self._target_ip}:{self._target_port}"

    @property
    def send_count(self) -> int:
        """已发送的数据包数量"""
        return self._send_count

    @property
    def last_sent(self) -> Optional[Tuple[float, float, float, float]]:
        """最近一次成功发送的位姿 (x_mm, y_mm, yaw_deg, z_m)"""
        return self._last_sent

    # ── 发送 ──────────────────────────────────────────────

    def send(self, x_mm: float, y_mm: float, yaw_deg: float = 0.0,
             z_m: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0):
        """
        发送小车位姿到目标设备。

        Args:
            x_mm: 世界坐标 X，毫米
            y_mm: 世界坐标 Y，毫米
            yaw_deg: 偏航角，度（默认 0）
            z_m: Z 坐标，米（默认 0，地面高度）
            pitch_deg: 俯仰角，度（默认 0）
            roll_deg: 翻滚角，度（默认 0）
        """
        if not self._enabled:
            return

        # 坐标系转换：tag 2D 坐标系 → 3D 世界坐标系
        #   tag 坐标系: 地面 2D (X, Y)，yaw 从 X 轴起算
        #   目标坐标系: 3D (X→右, Y→上=高度, Z→前)，yaw 在 XZ 平面
        # 映射关系:
        #   world_X = tag_X / 1000
        #   world_Y = 0         (地面高度)
        #   world_Z = tag_Y / 1000
        #   world_yaw = -tag_yaw (Z 翻转导致 yaw 方向也翻转)
        pos_x = y_mm / 1000.0
        pos_y = -x_mm / 1000.0              # 高度，默认 0
        pos_z = z_m     # tag_Y → 目标_Z
        remapped_yaw = yaw_deg   # Z 轴翻转，yaw 也翻转

        message = json.dumps({
            "type": "robot_position",
            "pos": [pos_x, pos_y, pos_z],
            "euler": [pitch_deg, remapped_yaw, roll_deg],
            "ts": time.time()  # 发送端时间戳（秒），用于接收端计算延迟和外推
        })

        try:
            if self._sock is None:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.sendto(message.encode('utf-8'),
                              (self._target_ip, self._target_port))
            self._last_sent = (x_mm, y_mm, yaw_deg, z_m)
            self._send_count += 1
        except Exception as e:
            logger.warning(f"UDP 发送失败: {e}")

    def close(self):
        """关闭 socket"""
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        logger.info("UDP socket 已关闭")


# ── 全局单例 ──────────────────────────────────────────────

_sender_instance: Optional[UdpPositionSender] = None


def get_sender() -> UdpPositionSender:
    """获取全局 UdpPositionSender 单例"""
    global _sender_instance
    if _sender_instance is None:
        _sender_instance = UdpPositionSender()
    return _sender_instance
