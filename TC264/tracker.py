"""
tracker.py —— SegmentTracker 语义分割循迹主类

整合模型推理、车道中心提取、岔道口处理、可视化的上层接口。
"""

import os
import sys
import logging

from .config import (
    LANE_CLASS_ID, TRACKING_HAND, SMOOTHING_FACTOR,
    BASE_SPEED, TURN_SPEED_REDUCTION, SPEED_PEAK_DECAY,
    STEER_KP, STEER_KP_LEFT, STEER_KP_RIGHT,
    STEER_KD, STEER_FF_GAIN, CURVATURE_SMOOTHING_FACTOR,
    STEER_DEAD_ZONE, STEER_OUTPUT_MAX,
    SERVO_MID, SERVO_MIN, SERVO_MAX,
    clamp,
)
from .model import infer_func, find_model_dir, resolve_model_path
from .lane import extract_lane_center
from .viz import draw_tracking_viz

logger = logging.getLogger("tracking.tracker")

# rknnPoolExecutor（仅目标设备可用）
# 先将 rknnpool.py 所在目录加入 Python 路径
_tracker_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_tracker_dir)
for _rp in [
    os.path.join(_project_root, "seg_python"),
    os.path.join(_project_root, "infer_wrap", "base"),
]:
    _rp = os.path.abspath(_rp)
    if os.path.isfile(os.path.join(_rp, "rknnpool.py")) and _rp not in sys.path:
        sys.path.insert(0, _rp)
        break
del _tracker_dir, _project_root, _rp

try:
    from rknnpool import rknnPoolExecutor
except ImportError:
    rknnPoolExecutor = None
    logger.warning("rknnPoolExecutor 不可用")


class SegmentTracker:
    """
    基于语义分割的视觉循迹跟踪器。

    整合模型推理、迷宫法边线提取、沿左边线固定偏移循迹、前瞻点控制、可视化。

    用法:
        tracker = SegmentTracker(model_dir="seg_python/model", TPEs=2)
        for frame in video_stream:
            offset, viz = tracker.process_frame(frame)
            servo = tracker.offset_to_servo(offset)

    属性:
        lane_class_id:     车道类别 ID（默认 1）
        hand:              循迹手性 "left"=沿左边线（默认）/"right"=沿右边线（预留）
        smoothing_factor:  帧间平滑系数 [0, 1]
        blend_alpha:       分割叠加透明度 [0, 1]
    """

    def __init__(self, model_dir=None, TPEs=1,
                 lane_class_id=LANE_CLASS_ID,
                 hand=TRACKING_HAND,
                 smoothing_factor=SMOOTHING_FACTOR,
                 blend_alpha=0.4):
        """
        初始化分割循迹器。

        参数:
            model_dir:       .rknn 模型目录（None 时自动搜索）
            TPEs:            推理线程数（建议 1~3）
            lane_class_id:   车道类别 ID
            hand:            循迹手性 "left"（沿左边线）/"right"（沿右边线，预留）
            smoothing_factor: 帧间平滑系数
            blend_alpha:     分割叠加透明度
        """
        self.lane_class_id = lane_class_id
        self.hand = hand
        self.smoothing_factor = smoothing_factor
        self.blend_alpha = blend_alpha

        # 加载模型
        if model_dir is None:
            model_dir = find_model_dir()
        model_path = resolve_model_path(model_dir)

        # 线程池
        if rknnPoolExecutor is None:
            raise RuntimeError("rknnPoolExecutor 不可用，请检查运行环境")
        self._pool = rknnPoolExecutor(
            rknnModel=model_path,
            TPEs=TPEs,
            func=infer_func,
        )
        self._pool_ready = False

        # 帧间状态
        self._prev_offset = 0.0
        self._prev_steer_error = 0.0  # PD 控制器 D 项用
        self._speed_peak_signal = 0.0  # 弯道速度峰值信号（上升快下降慢，防止弯道信号瞬小时提速）
        self._curvature = 0.0  # 目标线方向角 α（帧间平滑后，曲率前馈 + 预测减速用）

        logger.info("SegmentTracker 初始化完成: model=%s TPEs=%d lane_cls=%d hand=%s",
                     model_path, TPEs, lane_class_id, hand)

    # ----------------------------------------------------------------
    # 线程池预热
    # ----------------------------------------------------------------

    def _pool_warmup(self, img):
        """用首帧预热线程池，完成后排空队列避免残留旧帧"""
        for _ in range(self._pool.TPEs):
            self._pool.put(img)
        # 等待全部完成并排空队列，确保后续 get() 不会取到预热帧
        for _ in range(self._pool.TPEs):
            self._pool.get()
        self._pool_ready = True

    # ----------------------------------------------------------------
    # 主处理接口
    # ----------------------------------------------------------------

    def process_frame(self, frame_bgr, return_aux=False):
        """
        处理一帧图像，计算车道偏移。

        流程:
          输入帧 → 语义分割(RKNN) → seg_map
            → 车道掩码 → 迷宫法边线提取（左/右边界链）
            → 沿左边线法向偏移固定像素 → 目标线 → 前瞻点
            → 偏移 → 帧间平滑 → 可视化

        参数:
            frame_bgr:   BGR 图像 (H, W, 3)
            return_aux:  是否返回辅助信息（车道掩码等）

        返回:
            当 return_aux=False（默认）:
                offset: 归一化偏移 [-1.0, 1.0]
                viz:    可视化图（BGR）

            当 return_aux=True:
                offset: 归一化偏移 [-1.0, 1.0]
                viz:    可视化图（BGR）
                aux:    dict {
                    "lane_mask": 底部车道掩码,
                    "full_lane_mask": (H, W) 全帧车道掩码（避障重叠计算）,
                    "lane_center_x": 车道中心 x 坐标（锚点处）,
                    "fork_detected": 岔路/宽度突变标志（仅可视化）,
                    "seg_map": (H, W) 原始分割图,
                    "left_chain"/"right_chain": 左右边界链,
                    "midline"/"target"/"anchor": 中线/前瞻点/锚点,
                }
        """
        # 默认 fallback 值（推理失败时原样返回）
        offset = self._prev_offset
        viz = frame_bgr
        aux = {}

        if frame_bgr is None:
            return (offset, viz, aux) if return_aux else (offset, viz)

        # ---- 1. 推理 ----
        if not self._pool_ready:
            self._pool_warmup(frame_bgr)

        self._pool.put(frame_bgr)
        result, pool_ok = self._pool.get()

        if pool_ok and result is not None:
            if isinstance(result, tuple) and len(result) == 2:
                seg_map, infer_ok = result
                if not infer_ok:
                    seg_map = None
            else:
                seg_map = result

            if seg_map is not None:
                # ---- 2. 迷宫法循迹（沿左边线固定偏移 + 前瞻点） ----
                (_offset, lane_cx, lane_mask_bottom,
                 fork_detected, full_lane_mask, lane_info) = extract_lane_center(
                    seg_map, self.lane_class_id, hand=self.hand,
                )

                # ---- 3. 帧间平滑（丢线时保持上一帧偏移） ----
                if _offset is not None:
                    _offset = (self.smoothing_factor * self._prev_offset +
                               (1 - self.smoothing_factor) * _offset)
                    self._prev_offset = _offset
                    # 曲率前馈信号 α 帧间平滑（丢线时保持上一帧）
                    curv = lane_info.get("curvature")
                    if curv is not None:
                        self._curvature = (
                            CURVATURE_SMOOTHING_FACTOR * self._curvature +
                            (1 - CURVATURE_SMOOTHING_FACTOR) * curv)
                else:
                    _offset = self._prev_offset

                # ---- 4. 可视化 ----
                viz = draw_tracking_viz(
                    frame_bgr, seg_map, _offset, lane_cx,
                    fork_detected,
                    self.blend_alpha, self.lane_class_id,
                    chains=(lane_info["left_chain"],
                            lane_info["right_chain"]),
                    midline=lane_info["midline"],
                    target=lane_info["target"],
                    anchor=lane_info["anchor"],
                    curvature=self._curvature,
                    hand=self.hand,
                )

                offset = _offset
                aux = {
                    "lane_mask": lane_mask_bottom,       # 裁剪版，保持向后兼容
                    "full_lane_mask": full_lane_mask,    # 全帧版 (H,W)，用于避障重叠计算
                    "lane_center_x": lane_cx,
                    "fork_detected": fork_detected,
                    "seg_map": seg_map,
                    "left_chain": lane_info["left_chain"],
                    "right_chain": lane_info["right_chain"],
                    "midline": lane_info["midline"],
                    "target": lane_info["target"],
                    "anchor": lane_info["anchor"],
                    "curvature": self._curvature,
                }

        if return_aux:
            return offset, viz, aux
        return offset, viz

    # ----------------------------------------------------------------
    # 控制映射
    # ----------------------------------------------------------------

    def offset_to_servo(self, offset, curvature=None, dt=1.0):
        """
        归一化偏移 + 前方曲率 → 舵机 PWM 值（PD + 曲率前馈）

        控制链路:
          error = offset                              # [-1, 1]，正=偏右
          if |error| < dead_zone → error = 0          # 死区只作用于反馈项，消除直道抖动
          d_error = (offset - prev_offset) / dt       # 微分（误差变化率，用真实 offset）
          steer = Kp × error + Kd × d_error + Kff × α # PD 反馈 + 曲率前馈
          steer = clamp(steer, ±OUTPUT_MAX)           # 输出限幅
          servo = SERVO_MID + steer                   # 映射到 PWM
          servo = clamp(servo, SERVO_MIN, SERVO_MAX)  # 最终限幅

        curvature: 目标线方向角 α（弧度，正=前方路向右拐）。None 时用
                   最近一帧 process_frame 平滑后的值（正常循迹路径）。
                    避障 steer_override 等显式指令传 0.0 禁用前馈。
        dt: 距上一帧的时间（秒），不传时默认为 1（帧差），
            传实际 dt 可使 D 项不受帧率波动影响。
        """
        if curvature is None:
            curvature = self._curvature

        # 1. 死区只作用于反馈误差（前馈曲率项不受死区限制，
        #    保证入弯初期 e≈0 时 α 前馈仍能提前打角）
        error = 0.0 if abs(offset) < STEER_DEAD_ZONE else offset

        # 2. 选择 Kp（支持左右不对称补偿）
        if offset < 0 and STEER_KP_LEFT is not None:
            kp = STEER_KP_LEFT
        elif offset > 0 and STEER_KP_RIGHT is not None:
            kp = STEER_KP_RIGHT
        else:
            kp = STEER_KP

        # 3. 微分项（误差变化率，用真实 offset 差分，死区外恢复无阶跃）
        d_error = (offset - self._prev_steer_error) / max(dt, 0.001)

        # 4. PD + 曲率前馈 + 限幅
        steer = int(error * kp + d_error * STEER_KD + curvature * STEER_FF_GAIN)
        steer = clamp(steer, -STEER_OUTPUT_MAX, STEER_OUTPUT_MAX)

        self._prev_steer_error = offset

        # 5. 映射到舵机 PWM
        return clamp(SERVO_MID + steer, SERVO_MIN, SERVO_MAX)

    def compute_speed(self, offset, curvature=None, base_speed=BASE_SPEED):
        """
        弯道减速（预测性）：用峰值信号避免弯道信号瞬小时速度突然飙升。

        信号 = max(|offset|, |α|)：
          - |α| 是前方曲率，弯道一进前瞻窗即非零，先于 offset 出现
            → 入弯前提前减速（解决原"减速不明显/太晚"）
          - 直道上 α≈0、offset≈0，不影响直道速度

        算法：维护一个"上升快、下降慢"的峰值信号量 _speed_peak_signal
          - 上升：瞬时信号超过峰值时立即跟随
          - 下降：每帧按 SPEED_PEAK_DECAY 比例衰减，直至归零
        用峰值代替瞬时值计算速度，确保弯道上信号波动时速度稳定。
        """
        if curvature is None:
            curvature = self._curvature
        signal = max(abs(offset), abs(curvature))
        # 上升：新峰值更高 → 立即跟随
        if signal > self._speed_peak_signal:
            self._speed_peak_signal = signal
        else:
            # 下降：按比例衰减（同时设置一个 0.02 的死区，确保能完全归零）
            self._speed_peak_signal -= SPEED_PEAK_DECAY * (self._speed_peak_signal - 0.02)
            self._speed_peak_signal = max(self._speed_peak_signal, 0.0)

        reduction = int(self._speed_peak_signal * TURN_SPEED_REDUCTION * 2)
        return clamp(base_speed - reduction, 0, base_speed)

    # ----------------------------------------------------------------
    # 资源管理
    # ----------------------------------------------------------------

    def reset(self):
        """重置帧间状态（切赛道或重新开始时调用）"""
        self._prev_offset = 0.0
        self._prev_steer_error = 0.0
        self._speed_peak_signal = 0.0
        self._curvature = 0.0

    def release(self):
        """释放模型和线程池资源"""
        self._pool.release()
        logger.info("SegmentTracker 资源已释放")
