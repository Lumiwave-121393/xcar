"""
navigation.py —— 导航总控模块
==============================

集成语义分割循迹 + YOLO 目标检测 + 避障决策的主控制循环。

架构:
    Camera / SHM
        │
        ├──→ SegmentTracker (语义分割循迹)
        │       → lane_offset, lane_mask, viz
        │
        ├──→ ObstacleDetector (YOLO 目标检测, 每 N 帧)
        │       → boxes, classes, scores
        │       → ObjectTracker (多目标追踪)
        │           → tracked_objects (with velocity)
        │
        ├──→ AvoidancePlanner (避障决策)
        │       → action, steer_override, speed_override
        │
        ├──→ SignHandler (路牌 OCR + LLM, 异步)
        │       → fork_decision
        │
        └──→ VehicleController (串口控制)
                → servo_pwm, motor_speed

运行方式:
    python -m obstacle_detection.navigation -c 0       # 摄像头模式
    python -m obstacle_detection.navigation --shm       # 共享内存模式
"""

import os
import sys
import time
import logging
import argparse

import cv2
import numpy as np

# 确保项目路径
_nav_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_nav_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tracking import SegmentTracker
from vehicle_control import VehicleController, SERVO_MID, MOTOR_STOP, SERVO_MIN, SERVO_MAX
from serial_comm import list_available_ports

from .detector import ObstacleDetector
from .object_tracker import ObjectTracker
from .avoidance import AvoidancePlanner
from .sign_handler import SignHandler
from .config import (
    DETECT_EVERY_N_FRAMES, DEFAULT_SPEED, STOP_SPEED,
    DRAW_DETECT_BOXES, DRAW_TRACK_IDS, DRAW_LANE_OVERLAP,
    CLASS_HUMAN, CLASS_CAR, CLASS_NAMES, SIGN_CLASSES,
    SIGN_OCR_BBOX_AREA, SIGN_STOP_BBOX_AREA, SIGN_MIN_TRACK_HITS,
    SIGN_MAX_BBOX_AREA, PEDESTRIAN_MAX_BBOX_AREA,
    SIGN_COOLDOWN_SECONDS, FORK_RIGHT_BIAS,
    SIGN_TOGGLE_MODE,
    clamp,
)
from .ocr_client import OCRClient
from .llm_client import LLMClient

logger = logging.getLogger("navigation")


def _exceeds_max_area(tracked_obj, frame_w, frame_h):
    """检查检测框是否超过对应类别的面积上限。

    超过上限判定为场地外干扰，返回 True。
    路牌上限: SIGN_MAX_BBOX_AREA，行人上限: PEDESTRIAN_MAX_BBOX_AREA。
    """
    cls_id = tracked_obj.get("class_id")
    bbox = tracked_obj["bbox"]
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    area_ratio = (bbox_w * bbox_h) / (frame_w * frame_h)

    if cls_id in SIGN_CLASSES and area_ratio > SIGN_MAX_BBOX_AREA:
        return True
    if cls_id == CLASS_HUMAN and area_ratio > PEDESTRIAN_MAX_BBOX_AREA:
        return True
    return False


# ====================================================================
# NavigationController —— 导航总控
# ====================================================================

class NavigationController:
    """
    集成循迹 + 目标检测 + 避障的导航总控制器。

    用法:
        nav = NavigationController()
        nav.run_camera_mode(camera_id=0)     # 摄像头模式
        nav.run_shm_mode()                   # 共享内存模式
    """

    def __init__(self, serial_port=None, detection_tpes=1, tracking_tpes=1):
        """
        初始化导航控制器。

        参数:
            serial_port:    串口设备路径（None 则尝试自动发现）
            detection_tpes: 检测模型推理线程数（建议 1）
            tracking_tpes:  循迹模型推理线程数（建议 1~2）
        """
        # ── 循迹 ──
        logger.info("正在初始化 SegmentTracker...")
        self.tracker = SegmentTracker(TPEs=tracking_tpes)

        # ── 检测 ──
        logger.info("正在初始化 ObstacleDetector...")
        self.detector = ObstacleDetector(TPEs=detection_tpes)

        # ── 追踪 ──
        self.object_tracker = ObjectTracker()

        # ── 避障规划 ──
        self.avoidance = AvoidancePlanner()

        # ── 路牌处理 ──
        self.sign_handler = SignHandler()

        # ── 车辆控制 ──
        self._serial_port = serial_port
        self.controller = None
        self._init_serial()

        # ── 状态 ──
        self._frame_count = 0
        self._detect_frame_count = 0
        self._last_sign_decision = None
        self._sign_decision_applied = False
        self._sign_cooldown_until = 0.0  # 路牌冷却时间戳
        self._sign_stop_active = False    # True=强制停车等待LLM，控制层最高优先级
        self._fork_override = None  # LLM 岔路决策对循迹的覆盖

        # 路牌 Toggle 模式状态
        # 0=等待第一次路牌, 1=第一次完成等待第二次, 2=第二次已执行(忽略后续)
        self._sign_toggle_phase = 0
        self._first_sign_action = None    # 第一次 LLM 决策: "straight" 或 "fork_right"

        logger.info("NavigationController 初始化完成")

    def _init_serial(self):
        """初始化串口连接"""
        if self._serial_port:
            ports_to_try = [self._serial_port]
        else:
            # 自动发现
            available = list_available_ports()
            ports_to_try = [p["device"] for p in available]
            if not ports_to_try:
                logger.warning("未发现可用串口，将仅模拟运行")
                return

        for port in ports_to_try:
            try:
                ctrl = VehicleController(port=port)
                if ctrl.open():
                    ctrl.start_heartbeat()
                    self.controller = ctrl
                    logger.info(f"串口已连接: {port}")
                    return
            except Exception as e:
                logger.debug(f"尝试 {port} 失败: {e}")

        if not self.controller:
            logger.warning("无法连接任何串口，将仅模拟运行")

    # ----------------------------------------------------------------
    # 主控制循环
    # ----------------------------------------------------------------

    def process_frame(self, frame_bgr):
        """
        处理一帧图像：循迹 + 检测 + 避障 + 控制。

        参数:
            frame_bgr: BGR 图像 (H, W, 3)

        返回:
            viz:          可视化图像
            track_offset: 车道偏移
            decision:     避障决策
            control:      (servo, motor) 实际发送的控制量
        """
        self._frame_count += 1
        h, w = frame_bgr.shape[:2]

        # ================================================================
        # 1. 语义分割循迹
        # ================================================================
        track_offset, track_viz, track_aux = self.tracker.process_frame(
            frame_bgr, return_aux=True
        )
        lane_mask = track_aux.get("full_lane_mask",
                    np.zeros((h, w), dtype=np.uint8)) if track_aux else np.zeros((h, w), dtype=np.uint8)
        lane_center_x = track_aux.get("lane_center_x", w / 2.0) if track_aux else w / 2.0

        # 用于可视化的基础图像
        viz = track_viz if track_viz is not None else frame_bgr.copy()

        # ================================================================
        # 2. 目标检测（每 N 帧运行一次）
        # ================================================================
        tracked_objects = []
        detect_result = None
        should_detect = (self._frame_count % DETECT_EVERY_N_FRAMES == 0)

        if should_detect:
            self._detect_frame_count += 1
            detect_result = self.detector.detect(frame_bgr)

            if detect_result["ok"] and detect_result["boxes"] is not None:
                # 转换检测结果为对象列表
                detections = self._boxes_to_detections(
                    detect_result["boxes"],
                    detect_result["classes"],
                    detect_result["scores"],
                )
                # 更新追踪器
                tracked_objects = self.object_tracker.update(detections)

                # 绘制检测框
                if DRAW_DETECT_BOXES:
                    self._draw_on_viz(viz, detect_result, tracked_objects, lane_mask)
            else:
                # 检测失败或无结果 → 通知追踪器（保持轨迹老化）
                tracked_objects = self.object_tracker.update([])

        # ── 过滤场地外的大面积误检（超过上限的直接丢弃）──
        if tracked_objects:
            tracked_objects = [
                t for t in tracked_objects
                if not _exceeds_max_area(t, w, h)
            ]

        # ================================================================
        # 3. 路牌处理
        # ================================================================
        sign_decision = None
        has_sign_result = False

        if self.sign_handler.is_running():
            # 路牌正在后台处理中 → 停车等待
            pass
        elif self.sign_handler.is_ready():
            # 路牌处理完成 → 获取决策
            sign_decision = self.sign_handler.get_decision()
            if sign_decision:
                has_sign_result = True
                self._last_sign_decision = sign_decision
                self._sign_decision_applied = False
                action = sign_decision['action']
                action_cn = {"straight": "直行", "fork_right": "右转岔路"}
                act_str = action_cn.get(action, action)
                ocr_txt = sign_decision.get('ocr_text', '').strip()[:80]
                print(f"\n{'='*60}")
                print(f"  ✅ 路牌识别完成!")
                print(f"     OCR识别文字: {ocr_txt if ocr_txt else '(未识别到文字)'}")
                print(f"     LLM决策: {act_str}")
                print(f"     置信度:  {sign_decision['confidence']:.2f}")
                print(f"     理由:    {sign_decision.get('reason', '')}")
                print(f"  🚗 车辆将{'停车等待' if action == 'sign_detected' else '按决策行驶'}")
                print(f"{'='*60}\n")
                logger.info(
                    f"路牌决策: action={sign_decision['action']} "
                    f"confidence={sign_decision['confidence']:.2f} "
                    f"reason={sign_decision.get('reason', '')}"
                )
                # 进入冷却期，SIGN_COOLDOWN_SECONDS 秒内不再检测路牌
                self._sign_cooldown_until = time.time() + SIGN_COOLDOWN_SECONDS
                print(f"  ⏸️  进入路牌冷却期 ({SIGN_COOLDOWN_SECONDS:.0f}s)\n")
                # ── Toggle 模式: 记录第一次决策，进入 phase 1（等待第二次路牌）──
                if SIGN_TOGGLE_MODE and self._sign_toggle_phase == 0:
                    self._first_sign_action = action
                    self._sign_toggle_phase = 1
                    logger.info(
                        "🔀 Toggle模式: 第一次路牌决策已记录 (%s)，等待第二次路牌...",
                        action,
                    )

        # ── 路牌状态：计算面积和命中（每帧都算，用于 OCR 和停车两个阈值） ──
        sign_stop_triggered_this_frame = False
        signs = [t for t in tracked_objects if t["class_id"] in SIGN_CLASSES]
        if signs:
            best_sign = max(signs, key=lambda s: s["score"])
            bbox = best_sign["bbox"]
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]
            bbox_area_ratio = (bbox_w * bbox_h) / (w * h)
            hits = best_sign.get("hits", 0)
            hits_ok = hits >= SIGN_MIN_TRACK_HITS

            # ════════════════════════════════════════════════════════
            # Toggle 模式: phase >= 2 → 第三次及之后的路牌，完全忽略
            # ════════════════════════════════════════════════════════
            if SIGN_TOGGLE_MODE and self._sign_toggle_phase >= 2:
                pass  # 不打印、不处理、不设置任何状态

            # ════════════════════════════════════════════════════════
            # Toggle 模式: phase == 1 → 第二次路牌，到达 OCR 阈值即注入相反决策
            # ════════════════════════════════════════════════════════
            elif SIGN_TOGGLE_MODE and self._sign_toggle_phase == 1:
                # 必须等第一次的冷却期结束，防止同一路牌重复触发
                if (not time.time() < self._sign_cooldown_until
                        and bbox_area_ratio >= SIGN_OCR_BBOX_AREA and hits_ok):
                    opposite = (
                        "fork_right" if self._first_sign_action == "straight"
                        else "straight"
                    )
                    opposite_cn = {
                        "straight": "直行(不偏置，正常循迹)",
                        "fork_right": "右岔路(立即偏右循迹)",
                    }
                    sign_decision = {
                        "action": opposite,
                        "confidence": 1.0,
                        "reason": (
                            f"Toggle模式: 第二次路牌, "
                            f"与第一次({self._first_sign_action})取反"
                        ),
                        "ocr_text": "(toggle模式, 跳过OCR)",
                    }
                    has_sign_result = True
                    self._last_sign_decision = sign_decision
                    self._sign_decision_applied = False
                    self._sign_toggle_phase = 2
                    print(f"\n{'='*60}")
                    print(f"  🔀 Toggle模式: 第二次路牌达到阈值!")
                    print(f"     第一次决策: {self._first_sign_action}")
                    print(f"     取反执行:   {opposite} ({opposite_cn.get(opposite, opposite)})")
                    print(f"     → 不停车 | 不调OCR+LLM | 不进入冷却期")
                    print(f"{'='*60}\n")
                    logger.info(
                        "🔀 Toggle模式: 第二次路牌触发, 第一次=%s → 取反=%s",
                        self._first_sign_action, opposite,
                    )

            # ════════════════════════════════════════════════════════
            # 正常模式 / Toggle phase 0: 原始逻辑
            # ════════════════════════════════════════════════════════
            else:
                # 停车阈值（只依赖面积，不依赖 hits——停车无成本，不需过滤）
                if bbox_area_ratio >= SIGN_STOP_BBOX_AREA:
                    sign_stop_triggered_this_frame = True

                # ── 停车标志：路牌面积达标且在冷却期外 → 激活，控制层强制停车 ──
                if sign_stop_triggered_this_frame and not (time.time() < self._sign_cooldown_until):
                    if not self._sign_stop_active:
                        logger.info("🛑 路牌停车: 等待LLM决策...")
                    self._sign_stop_active = True

                # OCR 启动（更小阈值，先启动 OCR 再开近停车）
                if (not time.time() < self._sign_cooldown_until
                        and not self.sign_handler.is_running()
                        and not has_sign_result):
                    if bbox_area_ratio >= SIGN_OCR_BBOX_AREA and hits_ok:
                        # ✅ OCR 条件满足
                        print(f"\n{'='*60}")
                        print(f"  🚧 检测到路牌! 置信度={best_sign['score']:.2f}")
                        print(f"     类别: {best_sign['class']}  ID={best_sign.get('id', '?')}")
                        print(f"     位置: ({int(bbox[0])},{int(bbox[1])})→({int(bbox[2])},{int(bbox[3])})")
                        print(f"     面积: {bbox_area_ratio:.1%}  hits={hits}")
                        print(f"  ⏳ 正在启动 OCR+LLM 识别流水线...")
                        print(f"{'='*60}\n")
                        logger.info(f"检测到路牌 (置信度={best_sign['score']:.2f})，启动处理...")
                        self.sign_handler.start_processing(
                            frame_bgr, best_sign["bbox"], self._frame_count
                        )
                    else:
                        # ⏳ 条件未满足
                        area_str = f"{bbox_area_ratio:.1%}/{SIGN_OCR_BBOX_AREA:.0%}"
                        hits_str = f"{hits}" if hits_ok else f"{hits}/{SIGN_MIN_TRACK_HITS}"
                        print(f"  🚧 路牌 {best_sign['class']} (hits={hits_str}, 面积={area_str})  等待靠近中...")
        elif time.time() < self._sign_cooldown_until:
            remaining = self._sign_cooldown_until - time.time()
            if remaining > 0.5:
                # Toggle phase 1 时不打印冷却信息（正在等第二次路牌）
                if not (SIGN_TOGGLE_MODE and self._sign_toggle_phase == 1):
                    print(f"  ⏳ 路牌冷却中 ({remaining:.1f}s)，跳过检测")

        # ── 释放停车：LLM 结果就绪 → 清除停车标志，让车继续行驶 ──
        if has_sign_result:
            if self._sign_stop_active:
                logger.info("🟢 路牌停车释放: LLM决策已返回，恢复行驶")
            self._sign_stop_active = False

        # ================================================================
        # 4. 避障决策
        # ================================================================
        decision = self.avoidance.plan(
            tracked_objects=tracked_objects,
            lane_mask=lane_mask,
            frame_shape=(h, w),
            lane_center_x=lane_center_x,
            track_offset=track_offset,
            has_sign_result=has_sign_result,
            sign_decision=sign_decision,
            sign_cooldown_until=self._sign_cooldown_until,
            sign_triggered=(
                self._sign_stop_active
                or sign_stop_triggered_this_frame
                or has_sign_result
                or self.sign_handler.is_running()
                or time.time() < self._sign_cooldown_until
            ),
        )

        # ================================================================
        # 5. 计算最终控制量
        # ================================================================
        servo, motor = self._compute_control(track_offset, decision, w)

        # 发送控制
        if self.controller:
            self.controller.send_control(servo, motor)

        # ================================================================
        # 6. 可视化叠加
        # ================================================================
        self._draw_status(viz, decision, track_offset, servo, motor, sign_decision)

        return viz, track_offset, decision, (servo, motor)

    # ----------------------------------------------------------------
    # 控制量计算
    # ----------------------------------------------------------------

    def _compute_control(self, track_offset, decision, img_width):
        """
        根据循迹偏移和避障决策计算最终舵机和电机值。

        优先级:
          0. 路牌停车标志活跃 → 强制停止（最高优先级）
          1. 避障决策有明确覆盖 → 使用避障值
          2. 路牌决策为岔路 → 覆盖循迹偏移方向
          3. 正常情况 → 循迹偏移映射到舵机（含 track_offset_bias 偏置）
        """
        # ════════════════════════════════════════════════════════════
        # 路牌强制停车 —— 最高优先级，覆盖所有其他决策
        # ════════════════════════════════════════════════════════════
        if self._sign_stop_active:
            servo = self.tracker.offset_to_servo(track_offset)
            motor = STOP_SPEED
            return servo, motor

        action = decision["action"]
        steer_override = decision.get("steer_override")
        speed_override = decision.get("speed_override")
        track_offset_bias = decision.get("track_offset_bias", 0.0)

        # ── 速度计算 ──
        if speed_override is not None:
            motor = speed_override
        else:
            # 2026-08-16：基础速度恒为 DEFAULT_SPEED（删除首次避让前慢速阶段）
            # 必须用关键字 base_speed=（第二个位置参数是 curvature，误传会把速度恒压成 0）
            motor = self.tracker.compute_speed(track_offset, base_speed=DEFAULT_SPEED)

        # ── 转向计算 ──
        if steer_override is not None:
            # 避障有明确转向覆盖：显式指令不叠加曲率前馈
            servo = self.tracker.offset_to_servo(steer_override, curvature=0.0)
        elif decision.get("fork_right"):
            # 右岔路持续偏右循迹（由 avoidance planner 持久化）
            servo = self.tracker.offset_to_servo(track_offset + FORK_RIGHT_BIAS)
        elif action == "sign_detected" and decision.get("sign_decision"):
            # 路牌决策完成（直行）→ 正常循迹
            servo = self.tracker.offset_to_servo(track_offset + track_offset_bias)
        elif action == "sign_detected" and not decision.get("sign_decision"):
            # 路牌检测但未完成处理 → 停车，保持当前转向
            servo = self.tracker.offset_to_servo(track_offset)
            motor = STOP_SPEED
        else:
            # 正常循迹 / 行人 bypass 偏置循迹
            servo = self.tracker.offset_to_servo(track_offset + track_offset_bias)

        return servo, motor

    # ----------------------------------------------------------------
    # 可视化
    # ----------------------------------------------------------------

    def _boxes_to_detections(self, boxes, classes, scores):
        """将检测结果转换为统一格式"""
        detections = []
        for box, cls, score in zip(boxes, classes, scores):
            cls_id = int(cls)
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
            x1, y1, x2, y2 = [float(v) for v in box]
            detections.append({
                "class": cls_name,
                "class_id": cls_id,
                "score": float(score),
                "bbox": [x1, y1, x2, y2],
                "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            })
        return detections

    def _draw_on_viz(self, viz, detect_result, tracked_objects, lane_mask):
        """在可视化图像上叠加检测结果"""
        from .config import BOX_COLORS

        # 绘制追踪对象
        for obj in tracked_objects:
            bbox = obj["bbox"]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cls_name = obj["class"]
            track_id = obj["id"]
            is_moving = obj.get("is_moving", False)

            # 颜色
            if cls_name == "human":
                color = BOX_COLORS["pedestrian"]
            elif cls_name == "car":
                color = BOX_COLORS["vehicle"]
            elif cls_name in ("sign", "stop"):
                color = BOX_COLORS["sign"]
            else:
                color = BOX_COLORS["unknown"]

            # 移动中的行人用虚线框
            thickness = 2
            if cls_name == "human" and is_moving:
                thickness = 3  # 加粗

            cv2.rectangle(viz, (x1, y1), (x2, y2), color, thickness)

            # 标签
            label_parts = [cls_name]
            if DRAW_TRACK_IDS:
                label_parts.insert(0, f"ID{track_id}")
            if cls_name == "human" and is_moving:
                label_parts.append("MOVING")
            label = " ".join(label_parts)

            cv2.putText(viz, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # 高亮与车道重叠的区域
            if DRAW_LANE_OVERLAP and lane_mask is not None and lane_mask.any():
                overlap = self._compute_overlap(bbox, lane_mask)
                if overlap > 0.1:
                    # 画重叠警告
                    cv2.putText(viz, f"BLOCKING({overlap:.0%})",
                                (x1, y2 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    def _compute_overlap(self, bbox, lane_mask):
        """计算检测框与车道重叠比例"""
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        h, w = lane_mask.shape
        x2 = min(x2, w)
        y2 = min(y2, h)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        crop = lane_mask[y1:y2, x1:x2]
        total = crop.size
        if total == 0:
            return 0.0
        return np.count_nonzero(crop) / total

    def _draw_status(self, viz, decision, track_offset, servo, motor, sign_decision):
        """在图像上绘制状态信息"""
        h = viz.shape[0]
        y = 20

        def _put(text, color=(0, 255, 0)):
            nonlocal y
            cv2.putText(viz, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y += 18

        # 基本状态
        action = decision["action"]
        action_color = {
            "normal": (0, 255, 0),
            "stop": (0, 0, 255),
            "bypass": (0, 165, 255),
            "pedestrian_bypass": (255, 165, 0),
            "sign_detected": (0, 255, 255),
            "sign_approach": (255, 255, 0),
            "vehicle_approach": (0, 215, 255),
        }.get(action, (255, 255, 255))

        _put(f"Action: {action.upper()}", action_color)
        _put(f"Track Offset: {track_offset:+.3f}  |  Servo: {servo}  Motor: {motor}")
        _put(f"Debug: {decision.get('debug', '')}", (180, 180, 180))

        # 路牌决策
        if sign_decision:
            sa = sign_decision.get("action", "?")
            sc = sign_decision.get("confidence", 0)
            sr = sign_decision.get("reason", "")
            _put(f"Sign: {sa} (confidence={sc:.2f})", (0, 255, 255))
            if sr:
                _put(f"  Reason: {sr}", (200, 200, 0))

        # 障碍物信息
        obs_type = decision.get("obstacle_type")
        if obs_type:
            obs_info = decision.get("obstacle_info", {})
            _put(f"Obstacle: {obs_type}  ID={obs_info.get('id','?')}", (0, 165, 255))

    # ----------------------------------------------------------------
    # 运行模式
    # ----------------------------------------------------------------

    def run_camera_mode(self, camera_id=0, width=640, height=480):
        """摄像头直连模式"""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not cap.isOpened():
            logger.error(f"无法打开摄像头 {camera_id}")
            return

        logger.info(f"摄像头模式启动: device={camera_id} {width}x{height}")
        logger.info("按键: ESC=退出  r=重置  s=切换检测开关")

        detection_enabled = True

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("读取帧失败")
                    break

                # 如果关闭检测，则只做循迹
                if not detection_enabled:
                    offset, viz = self.tracker.process_frame(frame)
                    if self.controller:
                        servo = self.tracker.offset_to_servo(offset)
                        motor = self.tracker.compute_speed(offset)
                        self.controller.send_control(servo, motor)
                    cv2.imshow("Navigation (detection OFF)", viz)
                else:
                    viz, offset, decision, (servo, motor) = self.process_frame(frame)
                    cv2.imshow("Navigation", viz)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('r'):
                    self.reset()
                    logger.info("已重置所有状态")
                elif key == ord('s'):
                    detection_enabled = not detection_enabled
                    logger.info(f"检测: {'开启' if detection_enabled else '关闭'}")

        except KeyboardInterrupt:
            logger.info("用户中断")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.shutdown()

    def run_shm_mode(self):
        """共享内存模式（与 setup_webui 配合）"""
        from multiprocessing import shared_memory, resource_tracker
        from tracking.config import SHM_NAME, SHM_HEADER_SIZE
        import struct

        def _remove_shm_tracker():
            try:
                resource_tracker.unregister('/' + SHM_NAME, 'shared_memory')
            except Exception:
                pass

        logger.info(f"共享内存模式启动: shm={SHM_NAME}")
        logger.info("等待 setup_webui 写入共享内存...")
        logger.info("按键: ESC=退出  r=重置")

        while True:
            shm = None
            try:
                try:
                    shm = shared_memory.SharedMemory(name=SHM_NAME)
                    _remove_shm_tracker()
                    logger.info("已连接共享内存")
                except FileNotFoundError:
                    time.sleep(1.0)
                    continue

                last_fid = 0
                fps_t = time.time()
                fps_n = 0
                cur_fps = 0.0

                while True:
                    try:
                        header = bytes(shm.buf[:SHM_HEADER_SIZE])
                        fid, w, h = struct.unpack('QII', header)

                        if fid == last_fid:
                            time.sleep(0.002)
                            if cv2.waitKey(1) == 27:
                                raise KeyboardInterrupt
                            continue

                        last_fid = fid
                        size = w * h * 3
                        view = np.ndarray(
                            (h, w, 3), dtype=np.uint8,
                            buffer=shm.buf[SHM_HEADER_SIZE:SHM_HEADER_SIZE + size],
                        )
                        frame = view.copy()
                        del view

                        frame = cv2.flip(frame, 0)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        # 主处理
                        viz, offset, decision, (servo, motor) = self.process_frame(frame)

                        # FPS
                        fps_n += 1
                        if time.time() - fps_t >= 1.0:
                            cur_fps = fps_n / (time.time() - fps_t)
                            fps_n, fps_t = 0, time.time()

                        cv2.putText(viz, f"FPS: {cur_fps:.1f}",
                                    (viz.shape[1] - 130, viz.shape[0] - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.imshow("Navigation (SHM)", viz)

                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:
                            raise KeyboardInterrupt
                        elif key == ord('r'):
                            self.reset()
                            logger.info("已重置所有状态")

                    except (ValueError, struct.error, BufferError):
                        raise FileNotFoundError

            except KeyboardInterrupt:
                logger.info("用户中断")
                break
            except FileNotFoundError:
                logger.info("连接丢失，等待重连...")
                if shm:
                    shm.close()
                cv2.destroyAllWindows()
                time.sleep(1.0)
            finally:
                if shm:
                    try:
                        shm.close()
                    except Exception:
                        pass

        cv2.destroyAllWindows()
        self.shutdown()

    # ----------------------------------------------------------------
    # 资源管理
    # ----------------------------------------------------------------

    def reset(self):
        """重置所有组件状态"""
        self.tracker.reset()
        self.object_tracker.reset()
        self.avoidance.reset()
        self.sign_handler.reset()
        self._frame_count = 0
        self._last_sign_decision = None
        self._sign_decision_applied = False
        # 重置 Toggle 模式状态
        self._sign_toggle_phase = 0
        self._first_sign_action = None
        self._sign_cooldown_until = 0.0
        self._sign_stop_active = False

    def shutdown(self):
        """安全关闭所有组件"""
        logger.info("正在关闭...")
        if self.controller:
            self.controller.emergency_stop()
            self.controller.close()
        self.detector.release()
        self.tracker.release()
        self.sign_handler.cleanup()
        logger.info("已退出")

    def emergency_stop(self):
        """紧急停车"""
        if self.controller:
            self.controller.emergency_stop()
        logger.warning("紧急停车!")


# ====================================================================
# 命令行入口
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Navigation Controller —— 循迹 + 目标检测 + 避障",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('-c', '--camera', type=int, default=None, nargs='?', const=0,
                      help='摄像头设备号 (默认: 0)')
    mode.add_argument('--shm', action='store_true',
                      help='共享内存客户端模式 (与 setup_webui 配合)')
    mode.add_argument('-i', '--image', type=str, default=None,
                      help='单张图片测试模式')

    parser.add_argument('--port', type=str, default=None,
                        help='串口设备路径 (如 /dev/ttyUSB0)')
    parser.add_argument('--width', type=int, default=640,
                        help='图像宽度 (默认: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='图像高度 (默认: 480)')
    parser.add_argument('--detect-tpes', type=int, default=1,
                        help='检测模型推理线程数 (默认: 1)')
    parser.add_argument('--track-tpes', type=int, default=1,
                        help='循迹模型推理线程数 (默认: 1)')
    parser.add_argument('--detect-every', type=int, default=DETECT_EVERY_N_FRAMES,
                        help=f'检测间隔帧数 (默认: {DETECT_EVERY_N_FRAMES})')

    args = parser.parse_args()

    # 更新检测间隔
    import obstacle_detection.config as _cfg
    _cfg.DETECT_EVERY_N_FRAMES = args.detect_every

    # 创建控制器
    nav = NavigationController(
        serial_port=args.port,
        detection_tpes=args.detect_tpes,
        tracking_tpes=args.track_tpes,
    )

    try:
        if args.shm:
            nav.run_shm_mode()
        elif args.image:
            # 单张图片测试
            img = cv2.imread(args.image)
            if img is None:
                print(f"[错误] 无法读取图片: {args.image}")
                return
            viz, offset, decision, control = nav.process_frame(img)
            print(f"Offset: {offset:+.3f}")
            print(f"Decision: {decision}")
            print(f"Control: {control}")
            cv2.imshow("Navigation Test", viz)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cam_id = args.camera if args.camera is not None else 0
            nav.run_camera_mode(camera_id=cam_id,
                                width=args.width, height=args.height)
    except KeyboardInterrupt:
        print("\n[退出] 用户中断")
    finally:
        nav.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
