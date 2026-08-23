"""
full_pipeline.py —— 循迹 + 目标检测 + 避障 + 路牌 全流程集成管线
===============================================================

整合语义分割循迹、YOLO目标检测、多目标追踪、避障决策、路牌OCR+LLM、
车辆控制的完整自动驾驶流水线。

架构:
    摄像头 / 共享内存帧
        │
        ├──→ SegmentTracker (语义分割循迹)
        │       → lane_offset, lane_viz
        │
        ├──→ ObstacleDetector (YOLO, 后台线程非阻塞)
        │       → submit/poll 架构，永不阻塞主循环
        │
        ├──→ ObjectTracker (IoU 多目标追踪)
        │       → tracked_objects (含ID/速度/轨迹)
        │
        ├──→ SignHandler (路牌 OCR+LLM，异步后台线程)
        │       → sign_decision (直行/右岔)
        │
        ├──→ AvoidancePlanner (避障决策)
        │       → decision (停车/绕行/正常)
        │
        └──→ VehicleController (串口 → TC264 舵机+电机)

运行方式:
    python full_pipeline.py -c 0               # 摄像头模式
    python full_pipeline.py --shm              # 共享内存模式
    python full_pipeline.py -c 0 --no-control  # 仅显示模式

按键:
    ESC    退出
    r      重置循迹状态
    d      切换检测显示
"""

import os
import sys
import time
import struct
import logging
import argparse

import cv2
import numpy as np

# ── 添加项目根目录到路径 ──
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ── 有条件导入（避免在 PC 上因缺少 RKNN 库而崩溃） ──
try:
    from tracking import SegmentTracker
    HAS_TRACKING = True
except ImportError as e:
    SegmentTracker = None
    HAS_TRACKING = False
    _track_err = str(e)

try:
    from tracking.config import TRACKING_HAND  # 右岔路手性切换的恢复目标
    from tracking.viz import draw_control_hud   # 舵机/PID HUD（2026-08-16）
except ImportError:
    TRACKING_HAND = "left"
    draw_control_hud = None

try:
    from obstacle_detection import ObstacleDetector, ObjectTracker
    from obstacle_detection import AvoidancePlanner, SignHandler
    from obstacle_detection import OCRClient, LLMClient
    from obstacle_detection import is_hardware_ready
    HAS_DETECTION = True
except ImportError as e:
    ObstacleDetector = ObjectTracker = None
    AvoidancePlanner = SignHandler = None
    OCRClient = LLMClient = None
    HAS_DETECTION = False
    _detect_err = str(e)

try:
    from obstacle_detection.config import (
        CLASS_NAMES, CLASS_SIGN, CLASS_STOP, CLASS_HUMAN,
        SIGN_CLASSES, BOX_COLORS,
        DEFAULT_SPEED, STOP_SPEED,
        DRAW_DETECT_BOXES, DRAW_TRACK_IDS, DRAW_LANE_OVERLAP,
        DETECT_CONFIDENCE,
        SIGN_OCR_BBOX_AREA, SIGN_STOP_BBOX_AREA, SIGN_MIN_TRACK_HITS,
        SIGN_MAX_BBOX_AREA, PEDESTRIAN_MAX_BBOX_AREA,
        SIGN_COOLDOWN_SECONDS,
        FORK_RIGHT_HAND_SECONDS,
        SIGN_TOGGLE_MODE,
    )
    HAS_DETECT_CONFIG = True
except ImportError:
    CLASS_NAMES = ['car', 'gold', 'human', 'light', 'sign', 'stone', 'stop', 'zbx']
    CLASS_SIGN = 4
    CLASS_STOP = 6
    SIGN_CLASSES = {4, 6}
    SIGN_OCR_BBOX_AREA = 0.03
    SIGN_STOP_BBOX_AREA = 0.08
    SIGN_MIN_TRACK_HITS = 8
    SIGN_TOGGLE_MODE = False
    BOX_COLORS = {
        "pedestrian": (0, 0, 255),
        "vehicle":    (255, 0, 0),
        "sign":       (0, 255, 255),
        "unknown":    (128, 128, 128),
    }
    DEFAULT_SPEED = 1
    STOP_SPEED = -5
    SIGN_OCR_BBOX_AREA = 0.03
    SIGN_STOP_BBOX_AREA = 0.08
    SIGN_MIN_TRACK_HITS = 8
    SIGN_COOLDOWN_SECONDS = 15.0
    HAS_DETECT_CONFIG = False

try:
    from vehicle_control import (
        VehicleController, SERVO_MID, SERVO_MIN, SERVO_MAX, MOTOR_STOP,
    )
    from serial_comm import list_available_ports
    HAS_VEHICLE = True
except ImportError:
    HAS_VEHICLE = False

# ── 共享内存参数 ──
SHM_NAME = "shm_ar_video"
SHM_HEADER_SIZE = 16

logger = logging.getLogger("full_pipeline")


# ====================================================================
# 可视化辅助
# ====================================================================

def _get_color(cls_name):
    """根据类别名取 BGR 绘制颜色"""
    return BOX_COLORS.get(cls_name, BOX_COLORS["unknown"])


def _get_cls_name(cls_id):
    """根据类别 ID 取名称"""
    if HAS_DETECT_CONFIG and cls_id < len(CLASS_NAMES):
        return CLASS_NAMES[int(cls_id)]
    return {0: "car", 1: "gold", 2: "human", 3: "light",
            4: "sign", 5: "stone", 6: "stop", 7: "zbx"}.get(int(cls_id), f"cls{cls_id}")


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

    # 路牌: 上限 20%
    if cls_id in SIGN_CLASSES and area_ratio > SIGN_MAX_BBOX_AREA:
        return True
    # 行人: 上限 8%
    if cls_id == CLASS_HUMAN and area_ratio > PEDESTRIAN_MAX_BBOX_AREA:
        return True
    return False


def draw_detection_results(viz, tracked_objects, decision=None):
    """
    在可视化图上叠加检测框（2026-08-16：仅画框，不画文字标签；
    决策文字 Action/debug 由 draw_status_panel 统一绘制）。

    tracked_objects: ObjectTracker.get_active() 的返回值
    decision: AvoidancePlanner.plan() 的返回值（保留参数兼容）
    """
    if not tracked_objects:
        return

    for obj in tracked_objects:
        # 只绘制当前帧有新鲜检测匹配的目标，跳过追踪器保留的过期轨迹（消除残影）
        if obj.get("age", 0) > 0:
            continue
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        cls_name = obj["class"]
        is_moving = obj.get("is_moving", False)

        # 颜色
        if cls_name in ("human", "pedestrian", "person"):
            color = BOX_COLORS["pedestrian"]
        elif cls_name in ("car", "vehicle", "truck", "bus"):
            color = BOX_COLORS["vehicle"]
        elif cls_name in ("sign", "stop", "zbx"):
            color = BOX_COLORS["sign"]
        else:
            color = BOX_COLORS["unknown"]

        # 移动中的行人加粗
        thickness = 3 if (cls_name == "human" and is_moving) else 2
        cv2.rectangle(viz, (x1, y1), (x2, y2), color, thickness)


def draw_status_panel(viz, fps, frame_count, sign_status="", decision=None):
    """
    左下文字面板（2026-08-16 重排）：
      FPS/Frame、Sign、Action、debug 按行堆叠，位于舵机/PID HUD 下方，
      不与循迹层文字（Offset/方向/手性/Curv/Look，左上 y=35~85）和
      舵机/PID HUD（y=110~154，draw_control_hud 绘制）重叠。
    """
    y = 178
    lines = []

    lines.append((f"FPS: {fps:.1f}  Frame: {frame_count}",
                  (180, 180, 180), 0.5, 1))
    if sign_status:
        lines.append((f"Sign: {sign_status}", (0, 255, 255), 0.5, 1))

    if decision:
        action = decision.get("action", "normal")
        action_color = {
            "normal": (0, 255, 0),
            "stop": (0, 0, 255),
            "bypass": (0, 165, 255),
            "pedestrian_bypass": (255, 165, 0),
            "sign_detected": (0, 255, 255),
            "sign_approach": (255, 255, 0),
            "vehicle_approach": (0, 215, 255),
            "blind_box": (255, 105, 180),
        }.get(action, (255, 255, 255))
        lines.append((f"Action: {action.upper()}", action_color, 0.55, 2))
        debug = decision.get("debug", "")
        if debug:
            lines.append((debug, (180, 180, 180), 0.4, 1))

    for text, color, scale, thick in lines:
        cv2.putText(viz, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
        y += 22


# ====================================================================
# 共享内存辅助函数
# ====================================================================

def _remove_shm_tracker():
    """从资源跟踪器中注销共享内存"""
    try:
        from multiprocessing import resource_tracker
        resource_tracker.unregister('/' + SHM_NAME, 'shared_memory')
    except Exception:
        pass


def connect_shm():
    """连接到共享内存，返回 shm 对象或 None"""
    try:
        from multiprocessing import shared_memory
        shm = shared_memory.SharedMemory(name=SHM_NAME)
        _remove_shm_tracker()
        return shm
    except FileNotFoundError:
        return None


def read_shm_frame(shm):
    """
    从共享内存读取一帧。

    返回: (frame_bgr, fid) 或 None
    """
    try:
        header = bytes(shm.buf[:SHM_HEADER_SIZE])
        fid, w, h = struct.unpack('QII', header)

        size = w * h * 3
        view = np.ndarray(
            (h, w, 3), dtype=np.uint8,
            buffer=shm.buf[SHM_HEADER_SIZE:SHM_HEADER_SIZE + size],
        )
        frame = view.copy()
        del view

        frame = cv2.flip(frame, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame, fid
    except (ValueError, struct.error, BufferError, IndexError):
        return None


# ====================================================================
# 车辆控制辅助
# ====================================================================

def init_controller(serial_port=None):
    """初始化串口车辆控制器"""
    if not HAS_VEHICLE:
        logger.info("车辆控制模块不可用，仅显示模式")
        return None

    if serial_port:
        ports_to_try = [serial_port]
    else:
        available = list_available_ports()
        ports_to_try = [p["device"] for p in available]
        if not ports_to_try:
            logger.info("未发现串口，仅显示模式")
            return None

    for port in ports_to_try:
        try:
            ctrl = VehicleController(port=port)
            if ctrl.open():
                ctrl.start_heartbeat()
                logger.info(f"车辆控制器已连接: {port}")
                return ctrl
        except Exception as e:
            logger.debug(f"串口 {port} 连接失败: {e}")

    logger.info("无法连接串口，仅显示模式")
    return None


# ====================================================================
# FullPipeline —— 全流程集成管线
# ====================================================================

class FullPipeline:
    """
    全流程集成管线 —— 循迹 + 检测 + 追踪 + 避障 + 路牌 + 控制。

    用法:
        pipeline = FullPipeline()
        pipeline.run_camera(camera_id=0)
        pipeline.run_shm()
    """

    def __init__(self, serial_port=None,
                 tracking_tpes=1, detection_tpes=1):
        """
        初始化全流程管线。

        参数:
            serial_port:     串口设备路径（None=自动发现）
            tracking_tpes:   循迹推理线程数
            detection_tpes:  检测 NPU 核心数（1~3）
        """
        # ── 硬件依赖检查 ──
        if not HAS_TRACKING:
            raise RuntimeError(
                f"SegmentTracker 不可用: {_track_err}\n"
                f"请确认 RK3588 环境已正确安装驱动。"
            )
        if not HAS_DETECTION:
            raise RuntimeError(
                f"ObstacleDetector 不可用: {_detect_err}\n"
                f"请确认 RK3588 环境已正确安装驱动。"
            )

        # ════════════════════════════════════════════════════════════
        # 1. 语义分割循迹
        # ════════════════════════════════════════════════════════════
        logger.info("正在初始化 SegmentTracker（语义分割循迹）...")
        self.tracker = SegmentTracker(TPEs=tracking_tpes)

        # ════════════════════════════════════════════════════════════
        # 2. YOLO 目标检测（后台线程架构）
        # ════════════════════════════════════════════════════════════
        logger.info("正在初始化 ObstacleDetector（YOLO 目标检测）...")
        self.detector = ObstacleDetector(TPEs=detection_tpes)

        # ════════════════════════════════════════════════════════════
        # 3. 多目标追踪器
        # ════════════════════════════════════════════════════════════
        logger.info("正在初始化 ObjectTracker（IoU 多目标追踪）...")
        self.object_tracker = ObjectTracker()

        # ════════════════════════════════════════════════════════════
        # 4. 避障决策规划器
        # ════════════════════════════════════════════════════════════
        logger.info("正在初始化 AvoidancePlanner（避障决策）...")
        self.avoidance = AvoidancePlanner()

        # ════════════════════════════════════════════════════════════
        # 5. 路牌处理流水线（OCR + LLM）
        # ════════════════════════════════════════════════════════════
        logger.info("正在初始化 SignHandler（路牌 OCR+LLM）...")
        self.sign_handler = SignHandler()

        # ════════════════════════════════════════════════════════════
        # 6. 车辆控制
        # ════════════════════════════════════════════════════════════
        self.controller = init_controller(serial_port)

        # ════════════════════════════════════════════════════════════
        # 7. 状态变量
        # ════════════════════════════════════════════════════════════
        self.frame_count = 0
        self._last_frame_t = None   # 上一帧处理时刻（秒，D 项真实 dt 用）
        self.detection_enabled = True
        self.fps = 0.0
        self._fps_t = time.time()
        self._fps_n = 0

        # 缓存最近一次检测结果用于可视化
        self._last_detect_result = None

        # 缓存最近一次实际发出的舵机 PWM，用于停车时冻结转向（不回中）
        self._last_servo_pwm = SERVO_MID
        # 缓存最近一次 PID 分解（offset_to_servo 详情），停车/冻结帧 HUD 显示用
        self._last_pid_details = None

        # ── 缓存上一次障碍物状态，避免重复打印 ──
        self._last_obstacle_report = {
            "type": None,        # "pedestrian" | "vehicle" | None
            "action": "normal",
            "id": None,
            "timestamp": 0.0,
        }

        # 路牌防重复
        self._last_sign_report_ms = 0
        self._sign_cooldown_until = 0.0  # 路牌冷却时间戳
        self._last_cooldown_print_t = 0.0    # 冷却信息打印限流时间戳（每秒一次）
        self._last_sign_wait_report = None   # 路牌"等待靠近"打印去重键 (cls, hits_str, area_str)
        self._sign_stop_active = False    # True=强制停车等待LLM，控制层最高优先级

        # 路牌 Toggle 模式状态
        # 0=等待第一次路牌, 1=第一次完成等待第二次, 2=第二次已执行(忽略后续)
        self._sign_toggle_phase = 0
        self._first_sign_action = None    # 第一次 LLM 决策: "straight" 或 "fork_right"

        # 右岔路手性切换状态（2026-08-14：fork_right 决策 → 右手循迹 N 秒 → 恢复左手）
        self._fork_hand_pending = False   # 决策已消费，等待"恢复行驶帧"执行切换
        self._fork_right_hand_until = 0.0 # 右手循迹结束时刻（时间戳）

        logger.info("=" * 55)
        logger.info("FullPipeline 初始化完成！")
        logger.info("  循迹 TPEs: %d  |  检测 TPEs: %d", tracking_tpes, detection_tpes)
        logger.info("  车辆控制: %s", "已连接" if self.controller else "未连接（仅显示）")
        logger.info("  避障:     %s", "已启用")
        logger.info("  路牌:     %s", "已启用（OCR+LLM 异步流水线）")
        logger.info("=" * 55)

    # ----------------------------------------------------------------
    # 路牌决策 → 手性切换状态
    # ----------------------------------------------------------------

    def _apply_sign_decision(self, action):
        """
        路牌 LLM 决策 → 循迹手性切换（2026-08-14）。

        fork_right: 置 pending，等控制段在"恢复行驶帧"执行切换（见 process_frame 第 7 节）
        straight:   立即恢复左手（取消任何 pending）
        """
        if action == "fork_right":
            self._fork_hand_pending = True
            logger.info("🔀 路牌决策=右岔路: 待恢复行驶后切换右手循迹 %d s",
                        FORK_RIGHT_HAND_SECONDS)
        else:
            self._fork_hand_pending = False
            self._fork_right_hand_until = 0.0
            if self.tracker.hand != TRACKING_HAND:
                self.tracker.hand = TRACKING_HAND
                logger.info("路牌决策=直行，恢复左手循迹")

    # ----------------------------------------------------------------
    # 核心处理：每帧调用
    # ----------------------------------------------------------------

    def process_frame(self, frame_bgr):
        """
        处理一帧：循迹 + 检测 + 追踪 + 避障 + 路牌 + 控制。

        返回:
            viz:            融合可视化后的图像
            track_offset:   车道偏移量
            decision:       避障决策
            control:        (servo, motor) 控制量
        """
        h, w = frame_bgr.shape[:2]
        self.frame_count += 1

        # 真实帧间隔 dt（秒）：D 项按每秒差分，不受帧率波动影响（2026-08-16）
        now = time.time()
        dt = (now - self._last_frame_t) if self._last_frame_t is not None else (1.0 / 30.0)
        dt = min(max(dt, 0.005), 0.2)
        self._last_frame_t = now

        # ================================================================
        # 1. 语义分割循迹
        # ================================================================
        track_offset, lane_viz, track_aux = self.tracker.process_frame(frame_bgr, return_aux=True)

        # ================================================================
        # 2. 目标检测（非阻塞 submit/poll）
        # ================================================================
        if self.detection_enabled:
            self.detector.submit(frame_bgr)
            result = self.detector.poll()
            if result is not None and result.get("ok"):
                self._last_detect_result = result
            else:
                # 无新检测结果时清空缓存，防止旧框残影
                self._last_detect_result = None
        else:
            # 检测关闭时也清空缓存
            self._last_detect_result = None

        # ================================================================
        # 3. 多目标追踪
        # ================================================================
        tracked_objects = []
        if self._last_detect_result is not None and self._last_detect_result.get("ok"):
            boxes   = self._last_detect_result.get("boxes")
            classes = self._last_detect_result.get("classes")
            scores  = self._last_detect_result.get("scores")
            if boxes is not None and len(boxes) > 0:
                detections = _boxes_to_detections(boxes, classes, scores)
                tracked_objects = self.object_tracker.update(detections)
            else:
                tracked_objects = self.object_tracker.update([])
        else:
            tracked_objects = self.object_tracker.update([])

        # ── 过滤场地外的大面积误检（超过上限的直接丢弃）──
        if tracked_objects:
            tracked_objects = [
                t for t in tracked_objects
                if not _exceeds_max_area(t, w, h)
            ]

        # ================================================================
        # 4. 路牌检测与处理
        # ================================================================
        sign_decision = None
        has_sign_result = False
        sign_status = ""

        if self.sign_handler.is_running():
            sign_status = "PROCESSING..."
        elif self.sign_handler.is_ready():
            sign_decision = self.sign_handler.get_decision()
            if sign_decision:
                has_sign_result = True
                action = sign_decision.get("action", "straight")
                action_cn = {"straight": "STRAIGHT", "fork_right": "FORK RIGHT"}
                sign_status = f"{action_cn.get(action, action)}"
                print(f"\n{'='*55}")
                print(f"  ✅ 路牌识别完成!")
                print(f"     OCR识别: {sign_decision.get('ocr_text', '').strip()[:80]}")
                print(f"     LLM决策: {action_cn.get(action, action)}")
                print(f"     置信度:  {sign_decision.get('confidence', 0):.2f}")
                print(f"     理由:    {sign_decision.get('reason', '')}")
                print(f"{'='*55}\n")
                # 进入冷却期，SIGN_COOLDOWN_SECONDS 秒内不再检测路牌
                self._sign_cooldown_until = time.time() + SIGN_COOLDOWN_SECONDS
                print(f"  ⏸️  进入路牌冷却期 ({SIGN_COOLDOWN_SECONDS:.0f}s)\n")
                # ── 手性切换：fork_right → 右手循迹 pending；straight → 恢复左手 ──
                self._apply_sign_decision(action)
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
        signs = [t for t in tracked_objects
                 if t.get("class_id") in SIGN_CLASSES]
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
                        "straight": "STRAIGHT",
                        "fork_right": "FORK RIGHT",
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
                    self._sign_toggle_phase = 2
                    sign_status = f"Toggle->{opposite_cn.get(opposite, opposite)}"
                    # ── 手性切换：注入相反决策同样驱动（fork_right → 右手循迹 pending）──
                    self._apply_sign_decision(opposite)
                    # ── 注入后进入冷却期（2026-08-14 修复）──
                    # has_sign_result 只持续本帧，若不入冷却期，下一帧路牌还在画面时
                    # sign_triggered=False → avoidance 把它当"新路牌"重新进入
                    # sign_approach 减速(600) → 注入后持续低速直到路牌离开画面。
                    # 进冷却期后 avoidance P5 忽略路牌 → 立即恢复正常循迹速度
                    self._sign_cooldown_until = time.time() + SIGN_COOLDOWN_SECONDS
                    print(f"\n{'='*55}")
                    print(f"  🔀 Toggle模式: 第二次路牌达到阈值!")
                    print(f"     第一次决策: {self._first_sign_action}")
                    print(f"     取反执行:   {opposite} ({opposite_cn.get(opposite, opposite)})")
                    print(f"     → 不停车 | 不调OCR+LLM | 注入后进入冷却期({SIGN_COOLDOWN_SECONDS:.0f}s)")
                    print(f"{'='*55}\n")
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
                elif not signs:
                    # 路牌消失 → 释放停车（可能是检测抖动，但如果持续消失也不应死锁）
                    pass  # 由下方的统一释放逻辑处理（LLM结果就绪时清除）

                # OCR 启动（更小阈值，先启动 OCR 再开近停车）
                if (not time.time() < self._sign_cooldown_until
                        and not self.sign_handler.is_running()
                        and not has_sign_result):
                    if bbox_area_ratio >= SIGN_OCR_BBOX_AREA and hits_ok:
                        # ✅ OCR 条件满足
                        print(f"\n{'='*55}")
                        print(f"  🚧 检测到路牌! 类别={best_sign['class']}  置信度={best_sign['score']:.2f}")
                        print(f"     位置: ({int(bbox[0])},{int(bbox[1])})→({int(bbox[2])},{int(bbox[3])})")
                        print(f"     面积: {bbox_area_ratio:.1%}  hits={hits}")
                        print(f"  ⏳ 启动 OCR+LLM 识别流水线...")
                        print(f"{'='*55}\n")
                        logger.info(f"检测到路牌 (置信度={best_sign['score']:.2f})，启动处理...")
                        self.sign_handler.start_processing(
                            frame_bgr, bbox, self.frame_count
                        )
                        sign_status = "OCR+LLM..."
                    else:
                        # ⏳ 条件未满足（数值变化时才打印，避免每帧刷屏）
                        area_str = f"{bbox_area_ratio:.1%}/{SIGN_OCR_BBOX_AREA:.0%}"
                        hits_str = f"{hits}" if hits_ok else f"{hits}/{SIGN_MIN_TRACK_HITS}"
                        report_key = (best_sign['class'], hits_str, area_str)
                        if report_key != self._last_sign_wait_report:
                            self._last_sign_wait_report = report_key
                            print(f"  🚧 路牌 {best_sign['class']} (hits={hits_str}, 面积={area_str})  等待靠近中...")
        elif time.time() < self._sign_cooldown_until:
            # Toggle phase 1 时不打印冷却信息（正在等第二次路牌）
            if not (SIGN_TOGGLE_MODE and self._sign_toggle_phase == 1):
                now = time.time()
                remaining = self._sign_cooldown_until - now
                # 每秒最多打印一次，避免刷屏
                if remaining > 0.5 and now - self._last_cooldown_print_t >= 1.0:
                    self._last_cooldown_print_t = now
                    print(f"  ⏳ 路牌冷却中 ({remaining:.1f}s)，跳过检测")
        else:
            # 无路牌且非冷却期 → 清除"等待靠近"打印缓存
            self._last_sign_wait_report = None

        # ── 释放停车：LLM 结果就绪 → 清除停车标志，让车继续行驶 ──
        if has_sign_result:
            if self._sign_stop_active:
                logger.info("🟢 路牌停车释放: LLM决策已返回，恢复行驶")
            self._sign_stop_active = False

        # ================================================================
        # 5. 避障决策
        # ================================================================
        # 优先使用全帧车道掩码（避障 bbox 使用全帧坐标，需对齐）
        if track_aux:
            lane_mask = track_aux.get("full_lane_mask",
                          track_aux.get("lane_mask", np.zeros((h, w), dtype=np.uint8)))
        else:
            lane_mask = np.zeros((h, w), dtype=np.uint8)
        decision = self.avoidance.plan(
            tracked_objects=tracked_objects,
            lane_mask=lane_mask,
            frame_shape=(h, w),
            # 行人 v2（2026-08-16）：车道边界感知用，见 avoidance._check_pedestrians
            lane_center_x=(track_aux.get("lane_center_x") if track_aux else None),
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
            # 2026-08-15：车辆躲避用目标线在车辆 bbox 底部行的插值作参照
            midline=(track_aux.get("midline") if track_aux else None),
            # 行人 v2：占道判定/绕行空间计算的左右边界链
            left_chain=(track_aux.get("left_chain") if track_aux else None),
            right_chain=(track_aux.get("right_chain") if track_aux else None),
            # 2026-08-17：车辆绕行偏置按（循迹手性×绕行方向）四象限取幅值
            tracking_hand=self.tracker.hand,
        )

        # ── 终端输出避障状态（仅在状态变化时打印）──
        self._report_obstacle_status(decision)

        # 路牌决策更新sign_status
        action = decision.get("action", "")
        if action == "sign_detected" and sign_decision:
            action_cn = {"straight": "STRAIGHT", "fork_right": "FORK RIGHT"}
            sign_status = action_cn.get(sign_decision.get("action", ""), sign_decision.get("action", ""))
        elif action == "sign_detected" and not sign_decision:
            sign_status = "WAITING..."
        elif not sign_status:
            sign_status = ""

        # ================================================================
        # 6. 融合可视化（检测框；文字由 draw_status_panel + draw_control_hud 绘制）
        # ================================================================
        viz = lane_viz.copy() if lane_viz is not None else frame_bgr.copy()
        draw_detection_results(viz, tracked_objects, decision)

        self._update_fps()
        draw_status_panel(viz, self.fps, self.frame_count, sign_status, decision)

        # ================================================================
        # 7. 车辆控制（控制量每帧都计算：串口发送 + HUD 显示；--no-control 也显示）
        # ================================================================
        pid_details = self._last_pid_details   # 停车/冻结帧显示缓存

        # ════════════════════════════════════════════════════════════════
        # 盲盒任务 —— 最高优先级（2026-08-21 用户方案）
        # 锁存后任何情况都执行盲盒动作：舵机向右打固定角（绕过 PD 直出，
        # 瞬时切换）+ 低速行驶；覆盖路牌强制停车/行人/车辆避让。
        # ════════════════════════════════════════════════════════════════
        servo_override_us = decision.get("servo_override")
        if servo_override_us is not None:
            servo = int(SERVO_MID + servo_override_us)
            servo = max(SERVO_MIN, min(SERVO_MAX, servo))
            motor = int(decision.get("speed_override") or 0)
            # 合成 HUD 用 pid_details：无 PD/前馈参与，P/D/FF 显示 0
            pid_details = {
                "servo": servo,
                "steer": int(servo_override_us),
                "error": 0.0,
                "error_dz": 0.0,
                "p": 0,
                "d": 0,
                "ff": 0,
                "kp": 0.0,
                "kd": 0.0,
                "curve_signal": 0.0,
            }
        # ════════════════════════════════════════════════════════════════
        # 路牌强制停车 —— 第二优先级（盲盒任务之下），覆盖其余所有决策
        # 当 _sign_stop_active=True 时，无视 avoidance planner 的任何输出
        # ════════════════════════════════════════════════════════════════
        elif self._sign_stop_active:
            motor = int(STOP_SPEED)
            servo = self._last_servo_pwm  # 冻结舵机
        else:
            steer_override = decision.get("steer_override")
            speed_override = decision.get("speed_override")

            # ── 电机速度 ──
            if speed_override is not None:
                motor = int(speed_override)
            else:
                # 2026-08-16：基础速度恒为 DEFAULT_SPEED（删除首次避让前慢速阶段）
                # 注意：必须用关键字 base_speed= 传参——compute_speed 的第二
                # 个位置参数是 curvature，误传 base 会把曲率当成 800 → signal
                # 恒大 → speed 恒 0（2026-08-14 修复: 小车不走+手推阻碍感根因）
                motor = int(self.tracker.compute_speed(track_offset,
                                                       base_speed=DEFAULT_SPEED))

            # ── 舵机转向 ──
            # 判断当前是否处于 "停车保持" 状态（速度被覆盖为 0 或更小）
            is_stopping = (speed_override is not None and int(speed_override) <= 0)

            # ════════════════════════════════════════════════════════════════
            # 右岔路手性切换（2026-08-14）：fork_right 决策 → 右手循迹
            # FORK_RIGHT_HAND_SECONDS → 恢复左手。
            # 计时从"恢复行驶帧"开始：停车解除（非停车帧）才执行切换；
            # 切换后每帧检查超时，到点切回左手。
            # 时序：冷却期 8s > 右手 5s → 第二次路牌触发时右手早已结束，无冲突。
            # ════════════════════════════════════════════════════════════════
            now = time.time()
            # ① 超时切回左手（优先于新切换，防止连续右岔路决策重置计时）
            if (self.tracker.hand != TRACKING_HAND
                    and now >= self._fork_right_hand_until):
                self.tracker.hand = TRACKING_HAND
                self._fork_right_hand_until = 0.0
                logger.info("🔀 右手循迹 %d s 结束，恢复左手循迹",
                            FORK_RIGHT_HAND_SECONDS)
            # ② 决策 pending + 非停车帧 → 执行切换并开始计时
            if self._fork_hand_pending and not is_stopping:
                self.tracker.hand = "right"
                self._fork_right_hand_until = now + FORK_RIGHT_HAND_SECONDS
                self._fork_hand_pending = False
                logger.info("🔀 已恢复行驶，切换右手循迹 %d s",
                            FORK_RIGHT_HAND_SECONDS)

            # 获取循迹偏置（行人 bypas 等场景会叠加在 track_offset 上）
            track_offset_bias = decision.get("track_offset_bias", 0.0)

            if steer_override is not None:
                # 避障显式指定的转向（例如 vehicle bypass 绕行）：
                # 显式指令不叠加曲率前馈
                pid_details = self.tracker.offset_to_servo(
                    steer_override, curvature=0.0, dt=dt, return_details=True)
                servo = pid_details["servo"]
            elif is_stopping:
                # 停车且无显式转向 → 冻结舵机，保持停车前的角度
                servo = self._last_servo_pwm
                pid_details = self._last_pid_details
            else:
                # 正常循迹（含右岔路：右手循迹已由上方手性切换实现，
                # 目标线自然摆入支路，无需固定偏置）/ 行人 bypass 偏置循迹
                pid_details = self.tracker.offset_to_servo(
                    track_offset + track_offset_bias, dt=dt, return_details=True)
                servo = pid_details["servo"]

            # 仅在未停车时更新缓存（避免将冻结值覆盖为 SERVO_MID）
            if not is_stopping:
                self._last_servo_pwm = servo
                self._last_pid_details = pid_details

        if self.controller is not None:
            self.controller.send_control(servo, motor)
        control = (servo, motor)

        # ── 舵机打角 + PID HUD（visual_tracking 同款，2026-08-16）──
        if draw_control_hud is not None:
            draw_control_hud(viz, servo, pid_details, speed=motor,
                             origin=(15, 110))

        return viz, track_offset, decision, control

    # ----------------------------------------------------------------
    # 终端避障状态报告（仅在状态变化时打印）
    # ----------------------------------------------------------------

    def _report_obstacle_status(self, decision):
        """
        在终端输出避障状态，仅当状态变化时打印一次，避免刷屏。
        """
        action = decision.get("action", "normal")
        obs_type = decision.get("obstacle_type")
        obs_info = decision.get("obstacle_info")
        obs_id = obs_info.get("id") if obs_info else None

        # 根据 action 确定显示类型（pedestrian/vehicle 都视为 obstacle）
        if action in ("stop", "bypass", "pedestrian_bypass", "vehicle_approach", "blind_box") and obs_type:
            new_type = obs_type
        else:
            new_type = None

        last = self._last_obstacle_report

        # 判断是否状态变化：类型或 ID 变了
        changed = (
            new_type != last["type"]
            or (new_type is not None and obs_id != last["id"])
        )

        if not changed and new_type is None:
            return  # 无事发生，静默
        if not changed and action == last["action"]:
            return  # 相同状态，不重复打印

        # ── 更新缓存 ──
        self._last_obstacle_report = {
            "type": new_type,
            "action": action,
            "id": obs_id,
            "timestamp": time.time(),
        }

        # ── 打印状态 ──
        now = time.strftime("%H:%M:%S")

        if new_type is None and last["type"] is not None:
            # 从障碍物状态恢复
            emoji = {"pedestrian": "🚶", "vehicle": "🚗"}.get(last["type"], "🚧")
            print(f"\n{now}  ✅ {emoji} {last['type']}已离开车道，恢复正常循迹\n")
            return

        if obs_type == "pedestrian":
            debug = decision.get("debug", "")
            if obs_info:
                center_x = obs_info.get("center", (0, 0))[0]
                side = "左" if center_x < 320 else "右"
                moving = obs_info.get("is_moving", False)
            else:
                # 去抖保持帧（本帧无新鲜检测，obstacle_info=None）：无 bbox 可显示
                side = None
                moving = False

            if action == "pedestrian_bypass":
                action_text = "↩️ 偏置循迹通过中"
            elif action == "stop":
                action_text = "🛑 停车等待中"
            else:
                action_text = "⚠️ 绕行"

            print(f"\n{'─'*55}")
            print(f"  {now}  🚶 检测到行人!")
            print(f"     位置: 画面{side}侧" if side else "     位置: 未知（去抖保持中）")
            print(f"     状态: {'🚶 行走中' if moving else '🧍 静止'}")
            print(f"     动作: {action_text}")
            print(f"     详情: {debug}")
            print(f"{'─'*55}\n")

        elif obs_type == "vehicle":
            debug = decision.get("debug", "")
            bypass_dir = decision.get("bypass_direction", "?")
            dir_cn = {"left": "左", "right": "右", None: "?"}

            print(f"\n{'─'*55}")
            print(f"  {now}  🚗 检测到路边车辆!")
            print(f"     动作: {'绕行' if action == 'bypass' else '减速接近' if action == 'vehicle_approach' else '停车'}")
            print(f"     方向: 向{dir_cn.get(bypass_dir, '?')}侧绕行")
            print(f"     速度: 减速通过")
            print(f"     详情: {debug}")
            print(f"{'─'*55}\n")

        elif obs_type == "stop":
            if action == "blind_box" and decision.get("servo_override") is not None:
                su = int(decision.get("servo_override"))
                sp = decision.get("speed_override")
                print(f"\n{'─'*55}")
                print(f"  {now}  🎁 盲盒任务启动!")
                print(f"     动作: 右转固定角 +{su}us（舵机输出 {1500 + su}us，绕过PD直出）")
                print(f"     速度: {int(sp) if sp is not None else '?'}")
                print(f"     提示: 锁死保持，按 r 重置可解除")
                print(f"{'─'*55}\n")
            # "stop" 模式永久停车：logger 已打印，这里保持静默（与旧行为一致）

    def _update_fps(self):
        self._fps_n += 1
        if time.time() - self._fps_t >= 1.0:
            self.fps = self._fps_n / (time.time() - self._fps_t)
            self._fps_n = 0
            self._fps_t = time.time()

    # ----------------------------------------------------------------
    # 运行模式 1：直连摄像头
    # ----------------------------------------------------------------

    def run_camera(self, camera_id=0, width=640, height=480):
        """从摄像头读取视频流，运行全流程管线。"""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not cap.isOpened():
            logger.error(f"无法打开摄像头 {camera_id}")
            return

        print(f"\n{'='*55}")
        print(f"  🚗 FullPipeline 摄像头模式启动")
        print(f"     device={camera_id}  {width}x{height}")
        print(f"  按键: ESC=退出  r=重置  d=切换检测")
        print(f"  功能: 循迹 + 检测 + 避障 + 路牌OCR+LLM")
        print(f"{'='*55}\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("读取摄像头帧失败")
                    break

                viz, offset, decision, control = self.process_frame(frame)
                cv2.imshow("Full Pipeline - Track + Detect + Avoid + Sign", viz)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:      # ESC
                    break
                elif key == ord('r'):
                    self.reset()
                    print("  🔄 循迹状态已重置")
                elif key == ord('d'):
                    self.detection_enabled = not self.detection_enabled
                    status = "开启" if self.detection_enabled else "关闭"
                    logger.info(f"目标检测: {status}")

        except KeyboardInterrupt:
            logger.info("用户中断")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.shutdown()

    # ----------------------------------------------------------------
    # 运行模式 2：共享内存（与 setup_webui 配合）
    # ----------------------------------------------------------------

    def run_shm(self):
        """从共享内存读取服务端发布的视频流。"""
        print(f"\n{'='*55}")
        print(f"  🚗 FullPipeline 共享内存模式启动")
        print(f"     shm={SHM_NAME}")
        print(f"     等待 setup_webui 写入共享内存...")
        print(f"  功能: 循迹 + 检测 + 避障 + 路牌OCR+LLM")
        print(f"{'='*55}\n")

        last_fid = 0

        while True:
            shm = None
            try:
                while True:
                    shm = connect_shm()
                    if shm is not None:
                        logger.info("已连接共享内存")
                        break
                    time.sleep(1.0)

                while True:
                    result = read_shm_frame(shm)
                    if result is None:
                        raise FileNotFoundError

                    frame, fid = result

                    if fid == last_fid:
                        time.sleep(0.002)
                        if cv2.waitKey(1) == 27:
                            raise KeyboardInterrupt
                        continue

                    last_fid = fid

                    viz, offset, decision, control = self.process_frame(frame)
                    cv2.imshow("Full Pipeline - SHM", viz)

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        raise KeyboardInterrupt
                    elif key == ord('r'):
                        self.reset()
                        logger.info("循迹状态已重置")
                    elif key == ord('d'):
                        self.detection_enabled = not self.detection_enabled
                        logger.info(f"目标检测: {'开启' if self.detection_enabled else '关闭'}")

            except KeyboardInterrupt:
                logger.info("用户退出")
                break
            except FileNotFoundError:
                logger.info("共享内存连接断开，等待重连...")
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
    # 状态管理
    # ----------------------------------------------------------------

    def reset(self):
        """重置所有组件状态"""
        self.tracker.reset()
        self.tracker.hand = TRACKING_HAND  # 恢复默认循迹手性
        self.object_tracker.reset()
        self.avoidance.reset()
        self.sign_handler.reset()
        self._last_detect_result = None
        # 重置 Toggle 模式状态
        self._sign_toggle_phase = 0
        self._first_sign_action = None
        self._sign_cooldown_until = 0.0
        self._sign_stop_active = False
        # 重置右岔路手性切换状态
        self._fork_hand_pending = False
        self._fork_right_hand_until = 0.0
        self._last_servo_pwm = SERVO_MID
        self._last_pid_details = None
        logger.info("FullPipeline 已重置")

    def shutdown(self):
        """安全释放所有资源"""
        logger.info("正在释放资源...")
        try:
            self.tracker.release()
        except Exception:
            pass
        try:
            self.detector.release()
        except Exception:
            pass
        try:
            self.sign_handler.cleanup()
        except Exception:
            pass
        if self.controller:
            try:
                self.controller.emergency_stop()
                self.controller.close()
            except Exception:
                pass
        logger.info("FullPipeline 已安全退出")


# ====================================================================
# 辅助函数
# ====================================================================

def _boxes_to_detections(boxes, classes, scores):
    """
    将检测结果（numpy数组）转换为 ObjectTracker 所需的 dict 列表。
    """
    detections = []
    for box, cls, score in zip(boxes, classes, scores):
        cls_id = int(cls)
        cls_name = _get_cls_name(cls_id)
        x1, y1, x2, y2 = [float(v) for v in box]
        detections.append({
            "class": cls_name,
            "class_id": cls_id,
            "score": float(score),
            "bbox": [x1, y1, x2, y2],
            "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        })
    return detections


# ====================================================================
# 命令行入口
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FullPipeline —— 循迹 + 目标检测 + 避障 + 路牌 全流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python full_pipeline.py -c 0                  # 摄像头模式
  python full_pipeline.py -c 0 --no-control     # 仅显示模式
  python full_pipeline.py --shm                  # 共享内存模式
  python full_pipeline.py -c 0 --detect-tpes 2  # 使用 2 个 NPU 核心
        """,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('-c', '--camera', type=int, default=None, nargs='?', const=0,
                      help='摄像头设备号（默认 0）')
    mode.add_argument('--shm', action='store_true',
                      help='共享内存模式（与 setup_webui 配合）')

    parser.add_argument('--track-tpes', type=int, default=1,
                        help='循迹推理线程数（默认 1）')
    parser.add_argument('--detect-tpes', type=int, default=1,
                        help='检测 NPU 核心数（默认 1）')
    parser.add_argument('--width', type=int, default=640,
                        help='图像宽度（默认 640）')
    parser.add_argument('--height', type=int, default=480,
                        help='图像高度（默认 480）')
    parser.add_argument('--port', type=str, default=None,
                        help='串口设备路径（如 /dev/ttyUSB0）')
    parser.add_argument('--no-control', action='store_true',
                        help='不连接车辆控制，仅显示')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试日志')

    args = parser.parse_args()

    # 日志级别
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] full_pipeline: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── 硬件就绪检查 ──
    if not HAS_TRACKING:
        logger.error("=" * 55)
        logger.error("SegmentTracker 不可用！")
        logger.error("请确认在 RK3588 环境上运行。")
        logger.error("错误: %s", _track_err)
        logger.error("=" * 55)
        return 1

    if not HAS_DETECTION:
        logger.error("=" * 55)
        logger.error("ObstacleDetector 不可用！")
        logger.error("请确认在 RK3588 环境上运行。")
        logger.error("错误: %s", _detect_err)
        logger.error("=" * 55)
        return 1

    # ── 创建管线 ──
    try:
        pipeline = FullPipeline(
            serial_port=None if args.no_control else args.port,
            tracking_tpes=args.track_tpes,
            detection_tpes=args.detect_tpes,
        )
    except Exception as e:
        logger.error(f"初始化 FullPipeline 失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ── 运行 ──
    try:
        if args.shm:
            pipeline.run_shm()
        else:
            cam_id = args.camera if args.camera is not None else 0
            pipeline.run_camera(camera_id=cam_id,
                                width=args.width, height=args.height)
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"运行异常: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
