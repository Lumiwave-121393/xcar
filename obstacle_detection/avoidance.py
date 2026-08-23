"""
avoidance.py —— 避障决策逻辑
=============================

根据目标追踪结果和车道信息，生成避障控制决策。

决策优先级（从高到低）：
  0. 盲盒任务锁存（STOP_ACTION="blind_box"）→ 右打固定角+低速，锁死覆盖一切
  1. 路牌检测 → 触发停车 + OCR + LLM 决策
  2. 行人（v2 车道边界感知） → 占道停车等待 / 车道外绕行空间大的一侧
  3. 车辆接近（距离门控） → 固定偏置绕行，连续 N 帧不可见后恢复正常循迹
  4. stop 牌（STOP_ACTION="stop"） → 延时永久停车
  5. 无阻挡 → 正常循迹

路牌三级接近：
  检测到 → 减速接近（sign_approach） → 触发停车（sign_detected / stop）

用法:
    from obstacle_detection import AvoidancePlanner

    planner = AvoidancePlanner()
    decision = planner.plan(tracked_objects, frame_shape=(h, w),
                            lane_center_x=lane_cx)
    # → {"action": "stop"|"bypass"|"normal", "steer_override": 0.0, ...}
"""

import logging
import time

import numpy as np

from .config import (
    CLASS_HUMAN, CLASS_CAR, CLASS_STOP,
    VEHICLE_BYPASS_STEER_LH_LEFT, VEHICLE_BYPASS_STEER_LH_RIGHT,
    VEHICLE_BYPASS_STEER_RH_LEFT, VEHICLE_BYPASS_STEER_RH_RIGHT,
    VEHICLE_SLOW_SPEED,
    VEHICLE_RETURN_MISS_FRAMES,
    VEHICLE_JUDGE_Y2_RATIO, VEHICLE_DIR_FLIP_FRAMES, VEHICLE_DIR_LOCK_Y2_RATIO,
    SIGN_CLASSES,
    STOP_SPEED,
    STOP_GATE_SECONDS, STOP_DELAY_SECONDS, STOP_MIN_HITS,
    STOP_ACTION, BLIND_BOX_STEER_US, BLIND_BOX_SPEED,
    PEDESTRIAN_PROXIMITY_Y2_RATIO, PEDESTRIAN_PROXIMITY_AREA_RATIO,
    VEHICLE_PROXIMITY_Y2_RATIO, VEHICLE_PROXIMITY_AREA_RATIO,
    PEDESTRIAN_BYPASS_OFFSET,
    PEDESTRIAN_BYPASS_SPEED,
    PEDESTRIAN_MOVE_DIR_THRESHOLD,
    PEDESTRIAN_IN_LANE_RATIO, PEDESTRIAN_IN_LANE_SAMPLE_DOWN_PX,
    PEDESTRIAN_RETURN_MISS_FRAMES, PEDESTRIAN_DIR_FLIP_FRAMES,
    SIGN_SLOW_MIN_HITS, SIGN_SLOW_MIN_AREA, SIGN_APPROACH_SPEED,
)

logger = logging.getLogger("obstacle_detection.avoidance")

# 程序启动时刻（2026-08-19 stop 门控：STOP_GATE_SECONDS 从此刻起算）
_MODULE_START_T = time.time()


def _vehicle_bypass_magnitude(tracking_hand, bypass_dir):
    """按（循迹手性 × 绕行方向）取车辆绕行偏置幅值（正数）。

    2026-08-17：板测发现某方向绕行效果差（如右手循迹右绕蹭墙），
    拆成四象限分别可配；符号由绕行方向决定（左绕为负、右绕为正）。
    """
    hand_key = "right" if tracking_hand == "right" else "left"
    if bypass_dir > 0:  # 右绕
        return (VEHICLE_BYPASS_STEER_RH_RIGHT if hand_key == "right"
                else VEHICLE_BYPASS_STEER_LH_RIGHT)
    # 左绕
    return (VEHICLE_BYPASS_STEER_RH_LEFT if hand_key == "right"
            else VEHICLE_BYPASS_STEER_LH_LEFT)


def _mid_x_at_y(midline, y):
    """插值目标线 midline（(N,2) 点列）在 y 行的 x。

    2026-08-15 车辆躲避方案：参照"车辆 bbox 底部行 y2 处的拟合中线"，
    替代原画面底部 lane_center_x（透视/弯道下底部中心 ≠ 车辆行带中心）。

    2026-08-16 修复：原实现只按首尾 y 翻转后直接 np.interp。真实边界链
    （尤其右链在岔口/汇入区）存在 <MAZE_WRAP_DESCENT_PX 的局部下行起伏
    （防包绕只裁 >20px 的下行），目标线 y 非单调 → np.interp 要求 xp
    严格递增，未排序时返回垃圾值 → mid_x 错误 → 方向判断反向
    （板测：右手循迹"车在左却向左躲避"）。现改为按 y 排序 + 去重后再插值，
    对任意链序/起伏都正确。

    2026-08-19：y 超出目标线覆盖范围**上方**（目标线没画到车所在行，
    如顶部截断后/远处车）→ 返回 None（按丢线处理，不外推端点值——
    外推值不可靠，方向判断可能反向）；**下方**近处车（y2 略大于目标线
    底部，车在锚点附近）保持端点外推（外推量小，行为不变）。

    返回: 该行目标线 x（float）或 None（midline 不可用/未覆盖，由调用方兜底）。
    """
    if midline is None or len(midline) < 2:
        return None
    try:
        ys = np.asarray(midline[:, 1], dtype=np.float64)
        xs = np.asarray(midline[:, 0], dtype=np.float64)
        order = np.argsort(ys)
        ys, xs = ys[order], xs[order]
        uniq, idx = np.unique(ys, return_index=True)
        if y < uniq[0]:
            # 目标线未覆盖到该行（车在目标线顶部之上）→ 不外推，按丢线
            return None
        return float(np.interp(y, uniq, xs[idx]))
    except Exception:
        return None


def _x_at_y(chain, y):
    """边界链 (N,2) 在 y 行的 x 插值（与 _mid_x_at_y 同款排序去重，链序/起伏无关）。

    行人 v2（2026-08-16）用：占道判定与绕行空间计算都在"y2 下移采样行"
    取左右边界 x。链缺失/过短返回 None，由调用方兜底。

    返回: float 或 None。
    """
    if chain is None:
        return None
    try:
        pts = np.asarray(chain, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 2:
            return None
        ys = pts[:, 1]
        xs = pts[:, 0]
        order = np.argsort(ys)
        ys, xs = ys[order], xs[order]
        uniq, idx = np.unique(ys, return_index=True)
        if len(uniq) < 2:
            return None
        return float(np.interp(y, uniq, xs[idx]))
    except Exception:
        return None


class AvoidancePlanner:
    """
    避障决策规划器。

    综合分析：
    - 目标追踪结果（行人/车辆/路牌的位置和运动状态）
    - 车道中心坐标（从语义分割循迹得来）
    - 当前状态（是否在等待、等待了多久）

    输出统一的避障决策。
    """

    def __init__(self):
        # 行人 v2 状态（2026-08-16 车道边界感知：占道停车 + 空间大侧绕行）
        self._waiting_for_pedestrian = False   # 占道停车等待中
        self._wait_start_time = 0.0            # 停车开始时间（仅日志）
        self._wait_pedestrian_id = None
        self._pedestrian_bypass_active = False # 绕行中
        self._pedestrian_bypass_dir = 0        # -1=左绕, 0=未定, 1=右绕
        self._pedestrian_missing_frames = 0    # 连续无新鲜行人检测帧数（退出去抖）
        self._pedestrian_flip_pending_frames = 0  # 方向迟滞：连续相反判决帧数
        self._pedestrian_out_lane_frames = 0   # 停车→绕行迟滞：连续 must_stop=False（可通行）帧数
        self._pedestrian_was_in_lane = False   # 曾占道标志（2026-08-19）：进过车道内→True

        # 路牌处理状态
        self._handling_sign = False
        self._sign_decision = None
        # 注：右岔路"持续偏右"状态已移除（2026-08-14）——右岔路执行改为
        # full_pipeline 切换右手循迹 FORK_RIGHT_HAND_SECONDS，不再依赖 avoidance
        self._sign_approach_active = False  # 路牌减速接近中（带迟滞，防止面积抖动）

        # stop 牌状态（2026-08-19 延时触发；2026-08-21 加盲盒模式）：
        # 首次识别时刻（None=未触发）、锁存（STOP_ACTION 决定锁存后动作）
        self._stop_triggered = None
        self._stop_latched = False

        # 车辆绕行状态（v5：连续判断 + 迟滞翻转 + 并排锁死，不再一次锁死）
        self._vehicle_bypass_active = False   # 是否正在绕行同一辆车（方向已确定过至少一次）
        self._vehicle_bypass_dir = 0          # -1=左绕, 0=未确定, 1=右绕
        self._vehicle_bypass_bias = 0.0       # 绕行期间使用的偏置（固定值）
        self._vehicle_missing_frames = 0      # 连续"无新鲜车辆检测"的帧数（退出去抖；幽灵轨迹不计）
        self._vehicle_flip_pending_frames = 0 # 迟滞：连续出现"相反判决"的帧数（≥N 才翻转）
        self._vehicle_dir_locked = False      # 并排锁死：y2≥VEHICLE_DIR_LOCK_Y2_RATIO 后锁死到离开
        # 注：车辆回中阶段已删除（2026-08-16 二次修改）——去抖达标后直接
        # 恢复正常循迹（偏置瞬时归零），不再有回中偏置/超时/死区状态。

        # 统计
        self._decisions = {"normal": 0, "stop": 0, "bypass": 0, "approach": 0, "sign": 0}

    # ----------------------------------------------------------------
    # 主接口
    # ----------------------------------------------------------------

    def plan(self, tracked_objects, lane_mask=None, frame_shape=None,
             lane_center_x=None, track_offset=0.0,
             has_sign_result=False, sign_decision=None,
             sign_cooldown_until=0.0, sign_triggered=False,
             midline=None, left_chain=None, right_chain=None,
             tracking_hand="left"):
        """
        生成避障决策。

        参数:
            tracked_objects:     list[dict], ObjectTracker.get_active() 的返回值
            lane_mask:           (H, W) 二值图（已弃用，保留兼容）
            frame_shape:         (H, W) 帧尺寸
            lane_center_x:       float, 车道中心 x 坐标（像素），来自循迹（丢线兜底用）
            track_offset:        float, 循迹归一化偏移 [-1,1]（保留参数：车辆回中阶段已移除，当前不再用于任何决策判断）
            midline:             (N,2) 目标线点列（车辆 bbox 底部行插值参照，None 时兜底）
             left_chain/right_chain: 左右车道边界链（行人 v2 占道判定/绕行空间计算用，
                                  None 时退化为掩码法/目标线兜底）
             tracking_hand:      str, 当前循迹手性 "left"|"right"（车辆绕行偏置
                                  按手性×方向四象限取幅值；缺省左手）
            has_sign_result:     bool, SignHandler 是否已返回结果
            sign_decision:       dict, 路牌 LLM 决策（如果有）
            sign_cooldown_until: float, 路牌冷却截止时间戳
            sign_triggered:      bool, pipeline 是否已确认触发路牌处理（面积/hits 达标）

        返回:
            dict: {
                "action": "normal" | "stop" | "bypass" | "vehicle_approach" | "sign_detected" | "sign_approach" | "blind_box",
                "steer_override": float | None,   # 覆盖循迹转向 [-1, 1]
                "speed_override": int | None,     # 覆盖速度
                "servo_override": int | None,     # 盲盒任务：固定舵机角（相对中位 µs 偏移，正=右），
                                                  # 仅 action="blind_box" 时非 None，绕过 PD 直出
                "obstacle_type": str | None,      # "pedestrian" | "vehicle" | "sign"
                "obstacle_info": dict | None,     # 障碍物详情
                "bypass_direction": str | None,   # "left" | "right"
                "sign_decision": dict | None,     # 路牌 LLM 决策
                "fork_right": bool,               # 已废弃（恒 False）：右岔路执行改为
                                                  # pipeline 切换循迹手性（FORK_RIGHT_HAND_SECONDS）
                "debug": str,                     # 调试信息
            }
        """
        if frame_shape is None:
            h = w = 480
        else:
            h, w = frame_shape

        if lane_center_x is None:
            lane_center_x = w / 2.0

        # ── 优先级 0: 盲盒任务锁存（2026-08-21 用户方案）──
        # STOP_ACTION="blind_box" 且延迟期已到 → 锁存并返回盲盒决策，
        # 覆盖行人/车辆/路牌（含路牌强制停车），保持到 r 重置。
        if STOP_ACTION == "blind_box":
            if self._stop_latched or (
                    self._stop_triggered is not None
                    and time.time() - self._stop_triggered >= STOP_DELAY_SECONDS):
                if not self._stop_latched:
                    self._stop_latched = True
                    logger.info("stop 延迟结束，执行盲盒任务：右转固定角+低速")
                return self._blind_box_decision()

        # ── 优先级 1: 路牌检测 ──
        sign_result = self._check_signs(tracked_objects, has_sign_result, sign_decision,
                                        sign_cooldown_until, sign_triggered,
                                        frame_shape=(h, w))
        if sign_result:
            return sign_result

        # ── 优先级 2: 行人检测（v2 车道边界感知） ──
        pedestrian_result = self._check_pedestrians(
            tracked_objects, lane_center_x, w, h,
            lane_mask=lane_mask, left_chain=left_chain, right_chain=right_chain,
            midline=midline)
        if pedestrian_result:
            return pedestrian_result

        # ── 优先级 3: 车辆检测 ──
        vehicle_result = self._check_vehicles(tracked_objects, lane_center_x,
                                              midline, w, h,
                                              tracking_hand=tracking_hand)
        if vehicle_result:
            return vehicle_result

        # ── 优先级 4: stop 牌（2026-08-19：避让之后、正常循迹之前） ──
        stop_result = self._check_stop(tracked_objects)
        if stop_result:
            return stop_result

        # ── 优先级 5: 正常行驶 ──
        self._reset_wait_state()
        self._decisions["normal"] += 1
        return {
            "action": "normal",
            "steer_override": None,
            "speed_override": None,
            "obstacle_type": None,
            "obstacle_info": None,
            "bypass_direction": None,
            "sign_decision": None,
            "fork_right": False,
            "debug": "NORMAL",
        }

    # ----------------------------------------------------------------
    # stop 牌（2026-08-19）
    # ----------------------------------------------------------------

    def _check_stop(self, tracked_objects):
        """stop 牌延时处理：程序启动 STOP_GATE_SECONDS 秒后启用；
        识别到 stop（新鲜检测 + hits ≥ STOP_MIN_HITS）→ 首次识别起继续
        行驶 STOP_DELAY_SECONDS 秒（期间 stop 消失不重置计时）→ 执行
        STOP_ACTION 指定动作：
          - "stop"      → 永久停车（速度 0、舵机保持当前角度，不再恢复）
          - "blind_box" → 盲盒任务：右打固定角（BLIND_BOX_STEER_US）+
                          低速（BLIND_BOX_SPEED），锁存后由 plan() 优先级 0 锁定

        返回 stop/blind_box 决策或 None（未触发/延迟期内正常行驶）。
        """
        if self._stop_latched:
            if STOP_ACTION == "blind_box":
                return self._blind_box_decision()
            self._decisions["stop"] += 1
            return {
                "action": "stop",
                "steer_override": None,
                "speed_override": 0,
                "obstacle_type": "stop",
                "obstacle_info": None,
                "bypass_direction": None,
                "sign_decision": None,
                "fork_right": False,
                "debug": "STOP SIGN LATCHED",
            }

        now = time.time()
        if now - _MODULE_START_T < STOP_GATE_SECONDS:
            return None   # 门控期内忽略 stop

        stops = [t for t in tracked_objects
                 if t["class_id"] == CLASS_STOP
                 and t.get("age", 0) == 0
                 and t.get("hits", 0) >= STOP_MIN_HITS]
        if stops:
            if self._stop_triggered is None:
                self._stop_triggered = now
                logger.info(
                    "识别到 stop（hits=%d），%.1f 秒后执行 %s",
                    max(t.get("hits", 0) for t in stops), STOP_DELAY_SECONDS,
                    "盲盒任务" if STOP_ACTION == "blind_box" else "停车")
        if self._stop_triggered is not None:
            if now - self._stop_triggered >= STOP_DELAY_SECONDS:
                self._stop_latched = True
                if STOP_ACTION == "blind_box":
                    logger.info("stop 延迟结束，执行盲盒任务：右转固定角+低速")
                    return self._blind_box_decision()
                logger.info("stop 延迟结束，永久停车")
                self._decisions["stop"] += 1
                return {
                    "action": "stop",
                    "steer_override": None,
                    "speed_override": 0,
                    "obstacle_type": "stop",
                    "obstacle_info": None,
                    "bypass_direction": None,
                    "sign_decision": None,
                    "fork_right": False,
                    "debug": "STOP SIGN LATCHED",
                }
        return None

    def _blind_box_decision(self):
        """盲盒任务决策（2026-08-21 用户方案）：舵机向右打固定角 + 低速行驶。

        固定角 = SERVO_MID + BLIND_BOX_STEER_US（µs 偏移，正=右转），
        由 full_pipeline 直接输出、绕过 PD/前馈；速度 = BLIND_BOX_SPEED。
        """
        self._decisions["stop"] += 1
        return {
            "action": "blind_box",
            "steer_override": None,
            "speed_override": BLIND_BOX_SPEED,
            "servo_override": BLIND_BOX_STEER_US,
            "obstacle_type": "stop",
            "obstacle_info": None,
            "bypass_direction": None,
            "sign_decision": None,
            "fork_right": False,
            "debug": "BLIND BOX LATCHED",
        }

    # ----------------------------------------------------------------
    # 路牌检测
    # ----------------------------------------------------------------

    def _check_signs(self, tracked_objects, has_sign_result, sign_decision,
                     sign_cooldown_until=0.0, sign_triggered=False,
                     frame_shape=None):
        """检查是否需要处理路牌。

        接近状态机（从高到低）：
          1. LLM 决策刚返回 → 消费决策（fork_right 的手性切换由 full_pipeline 执行）
          2. 无路牌跟踪 → 无事可做
          3. 路牌存在但在冷却期 → 忽略（已处理过）
          4. 路牌稳定检测 → 减速接近（SIGN_APPROACH_SPEED，带面积迟滞）
          5. 路牌存在但未触发停车 → 继续靠近（不减速）
          6. 路牌触发停车条件 → 停车等待 OCR+LLM

        Parameters:
            frame_shape: (H, W) 帧尺寸，用于计算面积占比
        """
        signs = [t for t in tracked_objects if t["class_id"] in SIGN_CLASSES]

        # P1: LLM 决策已就绪 → 消费（执行侧：fork_right 由 pipeline 切换右手循迹，
        #     straight 由 pipeline 恢复左手循迹，avoidance 不再持有 fork 状态）
        if has_sign_result and sign_decision:
            self._handling_sign = False
            self._sign_approach_active = False
            sa = sign_decision.get("action", "straight")
            self._decisions["sign"] += 1
            return {
                "action": "sign_detected",
                "steer_override": None,
                "speed_override": None,
                "obstacle_type": "sign",
                "obstacle_info": signs[0] if signs else None,
                "bypass_direction": None,
                "sign_decision": sign_decision,
                "fork_right": False,   # 已废弃：右岔路改为 pipeline 切换循迹手性
                "debug": f"Sign decision: {sa}",
            }

        # P4: 无路牌跟踪 → 无事可做
        if not signs:
            self._handling_sign = False
            self._sign_approach_active = False
            return None

        # P5: 路牌存在但在冷却期 → 忽略（已处理过，不要重新停车/减速）
        if time.time() < sign_cooldown_until:
            self._sign_approach_active = False
            return None

        # ── 计算最显著路牌的属性 ──
        best_sign = max(signs, key=lambda s: s["score"])
        bbox = best_sign["bbox"]
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        if frame_shape is not None:
            fh, fw = frame_shape
        else:
            fh = fw = 480.0  # fallback
        bbox_area_ratio = (bbox_w * bbox_h) / (fw * fh)
        hits = best_sign.get("hits", 0)

        # ════════════════════════════════════════════════════════════════
        # P5.25: 稳定检测到但未到停车阈值 → 减速接近（带迟滞）
        #         进入阈值：hits >= SIGN_SLOW_MIN_HITS AND area >= SIGN_SLOW_MIN_AREA
        #         退出阈值：area < SIGN_SLOW_MIN_AREA × 0.7（防止面积抖动导致频繁切换）
        # ════════════════════════════════════════════════════════════════
        if not sign_triggered:
            approach_min_area = (
                SIGN_SLOW_MIN_AREA * 0.7 if self._sign_approach_active
                else SIGN_SLOW_MIN_AREA
            )
            if (hits >= SIGN_SLOW_MIN_HITS
                    and bbox_area_ratio >= approach_min_area):
                if not self._sign_approach_active:
                    self._sign_approach_active = True
                    logger.info(
                        f"路牌稳定检测 (area={bbox_area_ratio:.3f}, hits={hits})，"
                        f"减速接近 (速度={SIGN_APPROACH_SPEED})"
                    )
                self._decisions["sign"] += 1
                return {
                    "action": "sign_approach",
                    "steer_override": None,
                    "speed_override": SIGN_APPROACH_SPEED,
                    "obstacle_type": "sign",
                    "obstacle_info": best_sign,
                    "bypass_direction": None,
                    "sign_decision": None,
                    "fork_right": False,
                    "debug": (
                        f"Sign approaching area={bbox_area_ratio:.3f} hits={hits} "
                        f"slow to {SIGN_APPROACH_SPEED}"
                    ),
                }

            # 进入减速后又退出了 → 恢复（减速超过但还没停车就错过了）
            if self._sign_approach_active:
                self._sign_approach_active = False
                logger.info("路牌离开减速区，恢复正常速度")

        # P5.5: 路牌存在但 pipeline 还未确认触发（面积/hits 不够）→ 不停车，继续靠近
        if not sign_triggered:
            return None

        # P6: 路牌存在 + sign_triggered + 无决策 + 不在冷却期 → 停车等待 OCR+LLM
        self._sign_approach_active = False
        self._decisions["sign"] += 1
        return {
            "action": "sign_detected",
            "steer_override": None,        # 不覆盖转向: pipeline 会冻结当前舵机
            "speed_override": STOP_SPEED,  # 停车等待
            "obstacle_type": "sign",
            "obstacle_info": signs[0],
            "bypass_direction": None,
            "sign_decision": None,         # 还未处理
            "fork_right": False,
            "debug": "Sign detected, waiting for OCR",
        }

    # ----------------------------------------------------------------
    # 行人检测 & 避让
    # ----------------------------------------------------------------

    def _check_pedestrians(self, tracked_objects, lane_center_x, img_width, img_height,
                           lane_mask=None, left_chain=None, right_chain=None,
                           midline=None):
        """
        检查行人是否需要避让（v2.1：车道边界感知 + 让行/侵入运动判定，2026-08-16）。

        触发条件: 新鲜检测（age==0）+ y2 距离门控 + 面积门控，取 y2 最大的最近行人。
        占道判定: 行人中心 x 是否落在 y2 下移采样行处的左右边界之间；
                  缺链 → bbox 底部 25% 行带车道掩码占比兜底；仍不可用 → 保守占道。
        停车判定 must_stop（v2.1，行人恒速垂直车道平移、不回头）:
          ① 占道且非让行 → 停。让行 = 远离中线（右侧向右/左侧向左）且
             |side×vx| > PEDESTRIAN_MOVE_DIR_THRESHOLD；
          ② 车道外、朝车道内移动（|side×vx| 超阈值且方向朝中线）、且
             |ped_cx−mid_x| < 采样行车道宽 → 停（侵入预警）；
          ③ 车道信息不可用 → 保守停。
        状态机: must_stop=True 立即停车（安全优先，无迟滞）；解除需连续
                PEDESTRIAN_DIR_FLIP_FRAMES 帧 must_stop=False（迟滞，防 vx/边界
                抖动来回跳）。绕行方向=空间大侧，相反判决连续 N 帧才翻转。
        退出  : 连续 N 帧无新鲜检测 → 清状态直接恢复正常循迹（去抖期间保持
                 当前动作：绕行保持偏置/停车保持停车）。
        """
        # 只认本帧新鲜检测（age==0）：幽灵轨迹（检测消失后 bbox 冻结 ≤30 帧）
        # 一律视为"行人不在"，消失即开始去抖；偶发丢检由去抖兜底。
        pedestrians = [t for t in tracked_objects
                       if t["class_id"] == CLASS_HUMAN and t.get("age", 0) == 0]

        # ── 本帧无新鲜行人 → 缺席计数，连续 N 帧后恢复正常 ──
        if not pedestrians:
            if self._waiting_for_pedestrian or self._pedestrian_bypass_active:
                self._pedestrian_missing_frames += 1
                if self._pedestrian_missing_frames >= PEDESTRIAN_RETURN_MISS_FRAMES:
                    logger.info(
                        "行人连续 %d 帧不可见 (阈值 %d)，恢复正常循迹",
                        self._pedestrian_missing_frames, PEDESTRIAN_RETURN_MISS_FRAMES)
                    self._reset_pedestrian_state()
                    return None
                # 去抖期间保持当前动作
                if self._pedestrian_bypass_active:
                    return self._make_pedestrian_keep_decision()
                return self._make_pedestrian_stop_decision(
                    None, "MISS %d/%d" % (self._pedestrian_missing_frames,
                                          PEDESTRIAN_RETURN_MISS_FRAMES))
            self._pedestrian_missing_frames = 0
            return None

        # 本帧有新鲜行人 → 缺席计数清零
        self._pedestrian_missing_frames = 0

        # ── 距离门控 + 面积门控：取最近（y2 最大）行人 ──
        best = None
        best_y2 = 0.0
        for p in pedestrians:
            px1, py1, px2, py2 = p["bbox"]
            if py2 < img_height * PEDESTRIAN_PROXIMITY_Y2_RATIO:
                continue
            bbox_area = (px2 - px1) * (py2 - py1)
            if bbox_area / (img_width * img_height) < PEDESTRIAN_PROXIMITY_AREA_RATIO:
                continue
            if py2 > best_y2:
                best_y2 = py2
                best = p

        if best is None:
            # 行人仍可见但未达门控：不算缺席；保持当前动作（与车辆同款）
            if self._pedestrian_bypass_active:
                return self._make_pedestrian_keep_decision()
            if self._waiting_for_pedestrian:
                return self._make_pedestrian_stop_decision(None, "GATE-LOST")
            return None

        y2 = best["bbox"][3]
        ped_cx = best["center"][0]

        # ── 参照中线：目标线在 y2 行的插值（丢线兜底 lane_center_x） ──
        mid_x = _mid_x_at_y(midline, y2)
        if mid_x is None:
            mid_x = lane_center_x

        # ── 采样行（y2 下移 10px）：行人脚底附近，避开分割图空洞 ──
        y_sample = min(int(y2) + PEDESTRIAN_IN_LANE_SAMPLE_DOWN_PX,
                       img_height - 1)
        smp_lx = _x_at_y(left_chain, y_sample)
        smp_rx = _x_at_y(right_chain, y_sample)

        # ── 运动方向判定（v2.1，行人恒速垂直车道平移；vx=追踪器5帧滑动平均） ──
        vx = best["velocity"][0]
        side = 1 if ped_cx > mid_x else -1          # 行人相对车道中线的侧向
        sv = side * vx                               # + = 远离中线（向外），− = 朝中线
        moving_away = sv > PEDESTRIAN_MOVE_DIR_THRESHOLD
        moving_inbound = sv < -PEDESTRIAN_MOVE_DIR_THRESHOLD

        # ── 占道判定：行人中心 x vs 采样行左右边界 ──
        def _in_lane():
            """True=占道 / False=不占道 / None=判定不可用（保守按占道处理）"""
            if smp_lx is not None and smp_rx is not None:
                return smp_lx <= ped_cx <= smp_rx
            # 缺链兜底：bbox 底部 25% 行带内车道掩码占比
            if lane_mask is not None and int(np.count_nonzero(lane_mask)) > 0:
                px1, py1, px2, py2 = best["bbox"]
                bh = max(1.0, py2 - py1)
                r0 = max(0, int(round(py2 - 0.25 * bh)))
                r1 = min(img_height - 1, int(round(py2)))
                c0 = max(0, int(px1))
                c1 = min(img_width - 1, int(px2))
                if r1 > r0 and c1 > c0:
                    band = lane_mask[r0:r1 + 1, c0:c1 + 1]
                    ratio = float(np.count_nonzero(band)) / float(band.size)
                    return ratio >= PEDESTRIAN_IN_LANE_RATIO
            return None

        # ── 停车判定（2026-08-19 重写，用户方案：曾占道记忆 + 运动方向分类） ──
        def _must_stop(in_lane):
            """返回 (stop, reason)。
            ① 占道（车道内）→ 停（不管动静），并置曾占道标志；
            ② 车道外 + 曾占道 → 绕行（不管当前动静——"进过车道内再到
               车道外就开始绕行"）；
            ③ 车道外 + 从未占道 + 向车道内移动 → 停（无距离限制）；
            ④ 车道外 + 从未占道 + 静止 → 停；
            ⑤ 车道外 + 从未占道 + 向车道外移动 → 正常循迹（reason="NORMAL"，
               不处理该行人）；
            ⑥ 车道信息不可用 → 保守停。
            """
            if in_lane is None:
                return True, "NO-LANE"
            if in_lane:
                self._pedestrian_was_in_lane = True
                return True, "IN-LANE"
            if self._pedestrian_was_in_lane:
                return False, "BYPASS"
            if moving_inbound:
                return True, "INBOUND"
            if moving_away:
                return False, "NORMAL"
            return True, "INBOUND"

        # ── 绕行侧选择：空间大的一侧（链）；兜底按行人相对目标线 ──
        def _pick_dir():
            """返回 (dir, free_left, free_right)；dir: -1(左绕)/1(右绕)/None"""
            px1, py1, px2, py2 = best["bbox"]
            if smp_lx is not None and smp_rx is not None:
                free_left = px1 - smp_lx
                free_right = smp_rx - px2
                if free_right > free_left:
                    return 1, free_left, free_right
                if free_left > free_right:
                    return -1, free_left, free_right
                return None, free_left, free_right   # 空间相等 → 判定不了
            if abs(ped_cx - mid_x) < 1e-6:
                return None, None, None
            # 行人在目标线右侧 → 左绕，反之右绕
            return (-1 if ped_cx > mid_x else 1), None, None

        # ── 构造 bypass 决策 ──
        def _make_bypass(ped, bypass_dir, free_left=None, free_right=None,
                         yield_flag=False):
            sign = 1 if bypass_dir > 0 else -1
            bias = sign * PEDESTRIAN_BYPASS_OFFSET
            dir_str = "RIGHT" if bypass_dir > 0 else "LEFT"
            fl = (f" free={free_left:.0f}/{free_right:.0f}"
                  if free_left is not None else "")
            yld = " YIELD" if yield_flag else ""
            self._decisions["bypass"] += 1
            return {
                "action": "pedestrian_bypass",
                "steer_override": None,
                "speed_override": PEDESTRIAN_BYPASS_SPEED,
                "obstacle_type": "pedestrian",
                "obstacle_info": ped,
                "bypass_direction": "right" if bypass_dir > 0 else "left",
                "sign_decision": None,
                "fork_right": False,
                "track_offset_bias": bias,
                "debug": f"BYPASS {dir_str} bias={bias:+.2f}{fl}{yld}",
            }

        # ── 构造 stop 决策（ped=None 表示保持停车/去抖帧，无 bbox 信息） ──
        def _make_stop(ped, note):
            self._decisions["stop"] += 1
            return {
                "action": "stop",
                "steer_override": None,
                "speed_override": STOP_SPEED,
                "obstacle_type": "pedestrian",
                "obstacle_info": ped,
                "bypass_direction": None,
                "sign_decision": None,
                "fork_right": False,
                "debug": f"STOP {note}",
            }

        # ════════════════════════════════════════════════════════════
        # 状态机主逻辑（v2.1：must_stop 触发立即停车；解除连续 F 帧
        # must_stop=False 才转绕行；行人恒速横穿、不回头）
        # ════════════════════════════════════════════════════════════

        in_lane = _in_lane()
        stop_now, stop_reason = _must_stop(in_lane)

        # ── 停车等待中 ──
        if self._waiting_for_pedestrian:
            if stop_now:
                self._pedestrian_out_lane_frames = 0
                return _make_stop(best, stop_reason)
            # 可通行 → 迟滞计数，连续 F 帧才转绕行
            self._pedestrian_out_lane_frames += 1
            if self._pedestrian_out_lane_frames >= PEDESTRIAN_DIR_FLIP_FRAMES:
                new_dir, fl, fr = _pick_dir()
                if new_dir is None:
                    return _make_stop(best, "NO-DIR")
                logger.info(
                    "行人连续 %d 帧可通行，转入绕行 (%s)",
                    self._pedestrian_out_lane_frames,
                    "RIGHT" if new_dir > 0 else "LEFT")
                self._waiting_for_pedestrian = False
                self._wait_start_time = 0.0
                self._wait_pedestrian_id = None
                self._pedestrian_out_lane_frames = 0
                self._pedestrian_bypass_active = True
                self._pedestrian_bypass_dir = new_dir
                return _make_bypass(best, new_dir, fl, fr,
                                    yield_flag=(stop_reason == "YIELD"))
            return _make_stop(
                best, "CLEAR %d/%d" % (self._pedestrian_out_lane_frames,
                                       PEDESTRIAN_DIR_FLIP_FRAMES))

        # ── 绕行中 ──
        if self._pedestrian_bypass_active:
            if stop_now:
                # 让行中停下/回头、重新占道、侵入预警、判定不可用 → 立即停车
                logger.info("行人状态变化 (%s)，绕行中断，立即停车等待",
                            stop_reason)
                self._pedestrian_bypass_active = False
                self._pedestrian_bypass_dir = 0
                self._pedestrian_flip_pending_frames = 0
                self._waiting_for_pedestrian = True
                self._wait_start_time = time.time()
                self._wait_pedestrian_id = best["id"]
                self._pedestrian_out_lane_frames = 0
                return _make_stop(best, stop_reason)
            # 每帧重选方向（迟滞翻转）
            cand, fl, fr = _pick_dir()
            if cand is None:
                return _make_bypass(best, self._pedestrian_bypass_dir, fl, fr,
                                    yield_flag=(stop_reason == "YIELD"))
            if cand != self._pedestrian_bypass_dir:
                self._pedestrian_flip_pending_frames += 1
                if self._pedestrian_flip_pending_frames >= PEDESTRIAN_DIR_FLIP_FRAMES:
                    logger.info(
                        "相反绕行判决连续 %d 帧 (阈值 %d)，方向翻转 %+d → %+d",
                        self._pedestrian_flip_pending_frames,
                        PEDESTRIAN_DIR_FLIP_FRAMES,
                        self._pedestrian_bypass_dir, cand)
                    self._pedestrian_bypass_dir = cand
                    self._pedestrian_flip_pending_frames = 0
            else:
                self._pedestrian_flip_pending_frames = 0
            return _make_bypass(best, self._pedestrian_bypass_dir, fl, fr,
                                yield_flag=(stop_reason == "YIELD"))

        # ── 正常状态首次触发 ──
        if stop_now:
            self._waiting_for_pedestrian = True
            self._wait_start_time = time.time()
            self._wait_pedestrian_id = best["id"]
            logger.info("行人触发停车 (%s) (cx=%.0f, y2=%.0f)",
                        stop_reason, ped_cx, y2)
            return _make_stop(best, stop_reason)

        # 从未占道 + 向车道外移动 → 不处理（正常循迹，2026-08-19）
        if stop_reason == "NORMAL":
            return None

        # 可通行（含占道让行）→ 首帧直接绕行
        new_dir, fl, fr = _pick_dir()
        if new_dir is None:
            # 判不了方向 → 保守停车
            self._waiting_for_pedestrian = True
            self._wait_start_time = time.time()
            self._wait_pedestrian_id = best["id"]
            logger.info("行人可通行但绕行方向判定失败，保守停车")
            return _make_stop(best, "NO-DIR")
        self._pedestrian_bypass_active = True
        self._pedestrian_bypass_dir = new_dir
        self._pedestrian_flip_pending_frames = 0
        logger.info("行人可通行%s (cx=%.0f, y2=%.0f)，绕行 %s",
                    "（让行中）" if stop_reason == "YIELD" else "",
                    ped_cx, y2, "RIGHT" if new_dir > 0 else "LEFT")
        return _make_bypass(best, new_dir, fl, fr,
                            yield_flag=(stop_reason == "YIELD"))

    def _make_pedestrian_keep_decision(self):
        """行人去抖期间保持绕行：锁定方向 + 固定偏置 + 慢速。"""
        sign = 1 if self._pedestrian_bypass_dir > 0 else -1
        bias = sign * PEDESTRIAN_BYPASS_OFFSET
        self._decisions["bypass"] += 1
        return {
            "action": "pedestrian_bypass",
            "steer_override": None,
            "speed_override": PEDESTRIAN_BYPASS_SPEED,
            "obstacle_type": "pedestrian",
            "obstacle_info": None,
            "bypass_direction": "right" if self._pedestrian_bypass_dir > 0 else "left",
            "sign_decision": None,
            "fork_right": False,
            "track_offset_bias": bias,
            "debug": "MISS %d/%d KEEP BYPASS" % (self._pedestrian_missing_frames,
                                                 PEDESTRIAN_RETURN_MISS_FRAMES),
        }

    def _make_pedestrian_stop_decision(self, ped, note):
        """行人停车决策（ped=None 时为去抖/门控外保持停车，无 bbox 信息）。"""
        self._decisions["stop"] += 1
        return {
            "action": "stop",
            "steer_override": None,
            "speed_override": STOP_SPEED,
            "obstacle_type": "pedestrian",
            "obstacle_info": ped,
            "bypass_direction": None,
            "sign_decision": None,
            "fork_right": False,
            "debug": f"STOP {note}",
        }


    # ----------------------------------------------------------------
    # 车辆检测 & 避让
    # ----------------------------------------------------------------

    def _check_vehicles(self, tracked_objects, lane_center_x, midline,
                        img_width, img_height, tracking_hand="left"):
        """
        检查车辆是否需要避让。

        触发条件: 车辆 bbox 底部进入画面下半部分（距离门控 + 面积门控）
        退出条件: 连续 VEHICLE_RETURN_MISS_FRAMES 帧完全检测不到车辆
                  （去抖退出，防单帧丢检抖动；去抖期间继续绕行）。去抖达标后
                  直接恢复正常循迹（2026-08-16 二次修改，用户方案）：绕行偏置
                  瞬时归零、速度立即恢复 DEFAULT_SPEED，不再进入回中阶段
                  （无反向回中偏置），由正常循迹 PD 反馈按偏差比例自然回中。
        方向判断（2026-08-16 v5，用户方案：连续判断 + 迟滞 + 并排锁死）:
          - 轻量距离门控：门控内但 y2 < VEHICLE_JUDGE_Y2_RATIO 时只减速不判
            方向（远处判决不可靠），返回 vehicle_approach（无偏置正常循迹）；
          - y2 达到 VEHICLE_JUDGE_Y2_RATIO 后每帧连续判断：veh_cx 与 y2 行
            目标线插值 mid_x 比较（丢线兜底 lane_center_x），车在左→右绕、在右→左绕；
          - 首帧判断直接采用（进入绕行）；此后仅当"相反判决连续
            VEHICLE_DIR_FLIP_FRAMES 帧"才翻转方向（迟滞，防判决抖动）；
          - 并排锁死：y2 ≥ VEHICLE_DIR_LOCK_Y2_RATIO 后方向锁死直到车辆离开
            （并排时 bbox 被画面边缘截断、中心估计退化，防误翻转）；
          - 远处系统性错判由"靠近后连续正确判决 → 迟滞翻转"自动纠偏。

         绕行策略（2026-08-17 四象限固定偏置版）:
           - 偏置：固定幅值，按（循迹手性 × 绕行方向）四象限分别可配
             （VEHICLE_BYPASS_STEER_{LH,RH}_{LEFT,RIGHT}；不再按距离动态缩放——
             虚拟目标线固定 → 平行绕行，且 bias 恒定不进 D 项、不产生摆动）
           - 速度：接近/绕行/去抖全程 VEHICLE_SLOW_SPEED；恢复正常循迹后立即回 DEFAULT_SPEED
         """
        # 只认本帧新鲜检测（age==0）：多目标追踪会在检测消失后把轨迹
        # 保留最多 TRACK_MAX_AGE(30) 帧（bbox 冻结在最后位置），若把它当
        # 真车，miss 去抖要等轨迹死亡后才启动 → 躲避状态拖 30+5 帧
        # （虚拟地图过终点刷新后车仍歪着慢走 2~3 秒）。幽灵轨迹一律视为
        # "车辆不在"，消失即开始 5 帧去抖；真车偶发丢检由去抖兜底。
        vehicles = [t for t in tracked_objects
                    if t["class_id"] == CLASS_CAR and t.get("age", 0) == 0]

        # ── 本帧无新鲜车辆检测 → 缺席帧计数，连续 N 帧后恢复正常循迹 ──
        if not vehicles:
            if self._vehicle_bypass_active:
                self._vehicle_missing_frames += 1
                if self._vehicle_missing_frames >= VEHICLE_RETURN_MISS_FRAMES:
                    logger.info(
                        f"车辆连续 {self._vehicle_missing_frames} 帧不可见 "
                        f"(阈值 {VEHICLE_RETURN_MISS_FRAMES})，恢复正常循迹"
                        f"（绕行偏置瞬时归零，速度恢复 DEFAULT_SPEED）"
                    )
                    self._vehicle_bypass_active = False
                    self._vehicle_bypass_dir = 0
                    self._vehicle_bypass_bias = 0.0
                    self._vehicle_missing_frames = 0
                    self._vehicle_flip_pending_frames = 0
                    self._vehicle_dir_locked = False
                    # 返回 None → 落入"正常循迹"分支：无偏置、正常速度，
                    # 由正常循迹 PD 反馈自然回中（不再有反向回中偏置）
                    return None
                # 去抖期间：保持绕行（锁定方向 + 固定偏置 + 慢速）
                return self._make_vehicle_keep_decision(tracking_hand)
            self._vehicle_missing_frames = 0
            return None

        # 本帧有新鲜车辆检测 → 缺席计数清零
        self._vehicle_missing_frames = 0

        # 距离门控 + 面积门控：找到最近的且在门控范围内的车辆
        best_vehicle = None
        best_y2 = 0

        for veh in vehicles:
            vx1, vy1, vx2, vy2 = veh["bbox"]
            if vy2 < img_height * VEHICLE_PROXIMITY_Y2_RATIO:
                continue  # 太远，跳过
            bbox_area = (vx2 - vx1) * (vy2 - vy1)
            if bbox_area / (img_width * img_height) < VEHICLE_PROXIMITY_AREA_RATIO:
                continue  # 太小，跳过
            if vy2 > best_y2:
                best_y2 = vy2
                best_vehicle = veh

        if best_vehicle is None:
            # 车辆仍可见但未达门控：不算缺席（不计数）。
            # 若正在绕行则继续绕行，直到车辆完全不可见再走缺席计数。
            if self._vehicle_bypass_active:
                return self._make_vehicle_keep_decision(tracking_hand)
            return None

        # ── 轻量距离门控（v5，用户选择）：y2 未到判方向线 → 只减速、不判方向 ──
        y2 = best_vehicle["bbox"][3]
        if (not self._vehicle_bypass_active
                and y2 < img_height * VEHICLE_JUDGE_Y2_RATIO):
            self._decisions["approach"] += 1
            return {
                "action": "vehicle_approach",
                "steer_override": None,
                "speed_override": VEHICLE_SLOW_SPEED,
                "obstacle_type": "vehicle",
                "obstacle_info": best_vehicle,
                "bypass_direction": None,
                "sign_decision": None,
                "fork_right": False,
                "debug": (
                    f"Vehicle approach (y2={y2:.0f}, judge at "
                    f"{img_height * VEHICLE_JUDGE_Y2_RATIO:.0f}), slow no bias"
                ),
            }

        # ── 参照中线：优先插值车辆 bbox 底部行 y2 处的目标线 ──
        mid_x = _mid_x_at_y(midline, y2)
        line_lost = (mid_x is None)

        # ── 每帧连续判断（无死区直接比较，v5） ──
        # 2026-08-15 板测：原"±0.08×画宽 死区默认向左"导致贴边车
        # （车中心距目标线 <51px）被强制向左绕 → 撞向车。改为无死区：
        # 车在参照线左侧→右绕(+1)、右侧→左绕(-1)（占中 diff≈0 时首帧随机，
        # 之后由迟滞翻转纠偏，两侧都有空间，安全）。
        # 2026-08-19 丢线规则（用户方案）：循迹丢线无目标线可插值时——
        #   首帧（未判过方向）→ 只减速不判方向不给偏置，等目标线恢复；
        #   已绕行中 → 保持当前方向与偏置继续绕（半路不松），
        #   不再用 lane_center_x（丢线时为画面中心 320）兜底判方向。
        veh_center_x = best_vehicle["center"][0]   # bbox 底部中心 x（= 中心 x）

        if not self._vehicle_bypass_active:
            if line_lost:
                self._decisions["approach"] += 1
                return {
                    "action": "vehicle_approach",
                    "steer_override": None,
                    "speed_override": VEHICLE_SLOW_SPEED,
                    "obstacle_type": "vehicle",
                    "obstacle_info": best_vehicle,
                    "bypass_direction": None,
                    "sign_decision": None,
                    "fork_right": False,
                    "debug": (
                        f"Vehicle no lane (y2={y2:.0f}), slow no judge"
                    ),
                }
            # 首帧判断：直接采用并进入绕行（不等待确认，否则接近段一直无偏置）
            offset_from_lane = veh_center_x - mid_x
            judgment = 1 if offset_from_lane < 0 else -1
            self._vehicle_bypass_active = True
            self._vehicle_bypass_dir = judgment
            self._vehicle_flip_pending_frames = 0
        elif not self._vehicle_dir_locked:
            # 并排锁死线：y2 达标即锁（不再翻转，直到车辆离开；
            # 丢线时同样执行——锁死是防误翻保护，与目标线无关）
            if y2 >= img_height * VEHICLE_DIR_LOCK_Y2_RATIO:
                self._vehicle_dir_locked = True
                logger.info(
                    f"车辆并排 (y2={y2:.0f} ≥ 锁死线 "
                    f"{img_height * VEHICLE_DIR_LOCK_Y2_RATIO:.0f})，方向锁死"
                )
            elif not line_lost:
                offset_from_lane = veh_center_x - mid_x
                judgment = 1 if offset_from_lane < 0 else -1
                if judgment != self._vehicle_bypass_dir:
                    # 相反判决 → 迟滞计数，连续 N 帧才翻转（防判决抖动）
                    self._vehicle_flip_pending_frames += 1
                    if (self._vehicle_flip_pending_frames
                            >= VEHICLE_DIR_FLIP_FRAMES):
                        logger.info(
                            f"相反判决连续 {self._vehicle_flip_pending_frames} 帧 "
                            f"(阈值 {VEHICLE_DIR_FLIP_FRAMES})，绕行方向翻转 "
                            f"{self._vehicle_bypass_dir:+d} → {judgment:+d}"
                        )
                        self._vehicle_bypass_dir = judgment
                        self._vehicle_flip_pending_frames = 0
                else:
                    # 判决与当前方向一致 → 计数清零
                    self._vehicle_flip_pending_frames = 0
            # line_lost 且未锁死：保持当前方向与迟滞计数不变，
            # 等目标线恢复后继续判断

        bypass_dir = "right" if self._vehicle_bypass_dir > 0 else "left"

        # ── 固定偏置（2026-08-17：按手性×方向四象限取幅值，符号由方向决定） ──
        sign = 1 if self._vehicle_bypass_dir > 0 else -1
        magnitude = _vehicle_bypass_magnitude(tracking_hand,
                                              self._vehicle_bypass_dir)
        steer = sign * magnitude
        self._vehicle_bypass_bias = magnitude

        self._decisions["bypass"] += 1
        return {
            "action": "bypass",
            "steer_override": None,
            "speed_override": VEHICLE_SLOW_SPEED,
            "obstacle_type": "vehicle",
            "obstacle_info": best_vehicle,
            "bypass_direction": bypass_dir,
            "sign_decision": None,
            "fork_right": False,
            "track_offset_bias": steer,
            "debug": (
                f"Vehicle ahead (y2={y2:.0f}"
                f"{', no lane' if line_lost else ', dist=' + format(abs(veh_center_x - mid_x), '.0f') + 'px'}), "
                f"bypass {bypass_dir.upper()} bias={steer:+.2f}"
                + (" LOCKED" if self._vehicle_dir_locked else "")
            ),
        }

    def _make_vehicle_keep_decision(self, tracking_hand="left"):
        """车辆暂时不可见/未达门控时的保持决策：沿用锁定方向与固定偏置继续绕行。

        2026-08-16 去抖退出配套：只在"本帧完全检测不到车辆"时才会走到这里
        （miss 计数 1..N-1），或车辆可见但未达门控；速度保持 VEHICLE_SLOW_SPEED。
        2026-08-17：幅值按（手性×方向）四象限取（tracking_hand 取当前帧手性）。
        """
        sign = 1 if self._vehicle_bypass_dir > 0 else -1
        magnitude = _vehicle_bypass_magnitude(tracking_hand,
                                              self._vehicle_bypass_dir)
        steer = sign * magnitude
        self._vehicle_bypass_bias = magnitude
        bypass_dir = "right" if self._vehicle_bypass_dir > 0 else "left"
        self._decisions["bypass"] += 1
        return {
            "action": "bypass",
            "steer_override": None,
            "speed_override": VEHICLE_SLOW_SPEED,
            "obstacle_type": "vehicle",
            "obstacle_info": None,
            "bypass_direction": bypass_dir,
            "sign_decision": None,
            "fork_right": False,
            "track_offset_bias": steer,
            "debug": (
                f"Vehicle missing (miss={self._vehicle_missing_frames}/"
                f"{VEHICLE_RETURN_MISS_FRAMES}), keep bypass {bypass_dir.upper()}"
            ),
        }

    # ----------------------------------------------------------------
    # 状态管理
    # ----------------------------------------------------------------

    def _reset_wait_state(self):
        """重置等待状态（保留 bypass 状态）"""
        self._waiting_for_pedestrian = False
        self._wait_start_time = 0.0
        self._wait_pedestrian_id = None

    def _reset_pedestrian_state(self):
        """重置所有行人相关状态（车辆绕行状态由 _check_vehicles 自己管理）。

        注意（2026-08-15 修复）：原实现"车辆绕行状态也一并清除"导致
        _check_pedestrians 在"无行人"分支每帧调用本函数时误清
        _vehicle_bypass_active —— 车辆方向锁定与"消失→回中"永远无法生效。
        现改为只清行人状态；AvoidancePlanner.reset() 里显式清车辆状态。
        """
        self._waiting_for_pedestrian = False
        self._wait_start_time = 0.0
        self._wait_pedestrian_id = None
        self._pedestrian_bypass_active = False
        self._pedestrian_bypass_dir = 0
        self._pedestrian_missing_frames = 0
        self._pedestrian_flip_pending_frames = 0
        self._pedestrian_out_lane_frames = 0
        # 曾占道标志（2026-08-19）：行人进过车道内 → True；消失去抖恢复时重置
        self._pedestrian_was_in_lane = False

    def reset(self):
        """重置所有状态"""
        self._reset_pedestrian_state()
        self._handling_sign = False
        self._sign_decision = None
        self._sign_approach_active = False
        self._stop_triggered = None       # stop 延时状态（2026-08-19；盲盒锁存同用）
        self._stop_latched = False
        self._vehicle_bypass_active = False
        self._vehicle_bypass_dir = 0
        self._vehicle_bypass_bias = 0.0
        self._vehicle_missing_frames = 0
        self._vehicle_flip_pending_frames = 0
        self._vehicle_dir_locked = False
        logger.info("AvoidancePlanner 已重置")

    # ----------------------------------------------------------------
    # 统计
    # ----------------------------------------------------------------

    def get_stats(self):
        """返回决策统计"""
        return {
            "decisions": dict(self._decisions),
            "waiting_for_pedestrian": self._waiting_for_pedestrian,
            "handling_sign": self._handling_sign,
        }
