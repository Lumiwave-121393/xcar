"""
lane.py —— 迷宫法边线提取 + 沿左边线循迹（透视缩放偏移）

参考：第16届智能车智能视觉组-上海交通大学AuTop战队开源算法讲解(二)边线提取

算法链路:
  seg_map → 车道掩码 → 连通域滤波 + 闭运算
    → 迷宫法行走提取左/右边界链（见 maze.py，八邻域行走器）
    → 行扫描测路宽 + 鲁棒锥形拟合（剔除岔路区虚高行，透视自适应）
    → 逐行水平偏移：目标_x(y) = 左边线x(y) + RATIO×路宽(y)
    → 三角滤波 + 等距采样 → 前瞻点 → 归一化偏移 → 现有 PD

关键设计（板上实测修正）:
  1. 不用法向偏移：真实掩码的边界链是锯齿状，法向随切线翻转乱摆，
     偏移点以 40px 幅度左右乱跳、且法向带 y 分量导致下坠 → 目标线
     "混作一团"。逐行水平偏移与链同 y，无翻转、无下坠。
  2. 不用固定像素偏移：画面近大远小，固定 px 远处飘出路面、近处太小。
     平地面+固定相机下路宽(y) 近似线性（锥形），每帧鲁棒拟合：
     迭代剔除显著高于拟合的行（= 岔路区，行扫描跨越主路+岛+支路），
     拟合收敛到主干道路宽，目标点始终在主干道内 → 右侧汇出/汇入
     岔路直行通过，无需任何标定。
"""

import cv2
import numpy as np
import logging

from .config import (
    LANE_CLASS_ID, TRACKING_HAND, STEER_MODE,
    NOISE_MIN_AREA,
    MAZE_MAX_STEPS_LEFT, MAZE_MAX_STEPS_RIGHT,
    MAZE_SCAN_UP_TO_RATIO, MAZE_START_SKIP_BOTTOM_PX,
    MAZE_CLOSE_KSIZE, MIN_CHAIN_LEN,
    MAZE_WRAP_DESCENT_PX,
    TRIANGLE_PASSES, MIDLINE_RESAMPLE_STEP,
    WALL_RATIO, WIDTH_FIT_OUTLIER_RATIO, WIDTH_MIN_PX,
    WIDTH_MAX_TOP_ROW, WIDTH_MAX_TOP_PX,
    WIDTH_MAX_BOTTOM_ROW, WIDTH_MAX_BOTTOM_PX,
    LOOKAHEAD_PX,
    clamp,
)
from .maze import trace_boundary, find_start_points

logger = logging.getLogger("tracking.lane")


# ====================================================================
# 内部函数
# ====================================================================

def _filter_noise(lane_mask, min_area=NOISE_MIN_AREA):
    """用连通域分析滤除分割掩码中的孤立噪点"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        lane_mask, connectivity=8
    )
    filtered = np.zeros_like(lane_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255
    return filtered


def _trim_after_peak(chain):
    """最高点截断（2026-08-19 用户方案）：链自底向上爬升，
    保留到最高点（y 最小，含该点），其后的横向/下行段直接抛弃。
    """
    if not chain or len(chain) < 2:
        return chain
    pts = np.asarray(chain, dtype=np.float32)
    ys = pts[:, 1]
    top_y = float(ys.min())
    first_top = int(np.where(ys == top_y)[0][0])
    return chain[:first_top + 1]


def _triangular_filter(pts, passes=1):
    """沿链三角滤波：每列与 [1,2,1]/4 卷积 passes 遍（端点按边缘值扩展）"""
    pts = np.asarray(pts, dtype=np.float32)
    n = len(pts)
    if n < 3 or passes <= 0:
        return pts
    kernel = np.array([1.0, 2.0, 1.0], dtype=np.float32) / 4.0
    out = pts.copy()
    for _ in range(passes):
        padded = np.vstack([out[0], out, out[-1]])
        for c in range(2):
            out[:, c] = np.convolve(padded[:, c], kernel, mode="valid")
    return out


def _resample_chain(pts, step):
    """等距采样：沿折线按弧长 step 重新取点（含首末点）"""
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= step or step <= 0:
        return pts
    t_new = np.arange(0.0, total + 1e-6, step)
    xs = np.interp(t_new, cum, pts[:, 0])
    ys = np.interp(t_new, cum, pts[:, 1])
    return np.stack([xs, ys], axis=1)


def _measure_width_profile(mask, chain, scan_dir=1):
    """
    行扫描测路宽 + 鲁棒锥形拟合。

    对边界链经过的每一行，从该行边界点向路内侧扫到第一个非车道像素
    （墙 / 岔路口岛缘 / 障碍物），得到该行原始路宽。

    平地面+固定相机下，画面中路宽随行号近似线性（近大远小的锥形）。
    岔路区（右侧汇出/汇入）的行扫描跨越主路+岛+支路，宽度虚高——
    迭代剔除显著高于拟合的行后重新拟合，收敛到主干道路宽的锥形。
    拟合干净行不足时退化为中位数常量（中位数对离群天然鲁棒）。

    参数:
        mask:     (H, W) 二值图
        chain:    边界链 [(x, y), ...]
        scan_dir: 1  链为左边界，向右扫
                  -1 链为右边界，向左扫

    返回:
        (ys, pred, junction): 链行坐标、每行模型路宽、岔路标志（仅可视化）
    """
    h, w = mask.shape

    # 行 → 边界 x（链中首次出现该行时取）
    x_at = {}
    for x, y in chain:
        x_at.setdefault(int(y), int(x))
    if not x_at:
        return None

    ys = np.array(sorted(x_at.keys()), dtype=np.float32)
    ws = np.empty(len(ys), dtype=np.float32)

    for i, y in enumerate(ys):
        yi = int(y)
        row = mask[yi]
        xb = x_at[yi]
        if scan_dir > 0:
            black = np.where(row[xb:] == 0)[0]
            ws[i] = (w - xb) if black.size == 0 else float(black[0])
        else:
            black = np.where(row[:xb + 1] == 0)[0]
            ws[i] = xb if black.size == 0 else float(xb - black[-1])

    # ---- 鲁棒锥形拟合 w = a·y + b ----
    A = np.vstack([ys, np.ones(len(ys))]).T
    coef = np.linalg.lstsq(A, ws, rcond=None)[0]
    pred = A @ coef
    for _ in range(3):
        keep = ws < pred * WIDTH_FIT_OUTLIER_RATIO  # 剔除岔路区虚高行
        if int(keep.sum()) < 6:
            pred = np.full(len(ys), float(np.median(ws)), dtype=np.float32)
            break
        coef = np.linalg.lstsq(A[keep], ws[keep], rcond=None)[0]
        pred = A @ coef
    # 路宽上限随行号线性变化（透视合理性包络，锚点板上实测标定）：
    # 岔口区膨胀拟合在中部被压回真实锥形，限制中线向右突变幅度
    max_at = (WIDTH_MAX_TOP_PX
              + (WIDTH_MAX_BOTTOM_PX - WIDTH_MAX_TOP_PX)
              * (ys - WIDTH_MAX_TOP_ROW)
              / (WIDTH_MAX_BOTTOM_ROW - WIDTH_MAX_TOP_ROW))
    max_at = np.maximum(max_at, WIDTH_MIN_PX)  # 包络低于下限时按下限（顶点以上行）
    pred = np.clip(pred, WIDTH_MIN_PX, max_at)

    junction = bool(np.any(ws > pred * 1.5))
    return ys, pred, junction


def _target_line(chain, width_at, ratio, hand="left"):
    """
    边界链 → 循迹目标线：逐行水平偏移。

    target_x(y) = 左边线x(y) + ratio × 路宽(y)，与链同 y。

    为什么是水平偏移而不是法向偏移：
      真实掩码的边界链是锯齿状，法向随切线翻转乱摆，偏移点左右乱跳，
      且法向带 y 分量导致下坠——目标线会"混作一团"。水平偏移行对齐，
      无翻转、无下坠；路宽(y) 由鲁棒锥形拟合提供，近大远小自动适配。

    参数:
        chain:    边界链 [(x, y), ...]，行走顺序（自下而上）
        width_at: 可调用 width_at(y) → 该行模型路宽 (px)
        ratio:    目标横向位置 = 路宽比例（0.5=路中央，0.4=偏左贴墙）
        hand:     "left"  链为左边界，路在右侧（+偏移）
                  "right" 链为右边界，路在左侧（-偏移）

    返回:
        (N, 2) float32 目标线点列；失败返回 None
    """
    pts = _triangular_filter(chain, TRIANGLE_PASSES)
    if len(pts) < 3:
        return None

    sign = 1 if hand == "left" else -1
    xs = pts[:, 0] + sign * ratio * width_at(pts[:, 1])
    line = np.stack([xs, pts[:, 1]], axis=1)
    line = _resample_chain(line, MIDLINE_RESAMPLE_STEP)
    if len(line) < 2:
        return None
    return line


# ====================================================================
# 车道中心偏移提取（主入口）
# ====================================================================

def extract_lane_center(seg_map, lane_cls=LANE_CLASS_ID, hand=TRACKING_HAND,
                        lookahead_px=None):
    """
    从分割索引图中提取循迹偏移（迷宫法边线 + 沿左边线固定偏移 + 前瞻点）。

    参数:
        seg_map:  (H, W) 类别索引图
        lane_cls: 车道类别 ID
        hand:     "left" 沿左边线（默认） / "right" 沿右边线（预留）
        lookahead_px: 前瞻距离（None=用配置 LOOKAHEAD_PX；tracker 传速度自适应值）

    返回:
        offset:           [-1, 1]，正=目标点在图像中心右侧；None=丢线
                          （调用方保持上一帧偏移）
        lane_center_x:    底部目标线 x 坐标（供避障作近处参考）
        lane_mask_bottom: 底部车道掩码（仅用于可视化，向后兼容）
        fork_detected:    岔路标志（行扫描宽度显著高于锥形拟合，仅可视化）
        full_lane_mask:  (H, W) 完整尺寸车道掩码（避障重叠计算，全帧坐标对齐）
        info:             dict，含 left_chain/right_chain/midline(目标线)/
                          target/anchor/curvature(前方曲率 α，前馈用)，供可视化绘制
    """
    h, w = seg_map.shape
    if lookahead_px is None:
        lookahead_px = LOOKAHEAD_PX
    info = {
        "left_chain": [], "right_chain": [],
        "midline": None, "target": None, "anchor": None,
    }
    empty = (None, w // 2,
             np.zeros((max(1, h // 2), w), dtype=np.uint8),
             False, np.zeros((h, w), dtype=np.uint8), info)

    # ---- 1. 车道掩码 + 噪点过滤 + 闭运算 ----
    lane_mask = (seg_map == lane_cls).astype(np.uint8) * 255
    lane_mask = _filter_noise(lane_mask)
    if MAZE_CLOSE_KSIZE > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MAZE_CLOSE_KSIZE, MAZE_CLOSE_KSIZE))
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)

    full_lane_mask = lane_mask.copy()
    lane_mask_bottom = lane_mask[int(h * 0.5):, :]

    # ---- 2. 自底部向上找左右起点（至画面中部为止，找不到则丢线）----
    # 2026-08-16 追加：底部没有路时继续向上扫（范围 h×MAZE_SCAN_UP_TO_RATIO），
    # 不再局限固定扫描带；到上限仍找不到才返回 None（保持上一帧偏移）。
    # MAZE_START_SKIP_BOTTOM_PX：跳过画面最底边 N 行再开始找（0=不跳过）。
    left_start, right_start = find_start_points(
        lane_mask, scan_rows=int(h * MAZE_SCAN_UP_TO_RATIO),
        skip_bottom_px=MAZE_START_SKIP_BOTTOM_PX)

    # ---- 3. 迷宫法行走提取边界链（带整块掩码防包绕规则） ----
    # 2026-08-19：左右边线步数上限分开可配（驱动链用当前手性对应值，
    # 对侧链用另一侧值）；最高点截断（用户方案）——链自底向上爬升，
    # 保留到最高点（y 最小，含该点），其后的横向/下行段（画面顶边
    # 横走伪像/越过顶部后的包绕残留）直接抛弃，左右两链统一处理。
    drive_start = left_start if hand == "left" else right_start
    if drive_start is None:
        return empty

    drive_steps = (MAZE_MAX_STEPS_LEFT if hand == "left"
                   else MAZE_MAX_STEPS_RIGHT)
    drive_chain = trace_boundary(lane_mask, hand, drive_start,
                                 drive_steps, MAZE_WRAP_DESCENT_PX)
    drive_chain = _trim_after_peak(drive_chain)
    if len(drive_chain) < MIN_CHAIN_LEN:
        logger.debug("驱动边界链过短 (%d)，丢线", len(drive_chain))
        return empty

    # 另一侧链：可视化 + 将来切换手性用
    other_hand = "right" if hand == "left" else "left"
    other_start = right_start if hand == "left" else left_start
    if other_start is not None:
        other_steps = (MAZE_MAX_STEPS_RIGHT if hand == "left"
                       else MAZE_MAX_STEPS_LEFT)
        other_chain = trace_boundary(lane_mask, other_hand, other_start,
                                     other_steps, MAZE_WRAP_DESCENT_PX)
        other_chain = _trim_after_peak(other_chain)
    else:
        other_chain = []

    if hand == "left":
        info["left_chain"], info["right_chain"] = drive_chain, other_chain
    else:
        info["left_chain"], info["right_chain"] = other_chain, drive_chain

    # ---- 4. 路宽锥形拟合 + 目标线（逐行水平偏移） ----
    profile = _measure_width_profile(
        lane_mask, drive_chain, scan_dir=1 if hand == "left" else -1)
    if profile is None:
        return empty
    ys, pred, junction = profile

    def width_at(y):
        return np.interp(y, ys, pred, left=pred[0], right=pred[-1])

    line = _target_line(drive_chain, width_at, WALL_RATIO, hand)
    if line is None:
        return empty

    info["midline"] = line
    info["anchor"] = (float(line[0, 0]), float(line[0, 1]))

    # ---- 5. 前瞻点（沿目标线累积弧长 ≥ 前瞻距离） ----
    seg_len = np.linalg.norm(np.diff(line, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    idx = int(np.searchsorted(cum, lookahead_px))
    if idx >= len(line):
        idx = len(line) - 1  # 目标线可见长度不足 → 取终点
    target = (float(line[idx, 0]), float(line[idx, 1]))
    info["target"] = target

    # ---- 5.5 前方曲率估计（曲率前馈信号） ----
    # 锚点→前瞻点连线与竖直方向的夹角 α ≈ 前方曲率×半弧长，
    # 只依赖路面形状（两点都在目标线上），与车当前横向位置无关；
    # 弯道一进前瞻窗 α 立即非零，用于控制律前馈提前打角。
    # 正 = 前方路向右拐（与 offset 符号一致）。目标点在锚点上方，dy > 0。
    info["curvature"] = float(np.arctan2(target[0] - line[0, 0],
                                         line[0, 1] - target[1]))

    # ---- 6. 归一化偏移 ----
    if STEER_MODE == "pure_pursuit":
        # 纯跟踪：横向偏差 / 前瞻距离（曲率映射，sinα ≈ Δx/L）
        offset = (target[0] - w / 2.0) / float(lookahead_px)
    else:
        # 现有 PD 语义：横向偏差 / 半画宽
        offset = (target[0] - w / 2.0) / (w / 2.0)
    offset = clamp(offset, -1.0, 1.0)

    lane_center_x = float(line[0, 0])  # 底部目标线 x（近处参考）
    return (offset, lane_center_x, lane_mask_bottom, junction,
            full_lane_mask, info)
