"""
viz.py —— 循迹可视化

提供分割结果着色、半透明叠加、迷宫法边界链/中线/前瞻点绘制等功能。
"""

import cv2
import numpy as np

from .config import SEG_COLORS, clamp


def colorize_segmap(seg_map):
    """类别索引图 → 彩色可视化图"""
    h, w = seg_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(SEG_COLORS):
        mask = seg_map == cls_id
        colored[mask] = color
    return colored


def blend_overlay(orig_bgr, colored_rgb, alpha=0.5):
    """原图与分割结果半透明混合"""
    colored_bgr = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR)
    if orig_bgr.shape[:2] != colored_bgr.shape[:2]:
        colored_bgr = cv2.resize(
            colored_bgr,
            (orig_bgr.shape[1], orig_bgr.shape[0])
        )
    return cv2.addWeighted(orig_bgr, 1 - alpha, colored_bgr, alpha, 0)


def _draw_polyline(img, pts, color, thickness=2):
    """把点列画成折线（点数不足时退化为点）"""
    if pts is None:
        return
    arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
    if len(arr) >= 2:
        cv2.polylines(img, [arr], False, color, thickness)
    elif len(arr) == 1:
        cv2.circle(img, tuple(arr[0, 0]), 3, color, -1)


def draw_tracking_viz(frame_bgr, seg_map, offset, lane_cx,
                      blend_alpha=0.4, lane_class_id=1,
                      chains=None, midline=None, target=None,
                      anchor=None, curvature=0.0, hand="left",
                      lookahead_px=None):
    """
    绘制循迹可视化画面。

    要素：
      1. 原图 + 分割半透明叠加
      2. 迷宫法边界链：左边界（蓝）、右边界（红）
      3. 循迹目标线（黄，左边线+固定偏移）+ 锚点 + 前瞻点（圆点/箭头）
      4. 状态信息（偏移量、方向、曲率、手性）

    2026-08-16：已删除岔路检测文本（"! FORK / JUNCTION"）与右上角
    车道掩码小图；方向文字去掉箭头符号。舵机打角与 PID 分解由
    draw_control_hud() 绘制（见下方）。

    参数:
        frame_bgr:     原始 BGR 图像
        seg_map:       类别索引图
        offset:        归一化偏移 [-1, 1]
        lane_cx:       车道中心 x 坐标（锚点处）
        blend_alpha:   分割叠加透明度
        lane_class_id: 车道类别 ID
        chains:        (left_chain, right_chain) 边界链点列
        midline:       中线点列 (N, 2)
        target:        前瞻点 (x, y)
        anchor:        中线锚点 (x, y)
        curvature:     前方曲率 α（弧度，正=前方路向右拐，前馈用）
        hand:          "left" / "right"
        lookahead_px:  本帧使用的前瞻弧长（px，随速度自适应值；None 不显示，
                       显示在 Curv 行末，用于区分"弧长 vs 竖直跨度/链截断"）

    返回:
        overlay: 可视化图像 (BGR)
    """
    h, w = frame_bgr.shape[:2]

    # ---- 分割叠加 ----
    colored = colorize_segmap(seg_map)
    overlay = blend_overlay(frame_bgr, colored, blend_alpha)

    # ---- 边界链 ----
    if chains is not None:
        left_chain, right_chain = chains
        _draw_polyline(overlay, left_chain, (255, 140, 0), 2)   # 蓝：左边界
        _draw_polyline(overlay, right_chain, (0, 0, 255), 2)    # 红：右边界

    # ---- 中线 + 锚点 + 前瞻点 ----
    _draw_polyline(overlay, midline, (0, 255, 255), 2)          # 黄：中线

    if anchor is not None:
        ax, ay = int(anchor[0]), int(anchor[1])
        cv2.circle(overlay, (ax, ay), 5, (0, 255, 255), -1)     # 锚点（起点）

    if target is not None:
        tx, ty = int(target[0]), int(target[1])
        cv2.circle(overlay, (tx, ty), 7, (0, 165, 255), -1)     # 前瞻点
        cv2.circle(overlay, (tx, ty), 10, (0, 165, 255), 2)
        if anchor is not None:
            cv2.arrowedLine(overlay, (ax, ay), (tx, ty),
                            (0, 165, 255), 2, tipLength=0.2)

    # ---- 偏移指示（图像中心竖线 + 车道中心竖线） ----
    y_line = int(h * 0.92)
    cv2.line(overlay, (w // 2, y_line - 15), (w // 2, y_line + 15),
             (255, 255, 0), 2)
    cx = int(clamp(lane_cx, 0, w - 1))
    cv2.line(overlay, (cx, y_line - 15), (cx, y_line + 15),
             (0, 255, 255), 3)

    # ---- 状态信息 ----
    pct = offset * 100
    if abs(offset) < 0.03:
        direction = "STRAIGHT"
        color = (0, 255, 0)
    elif offset < 0:
        direction = "LEFT"
        color = (0, 200, 255)
    else:
        direction = "RIGHT"
        color = (0, 200, 255)

    cv2.putText(overlay, f"Offset: {pct:+.1f}%", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay, direction, (200, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    hand_text = f"Maze-Track: {hand.upper()}"
    cv2.putText(overlay, hand_text, (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    curv_text = f"Curv: {curvature:+.3f}"
    if lookahead_px is not None:
        curv_text += f"  Look:{lookahead_px:.0f}px"
    cv2.putText(overlay, curv_text, (15, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return overlay


def draw_control_hud(overlay, servo, pid=None, speed=None, origin=(15, 110)):
    """绘制舵机打角与 PID 分解 HUD（visual_tracking 用，2026-08-16）。

    显示三行：
      Servo: 1560us (+60)  Speed:1800      ← 舵机 PWM、相对中位的打角、电机速度
      PID  P:+45.2  D:+12.0  FF:+3.1       ← 三项各自贡献（µs）
      Gain Kp:95 Kd:27  e:+0.284           ← 当前增益与真实横向误差

    参数:
        overlay: 可视化图像（原地绘制）
        servo:   最终舵机 PWM（µs，来自 offset_to_servo）
        pid:     offset_to_servo(return_details=True) 返回的字典（None 时不画 PID 行）
        speed:   电机速度（None 时不显示）
        origin:  首行文字左上角坐标
    """
    x, y = origin
    steer = pid.get("steer", 0) if pid else 0

    # 打角颜色：直行绿、左转青、右转橙
    if abs(steer) < 5:
        scolor = (0, 255, 0)
    elif steer < 0:
        scolor = (255, 200, 0)
    else:
        scolor = (0, 200, 255)

    line1 = f"Servo: {servo}us ({steer:+d})"
    if speed is not None:
        line1 += f"  Speed:{int(speed)}"
    cv2.putText(overlay, line1, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, scolor, 2)

    if pid:
        line2 = (f"PID  P:{pid['p']:+.1f}  D:{pid['d']:+.1f}  "
                 f"FF:{pid['ff']:+.1f}")
        cv2.putText(overlay, line2, (x, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        line3 = (f"Gain Kp:{pid['kp']:.0f} Kd:{pid['kd']:.0f}  "
                 f"e:{pid['error']:+.3f}")
        cv2.putText(overlay, line3, (x, y + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return overlay
