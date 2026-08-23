#!/usr/bin/env python3
"""
============================================================
  小车定位系统 — 基于 AprilTag 的实时视觉定位
============================================================

原理：
  1. 摄像头画面四角固定贴有 4 个 AprilTag，它们定义了场地坐标系
  2. 小车上贴有 1 个 AprilTag
  3. 用四个角落 tag 的图像坐标 → 世界坐标 计算透视变换矩阵
  4. 将小车 tag 的图像坐标通过透视变换映射到世界坐标

用法：
  python main.py

配置文件：
  所有参数在 config.py 中修改
"""

import math
import sys
import time
import numpy as np

# ============================================================
# 导入配置文件
# ============================================================
try:
    import config
except ImportError:
    print("[ERROR] 找不到 config.py，请确保它与 main.py 在同一目录下")
    sys.exit(1)

# ============================================================
# 导入 AprilTag 检测库（支持多种后端）
# ============================================================
_APRILTAG_BACKEND = None
_Detector = None

# 优先尝试 pupil-apriltags（Windows 上有预编译包，安装最简单）
try:
    from pupil_apriltags import Detector as PupilDetector
    _APRILTAG_BACKEND = "pupil_apriltags"
except ImportError:
    pass

# 其次尝试官方 apriltag 绑定
if _APRILTAG_BACKEND is None:
    try:
        from apriltag import apriltag as ApriltagDetector
        _APRILTAG_BACKEND = "apriltag"
    except ImportError:
        pass

if _APRILTAG_BACKEND is None:
    print("=" * 60)
    print("[ERROR] 未找到 AprilTag Python 包！")
    print("请安装以下任一包：")
    print()
    print("  方式一（推荐，Windows 友好）：")
    print("    pip install pupil-apriltags")
    print()
    print("  方式二（官方绑定）：")
    print("    pip install apriltag")
    print("=" * 60)
    sys.exit(1)

# ============================================================
# OpenCV 导入
# ============================================================
try:
    import cv2
except ImportError:
    print("[ERROR] 未找到 OpenCV，请安装: pip install opencv-python")
    sys.exit(1)

# ============================================================
# UDP 发送器导入
# ============================================================
from udp_sender import UdpPositionSender, get_sender


# ============================================================
# AprilTag 检测器封装
# ============================================================
class TagDetector:
    """统一封装不同的 AprilTag Python 后端，对外提供一致的接口。"""

    def __init__(self, family="tagStandard41h12"):
        if _APRILTAG_BACKEND == "pupil_apriltags":
            self._detector = PupilDetector(families=family)
        else:
            self._detector = ApriltagDetector(family)

    def detect(self, gray_image):
        """检测图像中的 AprilTag，返回统一格式的列表。"""
        if _APRILTAG_BACKEND == "pupil_apriltags":
            raw = self._detector.detect(gray_image)
            return [
                {
                    "id": r.tag_id,
                    "center": (r.center[0], r.center[1]),
                    "corners": np.array(r.corners, dtype=np.int32),
                }
                for r in raw
            ]
        else:
            raw = self._detector.detect(gray_image)
            return [
                {
                    "id": d["id"],
                    "center": (d["center"][0], d["center"][1]),
                    "corners": np.array(d["corners"], dtype=np.int32),
                }
                for d in raw
            ]


# ============================================================
# 场地定位器
# ============================================================
class FieldLocalizer:
    """
    根据四个角落 tag 建立场地坐标系，
    然后将小车 tag 的图像坐标转换为场地世界坐标。
    """

    def __init__(self, corner_config, car_tag_id):
        """
        corner_config: CORNER_TAGS 字典，来自 config.py
        car_tag_id:    int, 小车 tag 的 ID
        """
        self.corner_config = corner_config
        self.car_tag_id = car_tag_id

    def get_corner_world_points(self):
        """提取四个角落的世界坐标（按 TL, TR, BR, BL 顺序）。"""
        names = ["top_left", "top_right", "bottom_right", "bottom_left"]
        pts = []
        for name in names:
            c = self.corner_config[name]
            pts.append([c["x"], c["y"]])
        return np.array(pts, dtype=np.float32), names

    def compute_transform(self, detections):
        """
        从检测结果中提取四个角落 tag 的图像坐标，
        计算图像 → 世界的透视变换矩阵。

        返回:
            M: 3x3 透视变换矩阵，若不足 4 个角落则返回 None
            info: 包含调试信息的字典
        """
        det_map = {d["id"]: d for d in detections}
        world_pts, corner_names = self.get_corner_world_points()

        image_pts = []
        matched_world = []
        matched_names = []

        for i, name in enumerate(corner_names):
            tag_id = self.corner_config[name]["id"]
            if tag_id in det_map:
                image_pts.append(det_map[tag_id]["center"])
                matched_world.append(world_pts[i])
                matched_names.append(name)
            else:
                matched_names.append(None)

        info = {
            "found_corners": sum(1 for n in matched_names if n is not None),
            "matched_names": matched_names,
        }

        if len(image_pts) < 4:
            return None, info

        image_pts = np.array(image_pts, dtype=np.float32)
        matched_world = np.array(matched_world, dtype=np.float32)

        M = cv2.getPerspectiveTransform(image_pts, matched_world)

        return M, info

    def localize_car(self, car_detection, M):
        """将小车 tag 的图像坐标映射为场地世界坐标。返回 (x, y, yaw_deg)。

        yaw_deg 为偏航角（度），基于 tag 的四个角点计算：
        tag 上边（corners[0]→corners[1]）中点相对 tag 中心的方向即为车头朝向。
        """
        # 变换中心点 → 世界坐标 (x, y)
        pt = np.array(
            [[[car_detection["center"][0], car_detection["center"][1]]]],
            dtype=np.float32,
        )
        world_pt = cv2.perspectiveTransform(pt, M)
        x, y = world_pt[0][0]

        # 变换四个角点 → 计算偏航角
        corners_img = car_detection["corners"].astype(np.float32).reshape(-1, 1, 2)
        corners_world = cv2.perspectiveTransform(corners_img, M)
        cw = corners_world.reshape(4, 2)  # cw[0..3] 按 CCW 排列

        # 上边中点 (tag 上边朝车头) → 中心 → 得到车头方向向量
        top_mid = (cw[0] + cw[1]) / 2.0
        center_world = cw.mean(axis=0)
        heading = top_mid - center_world

        yaw_rad = math.atan2(heading[1], heading[0])
        yaw_deg = math.degrees(yaw_rad)

        return x, y, yaw_deg


# ============================================================
# 可视化
# ============================================================
def draw_overlay(frame, detections, localizer, M, info, fps, sender):
    """
    在画面帧上绘制检测结果和定位信息。

    颜色约定：
      绿色  → 角落 tag（场地参照点）
      红色  → 小车 tag（定位目标）
      蓝色  → 场地边界
    """
    h, w = frame.shape[:2]
    det_map = {d["id"]: d for d in detections}
    corner_tag_ids = {cfg["id"] for cfg in localizer.corner_config.values()}

    # --- 1. 绘制所有检测到的 tag ---
    for det in detections:
        tid = det["id"]
        corners = det["corners"]
        center = (int(det["center"][0]), int(det["center"][1]))

        if tid in corner_tag_ids:
            color = (0, 255, 0)  # 绿色：角落 tag
            # 找到对应的角落名称
            label_parts = [f"ID:{tid}"]
            for name, cfg in localizer.corner_config.items():
                if cfg["id"] == tid:
                    label_parts.append(name)
                    break
            label = " ".join(label_parts)
        elif tid == localizer.car_tag_id:
            color = (0, 0, 255)  # 红色：小车 tag
            label = f"CAR ID:{tid}"
        else:
            color = (255, 255, 0)  # 青色：其他 tag
            label = f"ID:{tid}"

        # 绘制四边形轮廓
        cv2.polylines(frame, [corners], True, color, 2)
        # 绘制四个角点
        for pt in corners:
            cv2.circle(frame, tuple(pt), 3, color, -1)
        # 绘制中心点
        cv2.circle(frame, center, 5, color, -1)
        cv2.putText(
            frame, label, (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2,
        )

    # --- 2. 绘制场地边界（连接四个角落 tag） ---
    if info["found_corners"] >= 4:
        corner_order = ["top_left", "top_right", "bottom_right", "bottom_left"]
        pts = []
        for name in corner_order:
            tid = localizer.corner_config[name]["id"]
            if tid in det_map:
                pts.append(det_map[tid]["center"])
        if len(pts) == 4:
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (255, 150, 0), 2, cv2.LINE_AA)

    # --- 3. 绘制场地坐标网格（当变换矩阵可用时） ---
    if M is not None and config.SHOW_GRID and info["found_corners"] >= 4:
        draw_grid(frame, M, localizer)

    # --- 4. 小车定位信息 ---
    if M is not None and localizer.car_tag_id in det_map:
        car_pos = localizer.localize_car(
            det_map[localizer.car_tag_id], M
        )
        car_center = det_map[localizer.car_tag_id]["center"]
        car_x, car_y = int(car_center[0]), int(car_center[1])

        # 在画面上显示坐标
        pos_text = f"Car: ({car_pos[0]:.0f}, {car_pos[1]:.0f}) mm"
        cv2.putText(
            frame, pos_text, (car_x + 15, car_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
        )

        # 从中心画十字线到坐标轴
        cv2.line(frame, (car_x, car_y), (car_x, h - 1), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (car_x, car_y), (0, car_y), (0, 0, 255), 1, cv2.LINE_AA)

    # --- 5. 状态信息面板 ---
    draw_status_panel(frame, info, fps, localizer, det_map, M, sender)

    return frame


def draw_grid(frame, M, localizer):
    """在场地内绘制坐标网格线。"""
    h, w = frame.shape[:2]
    world_pts, _ = localizer.get_corner_world_points()
    x_min, x_max = world_pts[:, 0].min(), world_pts[:, 0].max()
    y_min, y_max = world_pts[:, 1].min(), world_pts[:, 1].max()
    spacing = config.GRID_SPACING

    # 从世界坐标反算到图像坐标
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return

    def world_to_image(wx, wy):
        pt = np.array([[[wx, wy]]], dtype=np.float32)
        ipt = cv2.perspectiveTransform(pt, M_inv)[0][0]
        return int(ipt[0]), int(ipt[1])

    # 垂直线
    x = x_min
    while x <= x_max + 1e-6:
        p1 = world_to_image(x, y_min)
        p2 = world_to_image(x, y_max)
        if 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h:
            cv2.line(frame, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)
        x += spacing

    # 水平线
    y = y_min
    while y <= y_max + 1e-6:
        p1 = world_to_image(x_min, y)
        p2 = world_to_image(x_max, y)
        if 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h:
            cv2.line(frame, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)
        y += spacing


def draw_status_panel(frame, info, fps, localizer, det_map, M, sender):
    """在画面左上角绘制状态信息面板。"""
    h, w = frame.shape[:2]

    lines = [
        f"FPS: {fps:.1f}",
        f"Corners: {info['found_corners']}/4",
        f"Car: {'Detected' if localizer.car_tag_id in det_map else 'Missing'}",
    ]

    # 如果有小车位置，追加坐标
    if M is not None and localizer.car_tag_id in det_map:
        car_pos = localizer.localize_car(det_map[localizer.car_tag_id], M)
        lines.append(f"X: {car_pos[0]:.1f} mm")
        lines.append(f"Y: {car_pos[1]:.1f} mm")

    # UDP 状态
    udp_status = "ON" if sender.enabled else "OFF"
    lines.append(f"UDP: {udp_status} -> {sender.target}")

    # 缺失的角落
    missing = []
    corner_order = ["top_left", "top_right", "bottom_right", "bottom_left"]
    for i, name in enumerate(corner_order):
        if info["matched_names"][i] is None:
            tid = localizer.corner_config[name]["id"]
            missing.append(f"ID:{tid}({name})")
    if missing:
        lines.append("Missing: " + ", ".join(missing))

    # 绘制半透明背景
    panel_h = 22 * len(lines) + 16
    panel_w = 280
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (panel_w + 5, panel_h + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # 绘制文字
    for i, line in enumerate(lines):
        y = 30 + i * 22
        cv2.putText(
            frame, line, (16, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 60)
    print("  小车定位系统 — Car Localization via AprilTags")
    print("=" * 60)
    print(f"  AprilTag 后端: {_APRILTAG_BACKEND}")
    print(f"  Tag 家族:     {config.TAG_FAMILY}")
    print(f"  角落 Tag:     { {cfg['id']: name for name, cfg in config.CORNER_TAGS.items()} }")
    print(f"  小车 Tag ID:  {config.CAR_TAG_ID}")
    print(f"  场地尺寸:     "
          f"{config.CORNER_TAGS['top_right']['x']:.0f} x "
          f"{config.CORNER_TAGS['bottom_left']['y']:.0f} mm")
    print(f"  摄像头索引:   {config.CAMERA_INDEX}")
    udp_state = "启用" if config.UDP_ENABLED else "禁用"
    print(f"  UDP 发送:     {udp_state} -> {config.UDP_TARGET_IP}:{config.UDP_TARGET_PORT}")
    print("=" * 60)
    print()
    print("按键说明:")
    print("  q / ESC → 退出")
    print("  s       → 保存当前画面截图")
    print("  d       → 打印当前检测到的所有 tag ID（用于调试）")
    print("  u       → 切换 UDP 发送 开/关")
    print()

    # --- 初始化检测器 ---
    print("[INFO] 正在初始化 AprilTag 检测器...")
    try:
        detector = TagDetector(family=config.TAG_FAMILY)
    except Exception as e:
        print(f"[ERROR] 检测器初始化失败: {e}")
        sys.exit(1)
    print("[INFO] 检测器初始化完成")

    # --- 初始化定位器 ---
    localizer = FieldLocalizer(config.CORNER_TAGS, config.CAR_TAG_ID)

    # --- 初始化 UDP 发送器 ---
    sender = get_sender()
    sender.configure(
        target_ip=config.UDP_TARGET_IP,
        target_port=config.UDP_TARGET_PORT,
    )
    sender.set_enabled(config.UDP_ENABLED)
    last_udp_send = 0  # 控制 UDP 发送频率

    # --- 打开摄像头 ---
    print(f"[INFO] 正在打开摄像头 (index={config.CAMERA_INDEX})...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 {config.CAMERA_INDEX}")
        print("  尝试修改 config.py 中的 CAMERA_INDEX")
        sys.exit(1)

    # 设置分辨率和帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] 摄像头分辨率: {actual_w:.0f} x {actual_h:.0f} @ {actual_fps:.1f} FPS")

    # --- 创建可缩放显示窗口 ---
    cv2.namedWindow("Car Localization System", cv2.WINDOW_NORMAL)

    # --- 主循环 ---
    fps = 0.0
    frame_count = 0
    last_print = 0  # 控制控制台输出频率

    print("[INFO] 开始检测，按 q 退出...")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 读取帧失败，重试中...")
            time.sleep(0.1)
            continue

        # 转灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测 AprilTag
        t0 = cv2.getTickCount()
        detections = detector.detect(gray)
        t1 = cv2.getTickCount()

        # 计算 FPS（指数移动平均）
        detect_ms = (t1 - t0) / cv2.getTickFrequency() * 1000
        if detect_ms > 0:
            fps = 0.85 * fps + 0.15 * (1000.0 / detect_ms)

        # 计算变换矩阵
        M, info = localizer.compute_transform(detections)

        # 控制台输出（每 500ms 输出一次）
        now = time.time()
        det_map = {d["id"]: d for d in detections}
        if now - last_print > 0.5 and M is not None and config.CAR_TAG_ID in det_map:
            car_pos = localizer.localize_car(det_map[config.CAR_TAG_ID], M)
            print(
                f"\r[Car] X={car_pos[0]:7.1f} mm  "
                f"Y={car_pos[1]:7.1f} mm  "
                f"Yaw={car_pos[2]:6.1f}°  "
                f"| Corners:{info['found_corners']}/4  "
                f"| FPS:{fps:5.1f}  "
                f"| UDP:{'ON' if sender.enabled else 'OFF'}  ",
                end="", flush=True,
            )
            last_print = now

        # UDP 发送小车坐标（按配置的间隔发送）
        if (M is not None and config.CAR_TAG_ID in det_map
                and now - last_udp_send >= config.UDP_SEND_INTERVAL):
            car_pos = localizer.localize_car(det_map[config.CAR_TAG_ID], M)
            sender.send(
                x_mm=float(car_pos[0]),
                y_mm=float(car_pos[1]),
                yaw_deg=float(car_pos[2]),
            )
            last_udp_send = now

        # 可视化
        frame = draw_overlay(frame, detections, localizer, M, info, fps, sender)

        # 显示画面（缩放到适合屏幕的大小）
        display = frame
        if config.DISPLAY_SCALE != 1.0:
            new_w = int(frame.shape[1] * config.DISPLAY_SCALE)
            new_h = int(frame.shape[0] * config.DISPLAY_SCALE)
            display = cv2.resize(frame, (new_w, new_h))
        cv2.imshow("Car Localization System", display)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q 或 ESC
            print("\n[INFO] 用户退出")
            break
        elif key == ord("s"):
            filename = time.strftime("screenshot_%Y%m%d_%H%M%S.png")
            cv2.imwrite(filename, frame)
            print(f"\n[INFO] 截图已保存: {filename}")
        elif key == ord("d"):
            # 调试：打印所有检测到的 tag
            print("\n[DEBUG] --- 当前检测到的所有 tag ---")
            if detections:
                for d in detections:
                    print(f"  ID={d['id']:2d}  center=({d['center'][0]:6.1f}, {d['center'][1]:6.1f})")
            else:
                print("  (未检测到任何 tag)")
            print("[DEBUG] ---")
        elif key == ord("u"):
            # 切换 UDP 发送
            sender.set_enabled(not sender.enabled)
            state = "启用" if sender.enabled else "禁用"
            print(f"\n[INFO] UDP 发送已{state}")

        frame_count += 1

    # --- 清理 ---
    sender.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 程序结束")


if __name__ == "__main__":
    main()
