"""
object_tracker.py —— 多目标追踪器
==================================

基于 IoU 的轻量级多目标追踪，用于：
  - 给每个检测到的目标分配唯一 ID
  - 追踪目标在帧间的运动轨迹
  - 判断行人是否正在移动（横向穿过车道）
  - 过滤误检（要求连续多帧命中）

用法:
    from obstacle_detection import ObjectTracker

    tracker = ObjectTracker()
    tracked = tracker.update(detections)  # detections 来自 ObstacleDetector.detect_objects()
    for obj in tracked:
        print(f"ID={obj['id']} {obj['class']} moving={obj['is_moving']}")
"""

import logging
import numpy as np

from .config import (
    TRACK_IOU_THRESHOLD, TRACK_MAX_AGE, TRACK_MIN_HITS,
    TRACK_VELOCITY_WINDOW, PEDESTRIAN_MOVE_THRESHOLD,
    CLASS_HUMAN,
)

logger = logging.getLogger("obstacle_detection.tracker")


def _compute_iou(box_a, box_b):
    """计算两个边界框的 IoU"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


class _Track:
    """单条追踪轨迹"""

    def __init__(self, track_id, bbox, cls_name, cls_id, score, frame_id):
        self.id = track_id
        self.cls_name = cls_name
        self.cls_id = cls_id
        self.bbox = list(bbox)          # [x1, y1, x2, y2]
        self.center = self._center(bbox)
        self.score = score

        self.hits = 1                   # 命中次数
        self.age = 0                    # 上次命中以来的帧数
        self.missed = 0                 # 连续丢失帧数
        self.first_seen = frame_id
        self.last_seen = frame_id

        # 速度计算
        self._centers = [self.center]   # 最近的 N 个中心点
        self.velocity = (0.0, 0.0)      # (vx, vy) 像素/帧
        self.is_moving = False

    @staticmethod
    def _center(bbox):
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def update(self, bbox, score, frame_id):
        """用新的检测更新轨迹"""
        self.bbox = list(bbox)
        self.center = self._center(bbox)
        self.score = score
        self.hits += 1
        self.age = 0
        self.missed = 0
        self.last_seen = frame_id

        # 维护中心点历史
        self._centers.append(self.center)
        if len(self._centers) > TRACK_VELOCITY_WINDOW:
            self._centers.pop(0)

        # 计算速度
        if len(self._centers) >= 2:
            c_old = self._centers[0]
            c_new = self._centers[-1]
            n = len(self._centers) - 1
            self.velocity = (
                (c_new[0] - c_old[0]) / max(n, 1),
                (c_new[1] - c_old[1]) / max(n, 1),
            )

        # 判断是否移动
        speed_px = (self.velocity[0]**2 + self.velocity[1]**2)**0.5
        self.is_moving = speed_px > PEDESTRIAN_MOVE_THRESHOLD

    def mark_missed(self):
        """标记一帧未匹配"""
        self.age += 1
        self.missed += 1

    def is_valid(self):
        """轨迹是否有效（达到最小命中次数）"""
        return self.hits >= TRACK_MIN_HITS

    def is_dead(self):
        """轨迹是否已死亡"""
        return self.missed > TRACK_MAX_AGE

    def to_dict(self):
        """转为字典输出"""
        return {
            "id": self.id,
            "class": self.cls_name,
            "class_id": self.cls_id,
            "bbox": self.bbox,
            "center": self.center,
            "score": self.score,
            "velocity": self.velocity,
            "is_moving": self.is_moving,
            "hits": self.hits,
            "age": self.age,
        }


class ObjectTracker:
    """
    基于 IoU 的轻量级多目标追踪器。

    每帧接收检测结果，与现有轨迹进行 IoU 匹配（2026-08-21 起仅同类别
    cls_id 可互配，防跨类串台）：
    - 匹配成功 → 更新轨迹
    - 未匹配的检测 → 创建新轨迹
    - 未匹配的轨迹 → 标记丢失，超时后删除

    用法:
        tracker = ObjectTracker()
        while True:
            detections = detector.detect_objects(frame)
            tracked = tracker.update(detections)
            for t in tracked:
                if t["is_moving"]:
                    print(f"行人 {t['id']} 正在移动!")
    """

    def __init__(self):
        self._tracks = []               # 活跃轨迹列表
        self._next_id = 0               # 下一个轨迹 ID
        self._frame_id = 0              # 当前帧编号

        # 统计
        self._total_tracks = 0

    # ----------------------------------------------------------------
    # 主接口
    # ----------------------------------------------------------------

    def update(self, detections):
        """
        用当前帧检测结果更新所有轨迹。

        参数:
            detections: list[dict], 来自 ObstacleDetector.detect_objects()

        返回:
            list[dict]: 当前活跃的有效轨迹（已确认的）
        """
        self._frame_id += 1

        if not detections:
            # 没有检测到，所有轨迹标记丢失
            for t in self._tracks:
                t.mark_missed()
            self._remove_dead()
            return self.get_active()

        # 提取检测框
        det_boxes = [d["bbox"] for d in detections]
        matched_det_indices = set()
        matched_track_indices = set()

        # IoU 匹配（2026-08-21：加类别一致性约束——cls_id 相同才允许匹配）
        # 此前只比 IoU：金币/车辆等不同类别框重叠时，金币轨迹可抢走车辆
        # 检测框（反之亦然），且轨迹类别永不更新 → 车辆永久显示为金币。
        for ti, track in enumerate(self._tracks):
            best_iou = TRACK_IOU_THRESHOLD
            best_di = -1

            for di, dbox in enumerate(det_boxes):
                if di in matched_det_indices:
                    continue
                # 类别一致性约束：不同类别不互配（防跨类串台）
                if track.cls_id != detections[di]["class_id"]:
                    continue
                iou = _compute_iou(track.bbox, dbox)
                if iou > best_iou:
                    best_iou = iou
                    best_di = di

            if best_di >= 0:
                # 匹配成功（同类别）
                d = detections[best_di]
                track.update(d["bbox"], d["score"], self._frame_id)
                matched_det_indices.add(best_di)
                matched_track_indices.add(ti)

        # 未匹配的轨迹标记丢失
        for ti, track in enumerate(self._tracks):
            if ti not in matched_track_indices:
                track.mark_missed()

        # 未匹配的检测创建新轨迹
        for di, d in enumerate(detections):
            if di not in matched_det_indices:
                new_track = _Track(
                    track_id=self._next_id,
                    bbox=d["bbox"],
                    cls_name=d["class"],
                    cls_id=d["class_id"],
                    score=d["score"],
                    frame_id=self._frame_id,
                )
                self._tracks.append(new_track)
                self._next_id += 1
                self._total_tracks += 1

        # 清理死亡轨迹
        self._remove_dead()

        return self.get_active()

    # ----------------------------------------------------------------
    # 查询接口
    # ----------------------------------------------------------------

    def get_active(self):
        """
        获取当前活跃的已确认轨迹。

        返回:
            list[dict]: 有效轨迹列表
        """
        return [t.to_dict() for t in self._tracks if t.is_valid() and not t.is_dead()]

    def get_all(self):
        """获取所有轨迹（包括未确认的）"""
        return [t.to_dict() for t in self._tracks if not t.is_dead()]

    def get_by_class(self, cls_name):
        """按类别名称筛选轨迹"""
        return [t.to_dict() for t in self._tracks
                if t.is_valid() and not t.is_dead() and t.cls_name == cls_name]

    def get_moving_pedestrians(self):
        """
        获取正在移动的行人。

        返回:
            list[dict]: 正在移动的行人轨迹
        """
        return [t.to_dict() for t in self._tracks
                if t.is_valid() and not t.is_dead()
                and t.cls_id == CLASS_HUMAN and t.is_moving]

    def get_stationary_vehicles(self):
        """
        获取静止的车辆（停在路边的）。

        返回:
            list[dict]: 静止车辆轨迹
        """
        return [t.to_dict() for t in self._tracks
                if t.is_valid() and not t.is_dead()
                and t.cls_name == "car" and not t.is_moving]

    def get_signs(self):
        """获取检测到的路牌"""
        from .config import SIGN_CLASSES
        return [t.to_dict() for t in self._tracks
                if t.is_valid() and not t.is_dead()
                and t.cls_id in SIGN_CLASSES]

    # ----------------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------------

    def _remove_dead(self):
        """移除死亡轨迹"""
        self._tracks = [t for t in self._tracks if not t.is_dead()]

    def reset(self):
        """重置追踪器"""
        self._tracks.clear()
        self._next_id = 0
        self._frame_id = 0
        logger.info("ObjectTracker 已重置")

    @property
    def frame_id(self):
        return self._frame_id
