"""
tracking —— 语义分割循迹包

提供基于 RKNN 语义分割模型的视觉循迹功能。

主要类:
    SegmentTracker : 语义分割循迹跟踪器

子模块:
    config  循迹参数配置
    maze    迷宫法边线提取（八邻域行走器）
    lane    沿左边线固定偏移目标线 + 前瞻点偏移
    model   RKNN 模型推理
    viz     循迹可视化
    cli     命令行入口

使用方式:
    from tracking import SegmentTracker
    tracker = SegmentTracker()
    offset, viz = tracker.process_frame(frame_bgr)
"""

from .tracker import SegmentTracker
from . import config
from . import lane
from . import maze
from . import model as inference
from . import viz

__all__ = ["SegmentTracker", "config", "lane", "inference", "viz"]
