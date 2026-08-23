"""
detector.py —— YOLO 目标检测封装
=================================

基于 infer_wrap/base 中的 rknnPoolExecutor + func.py 后处理逻辑，
提供返回结构化检测结果的检测器。

与 infer_wrap 的 InferWrap 不同：本模块返回 (boxes, classes, scores) 而非仅绘制图像。

用法:
    from obstacle_detection import ObstacleDetector

    detector = ObstacleDetector()
    boxes, classes, scores, annotated_img = detector.detect(frame_bgr)
    # boxes:  np.array (N, 4)  [x1, y1, x2, y2]
    # classes: np.array (N,)   类别索引
    # scores:  np.array (N,)   置信度
"""

import os
import sys
import time
import threading
import logging

import cv2
import numpy as np

# 确保 infer_wrap 可在路径中找到
_detector_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_detector_dir)
for _rp in [
    os.path.join(_project_root, "infer_wrap", "base"),
    os.path.join(_project_root, "infer_wrap"),
]:
    _rp = os.path.abspath(_rp)
    if os.path.isdir(_rp) and _rp not in sys.path:
        sys.path.insert(0, _rp)

from .config import (
    DETECT_CONFIDENCE, DETECT_IMG_SIZE, CLASS_NAMES,
    DETECT_NPU_CORE_OFFSET,
)

logger = logging.getLogger("obstacle_detection.detector")

# ====================================================================
# YOLO 后处理 —— 支持两种模型输出格式
# ====================================================================
#
# 格式 A（旧模型，多尺度输出）:
#   6 个张量: [pos_s1, cls_s1, pos_s2, cls_s2, pos_s3, cls_s3]
#   每个 pos 形状: (1, 64, H, W)  需要 DFL 解码
#   每个 cls 形状: (1, NC, H, W)
#
# 格式 B（新模型，展平输出）:
#   1 个张量: (1, N, K)
#   N = 80² + 40² + 20² = 8400（3 个尺度展平拼接）
#   K = 4 (cx,cy,w,h) + num_classes
#   bbox 已解码为模型输入空间像素坐标
# ====================================================================

OBJ_THRESH = DETECT_CONFIDENCE
NMS_THRESH = 0.45

# ── 模型输出格式自动检测结果（首次推理时确定） ──
_model_format = None          # "flat" | "multi_scale"
_num_classes_detected = None  # 从输出 shape 推断的类别数


def _detect_model_format(outputs):
    """
    检测模型输出格式。

    返回:
        "flat":        单张量格式 (1, N, K)，bbox 已解码
        "multi_scale": 多尺度格式（3或6个张量），需要 DFL 解码
    """
    global _model_format, _num_classes_detected

    if _model_format is not None:
        return _model_format

    n = len(outputs)

    # 单输出 → 展平格式
    if n == 1:
        _model_format = "flat"
        arr = np.asarray(outputs[0])
        # arr shape: (1, N, K) 或 (N, K)
        if arr.ndim == 3:
            k = arr.shape[-1]  # last dim
        elif arr.ndim == 2:
            k = arr.shape[-1]
        else:
            k = 0
        _num_classes_detected = k - 4  # K = 4(bbox) + num_classes
        logger.info(
            "检测到 展平输出格式: 1×%d×%d, num_classes=%d (推断)",
            arr.shape[-2] if arr.ndim >= 2 else 0, k, _num_classes_detected,
        )
        return "flat"

    # 多输出 → 多尺度格式
    _model_format = "multi_scale"
    _num_classes_detected = len(CLASS_NAMES)
    logger.info("检测到 多尺度输出格式: %d 个张量", n)
    return "multi_scale"


# ====================================================================
# 格式 A（旧模型）: 多尺度 DFL 解码
# ====================================================================

def _ensure_4d(tensor):
    """确保张量是 4D (N,C,H,W)"""
    if tensor.ndim == 3:
        return np.expand_dims(tensor, 0)
    return tensor


def _dfl(position):
    """DFL 解码"""
    position = _ensure_4d(position)
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num

    y = position.reshape(n, p_num, mc, h, w)
    exp_y = np.exp(y - np.max(y, axis=2, keepdims=True))
    y_softmax = exp_y / np.sum(exp_y, axis=2, keepdims=True)

    acc_metrix = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    y = np.sum(y_softmax * acc_metrix, axis=2)
    return y


def _box_process(position, size_im=DETECT_IMG_SIZE):
    """多尺度格式的边界框解码"""
    position = _ensure_4d(position)
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w).astype(np.float32)
    row = row.reshape(1, 1, grid_h, grid_w).astype(np.float32)
    grid = np.concatenate((col, row), axis=1)

    stride = np.array([size_im[1] // grid_h, size_im[0] // grid_w],
                      dtype=np.float32).reshape(1, 2, 1, 1)

    position = _dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def _post_process_multi_scale(input_data, img_shape):
    """多尺度格式后处理（旧模型兼容）"""
    boxes, scores, classes_conf = [], [], []
    default_branch = 3
    pair_per_branch = len(input_data) // default_branch

    for i in range(default_branch):
        boxes.append(_box_process(input_data[pair_per_branch * i], img_shape))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        _cls_tensor = _ensure_4d(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(_cls_tensor[:, :1, :, :], dtype=np.float32))

    def _sp_flatten(_in):
        _in = _ensure_4d(_in)
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [_sp_flatten(_v) for _v in boxes]
    classes_conf = [_sp_flatten(_v) for _v in classes_conf]
    scores = [_sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    boxes, classes, scores = _filter_boxes(boxes, scores, classes_conf)
    if boxes.size == 0:
        return None, None, None

    return _apply_nms(boxes, classes, scores)


# ====================================================================
# 格式 B（新模型）: 展平格式 (1, 8400, K)
# ====================================================================

def _post_process_flat(input_data, img_shape):
    """
    展平格式后处理。

    输入: 单个张量 (1, N, K) 或 (N, K)
          K = 4 (cx,cy,w,h) + num_classes
          bbox 坐标在模型输入空间 (DETECT_IMG_SIZE)
    """
    arr = np.asarray(input_data[0])

    # 去除 batch 维（如有）
    if arr.ndim == 3:
        arr = arr[0]  # (1, N, K) → (N, K)

    if arr.ndim != 2:
        logger.error(f"展平格式要求 2D 或 3D 张量, 实际 shape={arr.shape}")
        return None, None, None

    n_proposals = arr.shape[0]
    k_features = arr.shape[1]
    num_classes = k_features - 4

    if num_classes < 1:
        logger.error(f"K={k_features} 太小, 至少有 4 个 bbox + 1 个 class, 实际: {arr.shape}")
        return None, None, None

    logger.debug(
        "展平后处理: proposals=%d, features=%d, num_classes=%d",
        n_proposals, k_features, num_classes,
    )

    # 拆分 bbox 和 class scores
    bboxes = arr[:, :4].copy()          # (N, 4)  cx, cy, w, h
    class_scores = arr[:, 4:].copy()    # (N, num_classes)

    # ── 对 class scores 做 sigmoid（如果模型输出未激活） ──
    # 判断方式: 如果大量值落在 [0, 1] 之外，说明是 raw logits，需要 sigmoid
    _cls_min = class_scores.min()
    _cls_max = class_scores.max()
    if _cls_min < -0.5 or _cls_max > 1.5:
        # 值域不在 [0,1] 范围，认为是 raw logits，应用 sigmoid
        class_scores = 1.0 / (1.0 + np.exp(-np.clip(class_scores, -50, 50)))
        logger.debug("class_scores 已应用 sigmoid (raw logits 检测到)")
    # 如果值域在 [0,1] 内，假设已包含激活，不做处理

    # ── 置信度过滤 ──
    max_scores = class_scores.max(axis=1)   # (N,)
    class_ids = class_scores.argmax(axis=1) # (N,)

    mask = max_scores >= OBJ_THRESH
    if not mask.any():
        return None, None, None

    bboxes = bboxes[mask]
    class_ids = class_ids[mask]
    max_scores = max_scores[mask]

    # ── cx,cy,w,h → x1,y1,x2,y2（在模型输入空间） ──
    cx, cy, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    # ── 缩放到原始图像尺寸 ──
    orig_h, orig_w = img_shape
    scale_x = orig_w / DETECT_IMG_SIZE[0]
    scale_y = orig_h / DETECT_IMG_SIZE[1]

    x1 *= scale_x; x2 *= scale_x
    y1 *= scale_y; y2 *= scale_y

    # 裁剪到图像边界内
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # 过滤无效框 (x2<=x1 或 y2<=y1)
    valid = (x2 > x1) & (y2 > y1)
    if not valid.any():
        return None, None, None
    xyxy = xyxy[valid]
    class_ids = class_ids[valid]
    max_scores = max_scores[valid]

    if len(xyxy) == 0:
        return None, None, None

    # ── NMS ──
    return _apply_nms(xyxy, class_ids, max_scores)


# ====================================================================
# 通用后处理：NMS + 过滤
# ====================================================================

def _filter_boxes(boxes, box_confidences, box_class_probs):
    """过滤低置信度框（多尺度格式用）"""
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def _nms_boxes(boxes, scores):
    """NMS 非极大值抑制（x1,y1,x2,y2 格式输入）"""
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]

    return np.array(keep)


def _apply_nms(boxes, classes, scores):
    """按类别分别执行 NMS，返回 (boxes, classes, scores)"""
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)[0]
        b = boxes[inds]
        c_cls = classes[inds]
        s = scores[inds]
        keep = _nms_boxes(b, s)
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c_cls[keep])
            nscores.append(s[keep])

    if not nboxes:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


# ====================================================================
# 统一后处理入口
# ====================================================================

def _post_process(input_data, img_shape):
    """后处理入口：自动检测格式并调用对应处理函数"""
    fmt = _detect_model_format(input_data)

    if fmt == "flat":
        return _post_process_flat(input_data, img_shape)
    else:
        return _post_process_multi_scale(input_data, img_shape)


# ====================================================================
# 检测推理函数（供 rknnPoolExecutor 调用）
# ====================================================================

def _detect_func(rknn_lite, img_bgr):
    """
    线程池推理函数 —— 执行检测并返回结构化结果。

    参数:
        rknn_lite: RKNNLite 实例
        img_bgr:   BGR 图像 (H, W, 3)

    返回:
        dict: {
            "boxes":   np.array (N,4) 或 None,
            "classes": np.array (N,)  或 None,
            "scores":  np.array (N,)  或 None,
            "img":     绘制了检测框的图像,
            "ok":      bool,
        }
    """
    try:
        orig_h, orig_w = img_bgr.shape[:2]

        # 预处理
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, DETECT_IMG_SIZE)
        img_input = np.expand_dims(img_resized, 0)

        # 推理
        outputs = rknn_lite.inference(inputs=[img_input])

        # 后处理
        boxes, classes, scores = _post_process(outputs, (orig_h, orig_w))

        # 绘制
        img_annotated = img_bgr.copy()
        if boxes is not None:
            _draw_boxes(img_annotated, boxes, classes, scores)

        return {
            "boxes": boxes,
            "classes": classes,
            "scores": scores,
            "img": img_annotated,
            "ok": True,
        }
    except Exception as e:
        logger.error(f"检测推理失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            "boxes": None, "classes": None, "scores": None,
            "img": img_bgr, "ok": False,
        }


def _draw_boxes(img, boxes, classes, scores):
    """在图像上绘制检测框"""
    from .config import BOX_COLORS

    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        cls_id = int(cls)
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"

        # 颜色映射（按语义匹配，兼容不同类别数）
        if cls_name in ("human", "pedestrian", "person"):
            color = BOX_COLORS["pedestrian"]
        elif cls_name in ("car", "vehicle", "truck", "bus", "bicycle", "motorbike"):
            color = BOX_COLORS["vehicle"]
        elif cls_name in ("sign", "stop", "light", "zbx"):
            color = BOX_COLORS["sign"]
        else:
            color = BOX_COLORS["unknown"]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {score:.2f}"
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# ====================================================================
# ObstacleDetector —— 目标检测器（后台线程架构）
# ====================================================================

class ObstacleDetector:
    """
    YOLO 目标检测器（后台线程架构）。

    架构:
        主线程                   后台 daemon 线程
        ┌──────────┐            ┌──────────────────────┐
        │ submit() │──覆盖──→   │ _latest_frame (单槽) │
        └──────────┘            │         ↓            │
                                │    _detect_func()    │
        ┌──────────┐            │    (阻塞推理 NPU)     │
        │ poll()   │←──读取──   │         ↓            │
        └──────────┘            │ _latest_result (单槽) │
                                └──────────────────────┘

    优势:
      - submit() 永远只保留最新一帧，自动丢弃中间过期帧
      - poll() 非阻塞，主循环不卡顿
      - 多 NPU 核并行推理，延迟 = 单次推理时间（≈300ms@3核）
      - 无队列堆积问题

    用法（推荐 —— 非阻塞）:
        detector = ObstacleDetector(TPEs=2)
        for frame in video_stream:
            detector.submit(frame)          # 每帧提交最新画面
            result = detector.poll()        # 每帧检查结果
            if result and result["ok"]:
                # 使用 result["boxes"], result["classes"], result["scores"]
                pass

    用法（向后兼容 —— 阻塞）:
        detector = ObstacleDetector()
        result = detector.detect(frame)     # 阻塞等待，直到结果返回
    """

    def __init__(self, model_dir=None, TPEs=1):
        """
        初始化检测器。

        参数:
            model_dir: 模型目录路径（None 则自动搜索）
            TPEs:      NPU 核心数（1~3）。每个核心加载独立 RKNN 实例，
                       后台线程循环使用，实现多核并行推理。
        """
        # 动态导入 RKNN 相关（仅在目标设备可用）
        try:
            from rknnlite.api import RKNNLite
            self._RKNNLite = RKNNLite
        except ImportError:
            raise RuntimeError(
                "RKNNLite 不可用。ObstacleDetector 仅支持在 Rockchip NPU 设备上运行。"
            )

        # 模型路径
        self._model_path = self._find_model(model_dir)
        self._TPEs = max(1, min(TPEs, 3))  # 限制在 1~3

        # 初始化 RKNN 实例
        self._rknn_list = self._init_rknn()

        # 线程池（并行推理）
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=self._TPEs)

        # 单槽帧缓冲区：主线程写，后台线程读
        self._frame = None
        self._frame_lock = threading.Lock()

        # 单槽结果缓冲区：后台线程写，主线程读
        self._result = None
        self._result_lock = threading.Lock()

        # 启动后台检测线程
        self._running = True
        self._worker = threading.Thread(target=self._run, daemon=True,
                                        name="detect-worker")
        self._worker.start()

        self._frame_count = 0

        logger.info("ObstacleDetector 初始化完成: model=%s TPEs=%d cores=%s",
                     os.path.basename(self._model_path), TPEs,
                     [(i + DETECT_NPU_CORE_OFFSET) % 3 for i in range(TPEs)])

    # ----------------------------------------------------------------
    # 模型搜索
    # ----------------------------------------------------------------

    def _find_model(self, model_dir):
        """搜索 .rknn 模型文件"""
        import glob
        if model_dir and os.path.isdir(model_dir):
            candidates = [model_dir]
        else:
            _base = os.path.join(_project_root, "infer_wrap", "base", "model")
            candidates = [
                _base,
                os.path.join(_project_root, "model"),
                os.path.join(_project_root, "seg_python", "model"),
            ]

        for d in candidates:
            if os.path.isdir(d):
                files = glob.glob(os.path.join(d, "*.rknn"))
                if files:
                    logger.info("自动发现检测模型: %s", files[0])
                    return files[0]

        raise FileNotFoundError(
            f"未找到 .rknn 检测模型文件。请将模型放入 infer_wrap/base/model/ 目录"
        )

    # ----------------------------------------------------------------
    # NPU 核心初始化
    # ----------------------------------------------------------------

    def _init_rknn(self):
        """初始化 TPEs 个 RKNN 实例，分配到指定的 NPU 核心"""
        rknn_list = []
        for i in range(self._TPEs):
            core_id = (i + DETECT_NPU_CORE_OFFSET) % 3
            rknn = self._RKNNLite()
            ret = rknn.load_rknn(self._model_path)
            if ret != 0:
                raise RuntimeError(f"加载检测 RKNN 模型失败: {ret}")

            core_masks = {
                0: self._RKNNLite.NPU_CORE_0,
                1: self._RKNNLite.NPU_CORE_1,
                2: self._RKNNLite.NPU_CORE_2,
            }
            ret = rknn.init_runtime(core_mask=core_masks.get(core_id,
                                     self._RKNNLite.NPU_CORE_0_1_2))
            if ret != 0:
                raise RuntimeError(f"初始化检测 RKNN runtime 失败 (core {core_id}): {ret}")
            rknn_list.append(rknn)
            logger.info(f"检测 NPU 核心 {core_id} 初始化完成")

        logger.info("检测模型已加载，使用 %d 个 NPU 核心（偏移 %d）",
                     self._TPEs, DETECT_NPU_CORE_OFFSET)
        return rknn_list

    # ----------------------------------------------------------------
    # 后台检测线程
    # ----------------------------------------------------------------

    def _run(self):
        """
        后台检测循环。

        1. 检查所有在飞的推理是否完成 → 完成就存入 _result
        2. 如果有空闲槽位，从单槽帧缓冲取最新帧提交推理
        3. 5ms 轮询间隔
        """
        idx = 0
        pending = []  # 当前在飞的 Future 列表

        while self._running:
            # ── 1. 收集已完成的推理结果 ──
            still_pending = []
            for fut in pending:
                if fut.done():
                    try:
                        result = fut.result()
                        if result and result.get("ok"):
                            with self._result_lock:
                                self._result = result  # 覆盖旧结果
                    except Exception:
                        pass
                else:
                    still_pending.append(fut)
            pending = still_pending

            # ── 2. 有空闲槽位时，取最新帧提交推理 ──
            if len(pending) < self._TPEs:
                with self._frame_lock:
                    frame = self._frame
                    if frame is not None:
                        self._frame = None  # 消费掉

                if frame is not None:
                    rknn = self._rknn_list[idx % self._TPEs]
                    idx += 1
                    fut = self._executor.submit(_detect_func, rknn, frame)
                    pending.append(fut)

            # 5ms 轮询（避免忙等，同时保证低延迟）
            time.sleep(0.005)

    # ----------------------------------------------------------------
    # 公开接口
    # ----------------------------------------------------------------

    def submit(self, frame_bgr):
        """
        提交一帧用于检测（非阻塞，覆盖式）。

        将 frame_bgr 深拷贝后放入单槽帧缓冲区。如果缓冲区中已有未被
        后台线程取走的旧帧，直接覆盖。这保证了后台线程始终处理最新帧。

        调用频率建议：每帧调用。如果帧率 > 推理速率，中间的过期帧会
        自动被跳过，不会造成队列堆积。
        """
        with self._frame_lock:
            self._frame = frame_bgr.copy()

    def poll(self):
        """
        非阻塞获取最新的检测结果。

        返回:
            dict 或 None: 同 detect() 的返回值格式，无新结果时返回 None
        """
        with self._result_lock:
            result = self._result
            self._result = None
            return result

    def detect(self, frame_bgr):
        """
        阻塞式检测（向后兼容接口）。

        内部调用 submit() + 轮询 poll()，阻塞直到结果就绪或超时。

        新代码推荐使用 submit() + poll() 的非阻塞模式。
        """
        self._frame_count += 1
        self.submit(frame_bgr)

        deadline = time.time() + 5.0  # 5 秒超时
        while time.time() < deadline:
            result = self.poll()
            if result is not None:
                return result
            time.sleep(0.01)

        logger.warning("detect() 超时（5s），推理可能卡住")
        return {
            "boxes": None, "classes": None, "scores": None,
            "img": frame_bgr, "ok": False,
        }

    def detect_objects(self, frame_bgr):
        """
        便捷接口：直接返回检测到的对象列表。

        返回:
            list[dict]: [
                {"class": "human", "class_id": 2, "score": 0.85,
                 "bbox": [x1,y1,x2,y2], "center": (cx,cy)},
                ...
            ]
        """
        result = self.detect(frame_bgr)
        objects = []

        if result["boxes"] is not None:
            for box, cls, score in zip(result["boxes"],
                                        result["classes"],
                                        result["scores"]):
                cls_id = int(cls)
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
                x1, y1, x2, y2 = [float(v) for v in box]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                objects.append({
                    "class": cls_name,
                    "class_id": cls_id,
                    "score": float(score),
                    "bbox": [x1, y1, x2, y2],
                    "center": (cx, cy),
                })

        return objects

    # ----------------------------------------------------------------
    # 资源管理
    # ----------------------------------------------------------------

    def release(self):
        """释放 NPU 资源和后台线程"""
        self._running = False
        if self._worker.is_alive():
            self._worker.join(timeout=3.0)
        self._executor.shutdown(wait=False)
        for rknn in self._rknn_list:
            try:
                rknn.release()
            except Exception:
                pass
        logger.info("ObstacleDetector 资源已释放")
