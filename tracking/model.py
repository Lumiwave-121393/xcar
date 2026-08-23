"""
model.py —— RKNN 语义分割模型推理

提供模型加载、预处理、后处理和线程池推理函数。
与 ppseg_infer.py 的处理逻辑保持一致。
"""

import cv2
import numpy as np
import os
import glob
import logging

from .config import MODEL_IMG_SIZE, SEG_COLORS

logger = logging.getLogger("tracking.model")

# RKNN 库（仅目标设备可用，PC 开发环境不可用也不影响）
try:
    from rknnlite.api import RKNNLite
except ImportError:
    RKNNLite = None
    logger.debug("RKNNLite 不可用（仅在目标设备上可用）")


# ====================================================================
# 预处理
# ====================================================================

def preprocess(img_bgr):
    """
    与 ppseg_infer.py 一致的预处理：
      BGR → RGB → Resize(512×512) → Add batch dim
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, MODEL_IMG_SIZE,
                              interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(img_resized, axis=0)


# ====================================================================
# 后处理
# ====================================================================

def postprocess_to_segmap(outputs, original_hw):
    """
    与 ppseg_infer.py 一致的后处理：
      原始输出 → resize 回原图尺寸 → (H, W) 类别索引图
    """
    seg_out = outputs[0] if isinstance(outputs, list) else outputs
    seg_map = seg_out[0]

    # 如果输出是多通道概率图，取 argmax
    if seg_map.ndim == 3:
        seg_map = np.argmax(seg_map, axis=0).astype(np.uint8)

    orig_h, orig_w = original_hw
    seg_map = cv2.resize(
        seg_map.astype(np.float32),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.uint8)

    return seg_map


# ====================================================================
# 推理函数（供 rknnPoolExecutor 调用）
# ====================================================================

def infer_func(rknn_lite, img_bgr):
    """
    线程池推理函数。

    参数:
        rknn_lite: RKNNLite 实例
        img_bgr:   BGR 图像 (H, W, 3)

    返回:
        seg_map: (H, W) 类别索引图
        flag:    成功标志
    """
    try:
        h, w = img_bgr.shape[:2]
        input_data = preprocess(img_bgr)
        outputs = rknn_lite.inference(inputs=[input_data])

        seg_map = postprocess_to_segmap(outputs, (h, w))

        return seg_map, True
    except Exception as e:
        logger.error(f"推理失败: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8), False


# ====================================================================
# 模型文件搜索
# ====================================================================

def find_model_dir():
    """自动在常见位置搜索模型目录"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    candidates = [
        os.path.join(parent_dir, "seg_python", "model"),
        os.path.join(parent_dir, "infer_wrap", "base", "model"),
        os.path.join(parent_dir, "model"),
    ]
    for d in candidates:
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.rknn")):
            logger.info("自动发现模型目录: %s", d)
            return d
    fallback = candidates[0]
    logger.warning("未找到模型目录，使用默认: %s", fallback)
    return fallback


def resolve_model_path(model_dir):
    """找到 .rknn 模型文件"""
    model_dir = os.path.abspath(model_dir)
    model_files = glob.glob(os.path.join(model_dir, "*.rknn"))
    if not model_files:
        raise FileNotFoundError(
            f"在 {model_dir} 中未找到 .rknn 模型文件。\n"
            f"请确认模型文件存在，或通过 model_dir 参数指定正确路径。"
        )
    return model_files[0]
