"""
sign_handler.py —— 路牌检测→OCR→LLM 流水线
===========================================

当检测到路牌时：
  1. 从当前帧裁剪路牌区域
  2. 调用 PaddleOCR API 识别路牌文字
  3. 将 OCR 文本发送给 LLM 进行导航决策
  4. 返回决策结果（直行/右岔路）

整个流程在独立线程中异步执行，不阻塞主控制循环。

用法:
    from obstacle_detection import SignHandler

    handler = SignHandler()
    handler.start_processing(frame, sign_bbox)  # 异步启动
    ...
    if handler.is_ready():
        decision = handler.get_decision()
"""

import os
import sys
import logging
import threading
import time
import cv2

from .ocr_client import OCRClient
from .llm_client import LLMClient
from .config import (
    SIGN_CROP_MARGIN, SIGN_OCR_TIMEOUT, SIGN_LLM_TIMEOUT,
    SIGN_TEMP_DIR,
)

logger = logging.getLogger("obstacle_detection.sign_handler")


class SignHandler:
    """
    路牌处理流水线。

    检测到路牌 → 裁剪区域 → OCR 识别 → LLM 决策。

    使用异步模式：
      - start_processing() 在后台线程中执行 OCR+LLM
      - is_ready() 检查是否完成
      - get_decision() 获取结果

    也可使用同步模式：
      - process_sync() 阻塞等待完整结果
    """

    def __init__(self, ocr_client=None, llm_client=None):
        """
        初始化路牌处理器。

        参数:
            ocr_client: OCRClient 实例（None 则创建默认）
            llm_client: LLMClient 实例（None 则创建默认）
        """
        self.ocr = ocr_client or OCRClient()
        self.llm = llm_client or LLMClient()

        # 异步处理状态
        self._thread = None
        self._running = False
        self._result = None
        self._error = None
        self._lock = threading.Lock()

        # 防止重复处理同一路牌
        self._last_sign_frame_id = -1
        self._min_frames_between_signs = 30

        # 临时目录（相对于项目根目录）
        _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._temp_dir = os.path.join(_proj_root, SIGN_TEMP_DIR)
        os.makedirs(self._temp_dir, exist_ok=True)

        logger.info("SignHandler 初始化完成")

    # ----------------------------------------------------------------
    # 异步接口（推荐）
    # ----------------------------------------------------------------

    def start_processing(self, frame_bgr, sign_bbox, frame_id=0):
        """
        异步启动路牌处理。

        参数:
            frame_bgr: 当前帧（BGR 图像）
            sign_bbox: 路牌检测框 [x1, y1, x2, y2]
            frame_id:  帧编号（用于防止重复处理）
        """
        # 防止重复处理
        if frame_id > 0 and (frame_id - self._last_sign_frame_id) < self._min_frames_between_signs:
            logger.debug(f"跳过重复路牌处理 (frame_id={frame_id})")
            return

        if self._running:
            logger.debug("前一个路牌仍在处理中，跳过")
            return

        self._last_sign_frame_id = frame_id
        self._running = True
        self._result = None
        self._error = None

        # 裁剪路牌区域
        sign_crop = self._crop_sign(frame_bgr, sign_bbox)

        # 保存裁剪图到临时目录（供调试查看）
        try:
            _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _debug_dir = os.path.join(_proj_root, SIGN_TEMP_DIR)
            os.makedirs(_debug_dir, exist_ok=True)
            crop_path = os.path.join(_debug_dir, f"sign_{frame_id}_{int(time.time())}.jpg")
            cv2.imwrite(crop_path, sign_crop)
        except Exception:
            pass

        print(f"\n  ✂️  已裁剪路牌区域: {sign_bbox}")
        sys.stdout.flush()

        # 启动后台线程
        self._thread = threading.Thread(
            target=self._process_async,
            args=(sign_crop,),
            daemon=True,
            name="sign-handler",
        )
        self._thread.start()
        logger.info(f"路牌处理已启动 (frame_id={frame_id})")

    def is_ready(self):
        """检查处理是否完成"""
        with self._lock:
            return not self._running and self._result is not None

    def is_running(self):
        """检查是否正在处理"""
        return self._running

    def get_decision(self):
        """
        获取路牌决策结果。

        返回:
            dict: {"action": "straight"|"fork_right",
                   "confidence": float, "reason": str, "ocr_text": str}
            或 None（如果还未完成）
        """
        with self._lock:
            if self._result is not None:
                r = dict(self._result)
                # 取走结果后清除，允许下次处理
                self._result = None
                return r
            return None

    def get_error(self):
        """获取错误信息（如果有）"""
        with self._lock:
            return self._error

    # ----------------------------------------------------------------
    # 同步接口
    # ----------------------------------------------------------------

    def process_sync(self, frame_bgr, sign_bbox, timeout=30.0):
        """
        同步处理路牌（阻塞等待完整结果）。

        参数:
            frame_bgr: 当前帧
            sign_bbox: 路牌检测框 [x1, y1, x2, y2]
            timeout:   最长等待时间（秒）

        返回:
            dict: 决策结果，或默认直行决策
        """
        logger.info("开始同步路牌处理...")

        # 1. 裁剪路牌
        sign_crop = self._crop_sign(frame_bgr, sign_bbox)

        # 2. OCR
        ocr_start = time.time()
        ocr_text = self.ocr.recognize_from_image(sign_crop, timeout=SIGN_OCR_TIMEOUT)
        ocr_elapsed = time.time() - ocr_start
        logger.info(f"OCR 完成 ({ocr_elapsed:.1f}s): {ocr_text[:100]}")

        # 3. LLM 决策
        if not ocr_text.strip():
            logger.warning("OCR 未识别到文字，默认直行")
            return {
                "action": "straight",
                "confidence": 0.0,
                "reason": "OCR未识别到文字",
                "ocr_text": "",
            }

        llm_start = time.time()
        decision = self.llm.decide_track(ocr_text, timeout=SIGN_LLM_TIMEOUT)
        llm_elapsed = time.time() - llm_start
        logger.info(f"LLM 完成 ({llm_elapsed:.1f}s): {decision}")

        decision["ocr_text"] = ocr_text
        return decision

    # ----------------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------------

    def _crop_sign(self, frame_bgr, bbox):
        """从帧中裁剪路牌区域"""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame_bgr.shape[:2]

        # 扩展边距
        x1 = max(0, x1 - SIGN_CROP_MARGIN)
        y1 = max(0, y1 - SIGN_CROP_MARGIN)
        x2 = min(w, x2 + SIGN_CROP_MARGIN)
        y2 = min(h, y2 + SIGN_CROP_MARGIN)

        crop = frame_bgr[y1:y2, x1:x2]
        return crop

    def _process_async(self, sign_crop):
        """后台线程：执行 OCR + LLM"""
        try:
            # OCR
            print(f"\n  📷 正在 OCR 识别路牌文字...（超时 {SIGN_OCR_TIMEOUT}s）")
            sys.stdout.flush()
            ocr_start = time.time()
            ocr_text = self.ocr.recognize_from_image(
                sign_crop, timeout=SIGN_OCR_TIMEOUT
            )
            ocr_elapsed = time.time() - ocr_start
            ocr_display = ocr_text[:100].replace('\n', ' ') if ocr_text else '(空)'
            print(f"  ✅ OCR 完成 ({ocr_elapsed:.1f}s): \"{ocr_display}\"")
            sys.stdout.flush()
            logger.info(f"异步 OCR 完成 ({ocr_elapsed:.1f}s): {ocr_text[:100] if ocr_text else '(空)'}")

            # LLM
            if not ocr_text or not ocr_text.strip():
                print(f"  ⚠️  OCR 未识别到文字，默认直行")
                sys.stdout.flush()
                decision = {
                    "action": "straight",
                    "confidence": 0.0,
                    "reason": "OCR未识别到文字",
                    "ocr_text": ocr_text or "",
                }
            else:
                print(f"  🤖 正在调用 LLM 分析路牌含义...（超时 {SIGN_LLM_TIMEOUT}s）")
                sys.stdout.flush()
                llm_start = time.time()
                decision = self.llm.decide_track(ocr_text, timeout=SIGN_LLM_TIMEOUT)
                llm_elapsed = time.time() - llm_start
                act_cn = {"straight": "直行", "fork_right": "右转岔路"}
                act_str = act_cn.get(decision.get('action', ''), decision.get('action', ''))
                print(f"  ✅ LLM 决策完成 ({llm_elapsed:.1f}s): {act_str} (置信度={decision.get('confidence', 0):.2f})")
                sys.stdout.flush()
                logger.info(f"异步 LLM 完成 ({llm_elapsed:.1f}s): {decision}")
                decision["ocr_text"] = ocr_text

            with self._lock:
                self._result = decision
                self._running = False

        except Exception as e:
            logger.error(f"路牌处理异常: {e}")
            with self._lock:
                self._error = str(e)
                self._result = {
                    "action": "straight",
                    "confidence": 0.0,
                    "reason": f"处理异常: {e}",
                    "ocr_text": "",
                }
                self._running = False

    def reset(self):
        """重置处理器状态"""
        with self._lock:
            self._running = False
            self._result = None
            self._error = None
            self._last_sign_frame_id = -1
        logger.info("SignHandler 已重置")

    def cleanup(self):
        """清理临时文件"""
        try:
            import shutil
            if os.path.isdir(self._temp_dir):
                shutil.rmtree(self._temp_dir, ignore_errors=True)
                logger.info(f"已清理临时目录: {self._temp_dir}")
        except Exception as e:
            logger.warning(f"清理临时目录失败: {e}")
