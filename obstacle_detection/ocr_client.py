"""
ocr_client.py —— PaddleOCR API 客户端
=====================================

封装 PaddleOCR-VL API 调用，支持本地图片上传和结果轮询。

基于 ocr/ocr.py 重构，提供可复用的 OCRClient 类。

用法:
    from obstacle_detection import OCRClient

    ocr = OCRClient()
    text = ocr.recognize("path/to/image.png")
    # 或直接传入 numpy 图像
    text = ocr.recognize_from_image(frame_bgr)
"""

import json
import os
import sys
import time
import logging
import tempfile

import cv2
import requests

logger = logging.getLogger("OCRClient")

# ====================================================================
# API 配置
# ====================================================================

JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
TOKEN = "eb8c46b5fdc8e4b01d2f0da3cbb2bb03e9f0dea1"
MODEL = "PP-OCRv6"


class OCRClient:
    """
    PaddleOCR-VL API 客户端。

    支持本地文件或内存中的 numpy 图像作为输入。
    提交 OCR 任务后轮询等待结果，返回识别到的 Markdown 文本。
    """

    def __init__(self, token=None, model=None):
        """
        参数:
            token: API 鉴权 token（None 则使用默认值）
            model: 模型名称（None 则使用默认 PaddleOCR-VL-1.6）
        """
        self.token = token or TOKEN
        self.model = model or MODEL
        self._headers = {"Authorization": f"bearer {self.token}"}
        self._optional_payload = {
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useTextlineOrientation": False,
        }

    # ----------------------------------------------------------------
    # 公共接口
    # ----------------------------------------------------------------

    def recognize(self, file_path, timeout=30.0):
        """
        识别图片中的文字。

        参数:
            file_path: 本地图片路径或 URL
            timeout: 最长等待时间（秒）

        返回:
            str: 识别到的文字内容。失败时返回空字符串。
        """
        if not os.path.exists(file_path) and not file_path.startswith("http"):
            logger.error(f"文件不存在: {file_path}")
            return ""

        # 提交任务
        job_id = self._submit_job(file_path)
        if not job_id:
            return ""

        # 轮询等待结果
        jsonl_url = self._poll_result(job_id, timeout)
        if not jsonl_url:
            return ""

        # 下载并解析结果
        return self._fetch_result(jsonl_url)

    def recognize_from_image(self, image_bgr, timeout=30.0):
        """
        从内存中的 BGR 图像识别文字。

        参数:
            image_bgr: numpy 数组 (H, W, 3) BGR 格式
            timeout: 最长等待时间（秒）

        返回:
            str: 识别到的文字内容
        """
        # 保存到临时文件（相对于项目根目录）
        _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _temp_dir = os.path.join(_proj_root, "obstacle_detection", "temp")
        os.makedirs(_temp_dir, exist_ok=True)
        temp_path = os.path.join(_temp_dir,
                                 f"ocr_{int(time.time() * 1000)}.png")
        cv2.imwrite(temp_path, image_bgr)

        try:
            return self.recognize(temp_path, timeout)
        finally:
            # 清理临时文件
            try:
                os.remove(temp_path)
            except OSError:
                pass

    # ----------------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------------

    def _submit_job(self, file_path):
        """提交 OCR 任务，返回 jobId"""
        try:
            if file_path.startswith("http"):
                headers = dict(self._headers)
                headers["Content-Type"] = "application/json"
                payload = {
                    "fileUrl": file_path,
                    "model": self.model,
                    "optionalPayload": self._optional_payload,
                }
                resp = requests.post(JOB_URL, json=payload, headers=headers)
            else:
                data = {
                    "model": self.model,
                    "optionalPayload": json.dumps(self._optional_payload),
                }
                with open(file_path, "rb") as f:
                    resp = requests.post(JOB_URL, headers=self._headers,
                                         data=data, files={"file": f})

            if resp.status_code != 200:
                logger.error(f"OCR 提交失败: {resp.status_code} {resp.text}")
                return None

            job_id = resp.json()["data"]["jobId"]
            logger.info(f"OCR 任务已提交: jobId={job_id}")
            return job_id

        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            logger.error(f"OCR 提交异常: {e}")
            return None

    def _poll_result(self, job_id, timeout):
        """轮询等待 OCR 任务完成，返回 jsonl URL"""
        start_time = time.time()
        url = f"{JOB_URL}/{job_id}"

        while True:
            if time.time() - start_time > timeout:
                logger.error(f"OCR 任务超时: jobId={job_id}")
                return None

            try:
                resp = requests.get(url, headers=self._headers)
                resp.raise_for_status()
                state = resp.json()["data"]["state"]

                if state == "done":
                    jsonl_url = resp.json()["data"]["resultUrl"]["jsonUrl"]
                    logger.info(f"OCR 任务完成: jobId={job_id}")
                    return jsonl_url
                elif state == "failed":
                    error_msg = resp.json()["data"].get("errorMsg", "未知错误")
                    logger.error(f"OCR 任务失败: {error_msg}")
                    return None
                else:
                    # pending 或 running，继续等待
                    logger.debug(f"OCR 状态: {state}")
                    time.sleep(3)

            except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
                logger.error(f"OCR 轮询异常: {e}")
                time.sleep(3)

    def _fetch_result(self, jsonl_url):
        """下载并解析 OCR 结果，返回合并后的文本"""
        try:
            resp = requests.get(jsonl_url)
            resp.raise_for_status()

            lines = resp.text.strip().split("\n")
            texts = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                result = json.loads(line)["result"]
                for res in result.get("ocrResults", []):
                    pruned = res.get("prunedResult", {})
                    # PP-OCRv6 格式：rec_texts 是识别到的文字列表
                    for t in pruned.get("rec_texts", []):
                        if t and t.strip():
                            texts.append(t.strip())

            full_text = "\n".join(texts)
            logger.info(f"OCR 结果 ({len(texts)} 段): {full_text[:200]}...")
            return full_text

        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            logger.error(f"OCR 结果解析异常: {e}")
            return ""
