"""
llm_client.py —— 文心大模型（千帆）API 客户端
============================================

封装百度千帆 文心大模型 API，用于路牌识别后的导航决策。

基于 ocr/llm.py 重构，提供可复用的 LLMClient 类。

用法:
    from obstacle_detection import LLMClient

    llm = LLMClient()
    decision = llm.decide_track("前方直行，右岔路通往停车场")
    # → {"action": "straight", "confidence": 0.9, "reason": "路牌指示直行"}
"""

import json
import logging
import requests

logger = logging.getLogger("LLMClient")

# ====================================================================
# API 配置
# ====================================================================

API_URL = "https://qianfan.baidubce.com/v2/chat/completions"
API_KEY = "bce-v3/ALTAK-gfaNk2DBB17lvU41Q3xXr/c198ffa474535b8921239f309c15ed91369a292d"
DEFAULT_MODEL = "ernie-4.5-turbo-32k"

# 决策用的 system prompt
SYSTEM_PROMPT = (
    "你是一个自动驾驶小车的导航决策系统。"
    "小车正在沿主路循迹行驶，前方岔路口有两条路：左侧道路和右侧岔路。\n"
    "\n"
    "分析路牌OCR识别出的文字，根据路牌的真实含义（不限于字面关键词），"
    "判断路牌指示小车应该：\n"
    "- left —— 向左走\n"
    "- right —— 向右走\n"
    "\n"
    "因为OCR结果可能有错别字导致路牌语气不正式或者谐音,你要先将OCR结果转变成可能要表达的意思。"
    "根据转变的结果给出回复，不要仅按'左'或'右'字来判断"
    "请以JSON格式回复，必须包含 action、confidence 和 reason 字段；"
    "confidence 必须是 0.0 到 1.0 之间的数字（例如 0.85），不要使用 high/low 等文字。"
)


class LLMClient:
    """
    文心大模型（千帆）API 客户端。

    用于根据路牌 OCR 文字进行导航决策。
    """

    def __init__(self, api_key=None, model=None):
        """
        参数:
            api_key: API 鉴权 key（None 则使用默认值）
            model: 模型名称（None 则使用默认 ernie-5.0）
        """
        self.api_key = api_key or API_KEY
        self.model = model or DEFAULT_MODEL
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    # ----------------------------------------------------------------
    # 公共接口
    # ----------------------------------------------------------------

    def decide_track(self, ocr_text, timeout=10.0):
        """
        根据路牌 OCR 文字，调用 LLM 决策行驶方向。

        参数:
            ocr_text: 路牌 OCR 识别出的文字
            timeout: 请求超时（秒）

        返回:
            dict: {"action": "straight"|"fork_right",
                   "confidence": float,
                   "reason": str}
            失败时返回默认直行决策。
        """
        if not ocr_text or not ocr_text.strip():
            logger.warning("OCR 文本为空，默认直行")
            return self._default_decision("OCR未识别到文字")

        # 构建用户消息
        user_content = (
            f"路牌OCR识别内容：\n{ocr_text}\n\n"
            f"请判断路牌指示小车向左走(left)、向右走(right)。"
            f"以JSON格式回复。"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,  # 低温度，更确定性
        }

        try:
            logger.info("正在调用 LLM 决策...")
            resp = requests.post(
                API_URL,
                headers=self._headers,
                data=json.dumps(payload),
                timeout=timeout,
            )

            if resp.status_code != 200:
                logger.error(f"LLM 请求失败: {resp.status_code} {resp.text}")
                return self._default_decision(f"API返回{resp.status_code}")

            result = resp.json()
            content = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            logger.info(f"LLM 原始返回: {content}")
            return self._parse_decision(content, ocr_text)

        except requests.Timeout:
            logger.error("LLM 请求超时")
            return self._default_decision("LLM超时")
        except requests.RequestException as e:
            logger.error(f"LLM 请求异常: {e}")
            return self._default_decision(f"网络异常: {e}")
        except Exception as e:
            logger.error(f"LLM 调用未知异常: {e}")
            return self._default_decision(f"未知异常: {e}")

    # ----------------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------------

    def _parse_decision(self, content, ocr_text=""):
        """
        解析 LLM 返回的 JSON，提取决策。

        能处理以下情况：
        1. 标准 JSON 格式
        2. JSON 外层包裹了 markdown 代码块
        3. 非 JSON 文本中包含了关键方向词
        """
        # 尝试直接解析
        try:
            decision = json.loads(content)
            parsed = self._try_validate(decision)
            if parsed is not None:
                return parsed
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块中提取
        import re
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if json_match:
            try:
                decision = json.loads(json_match.group(1))
                parsed = self._try_validate(decision)
                if parsed is not None:
                    return parsed
            except json.JSONDecodeError:
                pass

        # 尝试从文本中找到 JSON 对象
        json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', content, re.DOTALL)
        if json_match:
            try:
                decision = json.loads(json_match.group(0))
                parsed = self._try_validate(decision)
                if parsed is not None:
                    return parsed
            except json.JSONDecodeError:
                pass

        # 最终回退：关键词匹配
        logger.warning(f"无法解析 LLM JSON，尝试关键词匹配: {content}")
        text_lower = content.lower()

        if any(w in text_lower for w in ["右", "right", "向右"]):
            return {"action": "fork_right", "confidence": 0.6,
                    "reason": f"关键词推断右（原始: {content[:100]}）"}
        elif any(w in text_lower for w in ["左", "left", "向左"]):
            return {"action": "straight", "confidence": 0.6,
                    "reason": f"关键词推断左（原始: {content[:100]}）"}
        else:
            return self._default_decision(f"无法解析LLM输出，默认直行")

    def _validate_decision(self, decision):
        """
        验证并规范化决策字典。

        LLM 返回 left/right/straight → 映射为最终动作:
          left/straight → straight（左拐和直行都走主路）
          right         → fork_right（右拐走右侧岔路）

        2026-08-16 加固：
          - confidence 可能返回 "high"/"低" 等非数字字符串，原 float() 直接
            抛 ValueError → 异常冒泡到 decide_track 外层 except → 整个决策
            （包括正确的 action）被丢弃、兜底默认直行。现改宽容解析：
            数字照用，定性词映射，其他非法值给 0.5，永不抛异常。
          - action 宽容匹配 right/fork_right/右/向右/右转 等写法。
        """
        raw_action = str(decision.get("action", "straight")).strip().lower()

        # 映射表（宽容：接受 right/fork_right/右 等写法）
        if raw_action in ("right", "fork_right", "fork right", "右", "向右", "右转", "右拐"):
            action = "fork_right"
        else:
            # left / straight / 其他 → 都走主路直行
            action = "straight"

        confidence = self._coerce_confidence(decision.get("confidence", 0.5))

        reason = decision.get("reason", "")

        logger.info(f"LLM 决策规范化: raw_action={raw_action!r} → {action}, confidence={confidence}")
        return {"action": action, "confidence": confidence, "reason": reason,
                "raw_action": raw_action}

    @staticmethod
    def _coerce_confidence(value):
        """宽容地把 confidence 转成 [0,1] 浮点数（2026-08-16）。

        数字/数字字符串 → float；定性词（high/中/低…）→ 映射；
        其他非法值 → 0.5。任何输入都不抛异常。
        """
        if isinstance(value, (int, float)):
            conf = float(value)
        elif isinstance(value, str):
            s = value.strip().lower()
            try:
                conf = float(s)
            except ValueError:
                conf = {
                    "high": 0.8, "very high": 0.9, "高": 0.8,
                    "medium": 0.6, "mid": 0.6, "中": 0.6,
                    "low": 0.4, "低": 0.4, "very low": 0.2,
                }.get(s, 0.5)
        else:
            conf = 0.5
        return max(0.0, min(1.0, conf))

    def _try_validate(self, decision):
        """_validate_decision 的兜底包装：校验异常不再向上抛，
        返回 None 让 _parse_decision 继续尝试下一条解析路径。"""
        try:
            return self._validate_decision(decision)
        except Exception as e:
            logger.warning(f"决策字典校验异常（继续尝试其他解析路径）: {e}")
            return None

    def _default_decision(self, reason=""):
        """返回默认直行决策"""
        return {"action": "straight", "confidence": 0.3, "reason": reason}

    # ----------------------------------------------------------------
    # 通用对话接口（预留）
    # ----------------------------------------------------------------

    def chat(self, messages, temperature=0.7, timeout=10.0):
        """
        通用对话接口，可用于调试或其他扩展场景。

        参数:
            messages: [{"role": "user", "content": "..."}, ...]
            temperature: 温度参数
            timeout: 超时（秒）

        返回:
            str: 模型回复文本
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        try:
            resp = requests.post(
                API_URL,
                headers=self._headers,
                data=json.dumps(payload),
                timeout=timeout,
            )
            if resp.status_code == 200:
                result = resp.json()
                return (result.get("choices", [{}])[0]
                        .get("message", {}).get("content", ""))
            else:
                logger.error(f"LLM chat 失败: {resp.status_code} {resp.text}")
                return ""
        except Exception as e:
            logger.error(f"LLM chat 异常: {e}")
            return ""
