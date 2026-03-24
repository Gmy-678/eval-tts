"""Gemini ASR 转写

识别效果差时可按以下方向排查：
- 上传后是否等到 state=ACTIVE 再调用 generate_content（本模块已加等待）
- 音频格式/采样率是否符合模型要求（Gemini 支持常见格式，过短/过长可能影响效果）
- 多语言/混合语种、专有名词、符号读法（如 +、=）易导致误识别
- 环境变量 GEMINI_ASR_MODEL（默认 gemini-3.0-pro）可更换为其他支持音频的模型

网络/代理：
- 强制使用 REST 传输，解决代理环境不稳定
- 失败会自动重新上传并转写（最多 5 次，指数退避）
- 请求超时默认 180 秒，可通过 GEMINI_ASR_TIMEOUT、GEMINI_ASR_MAX_RETRIES 调整
"""
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import google.generativeai as genai

from eval.core.config import GOOGLE_API_KEY, GEMINI_ASR_MODEL, GEMINI_ASR_TIMEOUT, GEMINI_ASR_MAX_RETRIES

logger = logging.getLogger(__name__)

MIME_MAP = {".wav": "audio/wav", ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".mp3": "audio/mpeg"}

ASR_PROMPT = """作为客观的语音转写专家，将提供的音频完整转写为纯文本。请严格遵守以下执行准则：

1. **绝对忠于原音**：你听到什么就转写什么。不需要理会句子逻辑是否通顺，不要自行补充、删除或修饰任何词汇。只关注文本与音频发音的绝对对应。
2. **零格式化输出**：只输出纯文本。不要生成任何总结，不要添加时间戳。除非有极其明显的多人对话交替，否则不要添加任何说话人标签（如"说话人A："）。
3. **原生多语言保留**：遇到多语言混合或外语时，直接转写为对应的原始语言文本。严禁进行任何形式的翻译，无需在意不同语言在同一句子中的混合。
4. **统一数字规范**：听到任何表达数字的声音（如"一百"、"一万"、"one hundred"），请一律转换为阿拉伯数字格式（如"100"、"10000"）进行输出。"""


def init_genai():
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY 未配置")
    # 强制 REST 传输，解决代理环境不稳定
    genai.configure(api_key=GOOGLE_API_KEY, transport="rest")
    logger.info("USE_REST_TRANSPORT=True (transport=rest)")


def transcribe_audio(audio_path: Path) -> Optional[str]:
    """
    使用 Gemini 对音频进行 ASR 转写，返回纯文本。
    支持 mp3, wav, m4a, ogg 等格式。失败会自动重新上传并转写。
    """
    if not audio_path.exists():
        logger.error("【跳过】文件不存在: %s", audio_path)
        return None

    init_genai()
    model = genai.GenerativeModel(GEMINI_ASR_MODEL)
    mime = MIME_MAP.get(audio_path.suffix.lower(), "audio/mpeg")
    max_retries = GEMINI_ASR_MAX_RETRIES or 5

    for attempt in range(max_retries):
        audio_file = None
        try:
            # 步骤 A: 上传
            audio_file = genai.upload_file(path=str(audio_path), mime_type=mime)

            # 步骤 B: 状态轮询（带 180 秒硬超时）
            wait_start = time.time()
            first_poll = True
            while True:
                audio_file = genai.get_file(audio_file.name)
                state = getattr(getattr(audio_file, "state", None), "name", None)
                if state == "ACTIVE":
                    break
                if state == "FAILED":
                    raise ValueError("Gemini 侧处理音频失败")
                if time.time() - wait_start > 180:
                    raise TimeoutError("文件处理 ACTIVE 超时")
                
                if first_poll:
                    time.sleep(2)
                    first_poll = False
                else:
                    time.sleep(5)

            # 步骤 C: 生成内容
            response = model.generate_content(
                [ASR_PROMPT, audio_file],
                request_options={"timeout": GEMINI_ASR_TIMEOUT},
            )
            if response and response.text:
                return response.text.strip()
            raise ValueError("响应内容为空")

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2**attempt) * 2 + random.uniform(1, 3)
                logger.warning(
                    "【捕获失败】%s 第 %d 次尝试失败，%.1fs 后重新转写。错误: %s",
                    audio_path.name,
                    attempt + 1,
                    wait_time,
                    e,
                )
                time.sleep(wait_time)
            else:
                logger.error("【彻底失败】%s 在 %d 次尝试后仍未成功。", audio_path.name, max_retries)
                return None
        finally:
            if audio_file and getattr(audio_file, "name", None):
                try:
                    genai.delete_file(audio_file.name)
                except Exception:
                    pass
    return None
