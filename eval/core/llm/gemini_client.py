import time
import logging
from typing import Dict, Any, Optional
import google.generativeai as genai
from eval.core.llm.json_parser import parse_and_fix_json
from eval.core.config import GOOGLE_API_KEY

logger = logging.getLogger(__name__)


# API Key 配置方式（任选其一）:
# 1. 环境变量：在 .env 或 .env.test 中设置 GEMINI_API_KEY=xxx 或 GOOGLE_API_KEY=xxx
# 2. 构造参数：GeminiClient(api_key="xxx", ...) 显式传入


class GeminiClient:
    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        max_retries: int = 3,
        api_key: Optional[str] = None,
    ):
        key = api_key or GOOGLE_API_KEY
        if not key:
            logger.warning("GEMINI_API_KEY or GOOGLE_API_KEY is not set in environment. Gemini plugin will fail.")
        else:
            genai.configure(api_key=key)
            
        self.model = genai.GenerativeModel(model_name)
        self.max_retries = max_retries

    def generate_json(self, prompt: str, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """带重试机制的 LLM JSON 生成"""
        if fallback is None:
            fallback = {"error": "All retries failed"}
            
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(
                    prompt, 
                    request_options={"timeout": 60},
                    generation_config={"response_mime_type": "application/json"}
                )
                result = parse_and_fix_json(response.text, fallback=fallback)
                if "error" not in result or result != fallback:
                    return result
            except Exception as e:
                logger.warning(f"Gemini call failed (attempt {attempt+1}/{self.max_retries}): {e}")
                time.sleep(2 ** attempt)  # 指数退避重试
                
        logger.error("All Gemini API retries exhausted.")
        return fallback