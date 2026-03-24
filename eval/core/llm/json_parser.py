import json
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def parse_and_fix_json(text: str, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """鲁棒的 JSON 解析，带兜底和自动修复机制"""
    if fallback is None:
        fallback = {"error": "Failed to parse JSON"}

    # 1. 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
        
    # 2. 尝试从 Markdown 代码块提取
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            text = json_str # 提取失败则保留内容进行后续修复

    # 3. 兜底修复：自动补齐外层大括号
    text = text.strip()
    if not text.startswith("{"):
        text = "{" + text
    if not text.endswith("}"):
        text = text + "}"
        
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON Auto-fix failed: {e}. Raw text: {text}")
        return fallback