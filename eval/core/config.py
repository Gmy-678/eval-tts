"""
配置：BQ、GCS、Gemini 模型等
参考 ttsManagementBack/app/config/config.py 的环境加载方式
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 根据 ENVIRONMENT 加载对应配置文件（与 ttsManagementBack 一致）
env_name = f".env.{os.getenv('ENVIRONMENT', 'test')}"
env_path = BASE_DIR / env_name
if env_path.is_file():
    logger.info("Loading environment from %s", env_path)
    load_dotenv(dotenv_path=env_path, override=True)
else:
    # 回退：尝试 .env
    fallback = BASE_DIR / ".env"
    if fallback.is_file():
        load_dotenv(dotenv_path=fallback, override=True)
    else:
        load_dotenv()
        if not (BASE_DIR / ".env").exists():
            logger.warning(
                "No .env or .env.test found. Using system env. "
                "GCS/BigQuery/Gemini may fail if vars not set."
            )

ENV = os.getenv("ENVIRONMENT", "test")

# BigQuery
BQ_PROJECT = os.getenv("BQ_PROJECT", "noiz-430406")
PUBLIC_DATASET = "public"
BQ_LOCATION = "us-central1"

# GCS
GCS_BUCKET = os.getenv("GCS_BUCKET", os.getenv("BUCKET_NAME", "noiz_data"))
GCS_ALT_BUCKET = os.getenv("PUBLIC_VOICE_BUCKET_NAME", "noiz_voices")

# GCP 凭证：与 ttsManagementBack 一致，支持相对路径、校验文件存在
_cred_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
_cred_in_base = BASE_DIR / "cred.json"
_cred_path = None
if _cred_env:
    _p = Path(_cred_env)
    _resolved = (BASE_DIR / _cred_env) if not _p.is_absolute() else _p
    if _resolved.is_file():
        _cred_path = str(_resolved.resolve())
    else:
        logger.warning("GCP 凭证文件不存在: %s，BigQuery/GCS 将不可用", _resolved)
elif _cred_in_base.is_file():
    _cred_path = str(_cred_in_base.resolve())
if _cred_path:
    GOOGLE_APPLICATION_CREDENTIALS = _cred_path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
else:
    GOOGLE_APPLICATION_CREDENTIALS = None
    os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

# Gemini：支持 GEMINI_API_KEY（与 ttsManagementBack 一致）或 GOOGLE_API_KEY
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
# 默认 gemini-3.0-pro
GEMINI_TAG_MODEL = os.getenv("GEMINI_TAG_MODEL", "gemini-2.5-pro")
GEMINI_ASR_MODEL = os.getenv("GEMINI_ASR_MODEL", "gemini-2.5-pro")
# ASR 超时（秒）与重试：长音频/网络慢时建议 180–300
GEMINI_ASR_TIMEOUT = int(os.getenv("GEMINI_ASR_TIMEOUT", "180"))
GEMINI_ASR_MAX_RETRIES = int(os.getenv("GEMINI_ASR_MAX_RETRIES", "2"))

# Postgres
POSTGRES_DSN = os.getenv("POSTGRES_DSN", os.getenv("DATABASE_URL", ""))

# Output
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "output")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# jiwer（WER/CER 评估，用于 ASR 转写结果与 target_text 对齐）
WER_LANG = os.getenv("WER_LANG", "en")  # ITN/文本规范化语言: en, zh, de, ...
USE_NEMO_ITN = os.getenv("USE_NEMO_ITN", "true").lower() in ("true", "1", "yes")
