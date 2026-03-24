"""GCS 音频下载"""
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from google.cloud import storage

from eval.core.config import GCS_BUCKET, GCS_ALT_BUCKET

logger = logging.getLogger(__name__)

_client: Optional[storage.Client] = None


def _get_storage_client() -> storage.Client:
    global _client
    if _client is None:
        _client = storage.Client()
    return _client


def _find_blob(bucket: storage.Bucket, file_path: str) -> Optional[tuple]:
    """
    在 bucket 中查找 blob，支持多种路径尝试。
    参考 ttsManagementBack/app/routers/gcs.py
    """
    blob = bucket.blob(file_path)
    if blob.exists():
        return blob, file_path

    candidates = [file_path]
    if file_path.startswith("voices/"):
        candidates.append(file_path.replace("voices/", "", 1))
    if "/" in file_path:
        candidates.append(file_path.split("/")[-1])
    if not file_path.startswith("uploads/"):
        candidates.append(f"uploads/{file_path}")

    for cand in candidates:
        alt = bucket.blob(cand)
        if alt.exists():
            return alt, alt.name

    file_name = file_path.split("/")[-1] if "/" in file_path else file_path
    prefix = file_name[: min(8, len(file_name))]
    for b in bucket.list_blobs(prefix=prefix, max_results=50):
        if b.name.endswith(file_name) or file_name in b.name:
            return b, b.name

    return None


def download_audio(
    file_path: str,
    out_dir: Optional[Path] = None,
    dest_name: Optional[str] = None,
) -> Optional[Path]:
    """
    从 GCS 下载音频到本地，返回本地文件路径。
    优先 noiz_data，找不到则尝试 noiz_voices。
    dest_name: 可选，指定本地文件名（如 gen_product_id.mp3），用于打包时统一命名。
    """
    if not file_path or not file_path.strip():
        logger.warning("file_path 为空")
        return None

    out_dir = out_dir or _PROJECT_ROOT / "tmp" / "eval_audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if dest_name:
        local_path = out_dir / dest_name
        if local_path.exists() and local_path.stat().st_size > 0:
            logger.info("Found cached audio: %s", local_path)
            return local_path

    client = _get_storage_client()
    bucket = client.bucket(GCS_BUCKET)
    result = _find_blob(bucket, file_path)

    if result is None and GCS_BUCKET != GCS_ALT_BUCKET:
        bucket = client.bucket(GCS_ALT_BUCKET)
        result = _find_blob(bucket, file_path)

    if result is None:
        logger.warning("未找到文件: %s (bucket: %s)", file_path, GCS_BUCKET)
        return None

    blob, resolved_path = result
    if not dest_name:
        safe_name = resolved_path.replace("/", "_").replace("\\", "_")
        local_path = out_dir / safe_name
        if local_path.exists() and local_path.stat().st_size > 0:
            logger.info("Found cached audio: %s", local_path)
            return local_path

    blob.download_to_filename(str(local_path))
    return local_path
