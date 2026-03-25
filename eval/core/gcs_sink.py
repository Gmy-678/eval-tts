import logging
import os
from pathlib import Path
from google.cloud import storage
from eval.core.config import GCS_BUCKET

logger = logging.getLogger(__name__)

def upload_to_gcs(local_file_path: str, gcs_prefix: str) -> str:
    """
    将本地文件上传到 GCS 对应的 bucket 中。
    返回上传后的完整 gs:// 路径
    """
    if not os.path.exists(local_file_path):
        logger.error(f"Local file {local_file_path} does not exist.")
        return ""

    if not GCS_BUCKET:
        logger.warning("GCS_BUCKET 未配置，跳过 GCS 上传。")
        return ""

    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        
        file_name = Path(local_file_path).name
        # 组装 destination blob 名字，例如 eval_results/final_results.csv
        # 实际调用的地方应该提前把本地文件重命名为比如 2026-03-24.csv，再传进来
        blob_name = f"{gcs_prefix.strip('/')}/{file_name}" if gcs_prefix else file_name
        
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_file_path)
        
        gcs_uri = f"gs://{GCS_BUCKET}/{blob_name}"
        logger.info(f"Successfully uploaded {local_file_path} to {gcs_uri}")
        return gcs_uri
        
    except Exception as e:
        logger.error(f"Failed to upload to GCS: {e}")
        return ""
