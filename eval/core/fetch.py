"""数据获取：BigQuery / Postgres"""
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

# 确保项目根目录在 path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from google.cloud import bigquery

from eval.core.config import BQ_PROJECT, BQ_LOCATION, POSTGRES_DSN, PUBLIC_DATASET

logger = logging.getLogger(__name__)

_client: Optional[bigquery.Client] = None


def _get_bq_client() -> bigquery.Client:
    global _client
    if _client is None:
        _client = bigquery.Client(project=BQ_PROJECT)
    return _client


def fetch_samples_bq(
    target_date: Optional[str] = None,
    limit: int = 2000,
    undownloaded_only: bool = True,
) -> list[dict]:
    """
    从 BigQuery 捞取指定日期样本：仅下载率 <50% 的用户，各类 creation_mode 均可，
    样本必须 is_downloaded 非 TRUE（NULL 或 FALSE），随机抽样。
    """
    if target_date is None:
        d = date.today() - timedelta(days=1)
        target_date = d.strftime("%Y%m%d")

    query = f"""
-- ============================================================
-- 捞取 {limit} 条：下载率 <50% 用户，各类 creation_mode，仅未下载样本，随机
-- ============================================================

DECLARE target_date STRING DEFAULT '{target_date}';

-- 用户下载率 = 该用户在该天的 gen_products 中已下载数/总数，不依赖 voices
WITH user_download_rates AS (
  SELECT
    gp.user_id,
    COUNT(*) AS total,
    COUNTIF(gp.is_downloaded = TRUE) AS downloaded,
    SAFE_DIVIDE(COUNTIF(gp.is_downloaded = TRUE), COUNT(*)) AS download_rate
  FROM `{BQ_PROJECT}.{PUBLIC_DATASET}.gen_products` gp
  WHERE gp.create_time >= UNIX_SECONDS(TIMESTAMP(FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%Y%m%d', target_date))))
    AND gp.create_time <  UNIX_SECONDS(TIMESTAMP_ADD(TIMESTAMP(FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%Y%m%d', target_date))), INTERVAL 1 DAY))
    AND (gp.delete_time IS NULL OR gp.delete_time = 0)
  GROUP BY gp.user_id
  HAVING SAFE_DIVIDE(COUNTIF(gp.is_downloaded = TRUE), COUNT(*)) < 0.50
),

eligible_products AS (
  SELECT
    gp.gen_product_id,
    gp.user_id,
    u.email,
    gp.target_text,
    gp.file_path,
    gp.create_time,
    gp.is_downloaded,
    gp.audio_product_id,
    udr.download_rate,
    ROW_NUMBER() OVER (ORDER BY RAND()) AS rn
  FROM `{BQ_PROJECT}.{PUBLIC_DATASET}.gen_products` gp
  LEFT JOIN `{BQ_PROJECT}.{PUBLIC_DATASET}.users` u ON gp.user_id = u.id
  INNER JOIN user_download_rates udr ON gp.user_id = udr.user_id
  WHERE gp.create_time >= UNIX_SECONDS(TIMESTAMP(FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%Y%m%d', target_date))))
    AND gp.create_time <  UNIX_SECONDS(TIMESTAMP_ADD(TIMESTAMP(FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%Y%m%d', target_date))), INTERVAL 1 DAY))
    AND (gp.delete_time IS NULL OR gp.delete_time = 0)
    {"AND (gp.is_downloaded IS NULL OR gp.is_downloaded = FALSE)" if undownloaded_only else ""}
    AND gp.file_path IS NOT NULL AND gp.file_path != ''
    AND gp.target_text IS NOT NULL AND gp.target_text != ''
)

SELECT
  gen_product_id,
  user_id,
  email,
  target_text,
  file_path,
  create_time,
  is_downloaded,
  audio_product_id,
  download_rate
FROM eligible_products
WHERE rn <= {limit}
ORDER BY rn
"""

    client = _get_bq_client()
    results = list(client.query(query, location=BQ_LOCATION).result())

    out = []
    for row in results:
        out.append({
            "gen_product_id": row.gen_product_id,
            "user_id": str(row.user_id) if row.user_id is not None else "",
            "email": getattr(row, "email", None) or "",
            "target_text": row.target_text or "",
            "file_path": row.file_path or "",
            "create_time": row.create_time,
            "is_downloaded": row.is_downloaded,
            "audio_product_id": str(row.audio_product_id) if row.audio_product_id is not None else "",
            "download_rate": float(row.download_rate) if row.download_rate is not None else 0.0,
        })
    return out


def fetch_samples_postgres(
    target_date: Optional[str] = None,
    limit: int = 2000,
    undownloaded_only: bool = True,
    skip_voice_filter: bool = False,
) -> list[dict]:
    """
    从 Postgres 捞取指定日期的样本，逻辑与 fetch_samples_bq 对齐。
    各类 creation_mode 均可，下载率<50% 用户，样本 is_downloaded 非 TRUE（NULL 或 FALSE），随机抽样。
    需配置 POSTGRES_DSN。
    """
    if not POSTGRES_DSN:
        logger.warning("POSTGRES_DSN 未配置，无法从 Postgres 捞取")
        return []

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        logger.warning("psycopg2 未安装，无法使用 Postgres")
        return []

    dsn = (POSTGRES_DSN or "").replace("postgresql+psycopg2://", "postgresql://")
    if not dsn:
        return []
        
    logger.info(f"Connecting to Postgres to fetch up to {limit} samples...")

    if target_date is None:
        d = date.today() - timedelta(days=1)
        target_date = d.strftime("%Y%m%d")

    start_ts = int(datetime.strptime(target_date, "%Y%m%d").timestamp())
    end_ts = start_ts + 86400

    undownloaded_clause = "AND (gp.is_downloaded IS NULL OR gp.is_downloaded = FALSE)" if undownloaded_only else ""

    # 与 BQ 对齐：user_download_rates = 该用户在该天的 gen_products 下载率，不依赖 voices
    query = f"""
WITH user_download_rates AS (
  SELECT
    gp.user_id,
    COUNT(*) AS total,
    COUNT(CASE WHEN gp.is_downloaded = TRUE THEN 1 END) AS downloaded,
    CASE WHEN COUNT(*) > 0
      THEN COUNT(CASE WHEN gp.is_downloaded = TRUE THEN 1 END)::float / COUNT(*)
      ELSE 0
    END AS download_rate
  FROM gen_products gp
  WHERE gp.create_time >= %(start_ts)s
    AND gp.create_time < %(end_ts)s
    AND (gp.delete_time IS NULL OR gp.delete_time = 0)
  GROUP BY gp.user_id
  HAVING CASE WHEN COUNT(*) > 0
    THEN COUNT(CASE WHEN gp.is_downloaded = TRUE THEN 1 END)::float / COUNT(*)
    ELSE 0
  END < 0.50
),
-- 优化：使用 TABLESAMPLE SYSTEM 从当日活跃块中快速随机获取候选行（按物理块抽样）
-- 假设 5% 的数据块足以覆盖足够数量的候选池，若不足可适当调高比例
sample_pool AS (
  SELECT *
  FROM gen_products TABLESAMPLE SYSTEM (5)
  WHERE create_time >= %(start_ts)s
    AND create_time < %(end_ts)s
    AND (delete_time IS NULL OR delete_time = 0)
    AND file_path IS NOT NULL
    AND file_path != ''
    AND target_text IS NOT NULL
    AND target_text != ''
),
eligible AS (
  SELECT
    gp.gen_product_id,
    gp.user_id,
    COALESCE(u.email, '') AS email,
    COALESCE(gp.target_text, '') AS target_text,
    COALESCE(gp.file_path, '') AS file_path,
    gp.create_time,
    gp.is_downloaded,
    COALESCE(gp.audio_product_id::text, '') AS audio_product_id,
    udr.download_rate
  FROM sample_pool gp
  LEFT JOIN users u ON gp.user_id = u.id
  INNER JOIN user_download_rates udr ON gp.user_id = udr.user_id
  WHERE 1=1
    {undownloaded_clause}
  -- 只有这少部分符合条件的候选才进行真正的全局行级随机排序
  ORDER BY RANDOM()
  LIMIT %(limit)s
)
SELECT * FROM eligible
"""

    conn = psycopg2.connect(dsn, connect_timeout=15)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, {"start_ts": start_ts, "end_ts": end_ts, "limit": limit})
            rows = cur.fetchall()
            logger.info(f"Successfully fetched {len(rows)} samples from Postgres.")
    except Exception as e:
        logger.error(f"Postgres query failed: {e}")
        return []
    finally:
        conn.close()

    out: list[dict] = []
    for row in rows:
        out.append({
            "gen_product_id": row["gen_product_id"],
            "user_id": str(row["user_id"]) if row["user_id"] is not None else "",
            "email": row.get("email") or "",
            "target_text": row.get("target_text") or "",
            "file_path": row.get("file_path") or "",
            "create_time": row.get("create_time"),
            "is_downloaded": row.get("is_downloaded"),
            "audio_product_id": str(row.get("audio_product_id") or ""),
            "download_rate": float(row.get("download_rate") or 0.0),
        })
    return out


def fetch_gen_product_postgres(gen_product_id: str) -> Optional[dict]:
    """
    从 Postgres 按 gen_product_id 查询单条 gen_product。
    需配置 POSTGRES_DSN。
    """
    if not POSTGRES_DSN:
        logger.warning("POSTGRES_DSN 未配置，跳过 Postgres 查询")
        return None

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        logger.warning("psycopg2 未安装，无法使用 Postgres")
        return None

    dsn = (POSTGRES_DSN or "").replace("postgresql+psycopg2://", "postgresql://")
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT gen_product_id, file_path, target_text
                FROM gen_products
                WHERE gen_product_id = %s
                  AND (delete_time IS NULL OR delete_time = 0)
                """,
                (gen_product_id,),
            )
            row = cur.fetchone()
            if row:
                return dict(row)
            return None
    finally:
        conn.close()


def save_samples(samples: list[dict], path: Path) -> None:
    """保存样本到 JSON 文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    logger.info("已保存 %d 条样本到 %s", len(samples), path)


def load_samples(path: Path) -> list[dict]:
    """从 JSON 文件加载样本"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
