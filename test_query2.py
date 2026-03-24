import sys, time
from datetime import date, timedelta, datetime
from eval.core.config import POSTGRES_DSN
import psycopg2
from psycopg2.extras import RealDictCursor

d = date.today() - timedelta(days=1)
target_date = d.strftime("%Y%m%d")
start_ts = int(datetime.strptime(target_date, "%Y%m%d").timestamp())
end_ts = start_ts + 86400

query = """
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
  FROM gen_products gp
  LEFT JOIN users u ON gp.user_id = u.id
  INNER JOIN user_download_rates udr ON gp.user_id = udr.user_id
  WHERE gp.create_time >= %(start_ts)s
    AND gp.create_time < %(end_ts)s
    AND (gp.delete_time IS NULL OR gp.delete_time = 0)
    AND gp.file_path IS NOT NULL
    AND gp.file_path != ''
    AND gp.target_text IS NOT NULL
    AND gp.target_text != ''
    AND (gp.is_downloaded IS NULL OR gp.is_downloaded = FALSE)
  ORDER BY RANDOM()
  LIMIT %(limit)s
)
SELECT * FROM eligible
"""
dsn = POSTGRES_DSN.replace("postgresql+psycopg2://", "postgresql://")
conn = psycopg2.connect(dsn, connect_timeout=15)
cur = conn.cursor(cursor_factory=RealDictCursor)
print("Running full query...")
t0 = time.time()
cur.execute(query, {"start_ts": start_ts, "end_ts": end_ts, "limit": 10})
res = cur.fetchall()
print(f"Returned {len(res)} rows, took {time.time() - t0:.2f}s")
conn.close()
