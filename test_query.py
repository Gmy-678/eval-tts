import sys, time
from datetime import date, timedelta, datetime
from eval.core.config import POSTGRES_DSN
import psycopg2
from psycopg2.extras import RealDictCursor

d = date.today() - timedelta(days=1)
target_date = d.strftime("%Y%m%d")
start_ts = int(datetime.strptime(target_date, "%Y%m%d").timestamp())
end_ts = start_ts + 86400

print(f"target_date={target_date}, start_ts={start_ts}, end_ts={end_ts}")

query = """
SELECT count(*) FROM gen_products
WHERE create_time >= %(start_ts)s
  AND create_time < %(end_ts)s
  AND (delete_time IS NULL OR delete_time = 0)
"""
dsn = POSTGRES_DSN.replace("postgresql+psycopg2://", "postgresql://")
conn = psycopg2.connect(dsn, connect_timeout=15)
cur = conn.cursor(cursor_factory=RealDictCursor)
print("Running count query...")
t0 = time.time()
cur.execute(query, {"start_ts": start_ts, "end_ts": end_ts})
res = cur.fetchone()
print(f"Count: {res['count']}, took {time.time() - t0:.2f}s")
conn.close()
