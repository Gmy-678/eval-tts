import sys, os, time
from eval.core.config import POSTGRES_DSN
import psycopg2
print(f"Testing DB connection to: {POSTGRES_DSN[:30]}...")
try:
    conn = psycopg2.connect(POSTGRES_DSN.replace("postgresql+psycopg2://", "postgresql://"), connect_timeout=5)
    print("Success")
    conn.close()
except Exception as e:
    print(f"Error: {e}")
