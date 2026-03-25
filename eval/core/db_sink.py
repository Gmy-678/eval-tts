import logging
import pandas as pd
from eval.core.config import POSTGRES_DSN

logger = logging.getLogger(__name__)

def save_to_postgres(df: pd.DataFrame, table_name: str) -> None:
    """
    将 DataFrame 数据批量插入到指定的 Postgres 数据表中。
    如果表不存在，会先创建表结构。
    """
    if df.empty:
        logger.warning("DataFrame is empty, nothing to save to Postgres.")
        return

    if not POSTGRES_DSN:
        logger.warning("POSTGRES_DSN 未配置，无法写入评测结果到 Postgres")
        return

    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError:
        logger.warning("psycopg2 未安装，无法使用 Postgres")
        return

    dsn = (POSTGRES_DSN or "").replace("postgresql+psycopg2://", "postgresql://")
    if not dsn:
        return

    # 处理一些特殊类型，将列转为原生 Python 类型，并处理 NA
    df_clean = df.copy()
    # 将字典/列表等转成 JSON 字符串（如果有的话）
    for col in df_clean.columns:
        if df_clean[col].apply(lambda x: isinstance(x, (dict, list))).any():
            import json
            df_clean[col] = df_clean[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)

    # 替换所有的 NaN/NaT 为 None，以便插入 Postgres NULL
    df_clean = df_clean.where(pd.notnull(df_clean), None)
    
    # 构建表创建语句，全部当做 TEXT 类型存储比较稳妥，或者部分列特化
    # 简便起见，全部字段设为 TEXT，首列 gen_product_id 作为唯一标识（若存在）
    columns = df_clean.columns.tolist()
    
    create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ("
    col_defs = []
    for col in columns:
        col_defs.append(f'"{col}" TEXT')
    create_table_sql += ", ".join(col_defs)
    create_table_sql += ");"
    
    insert_sql = f'INSERT INTO {table_name} ("' + '", "'.join(columns) + '") VALUES %s'
    
    data_tuples = [tuple(row) for row in df_clean.to_numpy()]

    conn = psycopg2.connect(dsn, connect_timeout=15)
    try:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
            execute_values(cur, insert_sql, data_tuples)
            conn.commit()
            logger.info(f"Successfully saved {len(df_clean)} records to {table_name} in Postgres.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to save results to Postgres: {e}")
    finally:
        conn.close()
