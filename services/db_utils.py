# app/services/db_utils.py
from db import get_conn

def q(sql, params=None, fetch="all"):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params or [])
        if fetch == "one":
            return cur.fetchone()
        if fetch == "all":
            return cur.fetchall()
        return None

def exec_sql(sql, params=None, returning=False):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params or [])
        if returning:
            return cur.fetchone()
        conn.commit()

def get_connection():
    """
    Return a new DB connection (caller is responsible for closing it).
    Mirrors the underlying db.get_conn() used by q/exec_sql.
    """
    return get_conn()