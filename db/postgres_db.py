import os
import psycopg2

from contextlib import contextmanager


def connection_params() -> dict:
    return dict(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5433)),
        dbname=os.getenv("POSTGRES_DB", "agnetic_ai_db"),
        user=os.getenv("POSTGRES_USER", "admin"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


@contextmanager
def postgres_db_connect():
    conn = psycopg2.connect(**connection_params())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()