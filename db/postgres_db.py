import asyncio
from agentic_ai_platform import Postgres_Engine
from contextlib import asynccontextmanager


@asynccontextmanager
async def postgres_db_connect():
    conn = Postgres_Engine.connect()

    try:
        yield conn
        conn.commit()
        await asyncio.sleep(0.5)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()



