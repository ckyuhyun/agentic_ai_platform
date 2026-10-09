from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

from agentic_ai_platform import Postgres_Engine


class PostgresCheckpointer:
    def __init__(self):
        self._checkpointer = None

        # Reuse the engine's connection settings, minus the SQLAlchemy
        # "+psycopg" driver suffix, which psycopg/libpq doesn't understand.
        self.db_url = Postgres_Engine.url.set(drivername="postgresql")\
                                         .render_as_string(hide_password=False)
        self._pool = None

    @property
    def checkpointer(self):
        return self._checkpointer

    async def setup(self):
        """
        Async Postgres Saver setup. Keeps the connection pool open for the
        lifetime of this wrapper -- AsyncPostgresSaver.from_conn_string() is a
        context manager that closes its connection on exit, so the
        checkpointer it yields becomes unusable once that block ends.

        Uses AsyncPostgresSaver (not the sync PostgresSaver) because the
        compiled graph is driven via app.astream()/app.ainvoke() -- the sync
        saver's async methods (aget_tuple/aput/...) aren't implemented and
        raise NotImplementedError as soon as the graph tries to checkpoint.
        """
        self._pool = AsyncConnectionPool(
            conninfo=self.db_url,
            kwargs={"autocommit": True, "row_factory": dict_row},
            open=False,
        )
        await self._pool.open()
        checkpointer = AsyncPostgresSaver(self._pool)
        await checkpointer.setup()
        self._checkpointer = checkpointer

    async def close(self):
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


