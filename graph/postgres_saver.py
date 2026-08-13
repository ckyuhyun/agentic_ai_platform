import os
import psycopg2
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, StateGraph, MessagesState
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from dotenv import load_dotenv


from agentic_ai_platform.db.postgres_db import connection_params



load_dotenv()

class PostgresSaverWrapper:
    def __init__(self):        
        self.url = connection_params().get('host')
        self.port =  connection_params().get('port')
        self.dbname = connection_params().get('dbname')
        self.user = connection_params().get('user')
        self.password = connection_params().get('password')
        self._checkpointer = None

        self.db_url = f"postgresql://{self.user}:{self.password}@{self.url}:{self.port}/{self.dbname}"
        self._pool = None

    @property
    def checkpointer(self):
        return self._checkpointer

    def setup(self):
        """
        Postgres Saver setup. Keeps the connection pool open for the
        lifetime of this wrapper -- PostgresSaver.from_conn_string() is a
        context manager that closes its connection on exit, so the
        checkpointer it yields becomes unusable once that block ends.
        """
        self._pool = ConnectionPool(conninfo=self.db_url, kwargs={"autocommit": True, "row_factory":dict_row})
        checkpointer = PostgresSaver(self._pool)
        checkpointer.setup()
        self._checkpointer = checkpointer

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool = None


