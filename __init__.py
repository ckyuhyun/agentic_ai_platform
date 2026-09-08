import logging
import os
from rich.logging import RichHandler
from sqlalchemy import create_engine, select, Column, Integer, String, func, Uuid
from sqlalchemy.orm import declarative_base, sessionmaker

from agentic_ai_platform.utils.logging_config import setup_logging

setup_logging()

## Logger
logger = logging.getLogger(__name__)

## Prompt Registry
from agentic_ai_platform.prompt_storage.prompt_registry import PromptRegistry
prompt_hub = PromptRegistry()


## PostGres DB
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", 5433)
POSTGRES_DB = os.getenv("POSTGRES_DB", "agnetic_ai_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

_post_db_url_ = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

Postgres_Engine = create_engine(_post_db_url_, pool_size=5, max_overflow=10)
