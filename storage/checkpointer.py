"""Checkpoint storage for persistent state and event logs."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseCheckpointer(ABC):
    """Abstract base for checkpoint storage implementations."""

    @abstractmethod
    def get_snapshot(self, state_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the latest state snapshot for a given state_id.

        Args:
            state_id: Unique run/state identifier

        Returns:
            Deserialized state dict or None if not found
        """
        pass

    @abstractmethod
    def write_snapshot(
        self,
        state_id: str,
        snapshot: Dict[str, Any],
        version: int,
    ) -> None:
        """
        Persist a state snapshot (with optimistic locking via version).

        Args:
            state_id: Unique run/state identifier
            snapshot: State dict to persist
            version: Expected current version (for optimistic locking)

        Raises:
            ValueError: If version mismatch (concurrent write detected)
        """
        pass

    @abstractmethod
    def append_event(
        self,
        state_id: str,
        event: Dict[str, Any],
    ) -> None:
        """
        Append an event to the immutable event log for a state.

        Args:
            state_id: Unique run/state identifier
            event: Event dict to append (e.g., node execution record)
        """
        pass

    @abstractmethod
    def get_events(self, state_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all events for a state.

        Args:
            state_id: Unique run/state identifier

        Returns:
            List of events (empty if none)
        """
        pass


class InMemoryCheckpointer(BaseCheckpointer):
    """
    Simple in-memory checkpoint store for development and testing.
    Not suitable for production.
    """

    def __init__(self):
        self._snapshots: Dict[str, Dict[str, Any]] = {}  # {state_id: snapshot}
        self._events: Dict[str, List[Dict[str, Any]]] = {}  # {state_id: [events]}
        self._versions: Dict[str, int] = {}  # {state_id: version}

    def get_snapshot(self, state_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve latest snapshot."""
        return self._snapshots.get(state_id)

    def write_snapshot(
        self,
        state_id: str,
        snapshot: Dict[str, Any],
        version: int,
    ) -> None:
        """Write snapshot with optimistic version check."""
        current_version = self._versions.get(state_id, 0)
        if current_version != version:
            raise ValueError(
                f"Version mismatch for state_id={state_id}: "
                f"expected {version}, got {current_version}"
            )

        snapshot_copy = snapshot.copy()
        snapshot_copy["_version"] = version + 1
        self._snapshots[state_id] = snapshot_copy
        self._versions[state_id] = version + 1
        logger.debug(f"Snapshot written for {state_id}, version now {version + 1}")

    def append_event(
        self,
        state_id: str,
        event: Dict[str, Any],
    ) -> None:
        """Append event to log."""
        event_copy = event.copy()
        event_copy["_timestamp"] = datetime.utcnow().isoformat()
        if state_id not in self._events:
            self._events[state_id] = []
        self._events[state_id].append(event_copy)
        logger.debug(f"Event appended for {state_id}: {event.get('type', 'unknown')}")

    def get_events(self, state_id: str) -> List[Dict[str, Any]]:
        """Retrieve all events."""
        return self._events.get(state_id, [])


class PostgresCheckpointer(BaseCheckpointer):
    """
    Production checkpoint store using PostgreSQL.
    Requires psycopg2 and a postgres connection string in environment.
    """

    def __init__(self, connection_string: str):
        """
        Args:
            connection_string: PostgreSQL connection URL (e.g., postgresql://user:pass@localhost/db)
        """
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            self.psycopg2 = psycopg2
            self.RealDictCursor = RealDictCursor
        except ImportError:
            raise ImportError(
                "psycopg2 required for PostgresCheckpointer. "
                "Install via: pip install psycopg2-binary"
            )

        self.connection_string = connection_string
        self._init_schema()
        logger.info("PostgresCheckpointer initialized")

    def _init_schema(self) -> None:
        """Create tables if not exist."""
        with self.psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # State snapshots table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS state_snapshots (
                        state_id TEXT PRIMARY KEY,
                        snapshot JSONB NOT NULL,
                        version INTEGER NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Event log table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS state_events (
                        id SERIAL PRIMARY KEY,
                        state_id TEXT NOT NULL,
                        event JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (state_id) REFERENCES state_snapshots(state_id) ON DELETE CASCADE
                    )
                """)

                # Index for fast event retrieval
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_state_events_state_id 
                    ON state_events(state_id)
                """)

                conn.commit()
                logger.debug("Schema initialized")

    def get_snapshot(self, state_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve latest snapshot."""
        try:
            with self.psycopg2.connect(self.connection_string) as conn:
                with conn.cursor(cursor_factory=self.RealDictCursor) as cur:
                    cur.execute(
                        "SELECT snapshot, version FROM state_snapshots WHERE state_id = %s",
                        (state_id,),
                    )
                    row = cur.fetchone()
                    if row:
                        snapshot = row["snapshot"]
                        snapshot["_version"] = row["version"]
                        return snapshot
        except Exception as e:
            logger.error(f"Failed to retrieve snapshot for {state_id}: {e}")
        return None

    def write_snapshot(
        self,
        state_id: str,
        snapshot: Dict[str, Any],
        version: int,
    ) -> None:
        """Write snapshot with optimistic locking."""
        try:
            with self.psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    # Check current version
                    cur.execute(
                        "SELECT version FROM state_snapshots WHERE state_id = %s",
                        (state_id,),
                    )
                    row = cur.fetchone()
                    current_version = row[0] if row else 0

                    if current_version != version:
                        raise ValueError(
                            f"Version mismatch for state_id={state_id}: "
                            f"expected {version}, got {current_version}"
                        )

                    # Insert or update
                    cur.execute(
                        """
                        INSERT INTO state_snapshots (state_id, snapshot, version)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (state_id) DO UPDATE
                        SET snapshot = EXCLUDED.snapshot, 
                            version = EXCLUDED.version,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (state_id, json.dumps(snapshot), version + 1),
                    )
                    conn.commit()
                    logger.debug(f"Snapshot written for {state_id}, version now {version + 1}")
        except Exception as e:
            logger.error(f"Failed to write snapshot for {state_id}: {e}")
            raise

    def append_event(
        self,
        state_id: str,
        event: Dict[str, Any],
    ) -> None:
        """Append event to log."""
        try:
            with self.psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO state_events (state_id, event) VALUES (%s, %s)",
                        (state_id, json.dumps(event)),
                    )
                    conn.commit()
                    logger.debug(f"Event appended for {state_id}")
        except Exception as e:
            logger.error(f"Failed to append event for {state_id}: {e}")
            raise

    def get_events(self, state_id: str) -> List[Dict[str, Any]]:
        """Retrieve all events."""
        try:
            with self.psycopg2.connect(self.connection_string) as conn:
                with conn.cursor(cursor_factory=self.RealDictCursor) as cur:
                    cur.execute(
                        "SELECT event, created_at FROM state_events WHERE state_id = %s ORDER BY id ASC",
                        (state_id,),
                    )
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve events for {state_id}: {e}")
        return []
