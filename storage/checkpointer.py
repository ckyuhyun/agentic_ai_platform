"""Checkpoint storage for persistent state and event logs."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
from agentic_ai_platform import logger



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
        event_copy["_timestamp"] = datetime.now().isoformat()
        if state_id not in self._events:
            self._events[state_id] = []
        self._events[state_id].append(event_copy)
        logger.debug(f"Event appended for {state_id}: {event.get('type', 'unknown')}")

    def get_events(self, state_id: str) -> List[Dict[str, Any]]:
        """Retrieve all events."""
        return self._events.get(state_id, [])

