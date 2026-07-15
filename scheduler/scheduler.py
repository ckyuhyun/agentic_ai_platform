"""Distributed scheduler for graph execution."""

import json
import logging
from typing import Optional, List
from uuid import uuid4
from datetime import datetime

from agentic_ai_platform.scheduler.task_schema import (
    NodeTask,
    GraphRunRequest,
    GraphRunResponse,
    TaskStatus,
)
from agentic_ai_platform.storage.checkpointer import BaseCheckpointer, InMemoryCheckpointer

logger = logging.getLogger(__name__)


class Scheduler:
    """
    Scheduler orchestrates graph execution by:
    1. Persisting initial state
    2. Enqueueing node tasks to a broker (or local queue)
    3. Managing state transitions
    """

    def __init__(
        self,
        checkpointer: Optional[BaseCheckpointer] = None,
        task_queue: Optional[List] = None,
        distributed: bool = False,
    ):
        """
        Args:
            checkpointer: State persistence layer (defaults to in-memory)
            task_queue: Task queue implementation (e.g., Celery, Kafka, or list for local)
            distributed: If True, enqueue tasks to broker; if False, return tasks for local runner
        """
        self.checkpointer = checkpointer or InMemoryCheckpointer()
        self.task_queue = [] if task_queue is None else task_queue
        self.distributed = distributed
        self.logger = logger

    def start_run(
        self,
        request: GraphRunRequest,
        start_node: str,
    ) -> GraphRunResponse:
        """
        Initialize and start a new graph run.

        Args:
            request: GraphRunRequest with query and optional run_id
            start_node: Name of the starting node

        Returns:
            GraphRunResponse with state_id and status
        """
        state_id = request.run_id or str(uuid4())
        session_id = request.session_id or state_id
        self.logger.info(f"Starting run: state_id={state_id}, session_id={session_id}, query={request.query}")

        # Persist initial state as simple serializable state dict.
        # Downstream node functions may normalize this into a SuperviseState model as needed.
        # state_id/session_id are carried through every node so LLM calls, tool calls,
        # and stored records can be grouped and replayed by run (see build_llm_config).
        initial_state = {
            "query": request.query,
            "tool_states": [],
            "messages": [],
            "state_id": state_id,
            "session_id": session_id,
        }

        # Persist initial snapshot
        try:
            self.checkpointer.write_snapshot(
                state_id=state_id,
                snapshot=initial_state,
                version=0,
            )
            self.logger.debug(f"Persisted initial snapshot for state_id={state_id}")
        except Exception as e:
            self.logger.error(f"Failed to persist initial state: {e}")
            return GraphRunResponse(
                state_id=state_id,
                status=TaskStatus.FAILED,
                message=f"Failed to initialize state: {e}",
            )

        # Enqueue first task
        start_task = NodeTask(
            state_id=state_id,
            node_name=start_node,
            snapshot_version=0,
        )

        try:
            self._enqueue_task(start_task)
            self.logger.info(f"Enqueued start task: {start_task.task_id}")
        except Exception as e:
            self.logger.error(f"Failed to enqueue start task: {e}")
            return GraphRunResponse(
                state_id=state_id,
                status=TaskStatus.FAILED,
                message=f"Failed to enqueue task: {e}",
            )

        return GraphRunResponse(
            state_id=state_id,
            status=TaskStatus.PENDING,
            message=f"Graph run started. Execute tasks from queue.",
        )

    def enqueue_tasks(self, tasks: List[NodeTask]) -> None:
        """
        Enqueue multiple downstream tasks (called after a node execution).

        Args:
            tasks: List of NodeTask to enqueue
        """
        for task in tasks:
            self._enqueue_task(task)

    def _enqueue_task(self, task: NodeTask) -> None:
        """
        Internal method to enqueue a task.

        For distributed mode, this would push to Celery/Kafka.
        For local mode, this appends to in-memory queue.
        """
        if self.distributed:
            # In production, use task_queue.apply_async() or broker.send_message()
            self.logger.info(f"[DISTRIBUTED] Enqueuing task: {task.task_id} -> {task.node_name}")
            # Placeholder: In step 6, replace with actual Celery/Kafka integration
            self.task_queue.append(task)
        else:
            # Local mode: append to queue
            self.logger.debug(f"[LOCAL] Enqueuing task: {task.task_id} -> {task.node_name}")
            self.task_queue.append(task)

    def dequeue_task(self) -> Optional[NodeTask]:
        """
        Dequeue the next task from queue.

        Called by worker or local runner.
        """
        if isinstance(self.task_queue, list):
            return self.task_queue.pop(0) if self.task_queue else None
        # In production, broker-specific dequeue logic
        return None

    def get_run_status(self, state_id: str) -> dict:
        """
        Get current status of a run.

        Args:
            state_id: Run ID

        Returns:
            Dict with state_id, latest_snapshot, and event_count
        """
        try:
            snapshot = self.checkpointer.get_snapshot(state_id)
            events = self.checkpointer.get_events(state_id)
            return {
                "state_id": state_id,
                "snapshot_version": snapshot.get("_version", 0) if snapshot else None,
                "event_count": len(events),
                "events": events[-5:] if events else [],  # Last 5 events
            }
        except Exception as e:
            self.logger.error(f"Failed to get run status: {e}")
            return {"state_id": state_id, "error": str(e)}
