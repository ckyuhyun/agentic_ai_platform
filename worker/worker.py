"""Worker that executes graph node tasks."""


import logging
import time
import os
from dotenv import load_dotenv
from typing import Any, Dict, Optional, List
from datetime import datetime

from agentic_ai_platform.scheduler.task_schema import (
    NodeTask,
    NodeExecution,
    TaskStatus,
)
from agentic_ai_platform.storage.checkpointer import BaseCheckpointer, InMemoryCheckpointer

try:
    from agentic_ai_platform.eval.langsmith.note_trace import post_trace
    from langsmith import Client as LangSmithClient
    LANGSMITH_AVAILABLE = True
except Exception:
    LANGSMITH_AVAILABLE = False

logger = logging.getLogger(__name__)
load_dotenv()

class Worker:

    """
    Worker executes a single node task, handles idempotency,
    persists results, and enqueues downstream tasks.
    """

    def __init__(
        self,
        checkpointer: Optional[BaseCheckpointer] = None,
        node_registry: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            checkpointer: State persistence layer
            node_registry: Dict mapping node_name -> node function
                          e.g., {"planner_agent": planner_agent_fn}
        """
        self.checkpointer = checkpointer or InMemoryCheckpointer()
        self.node_registry = node_registry or {}
        self.logger = logger

    def execute_task(self, task: NodeTask) -> tuple[bool, Optional[str]]:
        """
        Execute a single node task.

        Args:
            task: NodeTask to execute

        Returns:
            Tuple (success: bool, error_msg: Optional[str])
        """
        state_id = task.state_id
        node_name = task.node_name
        task_id = task.task_id

        self.logger.info(
            f"Executing task: task_id={task_id}, state_id={state_id}, node={node_name}"
        )

        start_time = time.time()

        try:
            # Step 1: Load current state snapshot
            snapshot = self.checkpointer.get_snapshot(state_id)
            if snapshot is None:
                error_msg = f"State snapshot not found for state_id={state_id}"
                self.logger.error(error_msg)
                return False, error_msg

            snapshot_version_before = snapshot.pop("_version", 0)
            self.logger.debug(f"Loaded snapshot version {snapshot_version_before}")

            # Step 2: Check idempotency (has this task already been executed?)
            events = self.checkpointer.get_events(state_id)
            for event in events:
                if (
                    event.get("task_id") == task_id
                    and event.get("type") == "node_execution"
                ):
                    self.logger.info(
                        f"Task {task_id} already executed; skipping (idempotent)"
                    )
                    return True, None

            # Step 3: Get node function from registry
            if node_name not in self.node_registry:
                error_msg = f"Node '{node_name}' not found in registry"
                self.logger.error(error_msg)
                return False, error_msg

            node_fn = self.node_registry[node_name]
            self.logger.debug(f"Executing node function: {node_fn}")

            # Step 4: Execute node with state
            # Node functions should accept a serializable state dict and return an updated dict or model.
            try:
                updated_state = node_fn(snapshot)
                self.logger.debug(f"Node execution completed")
            except Exception as e:
                error_msg = f"Node execution failed: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return False, error_msg

            # Step 5: Atomically persist updated state and execution record
            try:
                snapshot_version_after = snapshot_version_before + 1
                self.checkpointer.write_snapshot(
                    state_id=state_id,
                    snapshot=updated_state,
                    version=snapshot_version_before,
                )

                duration_ms = int((time.time() - start_time) * 1000)
                execution_record = {
                    "type": "node_execution",
                    "task_id": task_id,
                    "node_name": node_name,
                    "snapshot_version_before": snapshot_version_before,
                    "snapshot_version_after": snapshot_version_after,
                    "status": TaskStatus.COMPLETED.value,
                    "duration_ms": duration_ms,
                }
                self.checkpointer.append_event(state_id, execution_record)
                self.logger.info(
                    f"Task {task_id} completed in {duration_ms}ms; "
                    f"state version bumped from {snapshot_version_before} to {snapshot_version_after}"
                )

            except ValueError as e:
                # Version mismatch (concurrent write)
                error_msg = f"Concurrency error: {str(e)}"
                self.logger.warning(error_msg)
                return False, error_msg
            except Exception as e:
                error_msg = f"Failed to persist execution: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return False, error_msg

            return True, None

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            # Record failed execution
            try:
                execution_record = {
                    "type": "node_execution",
                    "task_id": task_id,
                    "node_name": node_name,
                    "status": TaskStatus.FAILED.value,
                    "error": error_msg,
                    "duration_ms": duration_ms,
                }
                self.checkpointer.append_event(state_id, execution_record)
            except Exception:
                pass

            return False, error_msg


class LocalWorker:
    """
    LocalWorker executes tasks from an in-memory queue sequentially.
    Useful for development and local testing.
    """

    def __init__(
        self,
        project_name : str,
        checkpointer: Optional[BaseCheckpointer] = None,
        node_registry: Optional[Dict[str, Any]] = None,
        graph_edges: Optional[Dict[str, List[str]]] = None,
    ):
        self.worker = Worker(checkpointer=checkpointer, 
                             node_registry=node_registry)
        
        self.project_name =  project_name
        self.graph_edges = graph_edges or {}
        self.logger = logger

    def run_queue(self,
                  task_queue: List[NodeTask],
                  max_iterations: Optional[int] = None) -> None:
        """
        Execute tasks from queue until empty (or max_iterations reached).
        On success, enqueues downstream tasks per graph_edges so the queue
        drains through the full graph rather than stopping after one node.

        Args:
            task_queue: List of NodeTask to process
            max_iterations: Optional max iterations to prevent infinite loops
        """
        iteration = 0
        while task_queue and (max_iterations is None or iteration < max_iterations):
            task = task_queue.pop(0)
            self.logger.info(
                f"[LocalWorker] Processing task {iteration + 1}: {task.task_id}"
            )

            success, error = self.worker.execute_task(task)


            if not success:
                self.logger.error(f"Task failed: {error}")
                # Optionally: re-queue with backoff or send to DLQ
            else:
                self.logger.info(f"Task succeeded")
                snapshot = self.worker.checkpointer.get_snapshot(task.state_id)
                next_nodes = self.graph_edges.get(task.node_name, [])(snapshot)
                
                if next_nodes[0] == "end":
                    self._post_traces_to_langsmith(task.state_id)
                    self.logger.info('### All Task ends ###')

                elif next_nodes:
                    next_version = snapshot.get("_version", 0) if snapshot else task.snapshot_version + 1
                    for next_node in next_nodes:
                        task_queue.append(NodeTask(
                            state_id=task.state_id,
                            node_name=next_node,
                            snapshot_version=next_version,
                        ))

            iteration += 1


    def _post_traces_to_langsmith(self, 
                                  state_id):
        snapshot = self.worker.checkpointer.get_snapshot(state_id)
        if not snapshot:
            return
        node_traces = snapshot.get("node_traces", [])
        if not node_traces:
            return
        ls = LangSmithClient()
        project = self.project_name

        variable = os.getenv("LANGSMITH_PROJECT")

        if not ls.has_project(project_name=project):
            ls.create_project(project_name=project)
        
        runs = list(ls.list_runs(project_name=project, limit=1, is_root=True))
        if runs:
            post_trace(str(runs[0].id), node_traces)