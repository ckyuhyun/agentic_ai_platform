"""Task schemas for distributed execution."""

from typing import Any, Dict, Optional
from enum import Enum
from pydantic import BaseModel, Field
import uuid
from datetime import datetime


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class NodeTask(BaseModel):
    """Task for executing a single node in the graph."""
    
    state_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique run/state identifier"
    )
    task_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique task identifier for idempotency"
    )
    node_name: str = Field(description="Graph node name to execute (e.g., 'planner_agent')")
    snapshot_version: int = Field(default=0, description="State snapshot version for optimistic locking")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Optional node-specific args")
    retry_count: int = Field(default=0, description="Number of retries so far")
    max_retries: int = Field(default=3, description="Max retries before DLQ")
    created_at: datetime = Field(default_factory=datetime.now)
    status: TaskStatus = Field(default=TaskStatus.PENDING)


class NodeExecution(BaseModel):
    """Record of a node execution for audit/replay."""
    
    task_id: str = Field(description="Task ID that executed this node")
    state_id: str = Field(description="State ID that was processed")
    node_name: str = Field(description="Node executed")
    snapshot_version_before: int = Field(description="State version before execution")
    snapshot_version_after: int = Field(description="State version after execution")
    status: TaskStatus = Field(description="Execution result status")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    duration_ms: int = Field(description="Execution time in milliseconds")
    executed_at: datetime = Field(default_factory=datetime.now)


class GraphRunRequest(BaseModel):
    """Request to start a new graph run."""
    
    query: str = Field(description="User query / input to graph")
    run_id: Optional[str] = Field(
        default=None,
        description="Optional run ID; if None, generates UUID"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional business-level grouping id; defaults to run_id when not given"
    )
    distributed: bool = Field(
        default=False,
        description="If True, use distributed scheduler; if False, run locally"
    )


class GraphRunResponse(BaseModel):
    """Response for a graph run request."""
    
    state_id: str = Field(description="State/run ID for tracking")
    status: TaskStatus = Field(description="Current status")
    message: str = Field(description="Human-readable status message")
