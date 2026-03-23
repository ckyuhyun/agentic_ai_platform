from pydantic import BaseModel
from typing import Any, Dict
from datetime import datetime
import uuid

class AgentLogTrace(BaseModel):
    trace_id: uuid.UUID
    parent_id: uuid.UUID | None  # For sub-agents or nested loops
    timestamp: datetime
    state_revision: int
    
    thought: str                
    action_tool: str            
    action_input: Dict[str, Any] 
    observation: Any           
    
    latency_ms: float          


class AgentLogTracker:
    def __init__(self):
        self.traces: Dict[uuid.UUID, AgentLogTrace] = {}
        self.current_revision = 0

    def start_trace_logging(self, 
                    parent_id: uuid.UUID | None, 
                    thought: str, 
                    action_tool: str, 
                    action_input: Dict[str, Any]) -> uuid.UUID:
        trace_id = uuid.uuid4()
        trace = AgentLogTrace(
            trace_id=trace_id,
            parent_id=parent_id,
            timestamp=datetime.now(),
            state_revision=self.current_revision,
            thought=thought,
            action_tool=action_tool,
            action_input=action_input,
            observation=None,
            latency_ms=0.0
        )
        self.traces[trace_id] = trace
        return trace_id

    def end_trace_logging(self, 
                  trace_id: uuid.UUID, 
                  observation: Any, 
                  latency_ms: float):
        if trace_id in self.traces:
            trace = self.traces[trace_id]
            trace.observation = observation
            trace.latency_ms = latency_ms
        else:
            raise ValueError(f"Trace ID {trace_id} not found")