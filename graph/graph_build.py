
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import StateSnapshot
from langchain_core.runnables import RunnableConfig

from typing import Any, Optional


class GraphBuild:
    def __init__(self):
        self.app = None
        self.config: Optional[RunnableConfig] = None

    def run_graph(self, graph: StateGraph, init_state: Any):
        checkpoints = InMemorySaver()
        self.app = graph.compile(checkpointer=checkpoints)
        
        self.config = {"configurable": {"thread_id": "1"}}
        self.app.invoke(init_state, config=self.config)

    def get_state(self) -> Optional[StateSnapshot]:
        """Return the latest StateSnapshot for the current thread."""
        if self.app is None or self.config is None:
            return None
        return self.app.get_state(self.config)

    
        