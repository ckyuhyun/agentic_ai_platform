from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import StateSnapshot
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from typing import Any, Literal, Optional


StreamMode = Literal["values", "messages", "custom", "updates"]


class GraphBuild:
    def __init__(self):
        self.app = None
        self.config: Optional[RunnableConfig] = None

    def run_graph(
        self,
        graph: StateGraph,
        init_state: Any,
        stream_mode: StreamMode = "values",
    ):
        checkpoints = InMemorySaver()
        self.app = graph.compile(checkpointer=checkpoints)
        self.config = {"configurable": {"thread_id": "1"}}

        try:
            for chunk in self.app.stream(init_state, 
                                        config=self.config, 
                                        stream_mode=stream_mode,
                                        version="v2"):
                self._handle_chunk(chunk)
        except ValueError as e:
            RuntimeError(f"Error during graph execution: {str(e)}")

    # ── chunk handlers ────────────────────────────────────────────────────────

    def _handle_chunk(self, chunk: Any):
        if chunk['type'] == "values":
            self._handle_values(chunk)
        elif chunk['type'] == "messages":
            self._handle_messages(chunk)
        elif chunk['type'] == "custom":
            self._handle_custom(chunk)
        elif chunk['type'] == "custom":
            self._handle_custom(chunk)
        elif chunk['type'] == "updates":
            self._handle_updates(chunk)
        else:
            raise ValueError(f"Unknown chunk type: {chunk['type']}")

    def _handle_values(self, chunk: dict):
        """chunk is the full state dict emitted after each node completes."""
        for key, value in chunk.items():
            print(f"  [values] {key}: {str(value)}")

    def _handle_messages(self, chunk: tuple):
        """chunk is (message, metadata) — emitted token by token."""
        for key, value in chunk.items():
            print(f"  [messages] {key}: {str(value)}")

    def _handle_custom(self, chunk: Any):
        """chunk is whatever the node emits via streamable() / dispatch_custom_event."""
        print(f"  [custom] {str(chunk)}")

    def _handle_updates(self, chunk: Any):
        """chunk is the updated state dict emitted after each node completes."""
        for key, value in chunk.items():
            print(f"  [updates] {key}: {str(value)}")


    # ── state access ──────────────────────────────────────────────────────────

    def get_state(self) -> Optional[StateSnapshot]:
        """Return the latest StateSnapshot for the current thread."""
        if self.app is None or self.config is None:
            return None
        return self.app.get_state(self.config)
