from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import StateSnapshot
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from typing import Any, Literal, Optional

try:
    from agentic_ai_platform.eval.langsmith.note_trace import post_trace
    from langsmith import Client as LangSmithClient
    LANGSMITH_AVAILABLE = True
except Exception:
    LANGSMITH_AVAILABLE = False


StreamMode = Literal["values", "messages", "custom", "updates"]


class GraphBuild:
    def __init__(self, enabled_persistentMemory=True):
        self.app = None
        self.config: Optional[RunnableConfig] = None
        self.enabled_persistentMemory = enabled_persistentMemory

    def run_graph(
        self,
        graph: StateGraph,
        init_state: Any,
        stream_mode: StreamMode = "values",
    ):
        checkpoints = InMemorySaver()
        if self.enabled_persistentMemory:
            self.app = graph.compile(checkpointer=checkpoints)
        else:
            self.app = graph.compile()

        self.config = {
            "configurable": {"thread_id": "1"},
            "metadata" : {"run_name": "LLM_service", "task": str(init_state.task)}
            }


        try:
            for chunk in self.app.stream(init_state,
                                        config=self.config,
                                        stream_mode=stream_mode,
                                        version="v2"):
                self._handle_chunk(chunk)
        except ValueError as e:
            RuntimeError(f"Error during graph execution: {str(e)}")

        #if LANGSMITH_AVAILABLE:
        #   self._post_traces_to_langsmith()

        

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
        pass
        #for key, value in chunk.items():
        #    print(f"  [values] {key}: {str(value)}")

    def _handle_messages(self, chunk: tuple):
        """chunk is (message, metadata) — emitted token by token."""
        pass
        # for key, value in chunk.items():
        #     print(f"  [messages] {key}: {str(value)}")

    def _handle_custom(self, chunk: Any):
        """chunk is whatever the node emits via streamable() / dispatch_custom_event."""
        pass
        #print(f"  [custom] {str(chunk)}")

    def _handle_updates(self, chunk: Any):
        """chunk is the updated state dict emitted after each node completes."""
        pass
        # for key, value in chunk.items():
        #     print(f"  [updates] {key}: {str(value)}")


    def _post_traces_to_langsmith(self):
        snapshot = self.get_state()
        if not snapshot:
            return
        node_traces = snapshot.values.get("node_traces", [])
        if not node_traces:
            return
        ls = LangSmithClient()
        project = (self.config or {}).get("metadata", {}).get("run_name", "default")
        runs = list(ls.list_runs(project_name=project, limit=1, is_root=True))
        if runs:
            post_trace(str(runs[0].id), node_traces)

    # ── state access ──────────────────────────────────────────────────────────

    def get_state(self) -> Optional[StateSnapshot]:
        """Return the latest StateSnapshot for the current thread."""
        if self.app is None or self.config is None:
            return None
        return self.app.get_state(self.config)
        
