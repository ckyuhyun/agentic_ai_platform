from langsmith import Client

from agentic_ai_platform.state_manager.draft_state import NodeTrace

ls = Client()


def post_trace(run_name : str,  node_trace : NodeTrace):
    for trace in node_trace:
        ls.create_run(
            name=run_name,
            metadata={
                "node": trace.node,
                "iteration": trace.iteration,
                "latency_ms": trace.latency_ms,
                "model": trace.model,
                "tool_calls_made": trace.tool_calls_made,
                "draft_len": trace.draft_len,
                "score": trace.score,
                "approved": trace.approved,
                "issue_count": trace.issue_count,
            }
        )