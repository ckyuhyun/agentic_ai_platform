from langgraph.graph import StateGraph, START, END

from agentic_ai_platform.state_manager.draft_state import DraftState
from agentic_ai_platform.graph.drafter import make_drafter_node
from agentic_ai_platform.graph.critic import make_critic_node, _route
from agentic_ai_platform.state_manager.draft_state import CriticFeedback







def build_drafter_critic_graph() -> StateGraph:
    critic_node = make_critic_node(CriticFeedback)
    drafter_node = make_drafter_node(DraftState)

    graph = StateGraph(DraftState)

    graph.add_node("drafter", drafter_node)
    graph.add_node("critic", critic_node)

    graph.add_edge(START, "drafter")
    graph.add_edge("drafter", "critic")
    graph.add_conditional_edges(
        "critic",
        _route,
        {"drafter": "drafter", "end": END},
    )

    return graph.compile()


def run(task: str, system_prompt: str = None, max_iterations: int = 3):
    app = build_drafter_critic_graph()

    initial_state = DraftState(
        task=task,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        approval_threshold=0.8,
    )

    final_state = app.invoke(initial_state)

    print(f"\n── Final output (after {final_state['iteration']} iteration(s)) ──")
    print(final_state["final_output"])

    critique = final_state.get("critique")
    if critique:
        print(f"\nCritic score : {critique.score:.2f}  |  approved: {critique.approved}")
        print(f"Reasoning    : {critique.reasoning}")


if __name__ == "__main__":
    run(
        task="Write a 3-sentence summary of why diversification matters in a stock portfolio.",
        max_iterations=3,
    )
