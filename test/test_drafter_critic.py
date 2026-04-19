from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from agentic_ai_platform.graph.graph_build import GraphBuild
from agentic_ai_platform.state_manager.draft_state import DraftConfig, DraftState
from agentic_ai_platform.agents.drafter_agent import create_drafter_agent
from agentic_ai_platform.agents.grader_agent import create_grader_agent, _route
from agentic_ai_platform.graph.human_in_loop import HITL
from agentic_ai_platform.state_manager.draft_state import CriticFeedback
from agentic_ai_platform.utils.snapshot_print import print_snapshot
from agentic_ai_platform.llm.llm import LLM

from agentic_ai_platform.tools.tool import Tools


def human_review_node(state: DraftState) -> DraftState:
    print("\n── Human Review Required ─────────────────────────")
    print(f"Task: {state.task}")
    print(f"Draft: {state.draft}")
    if state.critique:
        print(f"Critique score: {state.critique.score:.2f}")
        print(f"Critique issues: {state.critique.issues}")
        print(f"Critique suggestions: {state.critique.suggestions}")

    while True:
        user_input = input("Approve this draft? (y/n): ").strip().lower()
        if user_input in ("y", "yes"):
            state.final_output = state.draft
            break
        elif user_input in ("n", "no"):
            # Reset critique to force drafter to revise
            state.critique = None
            break
        else:
            print("Invalid input, please enter 'y' or 'n'.")

    return state



def build_drafter_critic_graph():

    drafter_node = create_drafter_agent(DraftState,
                                        tool_llm=LLM("llama3.1").llm_instance,
                                         graph_llm=LLM("llama3.1").llm_instance,
                                         tools=[Tools.search_rag, Tools.search_web])
    critic_node = create_grader_agent(CriticFeedback,
                                      graph_llm=LLM("llama3.1").llm_instance)
    

    graph = StateGraph(DraftState)

    graph.add_node("drafter", drafter_node)
    graph.add_node("critic", critic_node)
    graph.add_node("human_review", human_review_node)

    
    graph.set_entry_point("drafter")
    graph.add_edge("drafter", "critic")
    graph.add_conditional_edges(
        "critic",
        _route,
        {"drafter": "drafter", 
         "end": END, 
         "human_review": "human_review"},
    )

    return graph


def run(task: str, 
        system_prompt: str = None,
        max_iterations: int = 3):
    
    app = build_drafter_critic_graph()

    initial_state = DraftState(
        task=task,
        system_prompt=system_prompt,
        drafter_config=DraftConfig(max_iterations=max_iterations, approval_threshold=0.8),
    )

    graph = GraphBuild()
    graph.run_graph(app, initial_state, stream_mode=["values", "custom", "updates"])   
    
    

    


    # final_state = app.invoke(initial_state)

    # print(f"\n── Final output (after {final_state['iteration']} iteration(s)) ──")
    # print(final_state["final_output"])

    # critique = final_state.get("critique")
    # if critique:
    #     print(f"\nCritic score : {critique.score:.2f}  |  approved: {critique.approved}")
    #     print(f"Reasoning    : {critique.reasoning}")


if __name__ == "__main__":
    run(
        task="I would like to get some queries to ask how to invest in stocks well" \
        "",
        system_prompt=(
            "You are a professional financial advisor. you have speical and deep expertise in tech stock investment and primary have been invested in the states market. " \
            "Provide clear and concise guidance on stock investment strategies."
        ),
        max_iterations=3,
    )
