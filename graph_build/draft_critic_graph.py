from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.prebuilt import ToolNode

from agentic_ai_platform.graph.graph_build import GraphBuild
from agentic_ai_platform.state_manager.supervise_state import DraftConfig, SuperviseState
from agentic_ai_platform.agents.planner_execution_agents.planner_agent import create_planner_agent
from agentic_ai_platform.agents.drafter_critic_agents.drafter_agent import create_drafter_agent
from agentic_ai_platform.agents.drafter_critic_agents.grader_agent import create_grader_agent
from agentic_ai_platform.graph.human_in_loop import HITL
from agentic_ai_platform.state_manager.supervise_state import CriticFeedback
from agentic_ai_platform.state_manager.plan_state import PlanState
from agentic_ai_platform.tools.grader_tools import EvalsTools
from agentic_ai_platform.llm.llm import LLM
from agentic_ai_platform.tools.tool import Tools
from agentic_ai_platform import prompt_hub




def route(state: SuperviseState) -> str:
    if state.messages[-1].tool_calls[0]['name'] == "check_hallucinations":
        return "hallucination_check"
    elif state.iteration >= state.graph_config.max_iterations:
         return "human_review"
    elif state.critique and state.critique.approved:
        return "end"
    return "drafter"
    # if state.critique.hallucination_score > 0:
    #     return "hallucination_check"
    # if state.critique and state.critique.approved:
    #     return "end"
    # if state.iteration >= state.drafter_config.max_iterations:
    #     return "human_review"
    # return "drafter"

def human_review_node(state: SuperviseState) -> SuperviseState:
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
    """
    Build graph
    """
    plan_system_prompt = prompt_hub.get_prompt_by_type_version_tags(prompt_type="planner", 
                                                                    version_id="0").as_system_messages()
    plan_agent = create_planner_agent(schema=PlanState,
                                      system_prompt=plan_system_prompt,
                                      graph_llm=LLM("llama3.1").llm_instance)

    drafter_agent = create_drafter_agent(SuperviseState,
                                        tool_llm=LLM("llama3.1").llm_instance,
                                         graph_llm=LLM("llama3.1").llm_instance,
                                         tools=[Tools.search_rag, Tools.search_web])
    
    
    critic_prompt = prompt_hub.get_prompt_by_type_version_tags(prompt_type="critic", 
                                                               version_id="0").as_system_messages()
    critic_agent = create_grader_agent(CriticFeedback,
                                      system_prompt=critic_prompt,
                                      eval_tools=[EvalsTools.check_hallucinations],
                                      graph_llm=LLM("llama3.1").llm_instance)
    

    eval_tools = [EvalsTools.check_hallucinations]
    tools = ToolNode(eval_tools)

    graph = StateGraph(SuperviseState)

    graph.add_node("plan", plan_agent)
    graph.add_node("drafter", drafter_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("human_review", human_review_node)
    graph.add_node("eval_tools", tools)

    
    graph.set_entry_point("plan")
    graph.add_edge("plan","drafter")
    graph.add_edge("drafter", "critic")
    graph.add_conditional_edges(
        "critic",
        route,
        {
            "hallucination_check": "eval_tools",
            "end": END,
            "human_review": "human_review",
            "drafter": "drafter",
        },
    )
    graph.add_edge("eval_tools", "critic")
    graph.add_edge("human_review", END)

    return graph

