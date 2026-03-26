from typing import Annotated

from pydantic import BaseModel

from agentic_ai_platform.graph.graph_build import GraphBuild
from agentic_ai_platform.model.llm import llm

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class States(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = []


@tool("search_final_news")
def search_final_news(ticket: str, days_back: int) -> str:
    """Search news for a specific stock during a time period and return the most relevant articles."""
    return f"Search results for: {ticket} in the past {days_back} days"


tools = [search_final_news]

llm_instance = llm("llama3.1")
llm_instance.bind_tools(tools)


def call_llm(state: States) -> dict:
    """Invoke the LLM and append its response (with any tool_calls) to messages."""    
    response = llm_instance.llm_instance.invoke(state.messages)
    return {"messages": [response]}


def test_basic_graph():
    graph_build = GraphBuild()

    graph = StateGraph(States)

    graph.add_node("call_llm", call_llm)
    graph.add_node("tools", ToolNode(tools))

    
    graph.set_entry_point("call_llm")
    # Route to tools if LLM returned tool_calls, otherwise END
    graph.add_conditional_edges("call_llm", tools_condition)
    # After tool execution, loop back so LLM can process the result
    graph.add_edge("tools", "call_llm")

    initial_state = {"messages": [HumanMessage(content="Can you check news for TSLA over the last 2 days?")]}
    graph_build.run_graph(graph, initial_state)

    print("Basic graph test completed successfully.")


if __name__ == "__main__":
    test_basic_graph()

