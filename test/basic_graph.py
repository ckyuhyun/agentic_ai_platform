from agentic_ai_platform.graph.graph_build import GraphBuild
from agentic_ai_platform.rag.embedding import Embeddings
from agentic_ai_platform.model.llm import llm
from pydantic import BaseModel

from langgraph.graph import StateGraph, START, END

class States(BaseModel):
    query : str
    


def embedding(state: States) -> str:
    # In a real implementation, this would call an embedding model
    emd_model= Embeddings()
    query = state.query
    llm_instance = llm("llama3")
    llm_instance.chat(query)
    result = emd_model.generate_embedding(query)

    return f"embedding_of({state.query})"



def test_basic_graph():
    graph_build = GraphBuild()
    
    graph = StateGraph(States)
    
    graph.add_node("embedding", embedding)

    graph.add_edge(START, "embedding")    
    graph.add_edge("embedding", END)

    # Run the graph with an initial state (can be empty for this test)
    initial_state = {'query': "What is the stock price of AAPL?"}
    graph_build.run_graph(graph, initial_state)

    print("Basic graph test completed successfully.")


if __name__ == "__main__":
    test_basic_graph()

