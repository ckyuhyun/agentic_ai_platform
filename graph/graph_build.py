
from langgraph.graph import StateGraph, START, END
from typing import Any

class GraphBuild:
    def __init__(self):
        pass

    def run_graph(self,  
                  graph: StateGraph,
                  init_state : Any):
        
        # run the graph with the initial state
        app = graph.compile() 
        app.invoke(init_state)
        