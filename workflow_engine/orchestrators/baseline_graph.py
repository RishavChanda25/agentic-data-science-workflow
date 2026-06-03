import os
from langgraph.graph import StateGraph, START, END
from workflow_engine.state import DataScienceState

# CORRECT IMPORTS (Matching your linear_graph.py exactly)
from workflow_engine.agents.supervisor_agent import supervisor_node
from workflow_engine.agents.cleaning_agent import clean_data_node
from workflow_engine.agents.eda_agent import eda_agent_node
from workflow_engine.agents.feature_engineering_agent import feature_engineering_agent_node
from workflow_engine.agents.modelling_agent import modelling_agent_node
from workflow_engine.agents.reporting_agent import reporting_agent_node

def route_from_supervisor(state: DataScienceState):
    """Reads the 'next_node' chosen by the Supervisor and routes the graph to it."""
    # Failsafe: If the pipeline is in an error state, kill the graph
    if state.get("error_flag"):
        return END
        
    next_node = state.get("next_node", "FINISH")
    if next_node == "FINISH":
        return END
    return next_node

def build_baseline_graph():
    """Builds the LangGraph for Variant 2 (Baseline)."""
    workflow = StateGraph(DataScienceState)
    
    # Add nodes using your exact function names and string keys
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("data_cleaning", clean_data_node)
    workflow.add_node("eda", eda_agent_node)
    workflow.add_node("feature_engineering", feature_engineering_agent_node)
    workflow.add_node("modelling", modelling_agent_node)
    workflow.add_node("reporting", reporting_agent_node)
    
    # Entry point
    workflow.add_edge(START, "supervisor")
    
    # Conditional Edges from Supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "data_cleaning": "data_cleaning",
            "eda": "eda",
            "feature_engineering": "feature_engineering",
            "modelling": "modelling",
            "reporting": "reporting",
            END: END
        }
    )
    
    # Route all workers straight back to the supervisor
    workflow.add_edge("data_cleaning", "supervisor")
    workflow.add_edge("eda", "supervisor")
    workflow.add_edge("feature_engineering", "supervisor")
    workflow.add_edge("modelling", "supervisor")
    workflow.add_edge("reporting", "supervisor")
    
    return workflow.compile()