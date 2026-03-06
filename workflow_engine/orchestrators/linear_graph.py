from langgraph.graph import StateGraph, END
from workflow_engine.state import DataScienceState

# Import all our worker agents
from workflow_engine.agents.cleaning_agent import clean_data_node
from workflow_engine.agents.eda_agent import eda_agent_node
from workflow_engine.agents.feature_engineering_agent import feature_engineering_agent_node
from workflow_engine.agents.modelling_agent import modelling_agent_node
from workflow_engine.agents.reporting_agent import reporting_agent_node

def build_linear_pipeline():
    """
    Builds the rigid, linear multi-agent pipeline (Variant 1).
    No Supervisor agent is used here.
    """
    # 1. Initialize the Graph with our State schema
    workflow = StateGraph(DataScienceState)

    # 2. Add all the agent nodes
    workflow.add_node("data_cleaning", clean_data_node)
    workflow.add_node("eda", eda_agent_node)
    workflow.add_node("feature_engineering", feature_engineering_agent_node)
    workflow.add_node("modelling", modelling_agent_node)
    workflow.add_node("reporting", reporting_agent_node)

    # 3. Define the strict linear sequence (The Edges)
    workflow.set_entry_point("data_cleaning")
    workflow.add_edge("data_cleaning", "eda")
    workflow.add_edge("eda", "feature_engineering")
    workflow.add_edge("feature_engineering", "modelling")
    workflow.add_edge("modelling", "reporting")
    workflow.add_edge("reporting", END)

    # 4. Compile the graph into a runnable executable
    app = workflow.compile()
    
    return app