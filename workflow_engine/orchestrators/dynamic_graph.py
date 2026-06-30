from langgraph.graph import StateGraph, END
from workflow_engine.state import DataScienceState

# All agents safely imported from the same directory
from workflow_engine.agents.supervisor_agent import supervisor_node
from workflow_engine.agents.cleaning_agent import clean_data_node
from workflow_engine.agents.eda_agent import eda_agent_node
from workflow_engine.agents.feature_engineering_agent import feature_engineering_agent_node
from workflow_engine.agents.modelling_agent import modelling_agent_node
from workflow_engine.agents.reporting_agent import reporting_agent_node

def route_from_supervisor(state: DataScienceState):
    """Conditional edge router based on Supervisor's output."""
    next_node = state.get("next_node", "FINISH")
    if next_node == "FINISH":
        return END
    return next_node

def build_dynamic_pipeline():
    workflow = StateGraph(DataScienceState)

    # 1. Add Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("data_cleaning", clean_data_node)
    workflow.add_node("eda", eda_agent_node)
    workflow.add_node("feature_engineering", feature_engineering_agent_node)
    workflow.add_node("modelling", modelling_agent_node)
    workflow.add_node("reporting", reporting_agent_node)

    # 2. Add Edges (Workers always report back to the Dispatcher)
    workflow.add_edge("data_cleaning", "supervisor")
    workflow.add_edge("eda", "supervisor")
    workflow.add_edge("feature_engineering", "supervisor")
    workflow.add_edge("modelling", "supervisor")
    workflow.add_edge("reporting", "supervisor")

    # 3. Add Conditional Routing
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

    workflow.set_entry_point("supervisor")
    return workflow.compile()