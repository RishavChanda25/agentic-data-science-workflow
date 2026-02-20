import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional

class DataScienceState(TypedDict):
    """
    The shared state for the Multi-Agent Data Science Workflow.
    This dictionary is passed from node to node in the LangGraph.
    """
    # Standard LangGraph message tracking (appends new messages to the list)
    messages: Annotated[List[Any], operator.add]
    
    # User inputs and goals
    user_request: str
    target_variable: Optional[str]
    
    # Data pointers (Crucial: We pass paths, not raw DataFrames)
    raw_dataset_path: str
    current_dataset_path: str
    
    # Artifact tracking (Plots, models, and metrics)
    artifacts: Dict[str, str]  # e.g., {"correlation_matrix": "data/corr.png"}
    
    # Orchestration and error handling
    current_step: str
    error_flag: bool
    error_message: Optional[str]
    
    # Optional: For the future Critic/Debate Agent
    revision_count: int