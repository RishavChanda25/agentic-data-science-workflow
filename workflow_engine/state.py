import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional

class DataScienceState(TypedDict):
    """
    The shared state for the Multi-Agent Data Science Workflow.
    Updated for Variant 3: Neurosymbolic Routing.
    """
    # Standard LangGraph message tracking
    messages: Annotated[List[Any], operator.add]
    
    # User inputs and goals
    user_request: str
    target_variable: Optional[str]
    
    # Data pointers
    raw_dataset_path: str
    current_dataset_path: str
    
    # --- VARIANT 3: DYNAMIC ROUTING MEMORY ---
    active_preset: str                   # e.g., "RAPID_BASELINE"
    proposed_route: List[str]            # e.g., ["data_cleaning", "modelling", "reporting"]
    z3_verification_status: bool         # True if route passed the mathematical shield
    dataset_metadata: Dict[str, Any]     # Set by the Pre-Flight Profiler
    
    # Artifact tracking
    artifacts: Dict[str, str] 
    
    # Orchestration and error handling
    current_step: str
    error_flag: bool
    error_message: Optional[str]
    revision_count: int
    user_preferences: dict
    total_sleep_time: float

    # Observability Ledger
    total_input_tokens: int
    total_output_tokens: int
    api_call_timestamps: Annotated[list, operator.add]

    # Orchestration Tax trackers
    supervisor_latency: Annotated[float, operator.add]
    supervisor_tokens: Annotated[int, operator.add]
    supervisor_calls: Annotated[int, operator.add]
    
    next_node: str