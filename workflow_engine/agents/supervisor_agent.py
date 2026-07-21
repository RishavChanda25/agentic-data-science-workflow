import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from workflow_engine import state
from workflow_engine.state import DataScienceState
from workflow_engine.schemas.routing_schema import RouteProposal
from workflow_engine.utils.z3_verifier import verify_proposed_route

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

def supervisor_node(state: DataScienceState):
    """Variant 3 Supervisor: Neurosymbolic Routing with Zero-Tax Dispatch."""
    print("\n🤖 [Supervisor] Analyzing state to determine next step...")
    start_time = time.perf_counter()

    if state.get("error_flag"):
        print("🛑 [Supervisor] Error flag detected. Halting execution.")
        return {"next_node": "FINISH", "api_call_timestamps": [time.time()], "supervisor_latency": time.perf_counter() - start_time, "supervisor_calls": 1}
    
    current_step = state.get("current_step", "start")
    
    # MODE 1: ZERO-TAX DISPATCHER
    if state.get("z3_verification_status") and state.get("proposed_route"):
        route = state["proposed_route"]
        print(f"⚡ [Dispatcher] Using pre-verified route: {route}")
        
        if current_step == "start" and len(route) > 0:
            next_node = route[0]
        elif current_step in route:
            idx = route.index(current_step)
            if idx + 1 < len(route):
                next_node = route[idx + 1]
            else:
                next_node = "FINISH"
        else:
            next_node = "FINISH"
            
        return {
            "next_node": next_node,
            "supervisor_latency": time.perf_counter() - start_time,
            "supervisor_tokens": 0, 
            "supervisor_calls": 0
        }

    # MODE 2: THE LLM PLANNER
    print("🧠 [Planner] No verified route found. Invoking LLM and Z3 Solver...")
    
    sleep_duration = 5.0
    print(f"⏳ [Rate Limit Protocol] Pausing for {sleep_duration}s...")
    time.sleep(sleep_duration)
    current_sleep = state.get("total_sleep_time", 0.0)

    start_time = time.perf_counter() # Reset start time after sleep to measure LLM planning latency accurately.
    
    structured_llm = llm.with_structured_output(RouteProposal, include_raw=True)
    dataset_metadata = state.get(
        "dataset_metadata",
        {
            "rows": 0,
            "cols": 0,
            "has_nulls": False,
            "has_free_text_columns": False,
            "has_categorical_features": False,
            "has_nominal_ints": False,
            "task_type": "unknown",
            "is_imbalanced": False,
        }
    )
    
    system_prompt = f"""
You are the Master Orchestrator for a Multi-Agent Data Science Workflow.

Your responsibilities are:

1. Read the user's natural language request.
2. Classify it into EXACTLY ONE orchestration preset.
3. Design the SHORTEST execution route through the available worker nodes that satisfies:
   - the user's requested utility preferences,
   - the observed dataset characteristics, and
   - the workflow safety constraints.
4. Your proposed route will be formally verified by a Z3 theorem prover. If your proposal is rejected, you will receive feedback and must repair your route.

Available Worker Nodes:
['data_cleaning', 'eda', 'feature_engineering', 'modelling', 'reporting']

-------------------------
DATASET REALITY
-------------------------
Rows: {dataset_metadata.get('rows')}
Columns: {dataset_metadata.get('cols')}
Contains Null Values: {dataset_metadata.get('has_nulls')}
Contains Free-Text Columns: {dataset_metadata.get('has_free_text_columns')}
Contains Categorical Features: {dataset_metadata.get('has_categorical_features')}
Contains Integer Categoricals: {dataset_metadata.get('has_nominal_ints')}
Task Type: {dataset_metadata.get('task_type')}
Imbalanced Dataset: {dataset_metadata.get('is_imbalanced')}

-------------------------
PLANNING OBJECTIVE
-------------------------
Assume every worker node is unnecessary by default.

Only include a worker node when it is required by:
- the selected orchestration preset, or
- the dataset's physical characteristics.

If multiple valid routes exist, always propose the shortest one.

-------------------------
WORKFLOW CONSTRAINT GUIDANCE
-------------------------
The Z3 verifier enforces workflow legality.

To maximise the chance that your first proposal is accepted:

- Missing or null values require 'data_cleaning'.
- Free-text columns require 'data_cleaning'.
- Categorical features require 'feature_engineering' before modelling.
- 'reporting' must always be the final node.
- ENTERPRISE_STANDARD and REGULATORY_COMPLIANCE follow the complete end-to-end workflow.
- KAGGLE_COMPETITOR requires EDA, feature engineering and modelling for maximum predictive performance.
- C_SUITE_PITCH focuses on executive insights and does not perform feature engineering or modelling.
- RAPID_BASELINE and QUICK_EXPLAINABLE should avoid unnecessary worker nodes whenever the dataset allows.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state.get("user_request", "Please route my execution graph."))
    ]
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = structured_llm.invoke(messages)
            parsed_route: RouteProposal = result["parsed"]
            raw_response = result["raw"]
            
            preset = parsed_route.selected_preset
            proposed_path = parsed_route.proposed_route

            if preset == "OUT_OF_SCOPE":
                print(f"🎯 [Planner] Attempt {attempt+1}: Classified as {preset}")
                return {
                    "error_flag": True,
                    "error_message": "The request is outside the supported scope of this system. Only tabular machine learning workflows are supported.",
                    "active_preset": "OUT_OF_SCOPE",
                    "executed_route": []
                }
            
            print(f"🎯 [Planner] Attempt {attempt+1}: Classified as {preset}")
            print(f"🛤️ [Planner] Proposed Path: {proposed_path}")
            
            is_safe, z3_msg = verify_proposed_route(proposed_path, dataset_metadata, preset)
            print(z3_msg)
            
            if is_safe:
                usage = raw_response.usage_metadata or {}
                tax_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                
                next_node = proposed_path[0] if len(proposed_path) > 0 else "FINISH"
                latency = time.perf_counter() - start_time
                
                return {
                    "next_node": next_node,
                    "active_preset": preset,
                    "proposed_route": proposed_path,
                    "z3_verification_status": True,
                    "total_sleep_time": current_sleep + sleep_duration,
                    "total_input_tokens": state.get("total_input_tokens", 0) + usage.get("input_tokens", 0),
                    "total_output_tokens": state.get("total_output_tokens", 0) + usage.get("output_tokens", 0),
                    "supervisor_latency": latency,
                    "supervisor_tokens": tax_tokens,
                    "supervisor_calls": 1,
                    "api_call_timestamps": [time.time()]
                }
            else:
                messages.append(result["raw"]) 
                messages.append(HumanMessage(content=z3_msg)) 
                
        except Exception as e:
            print(f"❌ [Supervisor] API Error during planning: {e}")
            break
            
    print("🛑 [System] Supervisor failed to find a Z3-verified route.")
    return {
        "error_flag": True, 
        "error_message": "Supervisor could not find a mathematically safe route.",
        "next_node": "FINISH",
        "total_sleep_time": current_sleep + sleep_duration,
        "supervisor_latency": time.perf_counter() - start_time,
        "supervisor_calls": 1,
        "api_call_timestamps": [time.time()]
    }