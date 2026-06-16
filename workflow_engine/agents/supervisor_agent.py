import os
import time
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from workflow_engine.state import DataScienceState

# 1. Define the Structured Output with CORRECT node names
class Route(BaseModel):
    next_node: str = Field(
        description="The exact name of the next node to execute. Must be one of: 'data_cleaning', 'eda', 'feature_engineering', 'modelling', 'reporting', or 'FINISH'."
    )

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

def supervisor_node(state: DataScienceState):
    """The Baseline Supervisor with Latency and Token Tracking."""
    print("\n🤖 [Supervisor] Analyzing state to determine next step...")

    # --- THE EMERGENCY BRAKE ---
    if state.get("error_flag"):
        print("🛑 [Supervisor] Error flag detected from previous node. Halting execution to prevent Zombie Loop.")
        return {
            "next_node": "FINISH",
            "api_call_timestamps": [time.time()] # Log the final action time
        }
    
    # CRITICAL FIX: include_raw=True ensures we don't lose the token metadata
    structured_llm = llm.with_structured_output(Route, include_raw=True)
    
    # The Ironclad Baseline Prompt (No dynamic skipping)
    system_prompt = """You are a strict, deterministic Finite State Machine router. 
    You do not think. You do not analyze the data. You do not read user intent. 
    Your ONLY job is to look at the 'last completed step' provided by the user and map it to the next step using this exact dictionary:
    
    - If 'start' -> output 'data_cleaning'
    - If 'data_cleaning' -> output 'eda'
    - If 'eda' -> output 'feature_engineering'
    - If 'feature_engineering' -> output 'modelling'
    - If 'modelling' -> output 'reporting'
    - If 'reporting' -> output 'FINISH'
    
    You are strictly forbidden from skipping steps. Output ONLY the exact string name of the next node."""
    
    current_step = state.get("current_step", "start")
    
    # Retrieve the running total of sleep time (default to 0.0)
    current_sleep = state.get("total_sleep_time", 0.0)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"The last completed step was: {current_step}. What is the next node?")
    ]
    
    # Rate Limit Protocol & Latency Tracking
    sleep_duration = 10.0
    print(f"⏳ [Rate Limit Protocol] Pausing for {sleep_duration}s to protect quota...")
    time.sleep(sleep_duration)
    
    try:
        result = structured_llm.invoke(messages)
        
        # Unpack the raw message (for tokens) and parsed object (for routing)
        parsed_route = result["parsed"]
        raw_response = result["raw"]
        
        next_route = parsed_route.next_node
        print(f"🤖 [Supervisor] Proposed route: {next_route}")
        
        # --- OBSERVABILITY: Intercept Usage Metadata ---
        usage = raw_response.usage_metadata or {}
        node_input_tokens = usage.get("input_tokens", 0)
        node_output_tokens = usage.get("output_tokens", 0)
        
        return {
            "next_node": next_route,
            "total_sleep_time": current_sleep + sleep_duration,
            "total_input_tokens": state.get("total_input_tokens", 0) + node_input_tokens,
            "total_output_tokens": state.get("total_output_tokens", 0) + node_output_tokens,
            "api_call_timestamps": [time.time()] 
        }
    except Exception as e:
        print(f"❌ [Supervisor] Routing failed: {e}")
        return {
            "error_flag": True, 
            "error_message": f"Supervisor failed to route: {str(e)}",
            "next_node": "FINISH",
            "total_sleep_time": current_sleep + sleep_duration,
            "api_call_timestamps": [time.time()] # Log the time even on a failed API call
        }