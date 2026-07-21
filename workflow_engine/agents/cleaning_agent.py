import os
import time
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from workflow_engine.state import DataScienceState
from workflow_engine.tools.python_repl import DataScienceREPL

# Initialize your LLM (Choose the one you are actively using for this node)
# llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

def clean_data_node(state: DataScienceState) -> dict:
    """
    LangGraph node responsible for cleaning the raw dataset.
    Generates code to handle missing values, duplicates, and types.
    Features an internal 3-attempt self-correction loop and API cost tracking.
    """
    print("--- AGENT: DATA CLEANING ---")
    
    # 1. Dynamically resolve the absolute project root
    current_dir = os.getcwd()
    if current_dir.endswith('notebooks'):
        project_root = os.path.dirname(current_dir)
    else:
        project_root = current_dir

    # 2. Define absolute paths (normalized with forward slashes for Windows safety)
    raw_path = os.path.abspath(os.path.join(project_root, state["current_dataset_path"])).replace('\\', '/')
    processed_dir = os.path.join(project_root, "data", "processed").replace('\\', '/')
    processed_path = os.path.join(processed_dir, "cleaned_data.csv").replace('\\', '/')
    
    target_var = state.get("target_variable")
    
    preset = state.get("active_preset", "ENTERPRISE_STANDARD")

    # 3. Define the Agent's Persona and Rules
    system_prompt = f"""You are an expert Data Cleaning Agent.
Your task is to write Python code using the `pandas` library to clean the dataset located at '{raw_path}'.

Perform the following operations exactly:
1. Load the dataset using pandas.
2. CRITICAL DATA TYPING: Before checking for missing values, aggressively coerce hidden dirty strings. Iterate through all object/string columns. You MUST use a standard nested `for` loop to check if the column name contains 'charge', 'price', 'balance', 'amount', or 'fee' (case insensitive). If a match is found, force it to numeric using `df[col] = pd.to_numeric(df[col], errors='coerce')` and `break` the inner loop.
3. CRITICAL: Drop all columns flagged as unstructured free-text to preserve dimensionality limits.
4. Handle missing values based on your Persona Rule.
5. Remove exact duplicate rows.
6. Drop any rows where the target variable '{target_var}' is missing.
7. Save the cleaned dataframe exactly to '{processed_path}'.

CRITICAL RULES:
- Output ONLY valid Python code. Do not wrap it in markdown blockquotes (no ```python).
- Create the output directory first using `os.makedirs(r'{processed_dir}', exist_ok=True)`.
- PANDAS 3.0 COMPLIANCE: NEVER use `inplace=True`.
- DANGEROUS ENVIRONMENT QUIRK: You MUST NOT use list comprehensions or generator expressions. Use standard multi-line `for` loops.

--- DYNAMIC PERSONA INJECTION: {preset} ---
"""
    
    if preset == "RAPID_BASELINE":
        system_prompt += """
    MISSION: Produce a model-ready dataset as quickly as possible.

    PERSONA RULES:
    - Drop columns with >50% missing values.
    - Impute numerical columns using the median.
    - Impute categorical columns using the mode.
    - Never use iterative, distance-based or probabilistic imputers.
    - Prefer simple deterministic operations with minimal computational cost.
    """

    elif preset == "QUICK_EXPLAINABLE":
        system_prompt += """
    MISSION: Produce a clean dataset using transformations that are easy to explain.

    PERSONA RULES:
    - Drop columns with >50% missing values.
    - Impute numerical columns using the median.
    - Impute categorical columns using the mode.
    - Every cleaning decision should be intuitive and easily explained to non-technical stakeholders.
    - Avoid complex imputers entirely.
    """

    elif preset == "KAGGLE_COMPETITOR":
        system_prompt += """
    MISSION: Preserve as much predictive signal as possible regardless of execution time.

    PERSONA RULES:
    - Drop columns only if they contain >80% missing values.
    - Use `KNNImputer` from `sklearn.impute` for numerical features.
    - Impute categorical columns using the mode.
    - Preserve rare categories whenever possible.
    - Never sacrifice predictive information simply to reduce computation.
    """

    elif preset == "ENTERPRISE_STANDARD":
        system_prompt += """
    MISSION: Produce a robust production-quality dataset using widely accepted engineering practices.

    PERSONA RULES:
    - Drop columns with >70% missing values.
    - Use median imputation for skewed numerical columns.
    - Use mean imputation for approximately symmetric numerical columns.
    - Impute categorical columns using the mode.
    - Detect obvious numerical outliers using the IQR rule and clip them to the IQR bounds.
    - Prefer deterministic preprocessing.
    """

    elif preset == "REGULATORY_COMPLIANCE":
        system_prompt += """
    MISSION: Produce a completely transparent and auditable cleaned dataset.

    PERSONA RULES:
    - Drop rows containing missing values if they account for less than 5% of the dataset.
    - Otherwise use simple mean imputation for numerical columns.
    - Impute categorical columns using the mode.
    - Never use black-box imputers such as KNNImputer.
    - Every cleaning decision should be deterministic and suitable for inclusion in an audit report.
    """

    elif preset == "C_SUITE_PITCH":
        system_prompt += """
    MISSION: Clean the data only enough to support reliable visualisation.

    PERSONA RULES:
    - Drop columns with >50% missing values.
    - Impute numerical columns using the median.
    - Impute categorical columns using the mode.
    - Do not spend unnecessary computation preserving every possible feature.
    - Prioritise clean visualisations rather than modelling readiness.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please write the Python code to execute this task now.")
    ]
    
    # Initialize REPL and Retry logic
    repl = DataScienceREPL() # Ensure this is initialized/passed correctly in your actual scope
    max_retries = 3
    attempts = 0
    
    # --- OBSERVABILITY: Local Accumulators ---
    node_input_tokens = 0
    node_output_tokens = 0
    node_timestamps = []
    
    # 4. Intra-Node Execution and Self-Correction Loop
    while attempts < max_retries:
        attempts += 1
        print(f"\n--- ATTEMPT {attempts}/{max_retries} ---")
        
        # Get the code from the LLM
        response = llm.invoke(messages)
        
        # --- OBSERVABILITY: Intercept Usage Metadata ---
        usage = response.usage_metadata or {}
        node_input_tokens += usage.get("input_tokens", 0)
        node_output_tokens += usage.get("output_tokens", 0)
        node_timestamps.append(time.time())
        
        # --- SAFELY EXTRACT CONTENT (Handles both Strings and Multimodal Lists) ---
        raw_content = response.content
        if isinstance(raw_content, list):
            # Extract the text from the list of blocks
            text_content = "".join(block.get("text", "") for block in raw_content if isinstance(block, dict))
        else:
            text_content = str(raw_content)
            
        # Sanitize the output: strip out markdown blocks
        generated_code = text_content.replace("```python", "").replace("```", "").strip()
        print("Generated Code:\n", generated_code)
        
        # Execute the code locally via your REPL
        execution_result = repl.execute_code(generated_code)
        
        # 5. Evaluate and Update State
        if execution_result["success"]:
            print("Status: Success")
            return {
                "current_dataset_path": processed_path,
                "messages": [f"Data Cleaning Agent successfully cleaned the data after {attempts} attempt(s)."],
                "error_flag": False,
                "current_step": "data_cleaning",
                # Pass accumulated tokens and timestamps to the LangGraph state
                "total_input_tokens": state.get("total_input_tokens", 0) + node_input_tokens,
                "total_output_tokens": state.get("total_output_tokens", 0) + node_output_tokens,
                "api_call_timestamps": node_timestamps
            }
        else:
            error_msg = execution_result['output']
            print(f"Status: Failed - {error_msg}")
            
            # Append the error context back to the message history so the LLM can learn and retry
            correction_prompt = f"""The code you provided failed with the following error:
{error_msg}

Please fix the code and provide the complete, corrected Python script. Remember the critical rules."""
            
            messages.append(HumanMessage(content=correction_prompt))

    # 6. Fallback if the loop exhausts all retries
    print("\nStatus: Max retries reached. Node failed.")
    return {
        "error_flag": True,
        "error_message": f"Data Cleaning Agent failed after {max_retries} attempts. Last error: {execution_result['output']}",
        "messages": [f"Data Cleaning failed. Last error: {execution_result['output']}"],
        # Even on failure, bill the state for the tokens consumed during the failed retries
        "total_input_tokens": state.get("total_input_tokens", 0) + node_input_tokens,
        "total_output_tokens": state.get("total_output_tokens", 0) + node_output_tokens,
        "api_call_timestamps": node_timestamps
    }