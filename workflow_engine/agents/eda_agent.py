import os
import time
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from workflow_engine.state import DataScienceState
from workflow_engine.tools.python_repl import DataScienceREPL

# Sticking with flash for rapid prototyping
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

def eda_agent_node(state: DataScienceState) -> dict:
    """
    LangGraph node responsible for Exploratory Data Analysis.
    Generates Python code to create statistical visualizations and a JSON summary.
    Features an internal 3-attempt self-correction loop.
    """
    print("--- AGENT: EXPLORATORY DATA ANALYSIS ---")
    
    # 1. Dynamically resolve the absolute project root
    current_dir = os.getcwd()
    if current_dir.endswith('notebooks'):
        project_root = os.path.dirname(current_dir)
    else:
        project_root = current_dir

    # 2. Define absolute paths (normalized with forward slashes)
    input_path = os.path.abspath(os.path.join(project_root, state["current_dataset_path"])).replace('\\', '/')
    reports_dir = os.path.join(project_root, "reports", "figures").replace('\\', '/')
    artifacts_dir = os.path.join(project_root, "data", "artifacts").replace('\\', '/')
    
    # Extract target variable if provided in state
    target_var = state.get("target_variable")

    preset = state.get("active_preset", "ENTERPRISE_STANDARD")
    target_var = state.get("target_variable")

    system_prompt = f"""You are an expert Exploratory Data Analysis (EDA) Agent.
Your task is to write Python code using `pandas`, `matplotlib`, `seaborn`, and `json` to analyze the cleaned dataset located at '{input_path}'.

Perform the following operations exactly:
1. Load the dataset. The target variable is strictly '{target_var}'.
2. Create output directories using `os.makedirs(r'{reports_dir}', exist_ok=True)` for visualizations and `os.makedirs(r'{artifacts_dir}', exist_ok=True)` for the JSON summary.
3. Generate a comprehensive JSON summary of the dataset and save it exactly to
'{artifacts_dir}/eda_summary.json'.

The JSON MUST contain ALL of the following top-level fields:
- dataset_summary
- numerical_features
- categorical_features
- generated_figures

The 'generated_figures' field must be a list.

Every time you generate a figure, append an object containing:
- filename
- title
- description

Example:

{{
  "filename":"{reports_dir}/correlation_heatmap.png",
  "title":"Correlation Heatmap",
  "description":"Shows pairwise correlations between numerical features."
}}

The JSON serves as the primary communication artifact for BOTH the downstream Feature Engineering Agent and the Reporting Agent.
All Pandas/Numpy numeric values MUST be converted to standard Python types before serialization.

4. Generate visualisations according to the active execution environment and save them inside {reports_dir}.
Every visualisation that is successfully saved MUST also be recorded in generated_figures inside eda_summary.json.

CRITICAL RULES:
- Output ONLY valid Python code. Do not wrap it in markdown blockquotes (no ```python).
- Do not add explanations or text outside the code.
- DANGEROUS ENVIRONMENT QUIRK: You MUST NOT use list comprehensions or generator expressions (e.g., absolutely NO `[col for col in df.columns]`). You MUST use standard multi-line `for` loops and `.append()` methods to build lists. The Python REPL will crash if you use list comprehensions.
- ALWAYS use `plt.savefig(filepath, bbox_inches='tight')` to save your plots.
- ALWAYS call `plt.close()` or `plt.clf()` immediately after saving each plot to prevent overlapping axes and memory leaks.
- NEVER use `plt.show()`.

--- DYNAMIC PERSONA INJECTION: {preset} ---
"""
    
    if preset == "RAPID_BASELINE":
        system_prompt += f"""
    MISSION: Produce only the minimum exploratory analysis required for downstream modelling.

    PERSONA RULES:
    - Generate ONLY a correlation heatmap for numerical features.
    - Save it as 'correlation_heatmap.png'.
    - Do not generate target distributions, pair plots or additional exploratory visualisations.
    - Prioritise execution speed over exploratory insight.
    """

    elif preset == "QUICK_EXPLAINABLE":
        system_prompt += f"""
    MISSION: Produce a concise exploratory analysis that is easy to communicate.

    PERSONA RULES:
    - Generate a correlation heatmap ('correlation_heatmap.png').
    - Generate a target distribution plot for '{target_var}' ('target_distribution.png').
    - Do not generate computationally expensive exploratory plots.
    - Focus on visualisations that are immediately understandable to non-technical audiences.
    """

    elif preset == "KAGGLE_COMPETITOR":
        system_prompt += f"""
    MISSION: 
    Extract mathematical dataset realities for downstream feature engineering with absolute zero plotting overhead.

    PERSONA RULES:
    - Focus ENTIRELY on generating a highly detailed 'eda_summary.json' capturing cardinality, skewness, min/max, and distribution metrics.
    - YOU ARE STRICTLY FORBIDDEN from generating any visualisations or plots (do NOT use matplotlib or seaborn).
    - Skip all correlation heatmaps, pair plots, and distribution plots.
    - Save maximum compute time for the modelling agent.
    """

    elif preset == "ENTERPRISE_STANDARD":
        system_prompt += f"""
    MISSION: Produce a robust exploratory analysis suitable for production data science workflows.

    PERSONA RULES:
    - Generate a correlation heatmap ('correlation_heatmap.png').
    - Generate a target distribution plot for '{target_var}' ('target_distribution.png').
    - Generate boxplots for numerical variables.
    - Generate class balance plots for classification problems.
    - Focus on understanding data quality, feature relationships and distributions using standard industry practices.
    """

    elif preset == "REGULATORY_COMPLIANCE":
        system_prompt += f"""
    MISSION: Document the original characteristics of the dataset for audit purposes.

    PERSONA RULES:
    - Generate a correlation heatmap ('correlation_heatmap.png').
    - Generate a target distribution plot for '{target_var}' ('target_distribution.png').
    - Generate a missing value heatmap if applicable.
    - Generate class balance plots for classification problems.
    - Produce visualisations that serve as evidence of the dataset's original state rather than exploratory storytelling.
    """

    elif preset == "C_SUITE_PITCH":
        system_prompt += f"""
    MISSION: Produce executive-friendly visualisations that communicate business insights.

    PERSONA RULES:
    - Generate a target distribution plot for '{target_var}' ('target_distribution.png').
    - Generate clear bar charts for important categorical variables.
    - Generate histograms for key numerical variables.
    - Generate pie charts where appropriate.
    - Generate a correlation heatmap only if it supports the business narrative.
    - Prioritise clarity, presentation quality and executive storytelling over technical depth.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please write the Python code to execute this EDA task now.")
    ]
    
    repl = DataScienceREPL()
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
        
        # Execute the code locally via your upgraded REPL
        execution_result = repl.execute_code(generated_code)
        
        # 5. Evaluate and Update State
        if execution_result["success"]:
            print("Status: Success")
            
            # Update artifacts dictionary with the new JSON summary
            artifacts = state.get("artifacts", {})
            artifacts["eda_summary"] = f"{artifacts_dir}/eda_summary.json"
            
            return {
                "artifacts": artifacts,
                "messages": [f"EDA Agent successfully generated plots and JSON summary after {attempts} attempt(s)."],
                "error_flag": False,
                "current_step": "eda",
                # Pass accumulated tokens and timestamps to the LangGraph state
                "total_input_tokens": state.get("total_input_tokens", 0) + node_input_tokens,
                "total_output_tokens": state.get("total_output_tokens", 0) + node_output_tokens,
                "api_call_timestamps": node_timestamps
            }
        else:
            error_msg = execution_result['output']
            print(f"Status: Failed - {error_msg}")
            
            correction_prompt = f"""The code you provided failed with the following error:
{error_msg}

Please fix the code and provide the complete, corrected Python script. Remember to isolate numerical columns before calculating correlations, save the files correctly, and close the plots."""
            
            messages.append(HumanMessage(content=correction_prompt))

    # 6. Fallback if the loop exhausts all retries
    print("\nStatus: Max retries reached. Node failed.")
    return {
        "error_flag": True,
        "error_message": f"EDA Agent failed after {max_retries} attempts. Last error: {execution_result['output']}",
        "messages": [f"EDA failed. Last error: {execution_result['output']}"],
        # Pass accumulated tokens and timestamps to the LangGraph state
        "total_input_tokens": state.get("total_input_tokens", 0) + node_input_tokens,
        "total_output_tokens": state.get("total_output_tokens", 0) + node_output_tokens,
        "api_call_timestamps": node_timestamps
    }