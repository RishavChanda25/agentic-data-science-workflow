import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from workflow_engine.state import DataScienceState
from workflow_engine.tools.python_repl import DataScienceREPL

# Sticking with flash for rapid prototyping
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

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
    target_var = state.get("target_variable", "target") # Defaulted to 'target' for the heart disease dataset

    # 3. Define the Agent's Persona and Rules (UPDATED FOR JSON SERIALIZATION)
    system_prompt = f"""You are an expert Exploratory Data Analysis (EDA) Agent.
Your task is to write Python code using `pandas`, `matplotlib`, `seaborn`, and `json` to analyze the cleaned dataset located at '{input_path}'.

Perform the following operations:
1. Load the dataset.
2. Create output directories using `os.makedirs(r'{reports_dir}', exist_ok=True)` and `os.makedirs(r'{artifacts_dir}', exist_ok=True)`.
3. Generate a comprehensive JSON summary of the dataset's features to guide downstream Feature Engineering:
   - For every column (excluding '{target_var}'), determine if it is numerical or categorical.
   - HEURISTIC RULE: If a column has `dtype` 'object', 'category', 'bool', OR has <= 10 unique values, treat it as a 'categorical_feature' and record its cardinality.
   - Otherwise, treat it as a 'numerical_feature' and record its min, max, mean, and skewness.
   - CRITICAL: You MUST cast all Pandas/Numpy numeric types to standard Python types (e.g., use `int(val)` or `float(val)`) before saving to JSON, or it will crash with a TypeError.
   - Save this dictionary as a JSON file exactly to '{artifacts_dir}/eda_summary.json'.
4. Generate a correlation heatmap for numerical features ONLY. Save it exactly to '{reports_dir}/correlation_heatmap.png'.
5. Generate a distribution plot for the target variable '{target_var}' if it exists. Save it exactly to '{reports_dir}/target_distribution.png'.

CRITICAL RULES:
- Output ONLY valid Python code. Do not wrap it in markdown blockquotes (no ```python).
- Do not add explanations or text outside the code.
- ALWAYS use `plt.savefig(filepath, bbox_inches='tight')` to save your plots.
- ALWAYS call `plt.close()` or `plt.clf()` immediately after saving each plot to prevent overlapping axes and memory leaks.
- NEVER use `plt.show()`.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please write the Python code to execute this EDA task now.")
    ]
    
    repl = DataScienceREPL()
    max_retries = 3
    attempts = 0
    
    # 4. Intra-Node Execution and Self-Correction Loop
    while attempts < max_retries:
        attempts += 1
        print(f"\n--- ATTEMPT {attempts}/{max_retries} ---")
        
        response = llm.invoke(messages)
        
        # Sanitize the output
        generated_code = response.content.replace("```python", "").replace("```", "").strip()
        print("Generated Code:\n", generated_code)
        
        # Execute the code locally via your upgraded REPL
        execution_result = repl.execute_code(generated_code)
        
        # 5. Evaluate and Update State
        if execution_result["success"]:
            print("Status: Success")
            
            # Update artifacts dictionary with the new JSON summary
            artifacts = state.get("artifacts", {})
            artifacts["correlation_heatmap"] = f"{reports_dir}/correlation_heatmap.png"
            artifacts["target_distribution"] = f"{reports_dir}/target_distribution.png"
            artifacts["eda_summary"] = f"{artifacts_dir}/eda_summary.json"
            
            return {
                "artifacts": artifacts,
                "messages": [f"EDA Agent successfully generated plots and JSON summary after {attempts} attempt(s)."],
                "error_flag": False,
                "current_step": "Feature_Engineering_Agent" # Routing to your new agent
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
        "messages": [f"EDA failed. Last error: {execution_result['output']}"]
    }