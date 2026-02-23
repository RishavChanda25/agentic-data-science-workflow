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
    Generates Python code to create and save statistical visualizations.
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
    
    # Extract target variable if provided in state
    target_var = state.get("target_variable", "churn")

    # 3. Define the Agent's Persona and Rules
    system_prompt = f"""You are an expert Exploratory Data Analysis (EDA) Agent.
Your task is to write Python code using `pandas`, `matplotlib`, and `seaborn` to analyze the cleaned dataset located at '{input_path}'.

Perform the following operations:
1. Load the dataset.
2. Generate a correlation heatmap for numerical features. Save it exactly to '{reports_dir}/correlation_heatmap.png'.
3. Generate a distribution plot for the target variable '{target_var}'. Save it exactly to '{reports_dir}/target_distribution.png'.

CRITICAL RULES:
- Output ONLY valid Python code. Do not wrap it in markdown blockquotes (no ```python).
- Do not add explanations or text outside the code.
- Create the output directory first using `os.makedirs(r'{reports_dir}', exist_ok=True)`.
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
            
            # We can pass the generated artifact paths into the state for later nodes to use
            artifacts = state.get("artifacts", {})
            artifacts["correlation_heatmap"] = f"{reports_dir}/correlation_heatmap.png"
            artifacts["target_distribution"] = f"{reports_dir}/target_distribution.png"
            
            return {
                "artifacts": artifacts,
                "messages": [f"EDA Agent successfully generated plots after {attempts} attempt(s)."],
                "error_flag": False,
                "current_step": "Model_Agent" # Or whatever your next orchestrator step is
            }
        else:
            error_msg = execution_result['output']
            print(f"Status: Failed - {error_msg}")
            
            correction_prompt = f"""The code you provided failed with the following error:
{error_msg}

Please fix the code and provide the complete, corrected Python script. Remember to save the files and close the plots."""
            
            messages.append(HumanMessage(content=correction_prompt))

    # 6. Fallback if the loop exhausts all retries
    print("\nStatus: Max retries reached. Node failed.")
    return {
        "error_flag": True,
        "error_message": f"EDA Agent failed after {max_retries} attempts. Last error: {execution_result['output']}",
        "messages": [f"EDA failed. Last error: {execution_result['output']}"]
    }