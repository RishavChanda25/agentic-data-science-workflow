import os
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from workflow_engine.state import DataScienceState
from workflow_engine.tools.python_repl import DataScienceREPL

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def feature_engineering_agent_node(state: DataScienceState) -> dict:
    """
    Reads the EDA summary and writes Python code to encode categorical variables,
    scale numerical variables, and prepare the dataset for machine learning.
    """
    print("--- AGENT: FEATURE ENGINEERING ---")
    
    # 1. Resolve paths
    current_dir = os.getcwd()
    if current_dir.endswith('notebooks'):
        project_root = os.path.dirname(current_dir)
    else:
        project_root = current_dir

    input_path = os.path.abspath(os.path.join(project_root, state["current_dataset_path"])).replace('\\', '/')
    output_dir = os.path.join(project_root, "data", "processed").replace('\\', '/')
    output_path = os.path.join(output_dir, "engineered_data.csv").replace('\\', '/')
    
    target_var = state.get("target_variable", "target")
    
    # 2. Ingest the EDA Summary JSON
    eda_summary_path = state.get("artifacts", {}).get("eda_summary")
    eda_summary_text = "No EDA summary provided."
    
    if eda_summary_path and os.path.exists(eda_summary_path):
        with open(eda_summary_path, 'r') as f:
            eda_summary_text = f.read()

    # 3. Define the Agent's Persona and Rules
    system_prompt = f"""You are an expert Machine Learning Feature Engineering Agent.
Your task is to write Python code using `pandas` and `scikit-learn` to transform the dataset located at '{input_path}'.

Here is the EDA Summary detailing the dataset's columns:
{eda_summary_text}

Perform the following operations:
1. Load the dataset.
2. Ensure the output directory exists: `os.makedirs(r'{output_dir}', exist_ok=True)`.
3. Separate the target variable '{target_var}' from the features so it does not get scaled or encoded.
4. Using the EDA Summary above, identify the categorical features. You MUST apply One-Hot Encoding using `pd.get_dummies(df, columns=categorical_features, drop_first=True)`. Explicitly using the `columns` argument is CRITICAL because some categorical features are integers and will be ignored by pandas otherwise.
5. Using the EDA Summary above, identify the numerical features. Apply `StandardScaler` from `sklearn.preprocessing` to them.
6. Recombine the transformed features and the target variable '{target_var}' into a single DataFrame. Ensure the target column is placed at the very end.
7. Save the final engineered DataFrame EXACTLY to '{output_path}'.

CRITICAL RULES:
- Output ONLY valid Python code. Do not wrap it in markdown blockquotes (no ```python).
- Do not add explanations or text outside the code.
- Ensure all variable names align correctly when recombining dataframes.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please write the Python code to engineer the features now.")
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
        
        # Execute the code
        execution_result = repl.execute_code(generated_code)
        
        # 5. Evaluate and Update State
        if execution_result["success"]:
            print("Status: Success")
            
            return {
                # We update the current_dataset_path so the Modeling agent uses the engineered data!
                "current_dataset_path": f"data/processed/engineered_data.csv",
                "messages": [f"Feature Engineering Agent successfully transformed data after {attempts} attempt(s)."],
                "error_flag": False,
                "current_step": "Modelling_Agent" 
            }
        else:
            error_msg = execution_result['output']
            print(f"Status: Failed - {error_msg}")
            
            correction_prompt = f"""The code you provided failed with the following error:
{error_msg}

Please fix the code and provide the complete, corrected Python script. Pay close attention to index alignment when recombining pandas DataFrames after scaling."""
            
            messages.append(HumanMessage(content=correction_prompt))

    # 6. Fallback if the loop exhausts all retries
    print("\nStatus: Max retries reached. Node failed.")
    return {
        "error_flag": True,
        "error_message": f"Feature Engineering failed after {max_retries} attempts. Last error: {execution_result['output']}",
        "messages": [f"Feature Engineering failed. Last error: {execution_result['output']}"]
    }