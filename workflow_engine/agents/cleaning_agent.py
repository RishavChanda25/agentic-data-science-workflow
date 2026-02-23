import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from workflow_engine.state import DataScienceState
from workflow_engine.tools.python_repl import DataScienceREPL

# Initialize your LLM (Choose the one you are actively using for this node)
# llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def clean_data_node(state: DataScienceState) -> dict:
    """
    LangGraph node responsible for cleaning the raw dataset.
    Generates code to handle missing values, duplicates, and types.
    Features an internal 3-attempt self-correction loop.
    """
    print("--- AGENT: DATA CLEANING ---")
    
    # 1. Dynamically resolve the absolute project root
    current_dir = os.getcwd()
    if current_dir.endswith('notebooks'):
        project_root = os.path.dirname(current_dir)
    else:
        project_root = current_dir

    # 2. Define absolute paths (normalized with forward slashes for Windows safety)
    raw_path = os.path.abspath(os.path.join(project_root, state["raw_dataset_path"])).replace('\\', '/')
    processed_dir = os.path.join(project_root, "data", "processed").replace('\\', '/')
    processed_path = os.path.join(processed_dir, "cleaned_data.csv").replace('\\', '/')
    
    # 3. Define the Agent's Persona and Rules
    system_prompt = f"""You are an expert Data Cleaning Agent.
Your task is to write Python code using the `pandas` library to clean the dataset located at '{raw_path}'.

Perform the following operations:
1. Load the dataset using pandas.
2. Identify and handle missing values (e.g., impute numericals with median, drop columns with >50% missing).
3. Remove exact duplicate rows.
4. Save the cleaned dataframe exactly to '{processed_path}'.

CRITICAL RULES:
- Output ONLY valid Python code. Do not wrap it in markdown blockquotes (no ```python).
- Do not add explanations or text outside the code.
- Create the output directory first using `os.makedirs(r'{processed_dir}', exist_ok=True)`.
- PANDAS 3.0 COMPLIANCE: NEVER use `inplace=True` for filling missing values or dropping columns. Use reassignment instead (e.g., `df = df.drop(columns=['col'])` and `df['col'] = df['col'].fillna(val)`).
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please write the Python code to execute this task now.")
    ]
    
    repl = DataScienceREPL()
    max_retries = 3
    attempts = 0
    
    # 4. Intra-Node Execution and Self-Correction Loop
    while attempts < max_retries:
        attempts += 1
        print(f"\n--- ATTEMPT {attempts}/{max_retries} ---")
        
        # Get the code from the LLM
        response = llm.invoke(messages)
        
        # Sanitize the output: strip out markdown blocks if the LLM ignores instructions
        generated_code = response.content.replace("```python", "").replace("```", "").strip()
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
                "current_step": "EDA_Agent"
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
        "messages": [f"Data Cleaning failed. Last error: {execution_result['output']}"]
    }