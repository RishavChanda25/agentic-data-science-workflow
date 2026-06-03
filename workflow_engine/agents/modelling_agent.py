import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from workflow_engine.state import DataScienceState
from workflow_engine.tools.python_repl import DataScienceREPL
import time

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def modelling_agent_node(state: DataScienceState) -> dict:
    """
    Trains machine learning models on the engineered dataset, evaluates them, 
    and saves the best model and its evaluation metrics.
    """
    print("--- AGENT: MODELLING ---")
    
    # 1. Resolve paths
    current_dir = os.getcwd()
    if current_dir.endswith('notebooks'):
        project_root = os.path.dirname(current_dir)
    else:
        project_root = current_dir

    input_path = os.path.abspath(os.path.join(project_root, state["current_dataset_path"])).replace('\\', '/')
    artifacts_dir = os.path.join(project_root, "data", "artifacts").replace('\\', '/')
    figures_dir = os.path.join(project_root, "reports", "figures").replace('\\', '/')
    
    model_output_path = os.path.join(artifacts_dir, "best_model.pkl").replace('\\', '/')
    metrics_output_path = os.path.join(artifacts_dir, "model_metrics.json").replace('\\', '/')
    confusion_matrix_path = os.path.join(figures_dir, "confusion_matrix.png").replace('\\', '/')
    
    target_var = state.get("target_variable", "target")

    # 2. Define the Agent's Persona and Rules
    system_prompt = f"""You are an expert Machine Learning Modelling Agent.
Your task is to write Python code using `pandas`, `scikit-learn`, `xgboost`, `joblib`, and `matplotlib` to train and evaluate models on the dataset located at '{input_path}'.

Perform the following operations exactly:
1. Load the dataset.
2. Separate the features (X) and the target (y). The target variable column is EXACTLY named '{target_var}'. You MUST write `y = df['{target_var}']` and `X = df.drop(columns=['{target_var}'])`.
3. CRITICAL ENCODING: Check if the target variable (y) is of object/string type (e.g., 'Yes'/'No'). If it is, you MUST use `LabelEncoder` from `sklearn.preprocessing` to convert it to numeric (1/0) BEFORE splitting the data.
4. Convert any boolean columns in the features to floats using `X = X.astype(float)` to ensure complete compatibility with XGBoost.
5. Split the data into training and testing sets (80/20 split, random_state=42).
6. Train three models: a `LogisticRegression`, a `RandomForestClassifier(random_state=42)`, and an `XGBClassifier(random_state=42, eval_metric='logloss')`.
7. Evaluate all three models on the test set by calculating Accuracy, Precision, Recall, and F1-Score.
8. Determine the best model based strictly on the highest F1-Score to properly account for class imbalances.
9. Save the best model to '{model_output_path}' using `joblib.dump()`.
10. Save a JSON file containing the best model's name and its accuracy, precision, recall, and f1-score to '{metrics_output_path}'.
11. Generate a Confusion Matrix plot for the BEST model using `ConfusionMatrixDisplay` or `seaborn.heatmap`. Save the plot to '{confusion_matrix_path}'.

CRITICAL RULES:
- Output ONLY valid Python code. Do not wrap it in markdown blockquotes (no ```python).
- Do not add explanations or text outside the code.
- Ensure output directories exist using `os.makedirs()`.
- ALWAYS use `plt.savefig(filepath, bbox_inches='tight')` and then `plt.close()` for plots.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please write the Python code to train and evaluate the models now.")
    ]
    
    repl = DataScienceREPL()
    max_retries = 3
    attempts = 0
    
    # 3. Intra-Node Execution and Self-Correction Loop
    while attempts < max_retries:
        attempts += 1
        print(f"\n--- ATTEMPT {attempts}/{max_retries} ---")
        
        if attempts > 1:
            print("Cooling down for 15 seconds to respect API rate limits...")
            time.sleep(15)
            
        try:
            response = llm.invoke(messages)
            
            # Sanitize the output
            generated_code = response.content.replace("```python", "").replace("```", "").strip()
            print("Generated Code:\n", generated_code)
            
            # Execute the code
            execution_result = repl.execute_code(generated_code)
            
            # 4. Evaluate and Update State
            if execution_result["success"]:
                print("Status: Success")
                
                artifacts = state.get("artifacts", {})
                artifacts["best_model"] = model_output_path
                artifacts["model_metrics"] = metrics_output_path
                artifacts["confusion_matrix"] = confusion_matrix_path
                
                return {
                    "artifacts": artifacts,
                    "messages": [f"Modelling Agent successfully trained and saved models after {attempts} attempt(s)."],
                    "error_flag": False,
                    "current_step": "modelling"
                }
            else:
                error_msg = execution_result['output']
                print(f"Status: Failed - {error_msg}")
                
                correction_prompt = f"""The code failed with the following error:
{error_msg}

Please fix the code and provide the complete, corrected script. Ensure you import all necessary metrics from sklearn.metrics and handle file saving correctly."""
                messages.append(HumanMessage(content=correction_prompt))
                
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                print(f"⚠️ API Rate Limit Hit! Forcing a 65-second deep cooldown...")
                time.sleep(65.0)  # <-- INCREASE THIS TO 65
                # Note: Because this is a failed run, you don't need to track this 
                # sleep time for the final ablation study baseline. You only track 
                # the time of a fully successful, uninterrupted run.
            else:
                print(f"⚠️ API Error encountered: {error_str}")
                time.sleep(5.0) # Standard small delay for non-quota errors

    print("\nStatus: Max retries reached. Node failed.")
    return {
        "error_flag": True,
        "error_message": "Modelling Agent failed.",
    }