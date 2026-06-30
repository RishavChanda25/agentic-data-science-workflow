import os
import json
import time
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from workflow_engine.state import DataScienceState
from workflow_engine.tools.python_repl import DataScienceREPL

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

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
    eda_summary_text = "EDA was skipped. Infer feature types directly from the dataframe."
    
    if eda_summary_path and os.path.exists(eda_summary_path):
        with open(eda_summary_path, 'r') as f:
            eda_summary_text = f.read()

    preset = state.get("active_preset", "ENTERPRISE_STANDARD")

    system_prompt = f"""You are an expert Machine Learning Feature Engineering Agent.
Your task is to write Python code using `pandas` and `scikit-learn` to transform the dataset located at '{input_path}' into a model-ready feature matrix.

EDA Summary:
{eda_summary_text}

IMPORTANT (if EDA Summary is present):
- The JSON contains feature metadata, dataset statistics and generated visualisations.
- Use ONLY the feature metadata to guide your engineering decisions.
- Ignore the generated_figures section completely.

Perform the following operations exactly:
1. Load the dataset and create the output directory using:
   `os.makedirs(r'{output_dir}', exist_ok=True)`.
2. Separate the target variable '{target_var}' from the predictor variables.
3. CRITICAL MEMORY CONSTRAINT:
   If any predictor categorical feature contains more than 100 unique values,
   drop that feature.
   NEVER drop the target variable even if it exceeds this threshold.
4. Transform the predictor variables according to your Persona Rules.
5. Recombine the transformed predictors with the untouched target variable.
   The target variable MUST appear as the final column.
6. Save the engineered dataset EXACTLY to:
   '{output_path}'.

CRITICAL RULES:
- Output ONLY valid Python code. Do not wrap it in markdown blockquotes (no ```python).
- Do not add explanations or text outside the code.
- Ensure all variable names align correctly when recombining dataframes.
- NEVER transform the target variable.
- ALL feature engineering operations MUST be applied only to predictor variables.
- DANGEROUS ENVIRONMENT QUIRK: You MUST NOT use list comprehensions (e.g., [x for x in my_list]). You MUST use standard multi-line 'for' loops and '.append()' instead, otherwise the REPL will crash with a NameError.

--- DYNAMIC PERSONA INJECTION: {preset} ---
"""
    
    if preset == "RAPID_BASELINE":
        system_prompt += """
    MISSION:
    Produce a model-ready dataset with the absolute minimum computational overhead.

    PERSONA RULES:
    - One-hot encode categorical predictors.
    - Convert boolean predictors to numeric.
    - Leave numerical predictors unchanged.
    - No scaling.
    - No feature selection.
    - No PCA.
    - No polynomial or interaction features.
    - Avoid any transformation that is not strictly necessary for successful model training.
    """

    elif preset == "QUICK_EXPLAINABLE":
        system_prompt += """
    MISSION:
    Produce features that remain immediately understandable to non-technical stakeholders.

    PERSONA RULES:
    - One-hot encode categorical predictors.
    - StandardScale numerical predictors.
    - Do not create synthetic features.
    - No PCA.
    - No feature selection.
    - Prefer transparent transformations that can be easily explained.
    """

    elif preset == "KAGGLE_COMPETITOR":
        system_prompt += """
    MISSION:
    Maximise predictive performance regardless of execution time.

    PERSONA RULES:
    - One-hot encode categorical predictors.
    - StandardScale numerical predictors.
    - If the EDA Summary or dataset indicates severe class imbalance, apply SMOTE before model training preparation.
    - Generate second-degree PolynomialFeatures for numerical variables.
    - If the transformed feature space exceeds approximately 50 features, apply PCA retaining approximately 95% explained variance.
    - Optimise purely for predictive performance.
    """

    elif preset == "ENTERPRISE_STANDARD":
        system_prompt += """
    MISSION:
    Produce robust, production-quality engineered features following standard machine learning practice.

    PERSONA RULES:
    - One-hot encode categorical predictors.
    - StandardScale numerical predictors.
    - Preserve feature interpretability.
    - Do not create polynomial features.
    - Do not create interaction features.
    - Apply PCA ONLY if the post-encoding dimensionality exceeds approximately 100 features.
    - Prefer maintainability and robustness over aggressive optimisation.
    """

    elif preset == "REGULATORY_COMPLIANCE":
        system_prompt += """
    MISSION:
    Produce a fully transparent and auditable feature space.

    PERSONA RULES:
    - One-hot encode categorical predictors.
    - MinMaxScaler to numerical features.
    - Preserve a one-to-one mapping between engineered and original variables whenever possible.
    - Do not apply PCA.
    - Do not generate polynomial features.
    - Do not generate interaction features.
    - Every transformation must remain mathematically traceable.
    """

    elif preset == "C_SUITE_PITCH":
        system_prompt += """
    MISSION:
    Perform only the minimum preprocessing required for downstream compatibility.

    PERSONA RULES:
    - Leave the feature space largely unchanged.
    - One-hot encode categorical predictors only where they would otherwise prevent downstream processing.
    - Do not perform scaling.
    - Do not generate engineered features.
    - Do not apply PCA.
    - Do not perform feature selection beyond the mandatory high-cardinality rule.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please write the Python code to engineer the features now.")
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
        
        # Execute the code
        execution_result = repl.execute_code(generated_code)
        
        # 5. Evaluate and Update State
        if execution_result["success"]:
            print("Status: Success")
            
            return {
                # We update the current_dataset_path so the Modelling agent uses the engineered data!
                "current_dataset_path": output_path, # made absolute path earlier, but we want relative path in state
                "messages": [f"Feature Engineering Agent successfully transformed data after {attempts} attempt(s)."],
                "error_flag": False,
                "current_step": "feature_engineering",
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

Please fix the code and provide the complete, corrected Python script. Pay close attention to index alignment when recombining pandas DataFrames after scaling."""
            
            messages.append(HumanMessage(content=correction_prompt))

    # 6. Fallback if the loop exhausts all retries
    print("\nStatus: Max retries reached. Node failed.")
    return {
        "error_flag": True,
        "error_message": f"Feature Engineering failed after {max_retries} attempts. Last error: {execution_result['output']}",
        "messages": [f"Feature Engineering failed. Last error: {execution_result['output']}"],
        # Pass accumulated tokens and timestamps to the LangGraph state
        "total_input_tokens": state.get("total_input_tokens", 0) + node_input_tokens,
        "total_output_tokens": state.get("total_output_tokens", 0) + node_output_tokens,
        "api_call_timestamps": node_timestamps
    }