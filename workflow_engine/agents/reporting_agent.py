import os
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from workflow_engine.state import DataScienceState

# We can use a slightly more creative/higher temperature for reporting
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

def reporting_agent_node(state: DataScienceState) -> dict:
    """
    Synthesizes the EDA and Modelling artifacts into a human-readable 
    Markdown report.
    """
    print("--- AGENT: REPORTING ---")
    
    # 1. Resolve paths
    current_dir = os.getcwd()
    if current_dir.endswith('notebooks'):
        project_root = os.path.dirname(current_dir)
    else:
        project_root = current_dir

    final_reports_dir = os.path.join(project_root, "reports", "final_reports").replace('\\', '/')
    os.makedirs(final_reports_dir, exist_ok=True)
    report_output_path = os.path.join(final_reports_dir, "final_report.md").replace('\\', '/')
    
    # 2. Ingest the Artifacts
    artifacts = state.get("artifacts", {})
    eda_summary_path = artifacts.get("eda_summary")
    metrics_path = artifacts.get("model_metrics")
    
    eda_text = "No EDA data provided."
    metrics_text = "No modeling metrics provided."
    
    if eda_summary_path and os.path.exists(eda_summary_path):
        with open(eda_summary_path, 'r') as f:
            eda_text = json.dumps(json.load(f), indent=2)
            
    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics_text = json.dumps(json.load(f), indent=2)

    # Extract user goals
    target_var = state.get("target_variable", "target")
    user_request = state.get("user_request", "Clean the data and train a model.")

    # 3. Define the Agent's Persona and Rules
    system_prompt = f"""You are an Expert Data Science Communicator. 
Your task is to write a final, comprehensive Markdown report summarizing a completed machine learning pipeline.

Here is the context of the project:
- User's Original Request: {user_request}
- Target Variable to Predict: {target_var}

Here are the EDA Findings (Feature Data):
{eda_text}

Here are the Final Model Metrics:
{metrics_text}

Write a professional, well-formatted Markdown document. It should include:
1. An Executive Summary.
2. A brief overview of the dataset (mentioning categorical vs. numerical splits based on the EDA).
3. The modelling results and what the metrics (Accuracy, F1, etc.) mean in practical terms.
4. A brief concluding recommendation or insight.

CRITICAL RULES:
- Output ONLY valid Markdown text.
- Do NOT output Python code.
- Do NOT wrap your output in ```markdown blockquotes. Just output the raw text directly.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please write the final markdown report now.")
    ]
    
    # 4. Generate the Report
    try:
        response = llm.invoke(messages)
        report_content = response.content.replace("```markdown", "").replace("```", "").strip()
        
        # Save the report to disk
        with open(report_output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
            
        print(f"Status: Success - Report saved to {report_output_path}")
        
        # Update State
        artifacts["final_report"] = report_output_path
        
        return {
            "artifacts": artifacts,
            "messages": ["Reporting Agent successfully generated the final Markdown document."],
            "error_flag": False,
            "current_step": "end"
        }
        
    except Exception as e:
        print(f"Reporting Agent Failed: {e}")
        return {
            "error_flag": True,
            "error_message": f"Failed to generate report: {e}"
        }