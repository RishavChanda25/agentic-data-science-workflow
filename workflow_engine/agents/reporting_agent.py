import os
import time
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from workflow_engine.state import DataScienceState

# We can use a slightly more creative/higher temperature for reporting
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.4)

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
    figure_manifest = []

    # -----------------------------
    # Load EDA Summary
    # -----------------------------
    if eda_summary_path and os.path.exists(eda_summary_path):
        with open(eda_summary_path, "r", encoding="utf-8") as f:
            eda_json = json.load(f)

        eda_text = json.dumps(eda_json, indent=2)

        # Build a figure manifest with paths relative to the final report
        generated_figures = eda_json.get("generated_figures", [])

        for fig in generated_figures:

            absolute_path = fig.get("filename", "")

            if absolute_path and os.path.exists(absolute_path):

                relative_path = os.path.relpath(
                    absolute_path,
                    start=final_reports_dir
                ).replace("\\", "/")

                figure_manifest.append({
                    "path": relative_path,
                    "title": fig.get("title", ""),
                    "description": fig.get("description", "")
                })

    # -----------------------------
    # Always include the confusion matrix if present
    # -----------------------------
    confusion_matrix_path = artifacts.get("confusion_matrix")

    if confusion_matrix_path and os.path.exists(confusion_matrix_path):

        relative_path = os.path.relpath(
            confusion_matrix_path,
            start=final_reports_dir
        ).replace("\\", "/")

        figure_manifest.append({
            "path": relative_path,
            "title": "Confusion Matrix",
            "description": "Confusion matrix of the final selected predictive model."
        })

    # -----------------------------
    # Load modelling metrics
    # -----------------------------
    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_text = json.dumps(json.load(f), indent=2)

    # Extract user goals
    target_var = state.get("target_variable", "target")
    user_request = state.get("user_request", "Clean the data and train a model.")

    preset = state.get("active_preset", "ENTERPRISE_STANDARD")
    executed_route = state.get("proposed_route", [])

    print(f"--- PRESET: {preset} ---")

    system_prompt = f"""You are an Expert Data Science Communicator.
Your task is to produce the final Markdown report for a completed agentic data science workflow.

USER REQUEST
-------------
{user_request}

TARGET VARIABLE
---------------
{target_var}

ACTIVE PRESET
-------------
{preset}

EXECUTED WORKFLOW
-----------------
{executed_route}

EDA SUMMARY (JSON)
------------------
{eda_text}

The EDA Summary contains:
- dataset statistics
- feature metadata
- generated visualisations

FIGURE MANIFEST
---------------
{json.dumps(figure_manifest, indent=2)}

The Figure Manifest contains every figure that should appear in the report.
For EVERY figure in the manifest:
1. Embed it using standard Markdown:
   ![title](path)

2. Immediately below the figure, explain what it shows using its description.
3. Do NOT invent figures.
4. If the manifest is empty, omit the entire Visualisations section.

MODEL METRICS (JSON)
--------------------
{metrics_text}

IMPORTANT:
- If no model metrics were generated, do not fabricate modelling results.
- The model_training_time metric reflects the total time taken to train all models, not just the best one.
- If modelling was executed, always include the Confusion Matrix using:
  ![Confusion Matrix](../figures/confusion_matrix.png)
- Briefly explain what the confusion matrix indicates about the selected model.

CRITICAL RULES

- Output ONLY valid Markdown.
- Never output Python.
- Never use ```markdown fences.
- Only describe workflow stages that actually executed.
- Never invent datasets, figures or metrics.
- The report must faithfully reflect the priorities of the selected preset.
"""
    
    if preset == "RAPID_BASELINE":
        system_prompt += """
    MISSION:
    Deliver an immediate answer.

    PERSONA RULES:
    - Maximum 150 words.
    - Report only:
        - chosen model
        - Accuracy
        - F1-score
        - training time (seconds)
        - one-sentence conclusion
    - Do not discuss methodology.
    - Do not mention workflow stages that were skipped.
    """
    elif preset == "QUICK_EXPLAINABLE":
        system_prompt += """
    MISSION:
    Produce a concise report suitable for a non-technical manager.

    PERSONA RULES:
    Include:
    - Executive Summary
    - Brief explanation of the modelling approach
    - Key evaluation metrics
    - Plain-English interpretation

    Avoid excessive statistical detail.
    """
    elif preset == "KAGGLE_COMPETITOR":
        system_prompt += """
    MISSION:
    Write a ruthlessly technical, highly advanced machine learning evaluation report.

    PERSONA RULES:
    - You are writing STRICTLY for a panel of elite Data Scientists and Kaggle Grandmasters. 
    - You MUST use heavy ML jargon throughout the report (e.g., hyperparameter spaces, non-linear decision boundaries, gradient boosting architectures, precision-recall trade-offs, class imbalance distributions).
    - Discuss deep statistical trade-offs in technical detail (e.g., how Accuracy was sacrificed to maximize F1/Recall, or the effects of SMOTE on model variance).
    - If a simple model (like Logistic Regression) outperformed complex ensembles (like XGBoost), frame this as a significant statistical finding. Discuss linear vs. non-linear decision boundaries, Occam's razor, and the model's robustness to SMOTE-induced noise.
    - COMPLETELY OMIT all high-level business generalizations, executive summaries, or fluff. Do not talk about business impact, stakeholders, or revenue.
    - Provide a dense, mathematical analysis of the model's predictive capabilities.
    """
    elif preset == "ENTERPRISE_STANDARD":
        system_prompt += """
    MISSION:
    Produce a professional technical report.

    PERSONA RULES:
    Include:
    - Executive Summary
    - Dataset Overview
    - Data Preparation
    - Modelling Results
    - Key Takeaways

    Balance technical depth with readability.
    """
    elif preset == "REGULATORY_COMPLIANCE":
        system_prompt += """
    MISSION:
    Produce a formal audit document.

    PERSONA RULES:
    Document:
    - data cleaning decisions
    - feature transformations
    - modelling methodology
    - evaluation metrics

    Emphasize traceability, reproducibility and transparency.

    Avoid speculative or subjective language.
    """
    elif preset == "C_SUITE_PITCH":
        system_prompt += """
    MISSION:
    Produce an executive briefing.

    PERSONA RULES:
    Focus on:
    - dataset characteristics
    - business insights
    - important trends
    - actionable recommendations

    Avoid machine learning jargon.

    If modelling was intentionally skipped, do not mention missing metrics.
    """


    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please write the final markdown report now.")
    ]

    # --- OBSERVABILITY: Local Accumulators ---
    node_input_tokens = 0
    node_output_tokens = 0
    node_timestamps = []
    
    # 4. Generate the Report
    try:
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
        report_content = text_content.replace("```markdown", "").replace("```", "").strip()
        report_content = report_content.strip()  # Remove any remaining backticks
        
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
            "current_step": "reporting",
            # Pass accumulated tokens and timestamps to the LangGraph state
            "total_input_tokens": state.get("total_input_tokens", 0) + node_input_tokens,
            "total_output_tokens": state.get("total_output_tokens", 0) + node_output_tokens,
            "api_call_timestamps": node_timestamps
        }
        
    except Exception as e:
        print(f"Reporting Agent Failed: {e}")
        return {
            "error_flag": True,
            "error_message": f"Failed to generate report: {e}",
            # Pass accumulated tokens and timestamps to the LangGraph state
            "total_input_tokens": state.get("total_input_tokens", 0) + node_input_tokens,
            "total_output_tokens": state.get("total_output_tokens", 0) + node_output_tokens,
            "api_call_timestamps": node_timestamps
        }