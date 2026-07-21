import os
import sys
import time
import pandas as pd
from dotenv import load_dotenv
import json

# --- 1. PATH RESOLUTION & IMPORTS ---
# Pop potentially stale or corrupted SSL environment variables
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("SSL_CERT_DIR", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv()

from workflow_engine.orchestrators.dynamic_graph import build_dynamic_pipeline
from workflow_engine.utils.data_profiler import extract_dataset_metadata
from workflow_engine.utils.llm_judge import grade_report_with_llm

# --- 2. EXPERIMENTAL MATRIX CONFIGURATION ---
datasets_config = {
    "Heart_Disease_1K": {
        "path": os.path.join(project_root, "data", "raw", "heart_disease_statlog.csv").replace('\\', '/'),
        "target": "num"
    },
    "Telco_Churn_7K": {
        "path": os.path.join(project_root, "data", "raw", "telco_customer_churn.csv").replace('\\', '/'),
        "target": "Churn"
    },
    "PaySim_Fraud_100K": {
        "path": os.path.join(project_root, "data", "raw", "paysim_100k_sample.csv").replace('\\', '/'),
        "target": "isFraud"
    }
}

trigger_prompts = {
    "RAPID_BASELINE": "I have a tabular dataset and I need a baseline classification model as quickly as possible. I only care about getting a reasonable benchmark with minimal computation. Please avoid unnecessary processing and give me a quick performance summary.",
    "QUICK_EXPLAINABLE": "I need a predictive model that I can easily explain to non-technical stakeholders. The model should prioritise interpretability over predictive performance, and I don't want complex ensembles or neural networks.",
    "ENTERPRISE_STANDARD": "Build a production-ready machine learning pipeline for this dataset. I want a reliable end-to-end workflow including data cleaning, exploratory analysis, feature engineering, model training, evaluation and a comprehensive report suitable for deployment discussions.",
    "KAGGLE_COMPETITOR": "Maximise predictive performance on this dataset. Computational cost is not important. Train strong machine learning models, compare them thoroughly, and select whichever achieves the highest predictive performance regardless of interpretability.",
    "REGULATORY_COMPLIANCE": "I need a machine learning solution for a highly regulated environment. Every prediction should be easy to justify and audit, so transparency and explainability are much more important than maximising accuracy.",
    "C_SUITE_PITCH": "I don't need a predictive model. I just want an executive-level understanding of this dataset. Produce exploratory analysis, key insights and visualisations that I can present to senior management without discussing machine learning algorithms."
}

def run_evaluation():
    # --- 3. EXECUTION ENGINE ---
    evaluation_results = []
    app = build_dynamic_pipeline()

    print("🚀 Starting the Automated 18-Run Matrix...\n" + "="*60)

    for dataset_name, d_config in datasets_config.items():
        raw_path = d_config["path"]
        target_col = d_config["target"]
        
        # Pre-Flight Profiling
        dataset_metadata = extract_dataset_metadata(raw_path, target_col)
        
        for expected_preset, user_request in trigger_prompts.items():
            print(f"\n🧪 RUNNING: [{dataset_name}] | [{expected_preset}]")
            
            initial_state = {
                "messages": [f"Automated Eval: {dataset_name} - {expected_preset}"],
                "user_request": user_request,
                "target_variable": target_col,
                "raw_dataset_path": raw_path,
                "current_dataset_path": raw_path,
                "artifacts": {},
                "current_step": "start",
                "error_flag": False,
                "error_message": "",
                "dataset_metadata": dataset_metadata,
                "active_preset": "",
                "proposed_route": [],
                "z3_verification_status": False,
                "total_sleep_time": 0.0,
                "supervisor_latency": 0.0,
                "supervisor_tokens": 0,
                "supervisor_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "api_call_timestamps": []
            }

            start_time = time.time()
            final_state = initial_state
            executed_route = []
            
            try:
                # Use invoke to keep your metric tracking logic consistent
                final_state = app.invoke(initial_state, {"recursion_limit": 25})
                # To get executed route from final state if nodes were tracked
                executed_route = final_state.get("proposed_route", []) 
            except Exception as e:
                final_state["error_flag"] = True
                final_state["error_message"] = str(e)

            end_time = time.time()
            gross_runtime = end_time - start_time
            pure_compute_time = gross_runtime - final_state.get("total_sleep_time", 0.0)
            
            # --- Run OpenAI Judge ---
            judge_scores = {"Technical Score": 0, "Actionability Score": 0, "Alignment Score": 0}
            if not final_state.get("error_flag") and "final_report" in final_state.get("artifacts", {}):
                report_path = final_state["artifacts"]["final_report"]
                if os.path.exists(report_path):
                    with open(report_path, "r", encoding="utf-8") as f:
                        judge_scores = grade_report_with_llm(f.read(), expected_preset, user_request)

            # --- Extract Model Metrics (If Applicable) ---
            model_metrics = {
                "Model Name": "N/A",
                "Accuracy": "N/A",
                "Precision": "N/A",
                "Recall": "N/A",
                "F1 Score": "N/A",
                "Training Time (s)": "N/A"
            }
            
            # We look for the model metrics JSON in the artifacts dictionary
            if not final_state.get("error_flag") and "model_metrics" in final_state.get("artifacts", {}):
                metrics_path = final_state["artifacts"]["model_metrics"]
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        try:
                            metrics_data = json.load(f)
                            model_metrics["Model Name"] = metrics_data.get("model_name", "N/A")
                            # Round floats to 4 decimal places for clean tables
                            model_metrics["Accuracy"] = round(metrics_data.get("accuracy", 0.0), 4)
                            model_metrics["Precision"] = round(metrics_data.get("precision", 0.0), 4)
                            model_metrics["Recall"] = round(metrics_data.get("recall", 0.0), 4)
                            model_metrics["F1 Score"] = round(metrics_data.get("f1_score", 0.0), 4)
                            model_metrics["Training Time (s)"] = round(metrics_data.get("model_training_time", 0.0), 4)
                        except json.JSONDecodeError:
                            print(f"   ⚠️ Could not parse model_metrics.json at {metrics_path}")
            
            # --- Calculate Worker vs. Global Metrics (NEW) ---
            # Get raw totals
            total_in_tokens = final_state.get("total_input_tokens", 0)
            total_out_tokens = final_state.get("total_output_tokens", 0)
            total_tokens = total_in_tokens + total_out_tokens
            
            timestamps = final_state.get("api_call_timestamps", [])
            total_api_calls = len(timestamps)
            
            supervisor_tokens = final_state.get("supervisor_tokens", 0)
            supervisor_calls = final_state.get("supervisor_calls", 0)
            supervisor_latency = final_state.get("supervisor_latency", 0.0)
            
            # Calculate Worker specific costs
            worker_tokens = total_tokens - supervisor_tokens
            worker_calls = total_api_calls - supervisor_calls
            worker_compute_time = pure_compute_time - supervisor_latency

            # --- Append to Master List (UPDATED) ---
            evaluation_results.append({
                "Dataset": dataset_name,
                "Preset": expected_preset,
                "Executed Route": " -> ".join(executed_route),
                
                # Global Time Metrics
                "Runtime (s)": round(pure_compute_time, 2),
                
                # Orchestration Tax vs Worker Compute
                "Supervisor Latency (s)": round(supervisor_latency, 2),
                "Worker Compute (s)": round(worker_compute_time, 2),
                "Supervisor Tokens": supervisor_tokens,
                "Worker Tokens": worker_tokens,
                "Total Tokens": total_tokens,
                "Supervisor API Calls": supervisor_calls,
                "Worker API Calls": worker_calls,
                "Total API Calls": total_api_calls,
                
                # Predictive Metrics
                "Model Name": model_metrics["Model Name"],
                "Accuracy": model_metrics["Accuracy"],
                "Precision": model_metrics["Precision"],
                "Recall": model_metrics["Recall"],
                "F1 Score": model_metrics["F1 Score"],
                "Training Time (s)": model_metrics["Training Time (s)"],
                
                # Interpretability Metrics
                "Judge: Tech": judge_scores.get("Technical Score", 0),
                "Judge: Action": judge_scores.get("Actionability Score", 0),
                "Judge: Align": judge_scores.get("Alignment Score", 0)
            })
            
            print(f"   [DONE] Route: {' -> '.join(executed_route)} | F1: {model_metrics['F1 Score']} | Train Time: {model_metrics['Training Time (s)']}s | Align Score: {judge_scores.get('Alignment Score', 0)}/10")

    # --- 4. EXPORT TO reports/results ---
    df_results = pd.DataFrame(evaluation_results)
    results_dir = os.path.join(project_root, "reports", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, "eval_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n💾 Evaluation complete! Results exported to {csv_path}")
    print("\n--- Summary Preview ---")
    print(df_results[["Dataset", "Preset", "Model Name", "F1 Score", "Judge: Align"]].to_string(index=False))

if __name__ == "__main__":
    run_evaluation()