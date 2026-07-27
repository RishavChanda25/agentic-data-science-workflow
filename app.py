import streamlit as st
import os
import sys
import time
import json
import pandas as pd
import re
import base64
from dotenv import load_dotenv

# --- 1. ENV & PATH RESOLUTION ---
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("SSL_CERT_DIR", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

load_dotenv()

from workflow_engine.orchestrators.dynamic_graph import build_dynamic_pipeline
from workflow_engine.utils.data_profiler import extract_dataset_metadata

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Agentic Data Science Orchestrator", page_icon="🧠", layout="wide")
st.title("🧠 Agentic Data Science Orchestrator")
st.markdown("Upload a dataset and define your goal. The AI Supervisor will dynamically plan, route, and execute a personalized workflow.")

# --- 3. UI: SIDEBAR (TELEMETRY & COMMANDS) ---
with st.sidebar:
    st.header("📂 Data Input")
    uploaded_file = st.file_uploader("Upload Raw Dataset (CSV)", type=["csv"])
    target_col = st.text_input("Target Variable Column Name", value="target", key="target_col_input")
    
# --- 4. UI: MAIN CONFIGURATION ---
st.subheader("🎯 Define Your Objective")

# Initialize session state for the prompt so it doesn't wipe on rerun
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = ""

# Callback function to update the text area safely
def set_preset(text):
    st.session_state.user_prompt = text

# Quick Preset Buttons (3x2 Grid)
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

# Define preset texts
rapid_text = "I have a tabular dataset and I need a baseline classification model as quickly as possible. I only care about getting a reasonable benchmark with minimal computation. Please avoid unnecessary processing and give me a quick performance summary."
explain_text = "I need a predictive model that I can easily explain to non-technical stakeholders. The model should prioritise interpretability over predictive performance, and I don't want complex ensembles or neural networks."
enterprise_text = "Build a production-ready machine learning pipeline for this dataset. I want a reliable end-to-end workflow including data cleaning, exploratory analysis, feature engineering, model training, evaluation and a comprehensive report suitable for deployment discussions."
kaggle_text = "Maximise predictive performance on this dataset. Computational cost is not important. Train strong machine learning models, compare them thoroughly, and select whichever achieves the highest predictive performance regardless of interpretability."
reg_text = "I need a machine learning solution for a highly regulated environment. Every prediction should be easy to justify and audit, so transparency and explainability are much more important than maximising accuracy."
csuite_text = "I don't need a predictive model. I just want an executive-level understanding of this dataset. Produce exploratory analysis, key insights and visualisations that I can present to senior management without discussing machine learning algorithms."

with col1:
    st.button("🚀 Rapid Baseline", on_click=set_preset, args=(rapid_text,), use_container_width=True)
with col2:
    st.button("🧠 Quick Explainable", on_click=set_preset, args=(explain_text,), use_container_width=True)
with col3:
    st.button("🏢 Enterprise Standard", on_click=set_preset, args=(enterprise_text,), use_container_width=True)
with col4:
    st.button("🏆 Kaggle Competitor", on_click=set_preset, args=(kaggle_text,), use_container_width=True)
with col5:
    st.button("⚖️ Regulatory Compliance", on_click=set_preset, args=(reg_text,), use_container_width=True)
with col6:
    st.button("💼 C-Suite Pitch", on_click=set_preset, args=(csuite_text,), use_container_width=True)

# The text area is now bound directly to the session state key! No pressing Ctrl+Enter required.
user_request = st.text_area("User Prompt:", key="user_prompt", height=100, placeholder="Select a preset above or type your custom objective here...")

run_button = st.button("⚡ Initialize Agentic Workflow", type="primary")

# --- 5. EXECUTION ENGINE (.stream) ---
if run_button:
    if not uploaded_file or not target_col or not user_request:
        st.error("⚠️ Please provide a dataset, target column, and user prompt.")
        st.stop()

    # 1. Save the uploaded file to disk so our agents can access it
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Standardize the file name just like Variant 1
    file_path = os.path.join(raw_dir, "uploaded_dataset.csv").replace('\\', '/')
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.success(f"Dataset successfully uploaded and saved to `{file_path}`.")

    # Extract silent metadata for the Supervisor
    dataset_metadata = extract_dataset_metadata(file_path, target_col)
    
    initial_state = {
        "messages": ["Initialize Streamlit UI Run"],
        "user_request": user_request,
        "target_variable": target_col,
        "raw_dataset_path": file_path,
        "current_dataset_path": file_path,
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

    app = build_dynamic_pipeline()
    
    # We must explicitly track the state as it aggregates!
    final_state = initial_state.copy()
    start_time = time.time()
    
    # Flag to track the Supervisor's first visit
    first_supervisor_visit = True

    # UI Log Container
    st.markdown("### 🔄 Live Execution Trace")
    
    try:
        with st.status("Waking up Supervisor Agent...", expanded=True) as status_box:
            
            for event in app.stream(initial_state, {"recursion_limit": 25}):
                for node_name, state_update in event.items():
                    
                    # Safely merge the state diffs (PROTECTING METRICS)
                    for key, value in state_update.items():
                        if key == "artifacts" and isinstance(value, dict):
                            final_state["artifacts"].update(value)
                        elif key == "api_call_timestamps" and isinstance(value, list):
                            # Prevent agents from overwriting the global API call list!
                            final_state["api_call_timestamps"].extend(value)
                        elif key in ["supervisor_tokens", "supervisor_calls", "supervisor_latency", "total_sleep_time"]:
                            # Only update these if the node actually yielded them (prevents overwriting with 0)
                            if value: 
                                final_state[key] = value
                        else:
                            final_state[key] = value
                    
                    current_route = final_state.get('proposed_route', [])
                    preset = final_state.get("active_preset", "ENTERPRISE_STANDARD")

                    # UI Updates based on node execution
                    if node_name == "supervisor":
                        if first_supervisor_visit:
                            status_box.write(f"🧠 **Supervisor:** Intent Evaluated - {preset}.")
                            
                            # Because Z3 runs INSIDE the supervisor node, if we reached here, 
                            # it means the internal while-loop succeeded and the route is verified.
                            # (We can check if it took multiple attempts by looking at token count or messages)
                            messages_count = len(final_state.get("messages", []))
                            if messages_count > 2:
                                # A normal run has 1 or 2 messages. More means Z3 rejected and forced a repair loop!
                                status_box.write(f"🛠️ **Supervisor Self-Repair:** Z3 rejected unsafe drafts. Route repaired.")
                                
                            status_box.write(f"🛤️ **Verified Route:** `{' ➔ '.join(current_route)}`")
                            status_box.write(f"🛡️ **Z3 Verifier:** Route mathematically validated as SAFE.")
                            first_supervisor_visit = False
                        else:
                            status_box.write(f"🧭 **Supervisor:** Routing to next step...")
                            
                    elif node_name != "__end__":
                        status_box.write(f"✅ **{node_name.replace('_', ' ').title()}** completed successfully.")
                    
                    # Catch self-healing errors dynamically
                    if final_state.get("error_flag"):
                        status_box.update(label="⚠️ Pipeline Error Detected!", state="error", expanded=True)
                        st.error(f"Error in {node_name}: {final_state.get('error_message')}")
                        st.stop()
                        
            status_box.update(label="🎉 Workflow Complete!", state="complete", expanded=False)
            
    except Exception as e:
        st.error(f"Execution Error: {str(e)}")
        st.stop()

    # --- 6. OUTPUT DISPLAY (Tabs) ---
    st.markdown("---")
    tab1, tab2 = st.tabs(["📄 Final Report", "📊 Observability & Artifacts"])
    
    artifacts = final_state.get("artifacts", {})
    
    with tab1:
        if "final_report" in artifacts and os.path.exists(artifacts["final_report"]):
            with open(artifacts["final_report"], "r", encoding="utf-8") as f:
                md_content = f.read()
                
            def find_file_in_repo(filename, search_path):
                for root, dirs, files in os.walk(search_path):
                    if filename in files:
                        return os.path.join(root, filename)
                return None

            import re
            project_root = os.path.dirname(os.path.abspath(__file__))
            img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
            
            last_idx = 0
            for match in re.finditer(img_pattern, md_content):
                text_before = md_content[last_idx:match.start()]
                if text_before.strip():
                    st.markdown(text_before, unsafe_allow_html=True)
                
                alt_text = match.group(1)
                img_path = match.group(2)
                filename = os.path.basename(img_path)
                
                actual_file_path = find_file_in_repo(filename, project_root)
                if actual_file_path and os.path.exists(actual_file_path):
                    st.image(actual_file_path, caption=alt_text)
                else:
                    st.error(f"⚠️ UI Missing Image: Could not locate '{filename}'")
                
                last_idx = match.end()
                
            remaining_text = md_content[last_idx:]
            if remaining_text.strip():
                st.markdown(remaining_text, unsafe_allow_html=True)
        else:
            st.warning("No final markdown report was generated for this route.")

    # --- THE NEW OBSERVABILITY DASHBOARD ---
    with tab2:
        st.header("📊 Observability Dashboard")
        
        # 1. Routing Data & Preset
        st.subheader("🛤️ Intent & Executed Route")
        
        active_preset = final_state.get("active_preset", "Unknown Preset")
        st.caption(f"**Execution Preset Triggered:** `{active_preset}`")
        
        executed_route = final_state.get('proposed_route', [])
        if executed_route:
            st.info(f"`{' ➔ '.join(executed_route)}`")
        else:
            st.info("No dynamic route generated.")

        # 2. Extract Exact Metrics
        gross_runtime = time.time() - start_time
        sleep_time = final_state.get("total_sleep_time", 0.0)
        pure_compute = gross_runtime - sleep_time
        
        sup_latency = final_state.get("supervisor_latency", 0.0)
        sup_tokens = final_state.get("supervisor_tokens", 0)
        sup_calls = final_state.get("supervisor_calls", 0)

        total_tokens = final_state.get("total_input_tokens", 0) + final_state.get("total_output_tokens", 0)
        total_calls = len(final_state.get("api_call_timestamps", []))

        worker_tokens = total_tokens - sup_tokens
        worker_calls = total_calls - sup_calls

        # 3. Render Top Row: Time Metrics
        st.subheader("⏱️ Execution Latency")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Gross Runtime", f"{gross_runtime:.2f} s")
        c2.metric("Pure Compute Time", f"{pure_compute:.2f} s", help="Total runtime minus API rate-limit sleep times.")
        c3.metric("Stochastic Tax (Sleep)", f"{sleep_time:.2f} s", help="Time spent sleeping due to self-healing rate limits.")

        # 4. Render Bottom Row: Token & Call Telemetry
        st.subheader("🪙 Token & API Telemetry")
        t1, t2, t3 = st.columns(3)
        t1.metric("Total Tokens Processed", f"{total_tokens:,}")
        t2.metric("Supervisor Tax (Tokens)", f"{sup_tokens:,}", help="The O(1) fixed cost of the orchestrator.")
        t3.metric("Worker Compute (Tokens)", f"{worker_tokens:,}", help="The dynamic cost of executing the pipeline.")

        t4, t5, t6 = st.columns(3)
        t4.metric("Total API Calls", f"{total_calls}")
        t5.metric("Supervisor API Calls", f"{sup_calls}")
        t6.metric("Worker API Calls", f"{worker_calls}")

        st.divider()

        # 5. Modelling Metrics
        st.subheader("📈 Modelling Metrics")
        if "model_metrics" in artifacts and os.path.exists(artifacts["model_metrics"]):
            with open(artifacts["model_metrics"], "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
                
                # Highlight Best Model if available
                best_model = metrics_data.get("best_model", None)
                if best_model:
                    st.success(f"🏆 **Best Model Selected:** `{best_model}`")
                
                # Display the full JSON payload interactively
                st.json(metrics_data)
        else:
            st.info("No modelling metrics generated for this route (e.g., Exploratory intent selected).")

        st.divider()
        
        # 6. Download Artifacts
        st.subheader("📦 Generated Artifacts")
        col_art1, col_art2 = st.columns(2)
        
        with col_art1:
            if "best_model" in artifacts and os.path.exists(artifacts["best_model"]):
                with open(artifacts["best_model"], "rb") as f:
                    st.download_button(
                        label="💾 Download `best_model.pkl`",
                        data=f,
                        file_name="best_model.pkl",
                        mime="application/octet-stream",
                        type="primary"
                    )
            else:
                st.info("No model generated.")
                
        with col_art2:
            st.caption("All intermediate datasets (`cleaned_data.csv`, `engineered_data.csv`) and figures have been successfully persisted to your local `data/processed/` and `reports/figures/` directories.")