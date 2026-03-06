import streamlit as st
import os
import sys
import pandas as pd
from dotenv import load_dotenv

# --- LOAD ENVIRONMENT VARIABLES FIRST ---
load_dotenv()

# Ensure Python can find the workflow_engine module
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# NOW we can safely import our agents because the API key is loaded
from workflow_engine.orchestrators.linear_graph import build_linear_pipeline

# --- Page Config ---
st.set_page_config(page_title="AI Data Science Pipeline", page_icon="🤖", layout="wide")
st.title("🤖 Autonomous Data Science Pipeline")
st.markdown("Upload a dataset and let the multi-agent AI pipeline clean, engineer, model, and report on your data.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Pipeline Configuration")
    uploaded_file = st.file_uploader("Upload Raw Dataset (CSV)", type=["csv"])
    target_col = st.text_input("Target Variable Column Name", value="target")
    run_button = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

# --- Main Application Logic ---
if run_button:
    if uploaded_file is None:
        st.error("Please upload a CSV file first!")
    elif not target_col:
        st.error("Please specify the target variable!")
    else:
        # 1. Save the uploaded file to disk so our agents can access it
        raw_dir = os.path.join("data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        raw_path = os.path.join(raw_dir, "uploaded_dataset.csv").replace('\\', '/')
        
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.success(f"Dataset successfully uploaded and saved to `{raw_path}`.")

        # 2. Setup the Initial State
        initial_state = {
            "messages": ["Starting Streamlit UI pipeline run."],
            "user_request": f"Clean the data, engineer features, and train a classification model to predict '{target_col}'.",
            "target_variable": target_col,
            "raw_dataset_path": raw_path,
            "current_dataset_path": raw_path,
            "artifacts": {},
            "current_step": "start",
            "error_flag": False,
            "error_message": "",
            "revision_count": 0,
            "user_preferences": {},
            "next_node": ""
        }

        # 3. Build the Graph
        app = build_linear_pipeline()
        
        # 4. Execute with Progress Tracking
        st.markdown("### ⚙️ Pipeline Execution Log")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        nodes_executed = 0
        total_nodes = 5 # Cleaning, EDA, FE, Modelling, Reporting
        final_state = None
        
        with st.spinner("Agents are working..."):
            try:
                for output in app.stream(initial_state):
                    for node_name, state_update in output.items():
                        nodes_executed += 1
                        progress_percentage = int((nodes_executed / total_nodes) * 100)
                        progress_bar.progress(progress_percentage)
                        status_text.info(f"✅ Completed: **{node_name.replace('_', ' ').title()}**")
                        
                        final_state = state_update
                        
                        if state_update.get("error_flag"):
                            st.error(f"Pipeline failed at {node_name}: {state_update.get('error_message')}")
                            st.stop()
                            
            except Exception as e:
                st.error(f"A critical error occurred: {e}")
                st.stop()
                
        # 5. Display the Final Results
        st.success("🎉 Pipeline Execution Complete!")
        st.markdown("---")
        
        if final_state and not final_state.get("error_flag"):
            artifacts = final_state.get("artifacts", {})
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📄 AI-Generated Report")
                if "final_report" in artifacts and os.path.exists(artifacts["final_report"]):
                    with open(artifacts["final_report"], "r", encoding="utf-8") as f:
                        st.markdown(f.read())
                else:
                    st.warning("No final markdown report was generated.")
                    
            with col2:
                st.subheader("📊 Model Evaluation")
                if "confusion_matrix" in artifacts and os.path.exists(artifacts["confusion_matrix"]):
                    st.image(artifacts["confusion_matrix"], caption="Confusion Matrix")
                else:
                    st.warning("No confusion matrix available.")