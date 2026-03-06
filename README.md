🤖 Agentic Data Science Workflow
An autonomous, multi-agent AI pipeline built with LangGraph and the Gemini 2.5 Flash API to automate the end-to-end machine learning lifecycle.

This project transitions raw datasets into trained predictive models and synthesizes the mathematical metrics into human-readable business reports, completely autonomously.

✨ Key Features
End-to-End Automation: Seamlessly handles Data Cleaning, Exploratory Data Analysis (EDA), Feature Engineering, Model Training, and Reporting.

Intra-Node Self-Correction: Agents execute dynamically generated Python code in a secure REPL environment, featuring a 3-attempt LLM self-correction loop to debug pandas and scikit-learn runtime errors (e.g., dimensionality mismatches during One-Hot Encoding).

Interactive Frontend: A clean, user-friendly Streamlit web interface allowing users to upload CSVs, define target variables, and watch the pipeline execute in real-time.

Decoupled Architecture: Built using a state-driven LangGraph architecture, keeping worker agents isolated and modular.

🏗️ System Architecture (Variant 1: Linear Pipeline)
Currently, the system operates as a deterministic, sequential pipeline (baseline variant):

Cleaning Agent: Imputes missing values, drops duplicates, and normalizes column names.

EDA Agent: Generates statistical summaries, detects cardinality, and plots feature distributions.

Feature Engineering Agent: Applies dynamic One-Hot Encoding and Standard Scaling based on EDA insights.

Modelling Agent: Performs train-test splits, trains baseline models (Logistic Regression, Random Forest), and outputs serialized models (.pkl) and evaluation metrics.

Reporting Agent: Bypasses the REPL to safely synthesize raw JSON artifacts into a comprehensive Markdown report.

🚀 Getting Started
Prerequisites
Python 3.10+

A Google Gemini API Key

Installation
1. Clone the repository

Bash

git clone https://github.com/yourusername/data-science-workflow.git
cd data-science-workflow
2. Create and activate a Conda environment

Bash

conda create -p C:/venvs/data-science-workflow python=3.10 -y
conda activate C:/venvs/data-science-workflow
3. Install dependencies

Bash

pip install -r requirements.txt
4. Set up environment variables
Create a .env file in the root directory and add your Gemini API key:

Plaintext

GEMINI_API_KEY="your_actual_api_key_here"
Running the Application
Launch the interactive Streamlit UI by running:

Bash

streamlit run app.py
Upload a dataset (e.g., the Heart Disease dataset), specify your target variable, and click "Run Pipeline."

🔬 Future Research (MSc Dissertation Scope)
This repository will serve as the foundation for an ablation study on LLM orchestration techniques:

Variant 2 (Upcoming): Implementation of a central Supervisor Agent using strict, rule-based LangGraph Conditional Edges.

Variant 3 (Upcoming): A dynamic, preference-aware Supervisor capable of non-linear routing, conditional node bypassing, and intelligent workflow optimization based on user constraints.