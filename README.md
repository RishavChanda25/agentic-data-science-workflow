🤖 Agentic Data Science Workflow

An autonomous, multi-agent AI pipeline built with LangGraph and the Gemini 2.5 Flash API to automate the end-to-end machine learning lifecycle.

This project transitions raw datasets into trained predictive models and synthesizes the mathematical metrics into human-readable business reports, completely autonomously.

✨ Key Features

End-to-End Automation: Seamlessly handles Data Cleaning, Exploratory Data Analysis (EDA), Feature Engineering, Model Training, and Reporting.

Intra-Node Self-Correction: Agents execute dynamically generated Python code in a secure REPL environment, featuring a 3-attempt LLM self-correction loop to debug pandas and scikit-learn runtime errors.

Interactive Frontend: A clean, user-friendly Streamlit web interface allowing users to upload CSVs, define target variables, and watch the pipeline execute in real-time.

Decoupled Architecture: Built using a state-driven LangGraph architecture, keeping worker agents isolated and modular.

🏗️ System Architecture (Variant 1: Linear Pipeline)

Currently, the system operates as a deterministic, sequential pipeline:

Cleaning Agent: Imputes missing values, drops duplicates, and normalizes column names.

EDA Agent: Generates statistical summaries, detects cardinality, and plots feature distributions.

Feature Engineering Agent: Applies dynamic One-Hot Encoding and Standard Scaling.

Modelling Agent: Performs train-test splits, trains baseline models, and outputs serialized models (.pkl) and evaluation metrics.

Reporting Agent: Synthesizes raw JSON artifacts into a comprehensive Markdown report.

🚀 Getting Started

Prerequisites
Python 3.10+

A Google Gemini API Key

Installation
1. Clone the repository

2. Create and activate a Conda environment

3. Install dependencies

4. Set up environment variables
Create a .env file in the root directory and add your Gemini API key:

Running the Application
Launch the interactive Streamlit UI by running:

streamlit run app.py

Upload a dataset (e.g., the included Heart Disease dataset), specify your target variable, and click "Run Pipeline."
