# 🤖 Neurosymbolic Multi-Agent Data Science Pipeline

An autonomous, multi-agent artificial intelligence pipeline that dynamically routes, executes, and self-heals data science workflows based on natural language intent. 

Built using LangGraph and Streamlit, this system features a **Neurosymbolic Supervisor** that utilizes the Z3 Theorem Prover to mathematically guarantee the safety of LLM-generated execution paths before they are deployed to worker agents.

## ✨ Key Features

* **🧠 Dynamic Intent-Based Routing:** The system does not follow a rigid pipeline. The Supervisor Agent evaluates the user's natural language prompt (or selected persona) and dynamically constructs a Directed Acyclic Graph (DAG) for the exact workflow required (e.g., skipping modelling for C-Suite Pitch, or skipping deep EDA for Rapid Baseline).
* **🛡️ Neurosymbolic Safety Guardrails (Z3):** To prevent LLM hallucinations, the Supervisor's proposed DAG is intercepted by the Z3 Theorem Prover. The route is mathematically evaluated against strict chronological and dependency constraints. If a violation is detected, the workflow undergoes an autonomous self-healing repair loop until a safe route is generated.
* **📊 Real-Time Observability Dashboard:** A deeply integrated telemetry system measures the computational cost of the architecture. It decouples the $O(1)$ token cost of the Supervisor from the dynamic compute cost of the Workers, and tracks API latency and stochastic taxes required for autonomous error recovery.
* **📄 Dynamic Artifact Rendering:** The Reporting Agent synthesizes the workflow into a comprehensive Markdown document. The Streamlit frontend uses recursive repository scanning to autonomously locate and natively render all generated EDA and metric visualizations, bypassing standard browser security blocks.

## 🏗️ System Architecture (Variant 3)

1. **Supervisor Agent:** Classifies user request into a utility preset, drafts a route, verifies it via Z3, and dispatches tasks.
2. **Data Cleaning Agent:** Handles missing values, outliers, and data type normalization.
3. **EDA Agent:** Generates statistical summaries and visual figures (e.g., correlation matrices, distributions).
4. **Feature Engineering Agent:** Applies scaling, encoding, and SMOTE balancing.
5. **Modelling Agent:** Trains multiple models (Logistic Regression, Random Forest, XGBoost), selects the best performer, and saves the `.pkl` artifact.
6. **Reporting Agent:** Adapts its LLM persona (e.g., highly technical vs. executive summary) to generate a final markdown report tailored to the user's initial prompt.

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Google Gemini API Key (`GEMINI_API_KEY`)

### Installation
1. **Clone the repository:**
```bash
git clone [https://github.com/RishavChanda25/agentic-data-science-workflow.git](https://github.com/RishavChanda25/agentic-data-science-workflow.git)
cd agentic-data-science-workflow
```

2. **Create and activate a virtual environment (Optional but recommended):**
```bash
conda create -n data-science-workflow python=3.10
conda activate data-science-workflow
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
Create a `.env` file in the root directory and add your API key:
```env
GEMINI_API_KEY="your_api_key_here"
```

### Running the Application
Launch the interactive Streamlit UI by running:
```bash
streamlit run app.py
```
Upload a dataset (e.g., the included Heart Disease or PaySim dataset), specify your target variable, enter your request (or select one of the provided presets) and click "Initialize Agentic Workflow".

## 📁 Repository Structure

```text
├── data/
│   ├── raw/                 # Uploaded datasets stored here
│   ├── processed/           # Cleaned and engineered intermediate datasets
│   └── artifacts/           # Processed pipeline artifacts (model files, JSONs, etc.)
├── eval/                    # Evaluation scripts
├── notebooks/               # Miscellaneous Jupyter notebooks
├── reports/
│   ├── figures/             # Auto-generated EDA and metric plots
│   ├── final_reports/       # Synthesized Markdown reports
│   └── results/             # eval_results.csv and final thesis plots
├── workflow_engine/
│   ├── agents/              # Individual worker agent logic (Supervisor, EDA, Modelling, etc.)
│   ├── orchestrators/       # Orchestration graph logic
│   ├── schemas/             # Schemas for request preset classification and route determination
│   ├── tools/               # Source code for tools called by agents (REPL Sandbox)
│   ├── utils/               # Utility functions (Z3 Verifier, LLM-as-a-Judge)
│   └── state.py             # LangGraph state dictionary definitions
├── app.py                   # Streamlit frontend application
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## 🧪 Evaluation & Telemetry (Dissertation Results)
This repository includes the raw evaluation data (`reports/results/eval_results.csv`) and the plotting scripts used to evaluate the architecture across multiple datasets. It tracks F1-Scores, latency, token consumption, and qualitative LLM Judge scores across 18 distinct permutation runs for academic benchmarking.