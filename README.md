```
conda create -p C:/venvs/data-science-workflow (<- venv name) python=3.10 -y
```

```
conda activate C:/venvs/data-science-workflow
```

```
pip install -r requirements.txt
```

```
While running the app:
Terminal 1: streamlit run streamlit_ui.py
Terminal 2: uvicorn main:app --host 127.0.0.1 --port 8003 --reload
```