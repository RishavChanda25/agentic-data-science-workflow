import pandas as pd
import os

def extract_dataset_metadata(file_path: str, target_col: str = None) -> dict:
    """
    Pure-Python Pre-Flight Profiler to extract dataset realities.
    Runs with zero Orchestrator Tax (No LLM API calls).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f" 🛑 [Profiler Error] Dataset not found at: {file_path}")

    # Read the dataset
    df = pd.read_csv(file_path)

    # 1. Structural Dimensions & Memory Footprint
    rows = df.shape[0]
    cols = df.shape[1]
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    # 2. Fatal Constraint Triggers
    has_nulls = bool(df.isna().sum().sum() > 0)

    # 3. Categorical Feature & Unstructured Text Detection
    has_free_text_columns = False
    has_categorical_features = False
    max_cardinality = 0

    object_cols = df.select_dtypes(include=["object", "string", "category"]).columns

    if len(object_cols) > 0:
        has_categorical_features = True

    for col in object_cols:
        unique_count = df[col].nunique()

        if unique_count > max_cardinality:
            max_cardinality = unique_count

        # Heuristic: very high-cardinality string columns are likely free text
        if unique_count > 50:
            has_free_text_columns = True

    # NEW: 4. Integer Categoricals (Nominal Ints) Detection
    # If integer columns have very few unique values, they are likely categorical (e.g. 0=HR, 1=IT)
    has_nominal_ints = False
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if df[col].nunique() < 15:  
            has_nominal_ints = True
            break

    # 5. Target Variable Analysis (Task Type & Imbalance)
    task_type = "unknown"
    is_imbalanced = False
    minority_class_ratio = 1.0

    if target_col and target_col in df.columns:
        target_series = df[target_col]
        unique_targets = target_series.nunique()

        # Heuristic: If target has few unique values or is a string, it's Classification
        if unique_targets < 20 or pd.api.types.is_object_dtype(target_series):
            task_type = "classification"
            
            # Calculate class imbalance
            class_counts = target_series.value_counts(normalize=True)
            minority_class_ratio = float(class_counts.min())
            
            # Flag if the minority class is less than 5% of the data
            if minority_class_ratio < 0.05:  
                is_imbalanced = True
        else:
            task_type = "regression"

    return {
        "rows": rows,
        "cols": cols,
        "memory_mb": round(memory_mb, 2),
        "has_nulls": has_nulls,
        "has_free_text_columns": has_free_text_columns,
        "has_categorical_features": has_categorical_features,
        "has_nominal_ints": has_nominal_ints,
        "max_cardinality": int(max_cardinality),
        "task_type": task_type,
        "is_imbalanced": is_imbalanced,
        "minority_class_ratio": round(minority_class_ratio, 4)
    }