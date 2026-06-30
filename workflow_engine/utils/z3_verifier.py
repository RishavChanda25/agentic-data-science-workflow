from z3 import Solver, Bool, sat

def verify_proposed_route(proposed_route: list, dataset_metadata: dict, preset: str) -> tuple[bool, str]:
    """
    Formally verifies if an LLM-proposed route is mathematically safe and respects the preset boundaries.
    """
    if preset == "OUT_OF_SCOPE":
        return False, "🛑 [Trapdoor] Input classified as out of scope or prompt injection."

    solver = Solver()

    # Define Node Booleans
    visits_cleaning = Bool('visits_cleaning')
    visits_eda = Bool('visits_eda')
    visits_fe = Bool('visits_fe')
    visits_modelling = Bool('visits_modelling')
    visits_reporting = Bool('visits_reporting')

    # Map the proposed list to the booleans
    solver.add(visits_cleaning == ("data_cleaning" in proposed_route))
    solver.add(visits_eda == ("eda" in proposed_route))
    solver.add(visits_fe == ("feature_engineering" in proposed_route))
    solver.add(visits_modelling == ("modelling" in proposed_route))
    solver.add(visits_reporting == ("reporting" in proposed_route))

    # Extract Metadata
    has_nulls = dataset_metadata.get("has_nulls", False)
    has_free_text_columns = dataset_metadata.get("has_free_text_columns", False)

    requires_data_cleaning = (
        has_nulls or has_free_text_columns
    )

    has_nominal_ints = dataset_metadata.get("has_nominal_ints", False)
    has_categorical_features = dataset_metadata.get("has_categorical_features", False)

    requires_feature_engineering = (
        (has_nominal_ints or has_categorical_features) and preset != "C_SUITE_PITCH"
    )

    # -----------------------------------------
    # 1. GLOBAL FATAL DATA CONSTRAINTS
    # -----------------------------------------
    if requires_feature_engineering:
        solver.add(visits_fe == True) # required for one-hot encoding of categorical features
    if requires_data_cleaning:
        solver.add(visits_cleaning == True) # required for handling nulls or free text columns

    # The reporting node is mandatory for all valid workflows
    solver.add(visits_reporting == True)
        
    # -----------------------------------------
    # 2. PRESET TOPOLOGY CONSTRAINTS
    # -----------------------------------------
    if preset == "REGULATORY_COMPLIANCE":
        # Strict Audit Trail: Must visit every single node
        solver.add(visits_cleaning == True)
        solver.add(visits_eda == True)
        solver.add(visits_fe == True)
        solver.add(visits_modelling == True)

    elif preset == "ENTERPRISE_STANDARD":
        # Canonical end-to-end production workflow
        solver.add(visits_cleaning == True)
        solver.add(visits_eda == True)
        solver.add(visits_fe == True)
        solver.add(visits_modelling == True)
    
    elif preset == "C_SUITE_PITCH":
        # Pure EDA/Reporting: Must NOT model, MUST EDA
        solver.add(visits_modelling == False)
        solver.add(visits_fe == False)
        solver.add(visits_eda == True)
    
    else:
        # RAPID_BASELINE, QUICK_EXPLAINABLE, KAGGLE_COMPETITOR
        # These all require modelling, but EDA/Cleaning/FE are dynamic based on metadata & LLM choice
        solver.add(visits_modelling == True) 

    # -----------------------------------------
    # 3. VERIFICATION & ERROR ROUTING
    # -----------------------------------------
    if solver.check() == sat:
        return True, "✅ [Z3] Route formally verified as SAFE."
    else:
        # Determine exactly why the route was rejected for the LLM feedback loop
        reason = "Constraint violation detected."
        
        if preset == "REGULATORY_COMPLIANCE" and len(proposed_route) < 5:
            reason = "REGULATORY_COMPLIANCE requires a strict audit trail. All 5 nodes ('data_cleaning', 'eda', 'feature_engineering', 'modelling', 'reporting') must be visited."
        elif preset == "ENTERPRISE_STANDARD" and len(proposed_route) < 5:
            reason = "ENTERPRISE_STANDARD follows the canonical end-to-end workflow. All 5 nodes ('data_cleaning', 'eda', 'feature_engineering', 'modelling', 'reporting') must be visited."
        elif preset == "C_SUITE_PITCH" and "modelling" in proposed_route:
            reason = "C_SUITE_PITCH is for pure EDA. The 'modelling' node is strictly forbidden."
        elif preset == "C_SUITE_PITCH" and "feature_engineering" in proposed_route:
            reason = "C_SUITE_PITCH is for pure EDA. The 'feature_engineering' node is strictly forbidden."
        elif preset == "C_SUITE_PITCH" and "eda" not in proposed_route:
            reason = "C_SUITE_PITCH requires the 'eda' node to generate insights."
        elif requires_data_cleaning and "data_cleaning" not in proposed_route:
            reason = "Dataset contains Nulls. 'data_cleaning' node is mathematically mandatory."
        elif requires_feature_engineering and "feature_engineering" not in proposed_route:
            reason = "Dataset contains categorical features requiring encoding. 'feature_engineering' node is mandatory for One-Hot Encoding."
        elif "reporting" not in proposed_route:
            reason = "The 'reporting' node is mathematically mandatory at the end of all routes."
        elif preset not in ["C_SUITE_PITCH", "OUT_OF_SCOPE"] and "modelling" not in proposed_route:
            reason = f"The 'modelling' node is mandatory for the {preset} preset."

        return False, f"🛑 [Z3] Route REJECTED. {reason}"