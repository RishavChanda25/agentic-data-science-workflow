import os
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# --- GROUND TRUTH RUBRIC FROM ROUTING SCHEMA ---
PRESET_DEFINITIONS = {
    "RAPID_BASELINE": "(Speed > Accuracy > Interpretability) Produce a working baseline model in the shortest possible time. Extreme brevity. No extensive analysis.",
    "QUICK_EXPLAINABLE": "(Speed > Interpretability > Accuracy) Fast but transparent. Focus on educational, easily explainable insights for non-technical users.",
    "ENTERPRISE_STANDARD": "(Accuracy > Interpretability > Speed) The canonical end-to-end workflow. Balanced, highly detailed, production-ready ML pipeline report.",
    "KAGGLE_COMPETITOR": "(Accuracy > Speed > Interpretability) Ruthless predictive optimization. Heavy ML jargon, deep statistical trade-offs, omit high-level business generalizations.",
    "REGULATORY_COMPLIANCE": "(Interpretability > Accuracy > Speed) Fully auditable. Emphasize transparency, bias checking, and strict justification of the model choices.",
    "C_SUITE_PITCH": "(Interpretability > Speed > Accuracy) Pure EDA and executive storytelling. Focus on business impact, actionable insights, and visuals. Zero ML jargon."
}

class ReportGrades(BaseModel):
    """The structured grading rubric for the LLM Judge."""
    
    technical_appropriateness: int = Field(
        description="Score 1 to 10. 1 = Fails to match the required technical depth. 10 = Perfectly matches the expected technical jargon level of the preset."
    )
    actionability: int = Field(
        description="Score 1 to 10. 1 = Raw numbers with no context. 10 = Strong insights perfectly aligned with the preset's specific goals."
    )
    audience_alignment: int = Field(
        description="Score 1 to 10. Overall, how perfectly does the tone, length, and content match the requested orchestration preset definition?"
    )
    reasoning: str = Field(
        description="A strict 2-sentence justification for the scores given, directly referencing the preset definition."
    )

def grade_report_with_llm(report_text: str, preset: str, user_request: str) -> dict:
    """Evaluates the final markdown report using OpenAI's GPT-4o as an independent cross-vendor Judge."""
    
    if not report_text or not report_text.strip():
        return {
            "Technical Score": 0, "Actionability Score": 0, 
            "Alignment Score": 0, "Judge Reasoning": "No report text provided."
        }
        
    print(f"⚖️  [OpenAI Judge] Grading report against the strict {preset} rubric...")

    # Using gpt-4o as the independent frontier judge. 
    # Temperature 0.0 ensures highly deterministic grading.
    llm_judge = ChatOpenAI(model="gpt-4o", temperature=0.0)
    structured_judge = llm_judge.with_structured_output(ReportGrades)
    
    preset_definition = PRESET_DEFINITIONS.get(preset, "Standard ML workflow.")
    
    system_prompt = f"""You are an impartial, elite academic evaluator grading an AI-generated Data Science report.
    
    The user made the following request: "{user_request}"
    The pipeline was forced to execute under the following strict persona definition:
    
    PRESET: [{preset}]
    DEFINITION: {preset_definition}
    
    Your job is to read the final report and evaluate it strictly based on whether it conforms to that exact definition.
    Do not grade it on whether it is a "good" general data science report. Grade it on how well it obeys the constraints of the {preset} definition.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Here is the generated report:\n\n{report_text}")
    ]
    
    try:
        result: ReportGrades = structured_judge.invoke(messages)
        return {
            "Technical Score": result.technical_appropriateness,
            "Actionability Score": result.actionability,
            "Alignment Score": result.audience_alignment,
            "Judge Reasoning": result.reasoning
        }
    except Exception as e:
        print(f"❌ [OpenAI Judge] Grading failed: {e}")
        return {
            "Technical Score": 0, "Actionability Score": 0, 
            "Alignment Score": 0, "Judge Reasoning": f"Error: {str(e)}"
        }