"""
Extractor engine for trend analytics using LLM structured outputs.
"""
import json
import logging
import re
from typing import Optional, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

import config
from agent_module.providers import create_llm
from trend_module.schemas import ProjectTrendExtraction

logger = logging.getLogger("trend_analytics.extractor")

SYSTEM_PROMPT = """You are an expert AI Data Analyst specializing in engineering, procurement, and infrastructure tender projects (licitaciones).
Your task is to analyze project report summaries and extract structured statistical trend analytics.

Extract all requested metadata fields strictly adhering to the JSON schema:
1. engineering_field: Select the primary engineering discipline (e.g. Civil & Infrastructure, Water & Sanitation, Energy & Power, Transportation & Roads, Environmental, Buildings & Facilities, Telecom & IT, Industrial & Mining, Other).
2. project_type: Select the service model (e.g. Construction Supervision & Inspection, Feasibility, Audit & Design, Maintenance & Operation, Technical & Strategic Consulting, Turnkey / EPC Construction, Other).
3. country: Full official country name (e.g. Panama, Colombia, Spain, Peru, United States).
4. country_code: 2-letter ISO uppercase country code (e.g. PA, CO, ES, PE, MX, US).
5. region: Geographic region (e.g. LATAM, Central America, South America, Europe, North America, Global).
6. city_province: Specific city/province/department if mentioned.
7. budget_raw: Original raw budget phrase e.g. "B/. 12,500,000.00" or "$2.5M USD".
8. budget_usd_estimate: Estimated total budget converted to USD float value (e.g. 12500000.0).
9. budget_scale: Standard range bucket (< $500K, $500K - $2M, $2M - $10M, $10M - $50M, > $50M, Undisclosed / Not Stated).
10. funding_source: Public State Budget, Multilateral Development Bank (IDB, World Bank, CAF, CABEI), Private Capital, Public-Private Partnership (PPP), Unknown.
11. multilateral_entity_name: Name of multilateral bank if mentioned (e.g. BID, Banco Mundial, CAF, BCIE).
12. execution_period_months: Duration in months as integer (e.g. 24).
13. required_certifications: List of required ISO or technical certifications e.g. ["ISO 9001", "ISO 14001"].
14. consortium_allowed: true if consortium/joint venture is permitted, false if prohibited, null if omitted.
15. key_roles: List of top key personnel roles e.g. ["Director de Proyecto", "Especialista Ambiental", "Ingeniero Residente"].

Return ONLY a valid JSON object conforming to the schema."""


def extract_project_trends(
    report_summary: str,
    pliego_form_data: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
) -> ProjectTrendExtraction:
    """
    Extracts structured trend analytics from a project report summary and optional pliego form responses.
    """
    if not report_summary or not report_summary.strip():
        raise ValueError("Cannot extract trends from empty report summary")

    # Combine text context
    text_to_analyze = f"### PROJECT REPORT SUMMARY:\n{report_summary.strip()}"
    if pliego_form_data:
        text_to_analyze += f"\n\n### ADDITIONAL PROJECT FORM DETAILS:\n{json.dumps(pliego_form_data, ensure_ascii=False, indent=2)}"

    selected_model_name = model_name or getattr(config, "DEFAULT_CHAT_MODEL", "CLAUDE")
    
    try:
        model_cfg = config.MODEL_REGISTRY.get(selected_model_name)
        llm = create_llm(model_cfg)
    except Exception as e:
        logger.warning(f"Could not load configured model '{selected_model_name}', falling back to default LLM: {e}")
        # Try fallback to CLAUDE or first available model
        available_models = list(config.MODEL_REGISTRY._models.keys())
        if available_models:
            model_cfg = config.MODEL_REGISTRY.get(available_models[0])
            llm = create_llm(model_cfg)
        else:
            raise RuntimeError("No LLM model configuration available in MODEL_REGISTRY")

    # Attempt structured output via LangChain or direct JSON prompt
    try:
        if hasattr(llm, "with_structured_output"):
            structured_llm = llm.with_structured_output(ProjectTrendExtraction)
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=text_to_analyze),
            ]
            result = structured_llm.invoke(messages)
            if isinstance(result, ProjectTrendExtraction):
                return result
    except Exception as struct_err:
        logger.warning(f"Structured output call failed, falling back to manual JSON parsing: {struct_err}")

    # Fallback: Normal invocation with JSON extraction
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=text_to_analyze),
    ]
    response = llm.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)

    # Clean code blocks
    cleaned = re.sub(r"```json\s*", "", content)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()

    # Find JSON block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        json_str = match.group(0)
    else:
        json_str = cleaned

    data = json.loads(json_str)
    return ProjectTrendExtraction.model_validate(data)
