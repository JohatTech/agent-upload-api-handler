"""
Service layer for orchestrating trend analytics extraction and Supabase persistence.
"""
import logging
from typing import Dict, Any, List, Optional
from supabase_module.supabase_client import create_client, Client
import config
from trend_module.extractor import extract_project_trends
from trend_module.schemas import ProjectTrendExtraction

logger = logging.getLogger("trend_analytics.service")


def _get_supabase_admin_client() -> Client:
    """Helper to initialize Supabase client using service role key if available."""
    url = getattr(config, "SUPABASE_URL", None) or getattr(config, "AZURE_SUPABASE_URL", None)
    if not url:
        import os
        url = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    
    key = getattr(config, "SUPABASE_SERVICE_ROLE_KEY", None) or getattr(config, "SUPABASE_SERVICE_KEY", None) or getattr(config, "SUPABASE_KEY", None)
    if not key:
        import os
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

    if not url or not key:
        raise ValueError("Supabase URL and Key must be set in environment")

    return create_client(url, key)


def process_notebook_trend_analytics(notebook_id: str) -> Dict[str, Any]:
    """
    Extracts and stores trend analytics for a specific notebook.
    """
    logger.info(f"Starting trend analytics extraction for notebook_id: {notebook_id}")
    supabase = _get_supabase_admin_client()

    # 1. Fetch Notebook info
    nb_res = supabase.table("notebooks").select("*").eq("id", notebook_id).execute()
    if not nb_res.data:
        raise ValueError(f"Notebook with id '{notebook_id}' not found.")
    
    notebook = nb_res.data[0]
    project_name = notebook.get("name") or notebook_id
    tag = notebook.get("tag")

    # 2. Fetch Report Summary
    report_summary = None
    report_res = supabase.table("reports").select("*").eq("notebook_id", notebook_id).execute()
    if report_res.data:
        report_summary = report_res.data[0].get("summary")

    # Fallback to pipeline_jobs if report summary is not in reports table
    pipeline_job_id = None
    if not report_summary:
        pj_res = supabase.table("pipeline_jobs").select("*").eq("id", notebook_id).execute()
        if pj_res.data:
            pipeline_job_id = pj_res.data[0].get("id")
            report_summary = pj_res.data[0].get("report_summary")

    if not report_summary or not report_summary.strip():
        raise ValueError(f"No report summary found for notebook '{notebook_id}'. Generate report first.")

    # 3. Extract Trends via LLM
    extraction: ProjectTrendExtraction = extract_project_trends(
        report_summary=report_summary,
        pliego_form_data={"project_name": project_name, "sector_tag": tag},
    )

    # 4. Prepare row for Supabase project_trend_analytics
    trend_row = {
        "notebook_id": notebook_id,
        "pipeline_job_id": pipeline_job_id,
        "project_name": project_name,
        "tag": tag,
        "engineering_field": extraction.engineering_field.value,
        "project_type": extraction.project_type.value,
        "country": extraction.country,
        "country_code": extraction.country_code.upper(),
        "region": extraction.region,
        "city_province": extraction.city_province,
        "budget_raw": extraction.budget_raw,
        "budget_usd_estimate": extraction.budget_usd_estimate,
        "budget_scale": extraction.budget_scale.value,
        "funding_source": extraction.funding_source.value,
        "multilateral_entity_name": extraction.multilateral_entity_name,
        "execution_period_months": extraction.execution_period_months,
        "required_certifications": extraction.required_certifications,
        "consortium_allowed": extraction.consortium_allowed,
        "key_roles": extraction.key_roles,
        "raw_llm_json": extraction.model_dump(),
    }

    # 5. Upsert into Supabase
    upsert_res = supabase.table("project_trend_analytics").upsert(
        trend_row, on_conflict="notebook_id"
    ).execute()

    logger.info(f"Successfully processed trend analytics for notebook: {notebook_id}")
    return {
        "status": "success",
        "notebook_id": notebook_id,
        "project_name": project_name,
        "extraction": extraction.model_dump(),
        "db_record": upsert_res.data[0] if upsert_res.data else trend_row,
    }


def process_all_pending_trends() -> Dict[str, Any]:
    """
    Backfills trend analytics for all completed reports/notebooks that are missing from project_trend_analytics.
    """
    logger.info("Checking for notebooks pending trend analytics processing...")
    supabase = _get_supabase_admin_client()

    # Get existing analyzed notebook_ids
    existing_res = supabase.table("project_trend_analytics").select("notebook_id").execute()
    existing_ids = {row["notebook_id"] for row in existing_res.data} if existing_res.data else set()

    # Get all reports
    reports_res = supabase.table("reports").select("notebook_id, summary").execute()
    all_reports = reports_res.data if reports_res.data else []

    pending_notebook_ids = [
        r["notebook_id"] for r in all_reports if r["notebook_id"] not in existing_ids and r.get("summary")
    ]

    logger.info(f"Found {len(pending_notebook_ids)} pending notebooks to process.")

    results = []
    success_count = 0
    fail_count = 0

    for nb_id in pending_notebook_ids:
        try:
            res = process_notebook_trend_analytics(nb_id)
            results.append(res)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to process trends for notebook '{nb_id}': {e}", exc_info=True)
            results.append({"status": "error", "notebook_id": nb_id, "error": str(e)})
            fail_count += 1

    return {
        "total_pending": len(pending_notebook_ids),
        "success_count": success_count,
        "fail_count": fail_count,
        "details": results,
    }
