"""
Recovery & Manual Retry Service.
Provides:
  1. mark_crashed_jobs_on_startup(): One-time, non-blocking check on server startup that marks
     any previously in-flight jobs as 'failed' (interrupted by server crash/restart) so the user
     can see the state and click 'Reintentar'.
  2. retry_single_project_pipeline(): Explicit, user-driven on-demand retry for a single project.
Eliminates all automatic background monitoring, scanning threads, and autonomous loops.
"""

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import config
from core.formatting import sanitize_collection_name, set_collection_name
from core.azure_blob_service import AzureBlobService
from core.pipeline import process_project_folder, regenerate_report
from supabase_module.supabase_client import SupabaseModule

logger = logging.getLogger("recovery_service")

TEMP_UPLOAD_ROOT = Path("temp_uploads")


def mark_crashed_jobs_on_startup() -> int:
    """
    Synchronously reconciles pipeline jobs and notebooks that were left in non-terminal
    'processing' states when the server went down.
    
    1. If the job or notebook already has a completed report in Supabase, marks it as 'completed'/'ready'.
    2. If vectors exist but report generation was cut off, marks it as 'failed' with a message
       informing the user that vectors are saved and prompting them to click 'Reintentar'.
    3. If no vectors exist, marks it as 'failed' prompting the user to click 'Reintentar' to process from Azure.
    
    Does not spawn background threads, does not re-trigger jobs, and does not download files.
    """
    logger.info("[STARTUP] Checking for pipeline jobs interrupted by server restart...")
    crashed_count = 0
    recovered_count = 0

    try:
        supabase = SupabaseModule()

        stuck_jobs = supabase.get_stuck_pipeline_jobs()
        for job in stuck_jobs:
            job_id = job.get("id")
            if not job_id:
                continue

            vs_id = job.get("vector_store_id") or set_collection_name(job.get("project_name", ""))
            nb_id = job.get("notebook_id")

            # Check if report already exists
            if vs_id and supabase.has_report(notebook_id=nb_id, vector_store_id=vs_id):
                logger.info("[STARTUP] Job '%s' (vs: '%s') already has a completed report. Marking as completed.", job_id, vs_id)
                supabase.update_pipeline_job(job_id, status="completed")
                if nb_id:
                    supabase.client.table("notebooks").update({
                        "status": "ready",
                        "last_activity": datetime.now(timezone.utc).isoformat(),
                    }).eq("id", nb_id).execute()
                recovered_count += 1
                continue

            # Check if vectors already exist in documents table
            has_vectors = bool(vs_id and supabase.has_vectors_for_collection(vs_id))
            if has_vectors:
                logger.info("[STARTUP] Job '%s' has vectors saved but report was interrupted. Marking as failed with retry signal.", job_id)
                supabase.update_pipeline_job(
                    job_id,
                    status="failed",
                    report_summary="El servidor se reinició mientras se procesaba este proyecto. Los vectores están guardados. Haz clic en 'Reintentar' para generar el reporte sin volver a subir archivos.",
                )
            else:
                logger.info("[STARTUP] Job '%s' was interrupted before vectorization. Marking as failed with retry signal.", job_id)
                supabase.update_pipeline_job(
                    job_id,
                    status="failed",
                    report_summary="El servidor se reinició durante el procesamiento. Haz clic en 'Reintentar' para reanudar el procesamiento desde Azure.",
                )

            if nb_id:
                supabase.client.table("notebooks").update({
                    "status": "error",
                    "last_activity": datetime.now(timezone.utc).isoformat(),
                }).eq("id", nb_id).execute()

            crashed_count += 1

        # Check any leftover notebooks still in 'processing' status
        stuck_notebooks = supabase.get_stuck_notebooks()
        for nb in stuck_notebooks:
            nb_id = nb.get("id")
            if not nb_id:
                continue
            vs_id = nb.get("vector_store_id")

            if vs_id and supabase.has_report(notebook_id=nb_id, vector_store_id=vs_id):
                supabase.client.table("notebooks").update({
                    "status": "ready",
                    "last_activity": datetime.now(timezone.utc).isoformat(),
                }).eq("id", nb_id).execute()
                recovered_count += 1
            else:
                supabase.client.table("notebooks").update({
                    "status": "error",
                    "last_activity": datetime.now(timezone.utc).isoformat(),
                }).eq("id", nb_id).execute()
                crashed_count += 1

        logger.info(
            "[STARTUP] Interrupted jobs check complete │ recovered_as_ready=%d │ marked_failed_for_retry=%d",
            recovered_count,
            crashed_count,
        )
        return crashed_count

    except Exception as exc:
        logger.error("[STARTUP] Error while checking interrupted jobs on startup: %s", exc)
        return 0


def retry_single_project_pipeline(
    notebook_id: Optional[str] = None,
    vector_store_id: Optional[str] = None,
    project_name: Optional[str] = None,
    model_name: Optional[str] = None,
    user_email: Optional[str] = None,
) -> Dict[str, Any]:
    """
    User-directed manual retry of a single pipeline job.
    Called explicitly via POST /api/pipeline/retry.

    1. Resolves notebook, original project name, and vector_store_id.
    2. Updates notebook status to 'processing' and pipeline_jobs to 'triggered'.
    3. If vector embeddings already exist, skips ingestion and triggers report regeneration directly.
    4. If vectors do not exist, re-downloads project files from Azure Blob Storage and processes them.
    """
    supabase = SupabaseModule()

    # Resolve target notebook
    target_notebook = None
    if notebook_id:
        res = supabase.client.table("notebooks").select("*").eq("id", notebook_id).limit(1).execute()
        if res.data:
            target_notebook = res.data[0]

    if not target_notebook and vector_store_id:
        res = supabase.client.table("notebooks").select("*").eq("vector_store_id", vector_store_id).limit(1).execute()
        if res.data:
            target_notebook = res.data[0]

    original_folder_name = None
    if target_notebook:
        notebook_id = target_notebook.get("id")
        vector_store_id = target_notebook.get("vector_store_id") or vector_store_id
        project_name = project_name or target_notebook.get("name")
        project_source = target_notebook.get("project_source") or ""
        if project_source.startswith("blob/"):
            original_folder_name = project_source[len("blob/"):].strip()

    if not vector_store_id and project_name:
        vector_store_id = set_collection_name(project_name)

    if not vector_store_id:
        raise ValueError("Could not determine vector_store_id for retry.")

    project_name = project_name or vector_store_id

    logger.info(
        "[RETRY] User triggered manual retry for project '%s' (vector_store_id: '%s', notebook_id: '%s', original_folder: '%s')",
        project_name,
        vector_store_id,
        notebook_id,
        original_folder_name,
    )

    # 1. Update notebook status to processing
    if notebook_id:
        supabase.client.table("notebooks").update({
            "status": "processing",
            "last_activity": datetime.now(timezone.utc).isoformat(),
        }).eq("id", notebook_id).execute()

    # 2. Update existing pipeline job or create a new one
    job_id = None
    existing_jobs = supabase.client.table("pipeline_jobs").select("id, project_name").eq("vector_store_id", vector_store_id).limit(1).execute()
    if existing_jobs.data:
        job_id = existing_jobs.data[0]["id"]
        db_project_name = existing_jobs.data[0].get("project_name")
        if db_project_name and not original_folder_name:
            original_folder_name = db_project_name
        supabase.update_pipeline_job(
            job_id,
            status="triggered",
            completed_prompts=0,
            report_summary=None,
        )
    else:
        new_job = supabase.create_pipeline_job(
            project_name=original_folder_name or project_name,
            total_chunks=0,
            file_count=1,
            vector_store_id=vector_store_id,
            notebook_id=notebook_id,
            user_email=user_email,
        )
        job_id = new_job.get("id") if new_job else None

    # 3. Check if vectors exist in Supabase documents table
    has_vectors = supabase.has_vectors_for_collection(vector_store_id)

    if has_vectors:
        logger.info("[RETRY] Found existing vectors for '%s'. Regenerating report directly.", vector_store_id)
        regenerate_report(
            project_name=project_name,
            vector_store_id=vector_store_id,
            model_name=model_name,
            notebook_id=notebook_id,
            user_email=user_email,
        )
    else:
        logger.info("[RETRY] No vectors found for '%s'. Searching files in Azure Blob Storage...", vector_store_id)
        azure_service = AzureBlobService()
        
        # Search candidates in Azure Blob Storage
        blobs = []
        search_keys = [k for k in [original_folder_name, project_name] if k]
        for key in search_keys:
            blobs = azure_service.list_blobs(name_starts_with=f"{key}/")
            if blobs:
                break
            if "]" in key:
                clean_name = key.split("]", 1)[1].strip()
                blobs = azure_service.list_blobs(name_starts_with=f"{clean_name}/")
                if blobs:
                    break

        if not blobs:
            # Match any folder where sanitized name matches vector_store_id
            all_folders = azure_service.list_project_folders()
            for f_name, f_blobs in all_folders.items():
                if (
                    sanitize_collection_name(f_name) == vector_store_id
                    or set_collection_name(f_name) == vector_store_id
                    or (original_folder_name and f_name.lower() == original_folder_name.lower())
                ):
                    blobs = f_blobs
                    break

        if blobs:
            logger.info("[RETRY] Found %d file(s) in Azure Blob Storage for '%s'. Starting re-ingestion.", len(blobs), project_name)
            target_dir = TEMP_UPLOAD_ROOT / f"retry_{vector_store_id}"
            target_dir.mkdir(parents=True, exist_ok=True)
            try:
                for b_name in blobs:
                    fname = Path(b_name).name
                    azure_service.download_blob(b_name, target_dir / fname)
                process_project_folder(
                    target_dir,
                    model_name=model_name,
                    user_email=user_email,
                )
            finally:
                if target_dir.exists():
                    shutil.rmtree(target_dir, ignore_errors=True)
        else:
            logger.warning("[RETRY] No source files found in Azure Blob Storage for project '%s'.", project_name)
            if job_id:
                supabase.update_pipeline_job(
                    job_id,
                    status="failed",
                    report_summary="No se encontraron archivos en Azure Blob Storage para reintentar este proyecto.",
                )
            if notebook_id:
                supabase.client.table("notebooks").update({"status": "error"}).eq("id", notebook_id).execute()
            raise RuntimeError(f"No source files found in storage for project '{project_name}'.")

    return {
        "status": "accepted",
        "job_id": job_id,
        "vector_store_id": vector_store_id,
        "notebook_id": notebook_id,
        "project_name": project_name,
    }
