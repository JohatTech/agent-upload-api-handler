"""
Backend Startup Recovery Service.
Scans Supabase for stuck or interrupted pipeline jobs and notebooks,
discovers unregistered projects in Azure Blob Storage uploaded while the server was down,
and automatically resumes report processing and trend analytics backfilling.
"""

import logging
import threading
import shutil
from pathlib import Path
from typing import Dict, Any, List

import config
from core.utils import set_collection_name
from core.pipeline import process_project_folder, regenerate_report
from supabase_module.supabase_client import SupabaseModule

logger = logging.getLogger("recovery_service")

_RECOVERY_LOCK = threading.Lock()
_IS_RECOVERY_RUNNING = False

TEMP_UPLOAD_ROOT = Path("temp_uploads")


def scan_and_recover_stuck_jobs() -> Dict[str, Any]:
    """
    Main recovery routine executed on application startup.
    Scans for:
      1. Interrupted pipeline jobs & processing notebooks -> Resumes report generation or re-vectorizes.
      2. Unregistered project folders in Azure Blob Storage -> Downloads & ingests them.
      3. Completed reports missing trend analytics -> Backfills project_trend_analytics.
    """
    global _IS_RECOVERY_RUNNING

    if not _RECOVERY_LOCK.acquire(blocking=False):
        logger.info("[RECOVERY] Recovery scanner is already running in another thread. Skipping concurrent execution.")
        return {"status": "skipped", "reason": "already_running"}

    _IS_RECOVERY_RUNNING = True
    logger.info("=" * 70)
    logger.info("[RECOVERY] STARTING BACKEND RECOVERY SCANNER")
    logger.info("=" * 70)

    recovered_count = 0
    failed_count = 0
    discovered_count = 0

    try:
        supabase = SupabaseModule()

        # ----------------------------------------------------------------------
        # PHASE 1: RECOVER STUCK PIPELINE JOBS & NOTEBOOKS
        # ----------------------------------------------------------------------
        stuck_jobs = supabase.get_stuck_pipeline_jobs()
        stuck_notebooks = supabase.get_stuck_notebooks()

        logger.info(
            "[RECOVERY] Found %d stuck pipeline_job(s) and %d stuck notebook(s).",
            len(stuck_jobs),
            len(stuck_notebooks),
        )

        # Map project candidates to recover
        # Key: vector_store_id -> Dict of metadata
        candidates: Dict[str, Dict[str, Any]] = {}

        for job in stuck_jobs:
            p_name = job.get("project_name")
            if not p_name:
                continue
            vs_id = job.get("vector_store_id") or set_collection_name(p_name)
            candidates[vs_id] = {
                "project_name": p_name,
                "vector_store_id": vs_id,
                "notebook_id": job.get("notebook_id"),
                "job_id": job.get("id"),
                "status": job.get("status"),
            }

        for nb in stuck_notebooks:
            p_name = nb.get("name")
            vs_id = nb.get("vector_store_id") or (set_collection_name(p_name) if p_name else None)
            if not vs_id or not p_name:
                continue
            if vs_id not in candidates:
                candidates[vs_id] = {
                    "project_name": p_name,
                    "vector_store_id": vs_id,
                    "notebook_id": nb.get("id"),
                    "job_id": None,
                    "status": "processing",
                }

        for vs_id, info in candidates.items():
            project_name = info["project_name"]
            notebook_id = info["notebook_id"]
            job_id = info["job_id"]

            logger.info(
                "[RECOVERY] Processing recovery for project '%s' (vector_store_id: '%s')...",
                project_name,
                vs_id,
            )

            # Check if a report was already generated for this notebook/project
            already_has_report = False
            try:
                if notebook_id:
                    rep_res = supabase.client.table("reports").select("id").eq("notebook_id", notebook_id).execute()
                    if rep_res.data:
                        already_has_report = True
            except Exception as e:
                logger.error("[RECOVERY] Failed to check existing report: %s", e)

            if already_has_report:
                logger.info("[RECOVERY] Project '%s' already has a generated report. Marking job ready.", project_name)
                if job_id:
                    supabase.update_pipeline_job(job_id, status="completed")
                if notebook_id:
                    try:
                        supabase.client.table("notebooks").update({"status": "ready"}).eq("id", notebook_id).execute()
                    except Exception:
                        pass
                recovered_count += 1
                continue

            # Case A: Vector embeddings exist in Supabase 'documents' table
            if supabase.has_vectors_for_collection(vs_id):
                logger.info("[RECOVERY] Vector embeddings found for '%s'. Resuming report generation...", project_name)
                try:
                    ok = regenerate_report(
                        project_name=project_name,
                        vector_store_id=vs_id,
                        notebook_id=notebook_id,
                    )
                    if ok:
                        logger.info("[RECOVERY] Successfully recovered report for '%s'.", project_name)
                        recovered_count += 1
                    else:
                        logger.warning("[RECOVERY] Report regeneration returned False for '%s'.", project_name)
                        failed_count += 1
                except Exception as exc:
                    logger.exception("[RECOVERY] Exception resuming report for '%s': %s", project_name, exc)
                    failed_count += 1

            # Case B: Vectors missing, try to locate files locally or in Azure Blob Storage
            else:
                logger.info("[RECOVERY] No vector embeddings found for '%s'. Checking source files...", project_name)
                local_dir = TEMP_UPLOAD_ROOT / project_name
                file_found = False

                if local_dir.exists() and any(local_dir.iterdir()):
                    logger.info("[RECOVERY] Found local upload files in '%s'. Re-running pipeline...", local_dir)
                    try:
                        process_project_folder(local_dir)
                        file_found = True
                        recovered_count += 1
                    except Exception as exc:
                        logger.exception("[RECOVERY] Failed re-running pipeline for '%s': %s", project_name, exc)

                if not file_found:
                    # Check Azure Blob Storage for files
                    try:
                        from core.azure_blob_service import AzureBlobService
                        azure_service = AzureBlobService()
                        blob_files = azure_service.list_blobs(name_starts_with=f"{project_name}/")

                        if blob_files:
                            logger.info("[RECOVERY] Found %d file(s) in Azure Blob Storage for '%s'. Downloading...", len(blob_files), project_name)
                            local_dir.mkdir(parents=True, exist_ok=True)
                            for b_name in blob_files:
                                fname = Path(b_name).name
                                azure_service.download_blob(b_name, local_dir / fname)

                            process_project_folder(local_dir)
                            file_found = True
                            recovered_count += 1
                    except Exception as azure_err:
                        logger.error("[RECOVERY] Failed checking Azure Blob Storage for '%s': %s", project_name, azure_err)

                if not file_found:
                    logger.warning(
                        "[RECOVERY] Unable to recover '%s': No vectors or source files found. Setting status to failed/error.",
                        project_name,
                    )
                    if job_id:
                        supabase.update_pipeline_job(job_id, status="failed")
                    if notebook_id:
                        try:
                            supabase.client.table("notebooks").update({"status": "error"}).eq("id", notebook_id).execute()
                        except Exception:
                            pass
                    failed_count += 1

        # ----------------------------------------------------------------------
        # PHASE 2: DISCOVER UNREGISTERED AZURE BLOB STORAGE FOLDERS
        # ----------------------------------------------------------------------
        logger.info("[RECOVERY] Scanning Azure Blob Storage for unregistered project folders...")
        try:
            from core.azure_blob_service import AzureBlobService
            azure_service = AzureBlobService()
            blob_folders = azure_service.list_project_folders()

            if blob_folders:
                # Get all known notebooks & jobs to compare
                known_vs_ids = set()

                nb_res = supabase.client.table("notebooks").select("vector_store_id, name").execute()
                if nb_res.data:
                    for row in nb_res.data:
                        if row.get("vector_store_id"):
                            known_vs_ids.add(row["vector_store_id"])
                        if row.get("name"):
                            known_vs_ids.add(set_collection_name(row["name"]))

                pj_res = supabase.client.table("pipeline_jobs").select("vector_store_id, project_name").execute()
                if pj_res.data:
                    for row in pj_res.data:
                        if row.get("vector_store_id"):
                            known_vs_ids.add(row["vector_store_id"])
                        if row.get("project_name"):
                            known_vs_ids.add(set_collection_name(row["project_name"]))

                for proj_name, blob_list in blob_folders.items():
                    vs_id = set_collection_name(proj_name)
                    if vs_id not in known_vs_ids:
                        logger.info(
                            "[RECOVERY] Discovered unregistered project in Azure Blob Storage: '%s' (%d file(s)). Auto-ingesting...",
                            proj_name,
                            len(blob_list),
                        )
                        target_dir = TEMP_UPLOAD_ROOT / proj_name
                        target_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            for b_name in blob_list:
                                fname = Path(b_name).name
                                azure_service.download_blob(b_name, target_dir / fname)

                            process_project_folder(target_dir)
                            discovered_count += 1
                        except Exception as disc_err:
                            logger.exception(
                                "[RECOVERY] Failed auto-ingesting discovered Azure project '%s': %s",
                                proj_name,
                                disc_err,
                            )
                        finally:
                            if target_dir.exists():
                                shutil.rmtree(target_dir, ignore_errors=True)
        except Exception as azure_scan_err:
            logger.error("[RECOVERY] Azure Blob Storage folder scan failed: %s", azure_scan_err)

        # ----------------------------------------------------------------------
        # PHASE 3: BACKFILL MISSING TREND ANALYTICS
        # ----------------------------------------------------------------------
        logger.info("[RECOVERY] Triggering trend analytics backfill check...")
        try:
            from trend_module.service import process_all_pending_trends
            process_all_pending_trends()
        except Exception as trend_err:
            logger.error("[RECOVERY] Trend analytics backfill check failed: %s", trend_err)

    except Exception as overall_exc:
        logger.exception("[RECOVERY] Error during backend recovery scan: %s", overall_exc)
    finally:
        _IS_RECOVERY_RUNNING = False
        _RECOVERY_LOCK.release()
        logger.info("=" * 70)
        logger.info(
            "[RECOVERY] FINISHED SCAN  │  Recovered: %d  │  Failed: %d  │  Discovered & Ingested: %d",
            recovered_count,
            failed_count,
            discovered_count,
        )
        logger.info("=" * 70)

    return {
        "status": "completed",
        "recovered_count": recovered_count,
        "failed_count": failed_count,
        "discovered_count": discovered_count,
    }


def start_recovery_in_background() -> None:
    """
    Spawns the recovery scanner in a non-blocking daemon thread.
    """
    thread = threading.Thread(target=scan_and_recover_stuck_jobs, daemon=True)
    thread.start()
    logger.info("[RECOVERY] Recovery scanner thread started in background.")
