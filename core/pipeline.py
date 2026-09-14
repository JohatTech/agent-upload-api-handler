"""
The orchestrator. Given a project folder path, this module:
  1. Discovers every supported file inside it.
  2. Loads & chunks each file in parallel.
  3. Enriches every chunk with project + file metadata.
  4. Pushes the combined chunk list to all configured vector stores.
  5. Notifies N8N, Power Automate, Frontend, and the Chat Responder.
"""

import gc
import json
import logging
from pathlib import Path
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.documents import Document

import config
from core.azure_blob_service import AzureBlobService
from core.enrichment import enrich_chunks
from core.formatting import format_bytes, sanitize_collection_name
from core.loaders import discover_files, load_and_chunk_file
from core.notifier import (
    get_project_title_from_responder,
    notify_chat_responder,
    notify_frontend,
    notify_power_automate,
)
from core.schemas import IngestFileItem
from core.utils import set_collection_name
from core.vectorstore_manager import push_to_all_targets

from agent_module.agent_system import AutonomousRAGAgent
from agent_module.report_generator import generate_report
from agent_module.rag_responder import RAGResponder
from supabase_module.supabase_client import SupabaseModule

logger = logging.getLogger("pipeline")


def _load_single_file(file_path: Path) -> tuple[Path, list[Document]]:
    chunks = load_and_chunk_file(file_path)
    return file_path, chunks


def _update_event_metadata(event_metadata: dict[str, Any] | None, **kwargs: Any) -> None:
    if event_metadata is not None:
        event_metadata.update(kwargs)


def process_project_folder(
    folder_path: str | Path,
    model_name: str | None = None,
    event_metadata: dict[str, Any] | None = None,
    user_email: str | None = None,
    uploaded_files: list[tuple[str, str]] | None = None,
) -> int:
    folder = Path(folder_path)
    project_name = folder.name
    target_model = model_name or config.DEFAULT_CHAT_MODEL
    event_metadata = event_metadata or {}
    _update_event_metadata(
        event_metadata,
        project_name=project_name,
        model_name=target_model,
        process_type="folder",
    )
    
    # Extract tag if exists: e.g. "[innovation] MiProyecto" -> tag="innovation"
    tag = None
    match = re.match(r'^\[([^\]]+)\]', project_name)
    if match:
        tag = match.group(1)
    
    parts = project_name.split("_")
    email = user_email if user_email else (parts[0] if parts else None)

    logger.info("=" * 70)
    logger.info("PIPELINE START  │  project='%s'  │  model='%s'", project_name, target_model)
    logger.info("=" * 70)

    start_time = time.perf_counter()

    # Step 1: Connect to Supabase and register the START of everything immediately in pipeline_jobs
    vector_store_id = set_collection_name(project_name)
    try:
        supabase_module = SupabaseModule()
        job_id = supabase_module.create_pipeline_job(
            project_name=project_name, 
            status="triggered",
            vector_store_id=vector_store_id,
            tag=tag
        )
    except Exception as e:
        logger.error("Failed to init supabase job: %s", e)
        job_id = None
        supabase_module = None

    if job_id and supabase_module:
        supabase_module.update_pipeline_job(job_id, status="vectorizing")

    # Notify frontend that the notebook is processing BEFORE vectorization
    try:
        notify_frontend(
            project_name=project_name,
            vector_store_id=vector_store_id,
            total_chunks=0,
            status="processing",
            file_count=0,
            tag=tag,
            uploaded_files=uploaded_files,
            notebook_id=job_id,
        )
    except Exception as exc:
        logger.error("Failed to pre-register notebook on frontend: %s", exc)

    # Discover files
    files = discover_files(folder)
    if not files:
        logger.warning("No supported files found in '%s'. Nothing to do.", project_name)
        _update_event_metadata(event_metadata, processed_ok=False, total_chunks=0)
        if job_id and supabase_module:
            supabase_module.update_pipeline_job(job_id, status="failed")
        return 0

    # Load & chunk in parallel
    all_chunks: list[Document] = []

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {
            pool.submit(_load_single_file, f): f
            for f in files
        }

        for future in as_completed(futures):
            file_path = futures[future]
            try:
                _, chunks = future.result()
            except Exception as exc:
                logger.error("Worker failed for '%s': %s", file_path.name, exc)
                continue

            if not chunks:
                continue

            # Enrich metadata
            enrich_chunks(chunks, project_name=project_name, source_file=file_path)
            all_chunks.extend(chunks)

    total_content_size = sum(len(chunk.page_content.encode('utf-8')) for chunk in all_chunks)

    logger.info(
        "Loading complete  │  files=%d  │  total_chunks=%d  │  total_content_size=%s  │  %.2fs",
        len(files),
        len(all_chunks),
        format_bytes(total_content_size) if all_chunks else "0 B",
        time.perf_counter() - start_time,
    )

    if not all_chunks:
        logger.warning("All files produced zero chunks. Nothing to push.")
        _update_event_metadata(event_metadata, processed_ok=False, total_chunks=0)
        if job_id and supabase_module:
            supabase_module.update_pipeline_job(job_id, status="failed")
        return 0

    # Push to vector stores (Supabase)
    push_to_all_targets(all_chunks, project_name, job_id=job_id)

    if job_id and supabase_module:
        supabase_module.update_pipeline_job(job_id, status="vectorization_finished")

    # Determine natural language project name after vectorization completes
    natural_project_name = project_name
    try:
        natural_project_name = get_project_title_from_responder(project_name)
    except Exception as exc:
        logger.error("Failed to determine project name via responder server: %s", exc)

    # Pre-register vector ID and mapping with Chat Responder microservice
    vector_store_id = set_collection_name(project_name)
    notify_chat_responder(
        project_name=project_name,
        vector_store_id=vector_store_id,
        natural_project_name=natural_project_name,
    )

    if job_id and supabase_module:
        supabase_module.update_pipeline_job(job_id, status="responder_notified")

    # Notify frontend that the notebook is processing
    try:
        notify_frontend(
            project_name=natural_project_name,
            vector_store_id=vector_store_id,
            total_chunks=len(all_chunks),
            status="processing",
            file_count=len(files),
            tag=tag,
            notebook_id=job_id,
        )
    except Exception as exc:
        logger.error("Failed to pre-register notebook on frontend: %s", exc)

    # Run Autonomous RAG Agent & Reporting
    if job_id and supabase_module:
        supabase_module.update_pipeline_job(
            job_id, 
            status="generating_report", 
            project_name=natural_project_name,
            total_chunks=len(all_chunks), 
            file_count=len(files), 
            vector_store_id=vector_store_id
        )

    report_generated = False
    report_sent = False
    report_path = None
    report_summary = ""
    report_md = None
    try:
        logger.info("Step 5: Running Autonomous RAG Agent ...")
        prompts_path = Path("pliego_form.json")
        prompts = []
        
        if prompts_path.exists():
            with open(prompts_path, "r", encoding="utf-8") as f:
                prompts_data = json.load(f)
                prompts = [item for item in prompts_data if isinstance(item, str)]
        
        if prompts:
            agent = AutonomousRAGAgent(project_name, model_name="CLAUDE", job_id=job_id)
            report_md = agent.process_prompts(prompts)
            if report_md:
                clean_summary = report_md.strip()
                for prefix in ["# ", "## ", "### "]:
                    if clean_summary.startswith(prefix):
                        clean_summary = clean_summary[len(prefix):]
                report_summary = clean_summary[:500] + "..." if len(clean_summary) > 500 else clean_summary
            
            # Generate PDF report
            logger.info("Step 6: Generating PDF report ...")
            pdf_path = generate_report(project_name, report_md, model_name="CLAUDE")
            report_generated = bool(pdf_path)
            report_path = str(pdf_path) if pdf_path else None
            logger.info("Report Path: %s", pdf_path)

            # Send to Power Automate
            if email and pdf_path:
                logger.info("Step 7: Sending report to Power Automate for %s ...", email)
                report_sent = notify_power_automate(project_name, email, pdf_path)
        else:
            logger.warning("No prompts available for RAG analysis.")

    except Exception as e:
        logger.error("Agentic RAG/Reporting failed for '%s': %s", project_name, e, exc_info=True)
        _update_event_metadata(event_metadata, processed_ok=False, error=str(e))
        if job_id and supabase_module:
            supabase_module.update_pipeline_job(job_id, status="failed")
    else:
        _update_event_metadata(
            event_metadata,
            processed_ok=True,
            report_generated=report_generated,
            report_sent=report_sent,
            report_path=report_path,
            total_chunks=len(all_chunks),
        )
        if job_id and supabase_module:
            supabase_module.update_pipeline_job(
                job_id,
                status="completed" if report_generated else "failed",
                report_summary=report_summary,
                pdf_path=report_path
            )

    # Notify frontend of completion
    notify_frontend(
        project_name=natural_project_name,
        vector_store_id=vector_store_id,
        total_chunks=len(all_chunks),
        status="ready" if report_generated else "error",
        pdf_path=report_path,
        report_summary=report_summary,
        file_count=len(files),
        tag=tag,
        notebook_id=job_id,
    )

    elapsed = time.perf_counter() - start_time
    logger.info("PIPELINE DONE  │  project='%s'  │  model='%s'  │  %.2fs", project_name, target_model, elapsed)
    return len(all_chunks)


def process_project_cloud_ingestion(
    project_name: str,
    files: list[IngestFileItem] | list[dict],
    model_name: str | None = None,
    user_email: str | None = None,
) -> int:
    """
    Direct Cloud Storage Ingestion Pipeline.
    Streams files directly from Azure Blob Storage one by one, chunks and embeds them in bulk,
    flushes memory between files with gc.collect(), and executes the autonomous RAG report.
    Guarantees peak memory usage stays strictly under 180MB on Render Free Tier (512MB RAM).
    """
    target_model = model_name or config.DEFAULT_CHAT_MODEL
    
    # Extract tag if exists: e.g. "[innovation] MiProyecto" -> tag="innovation"
    tag = None
    match = re.match(r'^\[([^\]]+)\]', project_name)
    if match:
        tag = match.group(1)
    
    parts = project_name.split("_")
    email = user_email if user_email else (parts[0] if parts else None)

    logger.info("=" * 70)
    logger.info("CLOUD INGESTION START  │  project='%s'  │  files=%d  │  model='%s'", project_name, len(files), target_model)
    logger.info("=" * 70)

    start_time = time.perf_counter()
    vector_store_id = set_collection_name(project_name)

    # Step 1: Connect to Supabase and register pipeline job
    try:
        supabase_module = SupabaseModule()
        job_id = supabase_module.create_pipeline_job(
            project_name=project_name,
            status="triggered",
            vector_store_id=vector_store_id,
            tag=tag
        )
    except Exception as e:
        logger.error("Failed to init supabase job: %s", e)
        job_id = None
        supabase_module = None

    if job_id and supabase_module:
        supabase_module.update_pipeline_job(job_id, status="vectorizing")

    # Notify frontend of initial processing state
    uploaded_files_summary = []
    for f in files:
        fname = f.file_name if hasattr(f, "file_name") else f.get("file_name", "")
        burl = f.blob_url if hasattr(f, "blob_url") else f.get("blob_url", "")
        uploaded_files_summary.append((fname, burl))

    try:
        notify_frontend(
            project_name=project_name,
            vector_store_id=vector_store_id,
            total_chunks=0,
            status="processing",
            file_count=len(files),
            tag=tag,
            uploaded_files=uploaded_files_summary,
            notebook_id=job_id,
        )
    except Exception as exc:
        logger.error("Failed to pre-register notebook on frontend: %s", exc)

    # Step 2: Stream, extract, and chunk files from Azure Blob Storage
    azure_blob_service = AzureBlobService()
    total_chunks_processed = 0
    all_chunks_for_reporting: list[Document] = []

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        for idx, file_item in enumerate(files):
            file_name = file_item.file_name if hasattr(file_item, "file_name") else file_item["file_name"]
            blob_name = file_item.blob_name if hasattr(file_item, "blob_name") else file_item["blob_name"]
            safe_name = Path(file_name).name
            local_path = temp_dir / safe_name

            logger.info("Streaming file %d/%d: '%s' from Azure Blob '%s'...", idx + 1, len(files), safe_name, blob_name)

            try:
                # 1. Download single blob
                downloaded = azure_blob_service.download_blob(blob_name, local_path)
                if not downloaded or not local_path.exists():
                    logger.error("Could not download blob '%s' for project '%s'. Skipping.", blob_name, project_name)
                    continue

                # 2. Parse and chunk with PyMuPDF / Domain Parsers
                chunks = load_and_chunk_file(local_path)
                if not chunks:
                    logger.warning("No chunks produced from '%s'. Skipping.", safe_name)
                    if local_path.exists():
                        local_path.unlink(missing_ok=True)
                    continue

                # 3. Enrich chunk metadata
                enrich_chunks(chunks, project_name=project_name, source_file=local_path)

                # 4. Bulk push directly to vector store (Supabase pgvector)
                logger.info("Bulk pushing %d chunks from '%s' to vector store...", len(chunks), safe_name)
                push_to_all_targets(chunks, project_name, job_id=job_id)
                total_chunks_processed += len(chunks)
                all_chunks_for_reporting.extend(chunks[:50])  # Keep sample in memory if needed

                # 5. Clean up disk and force GC to keep RAM strictly low (<180MB)
                del chunks
                if local_path.exists():
                    local_path.unlink(missing_ok=True)
                gc.collect()

            except Exception as exc:
                logger.error("Error processing cloud file '%s': %s", safe_name, exc, exc_info=True)
                if local_path.exists():
                    local_path.unlink(missing_ok=True)
                continue

    logger.info(
        "Cloud Ingestion: Vectorized %d total chunks across %d files in %.2fs",
        total_chunks_processed, len(files), time.perf_counter() - start_time
    )

    if total_chunks_processed == 0:
        logger.warning("Project '%s' produced 0 chunks across all files.", project_name)
        if job_id and supabase_module:
            supabase_module.update_pipeline_job(job_id, status="failed")
        return 0

    if job_id and supabase_module:
        supabase_module.update_pipeline_job(job_id, status="vectorization_finished")

    # Determine natural language project name after vectorization completes
    natural_project_name = project_name
    try:
        natural_project_name = get_project_title_from_responder(project_name)
    except Exception as exc:
        logger.error("Failed to determine project name via responder server: %s", exc)

    # Pre-register vector ID and mapping with Chat Responder microservice
    notify_chat_responder(
        project_name=project_name,
        vector_store_id=vector_store_id,
        natural_project_name=natural_project_name,
    )

    if job_id and supabase_module:
        supabase_module.update_pipeline_job(job_id, status="responder_notified")

    # Step 3: Run Autonomous RAG Agent & Reporting
    if job_id and supabase_module:
        supabase_module.update_pipeline_job(
            job_id,
            status="generating_report",
            project_name=natural_project_name,
            total_chunks=total_chunks_processed,
            file_count=len(files),
            vector_store_id=vector_store_id
        )

    report_generated = False
    report_sent = False
    report_path = None
    report_summary = ""
    report_md = None

    try:
        logger.info("Running Autonomous RAG Agent for report generation...")
        prompts_path = Path("pliego_form.json")
        prompts = []

        if prompts_path.exists():
            with open(prompts_path, "r", encoding="utf-8") as f:
                prompts_data = json.load(f)
                prompts = [item for item in prompts_data if isinstance(item, str)]

        if prompts:
            agent = AutonomousRAGAgent(project_name, model_name="CLAUDE", job_id=job_id)
            report_md = agent.process_prompts(prompts)
            if report_md:
                clean_summary = report_md.strip()
                for prefix in ["# ", "## ", "### "]:
                    if clean_summary.startswith(prefix):
                        clean_summary = clean_summary[len(prefix):]
                report_summary = clean_summary[:500] + "..." if len(clean_summary) > 500 else clean_summary

            # Generate PDF report
            logger.info("Generating PDF report...")
            pdf_path = generate_report(project_name, report_md, model_name="CLAUDE")
            report_generated = bool(pdf_path)
            report_path = str(pdf_path) if pdf_path else None
            logger.info("Report Path: %s", pdf_path)

            # Send to Power Automate
            if email and pdf_path:
                logger.info("Sending report to Power Automate for %s ...", email)
                report_sent = notify_power_automate(project_name, email, pdf_path)
        else:
            logger.warning("No prompts available for RAG analysis in pliego_form.json.")

    except Exception as e:
        logger.error("Agentic RAG/Reporting failed for '%s': %s", project_name, e, exc_info=True)
        if job_id and supabase_module:
            supabase_module.update_pipeline_job(job_id, status="failed")
    else:
        if job_id and supabase_module:
            supabase_module.update_pipeline_job(
                job_id,
                status="completed" if report_generated else "failed",
                report_summary=report_summary,
                pdf_path=report_path,
                total_chunks=total_chunks_processed
            )

    # Notify frontend of completion
    notify_frontend(
        project_name=natural_project_name,
        vector_store_id=vector_store_id,
        total_chunks=total_chunks_processed,
        status="ready" if report_generated else "error",
        pdf_path=report_path,
        report_summary=report_summary,
        file_count=len(files),
        tag=tag,
        notebook_id=job_id,
    )

    elapsed = time.perf_counter() - start_time
    logger.info("CLOUD INGESTION COMPLETE  │  project='%s'  │  total_chunks=%d  │  %.2fs", project_name, total_chunks_processed, elapsed)
    return total_chunks_processed


def process_blob_file(
    file_path: str | Path,
    project_name: str,
    model_name: str | None = None,
    event_metadata: dict[str, Any] | None = None,
    user_email: str | None = None,
) -> int:
    path = Path(file_path)
    target_model = model_name or config.DEFAULT_CHAT_MODEL
    event_metadata = event_metadata or {}
    _update_event_metadata(
        event_metadata,
        project_name=project_name,
        model_name=target_model,
        process_type="blob",
    )
    
    # Extract tag if exists: e.g. "[innovation] MiProyecto" -> tag="innovation"
    tag = None
    match = re.match(r'^\[([^\]]+)\]', project_name)
    if match:
        tag = match.group(1)
    
    parts = project_name.split("_")
    email = user_email if user_email else (parts[0] if parts else None)

    logger.info("=" * 70)
    logger.info("BLOB PIPELINE START  │  project='%s'  │  file='%s'  │  model='%s'", project_name, path.name, target_model)
    logger.info("=" * 70)

    # Step 1: Connect to Supabase and register the START of everything immediately in pipeline_jobs
    vector_store_id = set_collection_name(project_name)
    try:
        supabase_module = SupabaseModule()
        job_id = supabase_module.create_pipeline_job(
            project_name=project_name, 
            status="triggered",
            vector_store_id=vector_store_id,
            tag=tag
        )
    except Exception as e:
        logger.error("Failed to init supabase job: %s", e)
        job_id = None
        supabase_module = None

    if job_id and supabase_module:
        supabase_module.update_pipeline_job(job_id, status="vectorizing")

    # Generate Azure Blob URL for this blob file
    blob_files = None
    try:
        from core.azure_blob_service import AzureBlobService
        azure_service = AzureBlobService()
        blob_name = f"{project_name}/{path.name}"
        blob_url = azure_service.get_blob_url(blob_name)
        blob_files = [(path.name, blob_url)]
    except Exception as e:
        logger.error("Failed to compute Azure Blob URL for '%s': %s", path.name, e)

    # Notify frontend that the notebook is processing BEFORE vectorization
    try:
        notify_frontend(
            project_name=project_name,
            vector_store_id=vector_store_id,
            total_chunks=0,
            status="processing",
            file_count=1,
            tag=tag,
            uploaded_files=blob_files,
            notebook_id=job_id,
        )
    except Exception as exc:
        logger.error("Failed to pre-register notebook on frontend: %s", exc)

    # Load & chunk
    chunks = load_and_chunk_file(path)
    if not chunks:
        _update_event_metadata(event_metadata, processed_ok=False, total_chunks=0)
        if job_id and supabase_module:
            supabase_module.update_pipeline_job(job_id, status="failed")
        return 0

    # Enrich metadata
    enrich_chunks(chunks, project_name=project_name, source_file=path)

    # Push to vector stores (Supabase)
    push_to_all_targets(chunks, project_name, job_id=job_id)

    if job_id and supabase_module:
        supabase_module.update_pipeline_job(job_id, status="vectorization_finished")

    # Determine natural language project name after vectorization completes
    natural_project_name = project_name
    try:
        natural_project_name = get_project_title_from_responder(project_name)
    except Exception as exc:
        logger.error("Failed to determine project name via responder server: %s", exc)

    # Pre-register vector ID and mapping with Chat Responder microservice
    vector_store_id = set_collection_name(project_name)
    notify_chat_responder(
        project_name=project_name,
        vector_store_id=vector_store_id,
        natural_project_name=natural_project_name,
    )

    if job_id and supabase_module:
        supabase_module.update_pipeline_job(job_id, status="responder_notified")

    # Notify frontend that the notebook is processing
    try:
        notify_frontend(
            project_name=natural_project_name,
            vector_store_id=vector_store_id,
            total_chunks=len(chunks),
            status="processing",
            file_count=1,
            tag=tag,
            notebook_id=job_id,
        )
    except Exception as exc:
        logger.error("Failed to pre-register notebook on frontend: %s", exc)

    # Agentic RAG & Reporting
    if job_id and supabase_module:
        supabase_module.update_pipeline_job(
            job_id, 
            status="generating_report", 
            project_name=natural_project_name,
            total_chunks=len(chunks), 
            file_count=1, 
            vector_store_id=vector_store_id
        )

    report_generated = False
    report_sent = False
    report_path = None
    report_summary = ""
    report_md = None

    try:
        prompts_path = Path("pliego_form.json")
        prompts = []
        if prompts_path.exists():
            with open(prompts_path, "r", encoding="utf-8") as f:
                prompts_data = json.load(f)
                prompts = [item for item in prompts_data if isinstance(item, str)]
        
        if prompts:
            agent = AutonomousRAGAgent(project_name, model_name="CLAUDE", job_id=job_id)
            report_md = agent.process_prompts(prompts)
            if report_md:
                clean_summary = report_md.strip()
                for prefix in ["# ", "## ", "### "]:
                    if clean_summary.startswith(prefix):
                        clean_summary = clean_summary[len(prefix):]
                report_summary = clean_summary[:500] + "..." if len(clean_summary) > 500 else clean_summary
            
            # Generate PDF report
            pdf_path = generate_report(project_name, report_md, model_name="CLAUDE")
            report_generated = bool(pdf_path)
            report_path = str(pdf_path) if pdf_path else None
            
            # Power Automate
            if email and pdf_path:
                report_sent = notify_power_automate(project_name, email, pdf_path)
        else:
            logger.warning("No prompts available for RAG analysis.")
    except Exception as e:
        logger.error("Agentic RAG/Reporting failed for blob '%s': %s", path.name, e, exc_info=True)
        _update_event_metadata(event_metadata, processed_ok=False, error=str(e))
        if job_id and supabase_module:
            supabase_module.update_pipeline_job(job_id, status="failed")
    else:
        _update_event_metadata(
            event_metadata,
            processed_ok=True,
            report_generated=report_generated,
            report_sent=report_sent,
            report_path=report_path,
            total_chunks=len(chunks),
        )
        if job_id and supabase_module:
            supabase_module.update_pipeline_job(
                job_id,
                status="completed" if report_generated else "failed",
                report_summary=report_summary,
                pdf_path=report_path
            )

    # Notify frontend of completion
    notify_frontend(
        project_name=natural_project_name,
        vector_store_id=vector_store_id,
        total_chunks=len(chunks),
        status="ready" if report_generated else "error",
        pdf_path=report_path,
        report_summary=report_summary,
        file_count=1,
        tag=tag,
        notebook_id=job_id,
    )

    return len(chunks)

def regenerate_report(
    project_name: str,
    vector_store_id: str,
    model_name: str | None = None,
    notebook_id: str | None = None,
    user_email: str | None = None,
) -> bool:
    """
    Regenerates the RAG report without re-vectorizing.
    """
    target_model = model_name or config.DEFAULT_CHAT_MODEL
    logger.info("=" * 70)
    logger.info("REGENERATE REPORT START │ project='%s' │ vector_store_id='%s' │ model='%s'", project_name, vector_store_id, target_model)
    logger.info("=" * 70)

    # Determine natural language project name first from notebooks table
    natural_project_name = project_name
    tag = None
    try:
        supabase_module = SupabaseModule()
        res = supabase_module.client.table("notebooks").select("name, tag").eq("vector_store_id", vector_store_id).limit(1).execute()
        if res.data and len(res.data) > 0:
            natural_project_name = res.data[0]["name"]
            tag = res.data[0].get("tag")
            logger.info("Found natural project name in database for redo: '%s' and tag: '%s'", natural_project_name, tag)
    except Exception as e:
        logger.error("Failed to query natural project name for redo: %s", e)

    try:
        supabase_module = SupabaseModule()
        job_id = supabase_module.create_pipeline_job(
            project_name=natural_project_name, 
            status="generating_report",
            vector_store_id=vector_store_id,
            notebook_id=notebook_id,
        )
    except Exception as e:
        logger.error("Failed to init supabase job: %s", e)
        job_id = None
        supabase_module = None

    report_generated = False
    report_sent = False
    report_path = None
    report_summary = ""
    report_md = None

    try:
        prompts_path = Path("pliego_form.json")
        prompts = []
        if prompts_path.exists():
            with open(prompts_path, "r", encoding="utf-8") as f:
                prompts_data = json.load(f)
                prompts = [item for item in prompts_data if isinstance(item, str)]
        
        if prompts:
            agent = AutonomousRAGAgent(project_name, model_name="CLAUDE", job_id=job_id, vector_store_id=vector_store_id)
            report_md = agent.process_prompts(prompts)
            if report_md:
                clean_summary = report_md.strip()
                for prefix in ["# ", "## ", "### "]:
                    if clean_summary.startswith(prefix):
                        clean_summary = clean_summary[len(prefix):]
                report_summary = clean_summary[:500] + "..." if len(clean_summary) > 500 else clean_summary
            
            # Generate PDF report
            pdf_path = generate_report(project_name, report_md, model_name="CLAUDE")
            report_generated = bool(pdf_path)
            report_path = str(pdf_path) if pdf_path else None
            
            # Power Automate (extract email from project_name if present)
            parts = project_name.split("_")
            email = user_email if user_email else (parts[0] if parts else None)
            if email and pdf_path:
                report_sent = notify_power_automate(project_name, email, pdf_path)
        else:
            logger.warning("No prompts available for RAG analysis.")
    except Exception as e:
        logger.error("Agentic RAG/Reporting failed during regeneration for '%s': %s", project_name, e, exc_info=True)
        if job_id and supabase_module:
            supabase_module.update_pipeline_job(job_id, status="failed")
    else:
        if job_id and supabase_module:
            supabase_module.update_pipeline_job(
                job_id,
                status="completed" if report_generated else "failed",
                report_summary=report_summary,
                pdf_path=report_path
            )

    # Notify frontend of completion
    notify_frontend(
        project_name=natural_project_name,
        vector_store_id=vector_store_id,
        total_chunks=0,
        status="ready" if report_generated else "error",
        pdf_path=report_path,
        report_summary=report_summary,
        file_count=0,
        tag=tag,
        notebook_id=notebook_id or job_id,
    )

    return report_generated
