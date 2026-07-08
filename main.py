import logging
import os
import shutil
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from agent_module.rag_responder import RAGResponder
from core.pipeline import process_project_folder, regenerate_report
from core.utils import set_collection_name
from supabase_module.supabase_client import SupabaseModule

# --- Logging --------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api.main")

# --- FastAPI Initialization ----------------------------------------------------
app = FastAPI(title="AgentLicitaciones API", version="1.0", redirect_slashes=False)

# Setup CORS with configurations from config.py if present, fallback to "*"
allowed_origins = getattr(config, "ALLOWED_ORIGINS", ["*"])
allow_creds = "*" not in allowed_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_creds,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_UPLOAD_ROOT = Path("temp_uploads")

# Ensure temporary upload directory exists
TEMP_UPLOAD_ROOT.mkdir(exist_ok=True)

# --- Background Task Wrapper ---------------------------------------------------
def process_project_folder_and_clean(
    folder_path: Path,
    project_name: str,
    model_name: Optional[str] = None
):
    """
    Background worker that runs the vectorization and RAG report generation,
    then cleans up the temporary directory from disk.
    """
    logger.info("Starting background processing for project folder: %s", folder_path)
    try:
        process_project_folder(folder_path, model_name=model_name)
    except Exception as exc:
        logger.exception("Pipeline execution failed for project: %s", project_name)
    finally:
        # Clean up temporary folder
        try:
            if folder_path.exists():
                shutil.rmtree(folder_path)
                logger.info("Successfully cleaned up temporary upload directory: %s", folder_path)
        except Exception as cleanup_exc:
            logger.error("Failed to clean up temporary directory %s: %s", folder_path, cleanup_exc)

# --- Pydantic Request Models ----------------------------------------------------
class AgentRequest(BaseModel):
    event_type: Optional[str] = "chat"
    model_name: Optional[str] = None
    
    # Chat fields
    project_name: Optional[str] = None
    vector_store_id: Optional[str] = None
    notebook_id: Optional[str] = None
    question: Optional[str] = None
    message: Optional[str] = None

class GenerateReportRequest(BaseModel):
    vector_store_id: str
    project_name: str
    model_name: Optional[str] = None
    notebook_id: Optional[str] = None

# --- Routes ---------------------------------------------------------------------
@app.get("/api/health")
def health():
    """Simple health-check endpoint."""
    return {"status": "ok"}

@app.get("/api/agent")
def agent_get():
    """Metadata/status endpoint for GET requests."""
    return {"status": "AgentLicitaciones API", "version": "1.0"}

@app.post("/api/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    project_name: str = Form(...),
    model_name: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
):
    """
    Receives direct file uploads from the frontend under form-data.
    Creates a temporary local folder, writes the files to it, and triggers 
    the vectorization & report-generation pipeline in the background.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for upload.")

    logger.info("Received upload request for project '%s' with %d files.", project_name, len(files))

    # Compute vector store ID
    vector_store_id = set_collection_name(project_name)

    # Setup local workspace for this project's files
    project_dir = TEMP_UPLOAD_ROOT / project_name
    project_dir.mkdir(exist_ok=True)

    try:
        supabase_module = SupabaseModule()
        # Save each uploaded file to the temporary directory
        for upload_file in files:
            file_ext = Path(upload_file.filename).suffix.lower()
            if file_ext not in config.SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format '{file_ext}'. Supported: {list(config.SUPPORTED_EXTENSIONS.keys())}"
                )
            
            dest_path = project_dir / upload_file.filename
            with dest_path.open("wb") as buffer:
                shutil.copyfileobj(upload_file.file, buffer)
            logger.info("Saved file: %s", dest_path)
            
            # Upload to Supabase Storage
            object_name = f"{project_name}/{upload_file.filename}"
            supabase_module.upload_file_to_storage("project_files", str(dest_path), object_name)
    except HTTPException:
        # Re-raise HTTP exceptions (like unsupported file types)
        if project_dir.exists():
            shutil.rmtree(project_dir)
        raise
    except Exception as exc:
        logger.exception("Error saving uploaded files")
        if project_dir.exists():
            shutil.rmtree(project_dir)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process and save uploaded files: {str(exc)}"
        )

    # Trigger pipeline in background
    background_tasks.add_task(
        process_project_folder_and_clean,
        folder_path=project_dir,
        project_name=project_name,
        model_name=model_name,
    )

    return {
        "status": "accepted",
        "message": "Files successfully uploaded. Ingestion pipeline triggered in the background.",
        "project_name": project_name,
        "vector_store_id": vector_store_id,
        "files_count": len(files),
    }

@app.post("/api/agent")
def handle_agent_event(payload: AgentRequest):
    """
    Chat endpoint for direct interaction with the RAG system.
    """
    event_type = payload.event_type.lower() if payload.event_type else "chat"
    model_name = payload.model_name

    if event_type == "chat":
        project_name = payload.project_name or payload.vector_store_id or payload.notebook_id
        question = payload.question or payload.message
        if not project_name or not question:
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: project_name (or vector_store_id/notebook_id) and question (or message)."
            )
        try:
            logger.info("RAG Query received for project '%s': %s", project_name, question)
            responder = RAGResponder(model_name=model_name)
            answer_data = responder.respond_chat(project_name, question)
            return answer_data
        except Exception as exc:
            logger.exception("Chat responder failed")
            raise HTTPException(
                status_code=500,
                detail={"error": "Chat responder failed.", "details": str(exc)}
            )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Event type '{event_type}' is not supported on this server. Use 'chat'."
        )

@app.post("/api/generate-report")
async def generate_report_endpoint(
    payload: GenerateReportRequest,
    background_tasks: BackgroundTasks
):
    """
    Endpoint to trigger report regeneration from an existing vector store ID.
    This skips the vectorization phase.
    """
    logger.info(
        "Received report regeneration request for project '%s' (vector_store_id: '%s')",
        payload.project_name,
        payload.vector_store_id
    )
    
    background_tasks.add_task(
        regenerate_report,
        project_name=payload.project_name,
        vector_store_id=payload.vector_store_id,
        model_name=payload.model_name,
        notebook_id=payload.notebook_id,
    )

    return {
        "status": "accepted",
        "message": "Report regeneration triggered in the background.",
        "project_name": payload.project_name,
        "vector_store_id": payload.vector_store_id
    }
