"""
Explicit Pydantic contracts for API requests, responses, and ingestion payloads.
Follows AI/ML Software Engineering Standard (Rule 5: Explicit Contracts).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Direct Upload Presign Contracts ──────────────────────────────────────────

class FilePresignRequestItem(BaseModel):
    """File metadata supplied by the frontend before uploading."""
    file_name: str = Field(..., description="Original name of the file (e.g. 'pliego_condiciones.pdf')")
    file_size_bytes: Optional[int] = Field(None, ge=0, description="Size of the file in bytes if known")
    content_type: Optional[str] = Field(None, description="MIME content type if known")


class PresignUploadRequest(BaseModel):
    """Request to generate pre-signed SAS upload URLs for a batch of files."""
    project_name: str = Field(..., min_length=1, description="Name of the project/notebook")
    files: List[FilePresignRequestItem] = Field(..., min_items=1, description="List of files to prepare for upload")


class FilePresignResponseItem(BaseModel):
    """Upload instructions for a single file."""
    file_name: str = Field(..., description="Original name of the file")
    blob_name: str = Field(..., description="Azure Blob Storage object name (e.g. 'project/file.pdf')")
    upload_sas_url: str = Field(..., description="Pre-signed Azure Blob SAS URL for direct PUT upload")
    blob_url: str = Field(..., description="Public/Canonical Azure Blob storage URL")


class PresignUploadResponse(BaseModel):
    """Response containing pre-signed upload URLs for all requested files."""
    project_name: str
    vector_store_id: str
    files: List[FilePresignResponseItem]


# ── Ingestion Trigger Contracts ──────────────────────────────────────────────

class IngestFileItem(BaseModel):
    """Metadata of a file that has been successfully uploaded to cloud storage."""
    file_name: str = Field(..., description="Original name of the file")
    blob_name: str = Field(..., description="Blob name in storage (e.g. 'project/file.pdf')")
    blob_url: str = Field(..., description="Accessible URL of the uploaded blob")
    file_size_bytes: Optional[int] = Field(None, ge=0, description="Size in bytes")


class IngestProjectRequest(BaseModel):
    """Request to trigger backend processing & vectorization of uploaded cloud files."""
    project_name: str = Field(..., min_length=1, description="Name of the project")
    files: List[IngestFileItem] = Field(..., min_items=1, description="Files already uploaded to cloud storage")
    model_name: Optional[str] = Field(None, description="Target LLM model name")
    user_email: Optional[str] = Field(None, description="User email for notification and sector tagging")


class IngestProjectResponse(BaseModel):
    """Immediate acceptance response for an ingestion request."""
    status: str = "accepted"
    message: str
    project_name: str
    vector_store_id: str
    files_count: int
