"""
Domain-specific exception hierarchy for the ingestion pipeline.
Follows AI/ML Software Engineering Standard (Rule 12 & 13: Error Handling Strategy).
"""

from typing import Any, Optional


class AppError(Exception):
    """Base exception for all application-level errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class StorageError(AppError):
    """Raised when an operation with Cloud Object Storage (Azure Blob / Supabase Storage) fails."""
    pass


class StorageAuthenticationError(StorageError):
    """Raised when credentials or SAS token generation fails."""
    pass


class DocumentParsingError(AppError):
    """Raised when a document fails to parse or decode."""
    pass


class DocumentSizeLimitError(DocumentParsingError):
    """Raised when a document exceeds maximum allowed bounds."""
    pass


class VectorizationError(AppError):
    """Raised when embedding generation or vector database upsert fails."""
    pass


class ValidationError(AppError):
    """Raised when payload or parameter validation fails."""
    pass
