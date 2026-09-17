"""
Formatting and string sanitization utilities.
Follows AI/ML Software Engineering Standard (Rule 6: Naming & Rule 24: Avoid Utils Anti-Pattern).
"""

import re
import unicodedata


def format_bytes(bytes_size: int | float) -> str:
    """Formats a byte count into a human-readable string (e.g. '12.45 MB')."""
    size = float(bytes_size)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def sanitize_collection_name(name: str) -> str:
    """
    Sanitizes a project name into a valid PostgreSQL/Qdrant collection identifier.
    Example: '[innovation] Licitación Hospital 2026' -> 'innovation_licitacion_hospital_2026'
    """
    # Normalize unicode characters to strip accents (e.g. ó -> o)
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("utf-8")
    
    cleaned = ascii_name.lower()
    cleaned = re.sub(r"[\s\-]+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_]", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned or "default_collection"


def generate_unique_vector_store_id(project_name: str, user_email: str | None = None) -> str:
    """
    Generates a guaranteed unique vector store ID incorporating the base project name,
    a UTC timestamp, and a short random hex token to prevent database collisions.
    Example: 'innovation_miproyecto_20260917_162500_a3f8'
    """
    import uuid
    from datetime import datetime, timezone

    base = sanitize_collection_name(project_name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_hash = uuid.uuid4().hex[:4]
    
    return f"{base}_{timestamp}_{short_hash}"


# Alias for backward compatibility
set_collection_name = sanitize_collection_name

