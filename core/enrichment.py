"""
Enriches every document chunk with metadata that ties it back to its source project and file.
"""

import logging
from pathlib import Path
from langchain_core.documents import Document

logger = logging.getLogger("enrichment")


def enrich_chunks(
    chunks: list[Document],
    project_name: str,
    source_file: Path | str,
) -> list[Document]:
    source_path = Path(source_file)
    file_name = source_path.name
    file_type = source_path.suffix.lstrip(".").lower()

    for chunk in chunks:
        chunk.metadata["project_name"] = project_name
        chunk.metadata["source_file"] = file_name
        chunk.metadata["file_type"] = file_type

    logger.debug(
        "Enriched %d chunks with project='%s', file='%s'.",
        len(chunks),
        project_name,
        file_name,
    )
    return chunks
