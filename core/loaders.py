"""
Responsible for:
  1. Discovering supported files inside a project folder.
  2. Loading & chunking each file using high-performance C-accelerated domain parsers.
Follows AI/ML Software Engineering Standard (Rule 26: Data Loading, Rule 33: Memory Management).
"""

import gc
import logging
from pathlib import Path
from typing import Optional
import time

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

import config
from core.formatting import format_bytes
from core.parsers import PyMuPDFParser, LightweightDocxLoader, LightweightExcelLoader

logger = logging.getLogger("loaders")


_LOADER_FACTORY = {
    "pdf":  lambda path: PyMuPDFParser(path),
    "docx": lambda path: LightweightDocxLoader(path),
    "xlsx": lambda path: LightweightExcelLoader(path),
    "txt":  lambda path: TextLoader(path, encoding="utf-8"),
}


def check_files_present(folder_path: str | Path, max_retries: int = 5, retry_delay: float = 1.0) -> bool:
    folder = Path(folder_path)
    
    for attempt in range(1, max_retries + 1):
        found_files = list(
            f for f in folder.rglob("*")
            if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS
        )
        
        if found_files:
            logger.info(
                "Files verified  │  folder='%s'  │  found=%d files on attempt %d/%d",
                folder.name,
                len(found_files),
                attempt,
                max_retries,
            )
            return True
        
        if attempt < max_retries:
            wait_time = retry_delay * (2 ** (attempt - 1))
            logger.info(
                "No files found yet  │  folder='%s'  │  attempt %d/%d  │  retrying in %.1fs",
                folder.name,
                attempt,
                max_retries,
                wait_time,
            )
            time.sleep(wait_time)
    
    logger.warning(
        "File check FAILED  │  folder='%s'  │  no supported files found after %d attempts",
        folder.name,
        max_retries,
    )
    return False


def discover_files(folder_path: str | Path) -> list[Path]:
    folder = Path(folder_path)
    if not folder.is_dir():
        logger.warning("discover_files: '%s' is not a directory – skipping.", folder)
        return []

    found: list[Path] = []
    total_size = 0
    
    for file in folder.rglob("*"):
        if file.is_file() and file.suffix.lower() in config.SUPPORTED_EXTENSIONS:
            found.append(file)
            file_size = file.stat().st_size
            total_size += file_size
            logger.info("File discovered: '%s'  │  size=%s", file.name, format_bytes(file_size))

    found.sort()
    logger.info(
        "Discovered %d supported files in '%s'  │  total_size=%s",
        len(found),
        folder.name,
        format_bytes(total_size) if found else "0 B"
    )
    return found


def load_and_chunk_file(
    file_path: Path,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[Document]:
    extension = file_path.suffix.lower()
    loader_key = config.SUPPORTED_EXTENSIONS.get(extension)

    if loader_key is None:
        logger.warning("No loader registered for extension '%s' – skipping %s", extension, file_path.name)
        return []

    factory = _LOADER_FACTORY.get(loader_key)
    if factory is None:
        logger.error("Loader key '%s' has no factory – this is a bug.", loader_key)
        return []

    raw_documents: list[Document] = []
    try:
        loader = factory(str(file_path))
        raw_documents = loader.load()
    except Exception as exc:
        logger.error("Failed to load '%s': %s", file_path.name, exc)
        return []

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or config.CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(raw_documents)
    content_size = sum(len(chunk.page_content.encode("utf-8")) for chunk in chunks)
    
    logger.info(
        "Loaded & chunked '%s' → %d raw pages/sections → %d chunks  │  content_size=%s",
        file_path.name,
        len(raw_documents),
        len(chunks),
        format_bytes(content_size) if chunks else "0 B"
    )
    
    # Explicit garbage cleanup
    del raw_documents
    gc.collect()
    
    return chunks
