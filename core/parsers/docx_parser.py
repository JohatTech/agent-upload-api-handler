"""
Lightweight DOCX document parser.
Extracts paragraphs and tables efficiently from Microsoft Word documents.
Follows AI/ML Software Engineering Standard (Rule 24: Domain Parsers).
"""

import gc
import logging
from pathlib import Path
from typing import Generator, List
import docx
from langchain_core.documents import Document

from core.exceptions import DocumentParsingError

logger = logging.getLogger("docx_parser")


class LightweightDocxLoader:
    """Lightweight loader for Word (.docx) files."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"DOCX file not found: {self.file_path}")

    def load_stream(self) -> Generator[Document, None, None]:
        """Yields section/paragraph chunks from the Word document."""
        try:
            doc = docx.Document(str(self.file_path))
            text_chunks: List[str] = []

            # 1. Paragraphs
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:
                    text_chunks.append(text)

            # 2. Tables
            for table in doc.tables:
                table_lines: List[str] = []
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_cells:
                        table_lines.append(" | ".join(row_cells))
                if table_lines:
                    text_chunks.append("\n".join(table_lines))

            full_text = "\n\n".join(text_chunks)
            if full_text.strip():
                yield Document(
                    page_content=full_text,
                    metadata={
                        "source": str(self.file_path),
                        "file_name": self.file_path.name,
                        "type": "docx",
                    }
                )
        except Exception as exc:
            logger.error("Failed to parse DOCX '%s': %s", self.file_path.name, exc)
            raise DocumentParsingError(f"Error parsing DOCX '{self.file_path.name}': {str(exc)}") from exc
        finally:
            gc.collect()

    def load(self) -> List[Document]:
        return list(self.load_stream())
