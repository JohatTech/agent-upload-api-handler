"""
High-Performance, C-Accelerated PDF Parser utilizing PyMuPDF (fitz).
Provides streaming page-by-page extraction with low memory overhead (<20MB RAM).
Follows AI/ML Software Engineering Standard (Rule 26: Data Loading, Rule 33: Memory Management).
"""

import gc
import logging
from pathlib import Path
from typing import Generator, List, Optional
import pymupdf as fitz  # PyMuPDF C-extension
from langchain_core.documents import Document

from core.exceptions import DocumentParsingError

logger = logging.getLogger("pdf_parser")


class PyMuPDFParser:
    """
    Streaming PDF document parser backed by the C-based PyMuPDF engine.
    Extracts text page-by-page without materializing large uncompressed memory trees.
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")

    def load_stream(self) -> Generator[Document, None, None]:
        """
        Yields LangChain Document instances page by page.
        Keeps RAM consumption minimal on low-memory containers (Render Free 512MB).
        """
        doc = None
        try:
            doc = fitz.open(self.file_path)
            total_pages = len(doc)
            logger.info("PyMuPDF: Opened '%s' (%d pages)", self.file_path.name, total_pages)

            for page_index in range(total_pages):
                page = doc.load_page(page_index)
                text = page.get_text("text") or ""
                
                # Only yield pages with content
                clean_text = text.strip()
                if clean_text:
                    metadata = {
                        "source": str(self.file_path),
                        "file_name": self.file_path.name,
                        "page": page_index + 1,
                        "total_pages": total_pages,
                    }
                    yield Document(page_content=clean_text, metadata=metadata)
                    
        except Exception as exc:
            logger.error("Failed to parse PDF '%s' with PyMuPDF: %s", self.file_path.name, exc)
            raise DocumentParsingError(f"Error parsing PDF '{self.file_path.name}': {str(exc)}") from exc
        finally:
            if doc is not None:
                doc.close()
            gc.collect()

    def load(self) -> List[Document]:
        """Loads and returns all pages as a list of Documents."""
        return list(self.load_stream())

    def split_pdf_by_half(self, output_dir: Optional[Path] = None) -> List[Path]:
        """
        Splits this PDF into two equal page halves losslessly.
        Returns a list of two output file paths: [part1_path, part2_path].
        """
        return self.split_file_by_half(self.file_path, output_dir)

    @classmethod
    def split_file_by_half(cls, input_path: str | Path, output_dir: Optional[Path] = None) -> List[Path]:
        """
        Class method to split any given PDF file into two halves.
        """
        in_path = Path(input_path)
        out_dir = output_dir or in_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(in_path)
        total_pages = len(doc)

        if total_pages <= 1:
            doc.close()
            return [in_path]

        half_point = total_pages // 2

        part1_path = out_dir / f"{in_path.stem}_part1{in_path.suffix}"
        part2_path = out_dir / f"{in_path.stem}_part2{in_path.suffix}"

        # Part 1: Pages 0 to half_point - 1
        doc1 = fitz.open()
        doc1.insert_pdf(doc, from_page=0, to_page=half_point - 1)
        doc1.save(part1_path)
        doc1.close()

        # Part 2: Pages half_point to total_pages - 1
        doc2 = fitz.open()
        doc2.insert_pdf(doc, from_page=half_point, to_page=total_pages - 1)
        doc2.save(part2_path)
        doc2.close()

        doc.close()
        logger.info(
            "Split PDF '%s' into 2 halves: '%s' (%d pages) and '%s' (%d pages)",
            in_path.name, part1_path.name, half_point, part2_path.name, total_pages - half_point
        )
        return [part1_path, part2_path]
