"""
Lightweight Excel document parser using openpyxl.
Follows AI/ML Software Engineering Standard (Rule 24: Domain Parsers).
"""

import gc
import logging
from pathlib import Path
from typing import Generator, List
import openpyxl
from langchain_core.documents import Document

from core.exceptions import DocumentParsingError

logger = logging.getLogger("excel_parser")


class LightweightExcelLoader:
    """Lightweight loader for Excel (.xlsx / .xls) files using openpyxl in read_only mode."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.file_path}")

    def load_stream(self) -> Generator[Document, None, None]:
        wb = None
        try:
            wb = openpyxl.load_workbook(self.file_path, read_only=True, data_only=True)
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                rows_text: List[str] = []
                for row in sheet.iter_rows(values_only=True):
                    row_values = [str(val).strip() for val in row if val is not None and str(val).strip()]
                    if row_values:
                        rows_text.append(" | ".join(row_values))

                sheet_content = "\n".join(rows_text)
                if sheet_content.strip():
                    yield Document(
                        page_content=sheet_content,
                        metadata={
                            "source": str(self.file_path),
                            "file_name": self.file_path.name,
                            "sheet_name": sheet_name,
                            "type": "xlsx",
                        }
                    )
        except Exception as exc:
            logger.error("Failed to parse Excel file '%s': %s", self.file_path.name, exc)
            raise DocumentParsingError(f"Error parsing Excel file '{self.file_path.name}': {str(exc)}") from exc
        finally:
            if wb is not None:
                wb.close()
            gc.collect()

    def load(self) -> List[Document]:
        return list(self.load_stream())
