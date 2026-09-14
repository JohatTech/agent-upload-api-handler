"""
Document parser registry and factory.
Follows AI/ML Software Engineering Standard (Rule 23: Factory Pattern & Rule 24: Domain Parsers).
"""

from pathlib import Path
from typing import Type
from core.parsers.pdf_parser import PyMuPDFParser
from core.parsers.docx_parser import LightweightDocxLoader
from core.parsers.excel_parser import LightweightExcelLoader

__all__ = [
    "PyMuPDFParser",
    "LightweightDocxLoader",
    "LightweightExcelLoader",
    "get_parser_for_file",
]

_PARSER_REGISTRY = {
    ".pdf": PyMuPDFParser,
    ".docx": LightweightDocxLoader,
    ".doc": LightweightDocxLoader,
    ".xlsx": LightweightExcelLoader,
    ".xls": LightweightExcelLoader,
}


def get_parser_for_file(file_path: str | Path):
    """Returns the appropriate document parser instance for the given file extension."""
    path = Path(file_path)
    ext = path.suffix.lower()
    parser_cls = _PARSER_REGISTRY.get(ext)
    if not parser_cls:
        raise ValueError(f"Unsupported file extension '{ext}' for file: {path.name}")
    return parser_cls(path)
