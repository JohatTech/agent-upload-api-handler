"""
Compatibility bridge for legacy callers.
Re-exports parsers and formatting from domain modules.
Follows AI/ML Software Engineering Standard.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document as LCDocument

from core.formatting import format_bytes, sanitize_collection_name, set_collection_name
from core.parsers.pdf_parser import PyMuPDFParser
from core.parsers.docx_parser import LightweightDocxLoader
from core.parsers.excel_parser import LightweightExcelLoader
from core.parsers import get_parser_for_file

__all__ = [
    "format_bytes",
    "sanitize_collection_name",
    "set_collection_name",
    "PyMuPDFParser",
    "LightweightDocxLoader",
    "LightweightExcelLoader",
    "get_parser_for_file",
    "load_doc_input",
    "get_text_file",
    "write_report",
]


def load_doc_input(path, doc_type, chunk_size, chunk_overlap):
    if doc_type in ["xlsx", "xls"]:
        loader = LightweightExcelLoader(path)
    elif doc_type in ["docx", "doc"]:
        loader = LightweightDocxLoader(path)
    elif doc_type == "pdf":
        loader = PyMuPDFParser(path)
    elif doc_type == "txt":
        loader = TextLoader(path)
    else:
        raise ValueError(f"Unsupported doc_type: {doc_type}")

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(separators=["\n"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    return documents, chunks


def get_text_file(example_path):
    with open(example_path, encoding="utf8") as file:
        data = file.readlines()
    return " ".join(data)


def write_report(text, file_name):
    import docx
    try:
        doc = docx.Document()
        doc.add_paragraph(text)
        doc.save(file_name)
    except Exception as e:
        import logging
        logging.getLogger("utils").error("Error writing report: %s", e)
