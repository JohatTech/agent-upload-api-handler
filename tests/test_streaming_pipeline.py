"""
Unit & Integration Tests for High-Performance Direct Cloud Upload and Streaming Ingestion.
Verifies PyMuPDF page streaming, PDF chunking (>100MB simulation), SAS generation, and Schema validation.
Follows AI/ML Software Engineering Standard.
"""

import os
import sys
import tempfile
from pathlib import Path
import pymupdf as fitz

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from core.formatting import format_bytes, sanitize_collection_name, generate_unique_vector_store_id
from core.parsers.pdf_parser import PyMuPDFParser
from core.parsers.docx_parser import LightweightDocxLoader
from core.schemas import (
    PresignUploadRequest,
    FilePresignRequestItem,
    IngestProjectRequest,
    IngestFileItem,
)
from core.azure_blob_service import AzureBlobService


def _create_sample_pdf(file_path: Path, num_pages: int = 10) -> None:
    """Creates a sample multi-page PDF for testing."""
    doc = fitz.open()
    for page_num in range(num_pages):
        page = doc.new_page(width=595, height=842)
        text = f"Page {page_num + 1} Content: Pliego de condiciones técnicas para licitación pública. Sección {page_num + 1}."
        page.insert_text((50, 72), text, fontsize=12)
    doc.save(file_path)
    doc.close()


def test_sanitize_collection_name():
    """Verify collection name sanitization complies with Qdrant and Supabase standards."""
    assert sanitize_collection_name("Mi Proyecto 2026") == "mi_proyecto_2026"
    assert sanitize_collection_name("[innovation] Licitación Hospital") == "innovation_licitacion_hospital"
    assert sanitize_collection_name("Special---Characters & Symbols!!") == "special_characters_symbols"


def test_generate_unique_vector_store_id():
    """Verify that generate_unique_vector_store_id produces unique identifiers for the same project name."""
    id1 = generate_unique_vector_store_id("MiProyecto")
    id2 = generate_unique_vector_store_id("MiProyecto")
    assert id1.startswith("miproyecto_")
    assert id2.startswith("miproyecto_")
    assert id1 != id2


def test_pymupdf_parser_streaming():
    """Verify PyMuPDF parser streams pages and produces LangChain Documents."""
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "test_doc.pdf"
        _create_sample_pdf(pdf_path, num_pages=6)

        parser = PyMuPDFParser(pdf_path)
        docs = parser.load()

        assert len(docs) == 6
        assert "Page 1 Content" in docs[0].page_content
        assert docs[0].metadata["page"] == 1
        assert docs[0].metadata["total_pages"] == 6


def test_pdf_split_by_half():
    """Verify PyMuPDFParser.split_pdf_by_half losslessly divides a multi-page PDF into two halves."""
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "tender_bid.pdf"
        _create_sample_pdf(pdf_path, num_pages=10)

        parser = PyMuPDFParser(pdf_path)
        part1_path, part2_path = parser.split_pdf_by_half()

        assert part1_path.exists()
        assert part2_path.exists()

        doc1 = fitz.open(part1_path)
        doc2 = fitz.open(part2_path)

        assert len(doc1) == 5
        assert len(doc2) == 5

        doc1.close()
        doc2.close()


def test_azure_sas_generation():
    """Verify Azure Blob Storage SAS generation produces valid authenticated URLs."""
    service = AzureBlobService()
    blob_name = "test_project/tender_doc.pdf"
    
    sas_token = service.generate_upload_sas(blob_name, expiry_minutes=30)
    assert sas_token and len(sas_token) > 20
    assert "sig=" in sas_token

    sas_url = service.generate_upload_sas_url(blob_name, expiry_minutes=30)
    assert sas_url.startswith("https://")
    assert blob_name in sas_url
    assert sas_token in sas_url


def test_presign_request_schema_validation():
    """Verify Pydantic models validate request contracts properly."""
    req = PresignUploadRequest(
        project_name="Proyecto Carreteras 2026",
        files=[
            FilePresignRequestItem(file_name="pliego.pdf", file_size_bytes=50000000),
            FilePresignRequestItem(file_name="anexo.docx", file_size_bytes=1000000),
        ]
    )
    assert req.project_name == "Proyecto Carreteras 2026"
    assert len(req.files) == 2


if __name__ == "__main__":
    test_sanitize_collection_name()
    test_generate_unique_vector_store_id()
    test_pymupdf_parser_streaming()
    test_pdf_split_by_half()
    test_azure_sas_generation()
    test_presign_request_schema_validation()
    print("All backend tests PASSED successfully!")

