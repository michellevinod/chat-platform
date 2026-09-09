from pathlib import Path

from app.ingestion.extractors.pdf_extractor import PDFExtractor
from app.ingestion.normalizers.pdf_normalizer import PDFNormalizer
from app.services.memory_service import MemoryService


def test_pdf_normalizer_reduces_excessive_text_blocks():
    pdf_path = Path("storage/uploads/PDF_BenchmarkTester_63Pages.pdf")
    assert pdf_path.exists(), "Benchmark PDF fixture is missing"

    document = PDFExtractor().extract(pdf_path)
    raw_text_blocks = sum(
        1
        for page in document.pages
        for block in page.blocks
        if block.block_type.value == "text"
    )

    normalized = PDFNormalizer().normalize(document)
    normalized_text_blocks = sum(
        1
        for page in normalized.pages
        for block in page.blocks
        if block.block_type.value == "text"
    )

    assert normalized_text_blocks < raw_text_blocks
    assert normalized_text_blocks > 0


def test_memory_service_handles_generic_follow_up_reference():
    memory = MemoryService()
    session_id = "pdf-followup"

    memory.save_turn(
        session_id=session_id,
        query="Show me the table on page 12",
        response="### Table from example.pdf (Page 12)\n\n| Name | Value |",
        project="Demo Project",
        document="example.pdf",
    )

    effective_query, resolved_doc, resolved_proj = memory.resolve_query(
        session_id=session_id,
        query="Explain this",
    )

    assert resolved_doc == "example.pdf"
    assert resolved_proj == "Demo Project"
    assert "Explain this" in effective_query
    assert "Show me the table on page 12" in effective_query
