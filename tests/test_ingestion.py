
import fitz
import pytest

from medical_rag.ingestion import (
    PDFIngestionError,
    clean_page_text,
    extract_pdf_pages,
)


def create_test_pdf() -> bytes:
    """Create a small two-page PDF in memory."""
    document = fitz.open()

    page1 = document.new_page()
    page1.insert_text((72, 72), "Medical guidelines for testing.")

    page2 = document.new_page()
    page2.insert_text((72, 72), "Dosage information on page two.")

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes


def test_extracts_text_from_all_pages():
    pdf_bytes = create_test_pdf()

    pages = extract_pdf_pages(
        pdf_bytes,
        document_name="test_medical.pdf",
    )

    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert pages[1]["page_number"] == 2

    assert "Medical guidelines" in pages[0]["text"]
    assert "Dosage information" in pages[1]["text"]


def test_preserves_document_name():
    pages = extract_pdf_pages(
        create_test_pdf(),
        document_name="clinical_guide.pdf",
    )

    assert all(
        page["document_name"] == "clinical_guide.pdf"
        for page in pages
    )


def test_accepts_pdf_file_path(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(create_test_pdf())

    pages = extract_pdf_pages(pdf_path)

    assert len(pages) == 2
    assert pages[0]["document_name"] == "sample.pdf"


def test_rejects_empty_bytes():
    with pytest.raises(PDFIngestionError, match="empty"):
        extract_pdf_pages(b"")


def test_rejects_invalid_pdf():
    with pytest.raises(PDFIngestionError, match="Could not open PDF"):
        extract_pdf_pages(b"this is not a PDF")


def test_rejects_missing_file(tmp_path):
    missing_path = tmp_path / "missing.pdf"

    with pytest.raises(PDFIngestionError, match="not found"):
        extract_pdf_pages(missing_path)


def test_cleans_whitespace():
    text = "  Medical   guidelines.  \n\n\nDosage   information.  "

    cleaned = clean_page_text(text)

    assert cleaned == "Medical guidelines.\n\nDosage information."
