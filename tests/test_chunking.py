
import pytest

from medical_rag.chunking import chunk_pages


def test_short_page_produces_one_chunk():
    pages = [
        {
            "document_name": "guide.pdf",
            "page_number": 1,
            "text": "Diabetes is a metabolic condition.",
        }
    ]

    chunks = chunk_pages(pages, chunk_size=100, overlap=20)

    assert len(chunks) == 1
    assert chunks[0]["text"] == pages[0]["text"]
    assert chunks[0]["page_number"] == 1


def test_long_page_produces_multiple_chunks():
    text = " ".join(["medical"] * 400)

    pages = [
        {
            "document_name": "guide.pdf",
            "page_number": 2,
            "text": text,
        }
    ]

    chunks = chunk_pages(pages, chunk_size=200, overlap=40)

    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= 200 for chunk in chunks)


def test_preserves_source_metadata():
    pages = [
        {
            "document_name": "clinical_guide.pdf",
            "page_number": 4,
            "text": "A clinical passage about treatment.",
        }
    ]

    chunks = chunk_pages(pages)

    assert chunks[0]["document_name"] == "clinical_guide.pdf"
    assert chunks[0]["page_number"] == 4
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["chunk_id"] == "clinical_guide.pdf:4:0"


def test_skips_empty_pages():
    pages = [
        {
            "document_name": "guide.pdf",
            "page_number": 1,
            "text": "   ",
        }
    ]

    assert chunk_pages(pages) == []


def test_rejects_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=0)


def test_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=100, overlap=100)

    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=100, overlap=-1)


def test_adjacent_chunks_share_context():
    text = " ".join(f"word{i}" for i in range(100))

    pages = [
        {
            "document_name": "guide.pdf",
            "page_number": 1,
            "text": text,
        }
    ]

    chunks = chunk_pages(pages, chunk_size=100, overlap=20)

    assert len(chunks) > 1

    # The next chunk should begin before the previous chunk ends.
    assert chunks[1]["start_char"] < chunks[0]["end_char"]
