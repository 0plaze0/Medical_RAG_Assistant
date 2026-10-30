
import numpy as np
import pytest

from medical_rag.retrieval import VectorRetriever


class FakeEmbeddingModel:
    """Deterministic embeddings for retrieval tests."""

    dimension = 3

    def embed_texts(self, texts, batch_size=32):
        vectors = []

        for text in texts:
            normalized_text = text.lower()

            if "hypertension" in normalized_text:
                vector = [1.0, 0.0, 0.0]
            elif "diabetes" in normalized_text:
                vector = [0.0, 1.0, 0.0]
            else:
                vector = [0.0, 0.0, 1.0]

            vectors.append(vector)

        return np.asarray(vectors, dtype=np.float32)


def make_chunk(chunk_id, document_name, page_number, text):
    return {
        "chunk_id": chunk_id,
        "document_name": document_name,
        "page_number": page_number,
        "chunk_index": 0,
        "start_char": 0,
        "end_char": len(text),
        "text": text,
    }


@pytest.fixture
def retriever():
    model = FakeEmbeddingModel()
    retriever = VectorRetriever(model)

    chunks = [
        make_chunk(
            "guide:1:0",
            "guide.pdf",
            1,
            "Hypertension increases cardiovascular risk.",
        ),
        make_chunk(
            "guide:2:0",
            "guide.pdf",
            2,
            "Diabetes affects blood glucose regulation.",
        ),
        make_chunk(
            "textbook:3:0",
            "textbook.pdf",
            3,
            "Hypertension requires appropriate evaluation.",
        ),
    ]

    retriever.build_index(chunks)
    return retriever


def test_builds_index(retriever):
    assert retriever.size == 3


def test_returns_results_in_similarity_order(retriever):
    results = retriever.search("hypertension", top_k=2)

    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]
    assert all("hypertension" in result["text"].lower()
               for result in results)


def test_preserves_source_metadata(retriever):
    results = retriever.search("hypertension", top_k=1)

    assert results[0]["document_name"] == "guide.pdf"
    assert results[0]["page_number"] == 1
    assert results[0]["chunk_id"] == "guide:1:0"


def test_limits_results_to_available_chunks(retriever):
    results = retriever.search("hypertension", top_k=10)

    assert len(results) == 3


def test_empty_index_returns_no_results():
    retriever = VectorRetriever(FakeEmbeddingModel())

    assert retriever.search("hypertension") == []


def test_rejects_empty_query(retriever):
    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.search("   ")


def test_rejects_invalid_top_k(retriever):
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("hypertension", top_k=0)


def test_can_rebuild_index(retriever):
    new_chunks = [
        make_chunk(
            "new:1:0",
            "new.pdf",
            1,
            "Hypertension information.",
        )
    ]

    retriever.build_index(new_chunks)

    assert retriever.size == 1

    results = retriever.search("hypertension", top_k=5)

    assert len(results) == 1
    assert results[0]["document_name"] == "new.pdf"
