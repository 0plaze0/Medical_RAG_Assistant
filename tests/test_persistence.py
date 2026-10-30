
import pytest

from medical_rag.persistence import (
    PersistenceError,
    load_knowledge_base,
    save_knowledge_base,
)


class FakeEmbeddingModel:
    dimension = 3

    def embed_texts(self, texts, batch_size=32):
        import numpy as np

        vectors = []
        for text in texts:
            vector = np.zeros(3, dtype=np.float32)
            if "heart" in text.lower():
                vector[0] = 1.0
            elif "lung" in text.lower():
                vector[1] = 1.0
            else:
                vector[2] = 1.0
            vectors.append(vector)

        return np.asarray(vectors, dtype=np.float32)


class FakeAnswerGenerator:
    def generate_answer(self, question, retrieved_chunks):
        return {"answer": "Test answer.", "citations": []}


@pytest.fixture
def pipeline():
    from medical_rag.pipeline import MedicalRAGPipeline

    return MedicalRAGPipeline(
        embedding_model=FakeEmbeddingModel(),
        answer_generator=FakeAnswerGenerator(),
    )


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "guide.pdf:1:0",
            "document_name": "guide.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 10,
            "text": "heart health",
        },
        {
            "chunk_id": "guide.pdf:2:0",
            "document_name": "guide.pdf",
            "page_number": 2,
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 10,
            "text": "lung health",
        },
    ]


def test_save_and_reload_knowledge_base(
    pipeline, sample_chunks, tmp_path
):
    pipeline.retriever.build_index(sample_chunks)
    pipeline.chunks = sample_chunks

    hashes = {"guide.pdf": "abc123"}

    save_knowledge_base(pipeline, hashes, tmp_path)

    assert (tmp_path / "index.faiss").is_file()
    assert (tmp_path / "metadata.json").is_file()

    restored = type(pipeline)(
        embedding_model=FakeEmbeddingModel(),
        answer_generator=FakeAnswerGenerator(),
    )

    loaded_hashes = load_knowledge_base(restored, tmp_path)

    assert loaded_hashes == hashes
    assert restored.chunk_count == 2
    assert restored.document_count == 1

    results = restored.retriever.search("heart", top_k=1)
    assert results[0]["chunk_id"] == "guide.pdf:1:0"


def test_missing_knowledge_base_returns_empty(pipeline, tmp_path):
    hashes = load_knowledge_base(pipeline, tmp_path)

    assert hashes == {}
    assert pipeline.chunk_count == 0


def test_incomplete_knowledge_base_is_rejected(pipeline, tmp_path):
    (tmp_path / "index.faiss").write_bytes(b"placeholder")

    with pytest.raises(PersistenceError, match="incomplete"):
        load_knowledge_base(pipeline, tmp_path)


def test_mismatched_dimension_is_rejected(
    pipeline, sample_chunks, tmp_path
):
    pipeline.retriever.build_index(sample_chunks)
    pipeline.chunks = sample_chunks

    save_knowledge_base(pipeline, {}, tmp_path)

    metadata_path = tmp_path / "metadata.json"
    import json

    metadata = json.loads(metadata_path.read_text())
    metadata["dimension"] = 999
    metadata_path.write_text(json.dumps(metadata))

    restored = type(pipeline)(
        embedding_model=FakeEmbeddingModel(),
        answer_generator=FakeAnswerGenerator(),
    )

    with pytest.raises(PersistenceError, match="dimensions"):
        load_knowledge_base(restored, tmp_path)
