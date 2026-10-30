
import fitz
import numpy as np

from medical_rag.pipeline import MedicalRAGPipeline


class FakeEmbeddingModel:
    dimension = 3

    def embed_texts(self, texts, batch_size=32):
        vectors = []

        for text in texts:
            text = text.lower()

            if "hypertension" in text:
                vector = [1.0, 0.0, 0.0]
            elif "diabetes" in text:
                vector = [0.0, 1.0, 0.0]
            else:
                vector = [0.0, 0.0, 1.0]

            vectors.append(vector)

        return np.asarray(vectors, dtype=np.float32)


class FakeAnswerGenerator:
    def generate_answer(self, question, retrieved_chunks):
        if not retrieved_chunks:
            return {
                "answer": "Insufficient evidence.",
                "citations": [],
            }

        chunk = retrieved_chunks[0]

        return {
            "answer": f"Retrieved evidence: {chunk['text']} [S1]",
            "citations": [
                {
                    "source_id": "S1",
                    "chunk_id": chunk["chunk_id"],
                    "document_name": chunk["document_name"],
                    "page_number": chunk["page_number"],
                }
            ],
        }


def make_pdf(text):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def make_pipeline():
    return MedicalRAGPipeline(
        embedding_model=FakeEmbeddingModel(),
        answer_generator=FakeAnswerGenerator(),
        chunk_size=200,
        overlap=30,
    )


def test_ingests_pdf_and_creates_chunks():
    pipeline = make_pipeline()

    count = pipeline.ingest_pdf(
        make_pdf("Hypertension is elevated blood pressure."),
        document_name="guide.pdf",
    )

    assert count >= 1
    assert pipeline.document_count == 1
    assert pipeline.chunk_count >= 1


def test_asks_question_and_returns_citations():
    pipeline = make_pipeline()

    pipeline.ingest_pdf(
        make_pdf("Hypertension is elevated blood pressure."),
        document_name="guide.pdf",
    )

    result = pipeline.ask("What is hypertension?", top_k=1)

    assert "answer" in result["answer"]
    assert result["answer"]["citations"][0]["document_name"] == "guide.pdf"
    assert result["answer"]["citations"][0]["page_number"] == 1


def test_ingests_multiple_documents():
    pipeline = make_pipeline()

    pipeline.ingest_pdf(
        make_pdf("Hypertension is elevated blood pressure."),
        document_name="guide.pdf",
    )

    pipeline.ingest_pdf(
        make_pdf("Diabetes affects blood glucose."),
        document_name="textbook.pdf",
    )

    assert pipeline.document_count == 2
    assert pipeline.chunk_count == 2


def test_reupload_replaces_document():
    pipeline = make_pipeline()

    pipeline.ingest_pdf(
        make_pdf("Hypertension is elevated blood pressure."),
        document_name="guide.pdf",
    )

    pipeline.ingest_pdf(
        make_pdf("Diabetes affects blood glucose."),
        document_name="guide.pdf",
    )

    assert pipeline.document_count == 1
    assert pipeline.chunk_count == 1
    assert "Diabetes" in pipeline.chunks[0]["text"]


def test_empty_collection_returns_insufficient_evidence():
    pipeline = make_pipeline()

    result = pipeline.ask("What is hypertension?")

    assert result["answer"]["answer"] == "Insufficient evidence."
    assert result["answer"]["citations"] == []
    assert result["retrieved_chunks"] == []


def test_invalid_chunk_configuration_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        MedicalRAGPipeline(
            embedding_model=FakeEmbeddingModel(),
            answer_generator=FakeAnswerGenerator(),
            chunk_size=100,
            overlap=100,
        )
