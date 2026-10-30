
from types import SimpleNamespace

import pytest

from medical_rag.generation import (
    GeminiAnswerGenerator,
    GenerationError,
)


def make_chunk(
    chunk_id="guide:1:0",
    document_name="guide.pdf",
    page_number=1,
    text="The document discusses blood pressure.",
):
    return {
        "chunk_id": chunk_id,
        "document_name": document_name,
        "page_number": page_number,
        "chunk_index": 0,
        "start_char": 0,
        "end_char": len(text),
        "text": text,
    }


class FakeModels:
    def __init__(self, response_text="Blood pressure is discussed [S1]."):
        self.response_text = response_text
        self.last_call = None

    def generate_content(self, **kwargs):
        self.last_call = kwargs
        return SimpleNamespace(text=self.response_text)


class FakeClient:
    def __init__(self, response_text="Blood pressure is discussed [S1]."):
        self.models = FakeModels(response_text)


def test_generate_answer_returns_citation_metadata():
    client = FakeClient()
    generator = GeminiAnswerGenerator(
        model_name="test-model",
        client=client,
    )

    result = generator.generate_answer(
        "What does the document discuss?",
        [make_chunk()],
    )

    assert result["answer"] == "Blood pressure is discussed [S1]."
    assert result["citations"] == [
        {
            "source_id": "S1",
            "chunk_id": "guide:1:0",
            "document_name": "guide.pdf",
            "page_number": 1,
        }
    ]


def test_no_chunks_returns_without_calling_gemini():
    client = FakeClient()
    generator = GeminiAnswerGenerator(client=client)

    result = generator.generate_answer("What is hypertension?", [])

    assert result["citations"] == []
    assert "could not find relevant information" in result["answer"]
    assert client.models.last_call is None


def test_invalid_citation_is_removed():
    client = FakeClient("The answer is supported [S1], not [S99].")
    generator = GeminiAnswerGenerator(client=client)

    result = generator.generate_answer(
        "What is discussed?",
        [make_chunk()],
    )

    assert "[S1]" in result["answer"]
    assert "[S99]" not in result["answer"]
    assert [c["source_id"] for c in result["citations"]] == ["S1"]


def test_empty_question_is_rejected():
    generator = GeminiAnswerGenerator(client=FakeClient())

    with pytest.raises(ValueError):
        generator.generate_answer("   ", [make_chunk()])


def test_empty_model_response_raises_error():
    generator = GeminiAnswerGenerator(client=FakeClient(""))

    with pytest.raises(GenerationError):
        generator.generate_answer("What is discussed?", [make_chunk()])


def test_api_failure_raises_generation_error():
    class FailingModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("Simulated API failure")

    client = SimpleNamespace(models=FailingModels())
    generator = GeminiAnswerGenerator(client=client)

    with pytest.raises(GenerationError):
        generator.generate_answer("What is discussed?", [make_chunk()])


def test_empty_model_name_is_rejected():
    with pytest.raises(ValueError):
        GeminiAnswerGenerator(model_name=" ", client=FakeClient())
