
import numpy as np
import pytest

import medical_rag.embeddings as embeddings


class FakeSentenceTransformer:
    """Small deterministic substitute for unit tests."""

    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device
        self.dimension = 3

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(
        self,
        texts,
        batch_size,
        show_progress_bar,
        convert_to_numpy,
        normalize_embeddings,
    ):
        vectors = np.array(
            [
                [3.0, 4.0, 0.0]
                for _ in texts
            ],
            dtype=np.float32,
        )

        if normalize_embeddings:
            vectors = vectors / np.linalg.norm(
                vectors,
                axis=1,
                keepdims=True,
            )

        return vectors


@pytest.fixture
def embedder(monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )
    return embeddings.LocalEmbeddingModel()


def test_embedding_shape(embedder):
    vectors = embedder.embed_texts(
        ["First medical passage", "Second medical passage"]
    )

    assert vectors.shape == (2, 3)


def test_embeddings_are_float32(embedder):
    vectors = embedder.embed_texts(["Medical passage"])

    assert vectors.dtype == np.float32


def test_embeddings_are_normalized(embedder):
    vectors = embedder.embed_texts(["Medical passage"])

    norms = np.linalg.norm(vectors, axis=1)

    np.testing.assert_allclose(norms, [1.0])


def test_empty_input_returns_empty_matrix(embedder):
    vectors = embedder.embed_texts([])

    assert vectors.shape == (0, 3)
    assert vectors.dtype == np.float32


def test_rejects_empty_text(embedder):
    with pytest.raises(ValueError, match="must not be empty"):
        embedder.embed_texts(["   "])


def test_rejects_non_string_input(embedder):
    with pytest.raises(TypeError, match="must be a string"):
        embedder.embed_texts(["valid text", None])


def test_rejects_invalid_batch_size(embedder):
    with pytest.raises(ValueError, match="batch_size"):
        embedder.embed_texts(["Medical passage"], batch_size=0)


def test_supports_explicit_device(monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    model = embeddings.LocalEmbeddingModel(device="cpu")

    assert model.model.device == "cpu"
