
from typing import TypedDict

import faiss
import numpy as np
from numpy.typing import NDArray

from medical_rag.chunking import ChunkRecord
from medical_rag.embeddings import LocalEmbeddingModel


class RetrievedChunk(ChunkRecord):
    """A chunk returned by similarity search."""

    score: float


class VectorRetriever:
    """Exact cosine-similarity retrieval using a FAISS inner-product index."""

    def __init__(self, embedding_model: LocalEmbeddingModel) -> None:
        self.embedding_model = embedding_model
        self.dimension = embedding_model.dimension

        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks: list[ChunkRecord] = []

    @property
    def size(self) -> int:
        """Number of chunks currently indexed."""
        return len(self.chunks)

    def build_index(self, chunks: list[ChunkRecord]) -> None:
        """Build the index from scratch using the supplied chunks."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []

        if not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]
        vectors = self.embedding_model.embed_texts(texts)

        if vectors.shape != (len(chunks), self.dimension):
            raise ValueError(
                "Embedding dimensions do not match the expected index shape."
            )

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain non-finite values.")

        self.index.add(vectors)
        self.chunks = [dict(chunk) for chunk in chunks]

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Return the top-k most similar chunks for a question.

        Results are ordered by descending inner-product similarity.
        Because the vectors are normalized, the score is cosine similarity.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.size == 0:
            return []

        query_vector = self.embedding_model.embed_texts([query])
        query_vector = np.ascontiguousarray(
            query_vector,
            dtype=np.float32,
        )

        if query_vector.shape != (1, self.dimension):
            raise ValueError(
                "Query embedding dimension does not match the index."
            )

        if not np.isfinite(query_vector).all():
            raise ValueError("Query embedding contains non-finite values.")

        k = min(top_k, self.size)
        scores, indices = self.index.search(query_vector, k)

        results: list[RetrievedChunk] = []

        for score, index_position in zip(scores[0], indices[0]):
            if index_position < 0:
                continue

            chunk = self.chunks[int(index_position)]

            results.append(
                {
                    **chunk,
                    "score": float(score),
                }
            )

        return results
