
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EmbeddingArray = NDArray[np.float32]


class LocalEmbeddingModel:
    """Generate normalized text embeddings using a local model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        """
        Load an embedding model.

        Args:
            model_name: Sentence Transformers model identifier.
            device: Optional device, such as "cpu" or "cuda".
        """
        self.model_name = model_name

        if device is None:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = SentenceTransformer(
                model_name,
                device=device,
            )

        self.dimension = self.model.get_sentence_embedding_dimension()

        if self.dimension is None:
            raise ValueError(
                f"Could not determine embedding dimension for {model_name}."
            )

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> EmbeddingArray:
        """
        Convert text strings into normalized float32 vectors.

        Returns:
            A NumPy array with shape (number_of_texts, dimension).
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Every input must be a string.")

        if any(not text.strip() for text in texts):
            raise ValueError("Input texts must not be empty or whitespace.")

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(vectors, dtype=np.float32)
