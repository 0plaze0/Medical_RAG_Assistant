import os
from dataclasses import dataclass

from dotenv import load_dotenv

from medical_rag.embeddings import DEFAULT_MODEL_NAME


def _read_positive_int(name: str, default: int) -> int:
    """Read a positive integer from an environment variable."""
    value = os.getenv(name, str(default))

    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer, but received {value!r}."
        ) from exc

    if result <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return result


@dataclass(frozen=True)
class Settings:
    """Application configuration loaded from environment variables."""

    gemini_api_key: str | None
    gemini_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int

    @classmethod
    def from_env(cls) -> "Settings":
        """Load application settings from the .env file and environment."""
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY", "").strip() or None

        gemini_model = os.getenv(
            "GEMINI_MODEL", "gemini-2.5-flash"
        ).strip()

        embedding_model = os.getenv(
            "EMBEDDING_MODEL", DEFAULT_MODEL_NAME
        ).strip()

        chunk_size = _read_positive_int("CHUNK_SIZE", 1200)
        chunk_overlap = _read_positive_int("CHUNK_OVERLAP", 200)
        top_k = _read_positive_int("TOP_K", 5)

        if not gemini_model:
            raise ValueError("GEMINI_MODEL must not be empty.")

        if not embedding_model:
            raise ValueError("EMBEDDING_MODEL must not be empty.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
            )

        return cls(
            gemini_api_key=api_key,
            gemini_model=gemini_model,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
        )