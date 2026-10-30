import pytest

from medical_rag.config import Settings


def test_default_settings(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    monkeypatch.delenv("CHUNK_OVERLAP", raising=False)
    monkeypatch.delenv("TOP_K", raising=False)

    settings = Settings.from_env()

    assert settings.gemini_api_key is None
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.embedding_model == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert settings.chunk_size == 1200
    assert settings.chunk_overlap == 200
    assert settings.top_k == 5


def test_custom_settings(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "custom-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "custom-embedding")
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    monkeypatch.setenv("TOP_K", "3")

    settings = Settings.from_env()

    assert settings.gemini_api_key == "test-key"
    assert settings.gemini_model == "custom-model"
    assert settings.embedding_model == "custom-embedding"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 100
    assert settings.top_k == 3


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("CHUNK_SIZE", "abc"),
        ("CHUNK_SIZE", "0"),
        ("CHUNK_OVERLAP", "-1"),
        ("TOP_K", "0"),
    ],
)
def test_invalid_integer_settings(monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValueError):
        Settings.from_env()


def test_overlap_must_be_smaller_than_chunk_size(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "200")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")

    with pytest.raises(ValueError, match="smaller than CHUNK_SIZE"):
        Settings.from_env()


def test_empty_gemini_model_is_rejected(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "   ")

    with pytest.raises(ValueError, match="GEMINI_MODEL"):
        Settings.from_env()