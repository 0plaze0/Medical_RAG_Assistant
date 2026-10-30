
import json
from pathlib import Path
from typing import TYPE_CHECKING

import faiss

if TYPE_CHECKING:
    from medical_rag.pipeline import MedicalRAGPipeline


class PersistenceError(Exception):
    """Raised when saving or loading the knowledge base fails."""


def save_knowledge_base(
    pipeline: "MedicalRAGPipeline",
    document_hashes: dict[str, str],
    storage_dir: str | Path,
) -> None:
    """Save the FAISS index and associated metadata to disk."""
    directory = Path(storage_dir)
    directory.mkdir(parents=True, exist_ok=True)

    index_path = directory / "index.faiss"
    metadata_path = directory / "metadata.json"

    index_temp = directory / "index.faiss.tmp"
    metadata_temp = directory / "metadata.json.tmp"

    metadata = {
        "version": 1,
        "dimension": pipeline.retriever.dimension,
        "chunks": pipeline.chunks,
        "document_hashes": document_hashes,
    }

    try:
        # Write temporary files first to avoid replacing valid files
        # until the new files have been written successfully.
        faiss.write_index(pipeline.retriever.index, str(index_temp))

        with metadata_temp.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

        index_temp.replace(index_path)
        metadata_temp.replace(metadata_path)

    except Exception as exc:
        index_temp.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)
        raise PersistenceError(
            f"Could not save the knowledge base to {directory}."
        ) from exc


def load_knowledge_base(
    pipeline: "MedicalRAGPipeline",
    storage_dir: str | Path,
) -> dict[str, str]:
    """Restore a previously saved index and return document hashes."""
    directory = Path(storage_dir)
    index_path = directory / "index.faiss"
    metadata_path = directory / "metadata.json"

    # A missing knowledge base is a normal first-run condition.
    if not index_path.exists() and not metadata_path.exists():
        return {}

    if not index_path.is_file() or not metadata_path.is_file():
        raise PersistenceError(
            "The knowledge base is incomplete. Both index.faiss "
            "and metadata.json must exist."
        )

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        if metadata.get("version") != 1:
            raise PersistenceError(
                "Unsupported knowledge base metadata version."
            )

        if metadata.get("dimension") != pipeline.retriever.dimension:
            raise PersistenceError(
                "Saved embedding dimensions do not match the "
                "configured embedding model."
            )

        chunks = metadata.get("chunks")
        document_hashes = metadata.get("document_hashes")

        if not isinstance(chunks, list):
            raise PersistenceError("Invalid chunk metadata.")

        if not isinstance(document_hashes, dict):
            raise PersistenceError("Invalid document hash metadata.")

        if not all(
            isinstance(name, str) and isinstance(digest, str)
            for name, digest in document_hashes.items()
        ):
            raise PersistenceError("Invalid document hash entries.")

        # Only load indexes created by your own trusted application.
        index = faiss.read_index(str(index_path))

        if index.d != pipeline.retriever.dimension:
            raise PersistenceError(
                "Saved FAISS index has an unexpected dimension."
            )

        if index.ntotal != len(chunks):
            raise PersistenceError(
                "Index vector count does not match chunk metadata."
            )

        # Validate basic chunk structure before restoring the pipeline.
        required_fields = {
            "chunk_id",
            "document_name",
            "page_number",
            "chunk_index",
            "start_char",
            "end_char",
            "text",
        }

        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise PersistenceError("Invalid chunk record.")

            if not required_fields.issubset(chunk):
                raise PersistenceError("Chunk record is missing fields.")

            if not isinstance(chunk["text"], str):
                raise PersistenceError("Chunk text must be a string.")

        # Restore state only after the files have been validated.
        pipeline.retriever.index = index
        pipeline.retriever.chunks = [dict(chunk) for chunk in chunks]
        pipeline.chunks = [dict(chunk) for chunk in chunks]

        return dict(document_hashes)

    except PersistenceError:
        raise
    except Exception as exc:
        raise PersistenceError(
            f"Could not load the knowledge base from {directory}."
        ) from exc
