
from pathlib import Path

from medical_rag.chunking import ChunkRecord, chunk_pages
from medical_rag.embeddings import LocalEmbeddingModel
from medical_rag.generation import GeminiAnswerGenerator, GenerationResult
from medical_rag.ingestion import extract_pdf_pages
from medical_rag.retrieval import RetrievedChunk, VectorRetriever


class MedicalRAGPipeline:
    """Coordinate document ingestion, retrieval, and answer generation."""

    def __init__(
        self,
        embedding_model: LocalEmbeddingModel,
        answer_generator: GeminiAnswerGenerator,
        chunk_size: int = 1200,
        overlap: int = 200,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "overlap must be non-negative and smaller than chunk_size."
            )

        self.embedding_model = embedding_model
        self.answer_generator = answer_generator
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.retriever = VectorRetriever(embedding_model)
        self.chunks: list[ChunkRecord] = []

    def ingest_pdf(
        self,
        source: str | Path | bytes,
        document_name: str | None = None,
    ) -> int:
        """
        Ingest a PDF and add its chunks to the searchable collection.

        If a document with the same name already exists, replace it.

        Returns:
            Number of chunks created for this document.
        """
        pages = extract_pdf_pages(source, document_name=document_name)

        new_chunks = chunk_pages(
            pages,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

        # Build the candidate collection before changing the active one.
        # This prevents a failed re-index from replacing the current collection.
        retained_chunks = [
            chunk
            for chunk in self.chunks
            if chunk["document_name"] != pages[0]["document_name"]
        ] if pages else list(self.chunks)

        candidate_chunks = retained_chunks + new_chunks

        self.retriever.build_index(candidate_chunks)
        self.chunks = candidate_chunks

        return len(new_chunks)

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict[str, GenerationResult | list[RetrievedChunk]]:
        """
        Retrieve relevant chunks and generate an evidence-grounded answer.
        """
        retrieved_chunks = self.retriever.search(
            query=question,
            top_k=top_k,
        )

        result = self.answer_generator.generate_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        return {
            "answer": result,
            "retrieved_chunks": retrieved_chunks,
        }

    @property
    def document_count(self) -> int:
        """Number of distinct documents in the active collection."""
        return len({chunk["document_name"] for chunk in self.chunks})

    @property
    def chunk_count(self) -> int:
        """Number of chunks in the active collection."""
        return len(self.chunks)
