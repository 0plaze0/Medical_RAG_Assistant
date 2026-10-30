
from typing import TypedDict


class PageRecord(TypedDict):
    document_name: str
    page_number: int
    text: str


class ChunkRecord(TypedDict):
    chunk_id: str
    document_name: str
    page_number: int
    chunk_index: int
    start_char: int
    end_char: int
    text: str


def chunk_pages(
    pages: list[PageRecord],
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[ChunkRecord]:
    """
    Split page records into overlapping text chunks.

    Chunk boundaries remain within individual pages so each chunk
    retains an unambiguous source page.

    Args:
        pages: Page records produced by PDF ingestion.
        chunk_size: Maximum approximate number of characters per chunk.
        overlap: Number of characters shared between adjacent chunks.

    Returns:
        Chunk records containing text and source metadata.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks: list[ChunkRecord] = []

    for page in pages:
        text = page["text"].strip()

        if not text:
            continue

        document_name = page["document_name"]
        page_number = page["page_number"]

        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Prefer ending at a word boundary when possible.
            if end < len(text):
                search_start = start + chunk_size // 2
                boundary = text.rfind(" ", search_start, end)

                if boundary > start:
                    end = boundary

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": (
                            f"{document_name}:"
                            f"{page_number}:{chunk_index}"
                        ),
                        "document_name": document_name,
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "start_char": start,
                        "end_char": end,
                        "text": chunk_text,
                    }
                )

                chunk_index += 1

            if end >= len(text):
                break

            # Move forward while retaining overlapping context.
            start = max(start + 1, end - overlap)

    return chunks
