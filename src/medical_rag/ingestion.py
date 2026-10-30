
from pathlib import Path
import re

import fitz  # PyMuPDF


class PDFIngestionError(Exception):
    """Raised when a PDF cannot be opened or processed."""


def clean_page_text(text: str) -> str:
    """Normalize whitespace without removing meaningful paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing whitespace from individual lines.
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse repeated spaces and tabs, but preserve newlines.
    text = re.sub(r"[^\S\n]+", " ", text)

    # Limit excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf_pages(
    source: str | Path | bytes,
    document_name: str | None = None,
) -> list[dict[str, str | int]]:
    """
    Extract text from a PDF, returning one record per page.

    Args:
        source: A filesystem path or PDF contents as bytes.
        document_name: Optional display name for the source document.

    Returns:
        A list of page records containing document_name, page_number,
        and text. Page numbers start at 1.

    Raises:
        PDFIngestionError: If the PDF cannot be opened or processed.
    """
    if isinstance(source, bytes):
        if not source:
            raise PDFIngestionError("The uploaded PDF is empty.")

        pdf_name = document_name or "uploaded.pdf"

        try:
            document = fitz.open(stream=source, filetype="pdf")
        except Exception as exc:
            raise PDFIngestionError(
                f"Could not open PDF '{pdf_name}'. "
                "Check that the file is a valid PDF."
            ) from exc

    else:
        path = Path(source)
        pdf_name = document_name or path.name

        if not path.is_file():
            raise PDFIngestionError(f"PDF file not found: {path}")

        try:
            document = fitz.open(path)
        except Exception as exc:
            raise PDFIngestionError(
                f"Could not open PDF '{pdf_name}'. "
                "Check that the file is a valid PDF."
            ) from exc

    try:
        if document.needs_pass:
            raise PDFIngestionError(
                f"PDF '{pdf_name}' is password-protected."
            )

        if document.page_count == 0:
            raise PDFIngestionError(
                f"PDF '{pdf_name}' contains no pages."
            )

        pages: list[dict[str, str | int]] = []

        for page_index, page in enumerate(document):
            try:
                raw_text = page.get_text("text")
            except Exception as exc:
                raise PDFIngestionError(
                    f"Failed to extract page {page_index + 1} "
                    f"from '{pdf_name}'."
                ) from exc

            pages.append(
                {
                    "document_name": pdf_name,
                    "page_number": page_index + 1,
                    "text": clean_page_text(raw_text),
                 }
            )

        return pages

    finally:
        document.close()
