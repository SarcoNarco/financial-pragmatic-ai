"""Convert supported uploaded transcript documents into plain text."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path


class DocumentExtractionError(ValueError):
    """A user-facing document validation or extraction failure."""


def extract_transcript_text(filename: str | None, content: bytes) -> str:
    """Extract transcript text from a TXT, PDF, or DOCX upload.

    The caller sends the returned text through the same analysis path used for
    pasted transcripts. This module deliberately does not perform OCR or any
    model-related work.
    """
    suffix = Path(filename or "").suffix.lower()

    if suffix == ".doc":
        raise DocumentExtractionError(
            "Legacy Microsoft Word (.doc) files are not supported. "
            "Please save the document as .docx and try again."
        )
    if suffix == ".txt":
        return _extract_txt(content)
    if suffix == ".pdf":
        return _extract_pdf(content)
    if suffix == ".docx":
        return _extract_docx(content)

    raise DocumentExtractionError(
        "Unsupported file type. Upload a .txt, .pdf, or .docx transcript."
    )


def _extract_txt(content: bytes) -> str:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentExtractionError("TXT file must be UTF-8 encoded.") from exc
    return _require_text(text, "No extractable text found in TXT file.")


def _extract_pdf(content: bytes) -> str:
    import pdfplumber

    try:
        with pdfplumber.open(BytesIO(content)) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
    except Exception as exc:
        raise DocumentExtractionError(
            "Could not read PDF document. Please upload a valid PDF file."
        ) from exc

    text = "\n".join(page_text for page_text in pages if page_text)
    return _require_text(text, "No extractable text found in PDF document.")


def _extract_docx(content: bytes) -> str:
    from docx import Document

    try:
        document = Document(BytesIO(content))
    except Exception as exc:
        raise DocumentExtractionError(
            "Could not read Word document. Please upload a valid .docx file."
        ) from exc

    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs]
    text = "\n\n".join(paragraph for paragraph in paragraphs if paragraph)
    return _require_text(text, "No extractable text found in Word document.")


def _require_text(text: str, empty_message: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise DocumentExtractionError(empty_message)
    return cleaned
