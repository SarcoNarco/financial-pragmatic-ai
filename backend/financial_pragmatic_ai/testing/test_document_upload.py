"""Focused upload-route tests that never initialize the inference model."""

import asyncio
import io
import unittest
from tempfile import SpooledTemporaryFile
from unittest.mock import patch

from docx import Document
from fastapi import HTTPException, UploadFile

from api import server


def make_upload(filename: str, content: bytes) -> UploadFile:
    file_handle = SpooledTemporaryFile()
    file_handle.write(content)
    file_handle.seek(0)
    return UploadFile(filename=filename, file=file_handle)


def make_pdf(text: str = "") -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    header = b"%PDF-1.4\n"
    body = b""
    offsets = [0]
    for number, value in enumerate(objects, start=1):
        offsets.append(len(header) + len(body))
        body += f"{number} 0 obj\n".encode() + value + b"\nendobj\n"
    xref_offset = len(header) + len(body)
    xref = b"xref\n0 6\n0000000000 65535 f \n" + b"".join(
        f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]
    )
    trailer = b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n" + str(xref_offset).encode()
    return header + body + xref + trailer + b"\n%%EOF\n"


def make_docx(paragraphs: list[str]) -> bytes:
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


class DocumentUploadTests(unittest.TestCase):
    def setUp(self):
        self.analysis_patch = patch.object(
            server,
            "_run_analysis",
            side_effect=lambda transcript: {"transcript": transcript},
        )
        self.mock_run_analysis = self.analysis_patch.start()

    def tearDown(self):
        self.analysis_patch.stop()

    def upload(self, filename: str, content: bytes):
        async def send_upload():
            upload = make_upload(filename, content)
            try:
                return await server.upload_transcript(upload)
            finally:
                await upload.close()

        return asyncio.run(send_upload())

    def assert_upload_error(self, filename: str, content: bytes, detail: str):
        with self.assertRaises(HTTPException) as raised:
            self.upload(filename, content)
        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.detail, detail)
        self.mock_run_analysis.assert_not_called()

    def test_txt_upload_uses_canonical_analysis_path(self):
        result = self.upload("transcript.txt", b"CEO: Revenue grew.")
        self.assertEqual(result["transcript"], "CEO: Revenue grew.")
        self.mock_run_analysis.assert_called_once_with("CEO: Revenue grew.")

    def test_pdf_upload_uses_canonical_analysis_path(self):
        result = self.upload("transcript.pdf", make_pdf("CEO: Revenue grew."))
        self.assertIn("CEO: Revenue grew.", result["transcript"])
        self.mock_run_analysis.assert_called_once_with("CEO: Revenue grew.")

    def test_docx_upload_uses_canonical_analysis_path(self):
        result = self.upload(
            "transcript.docx",
            make_docx(["CEO: Revenue grew.", "CFO: Margins improved."]),
        )
        self.assertEqual(
            result["transcript"],
            "CEO: Revenue grew.\n\nCFO: Margins improved.",
        )
        self.mock_run_analysis.assert_called_once_with(
            "CEO: Revenue grew.\n\nCFO: Margins improved."
        )

    def test_unsupported_extension_is_rejected(self):
        self.assert_upload_error(
            "transcript.rtf",
            b"not supported",
            "Unsupported file type. Upload a .txt, .pdf, or .docx transcript.",
        )

    def test_legacy_doc_is_rejected(self):
        self.assert_upload_error(
            "transcript.doc",
            b"legacy word",
            "Legacy Microsoft Word (.doc) files are not supported. Please save the document as .docx and try again.",
        )

    def test_empty_docx_is_rejected(self):
        self.assert_upload_error(
            "empty.docx",
            make_docx([]),
            "No extractable text found in Word document.",
        )

    def test_empty_pdf_is_rejected(self):
        self.assert_upload_error(
            "empty.pdf",
            make_pdf(),
            "No extractable text found in PDF document.",
        )

    def test_corrupted_documents_are_rejected_without_parser_details(self):
        self.assert_upload_error(
            "broken.pdf",
            b"not a PDF",
            "Could not read PDF document. Please upload a valid PDF file.",
        )
        self.assert_upload_error(
            "broken.docx",
            b"not a Word document",
            "Could not read Word document. Please upload a valid .docx file.",
        )

    def test_oversized_upload_uses_existing_transcript_limit(self):
        self.analysis_patch.stop()
        original_limit = server.MAX_FULL_TRANSCRIPT_CHARS
        server.MAX_FULL_TRANSCRIPT_CHARS = 10
        try:
            with self.assertRaises(HTTPException) as raised:
                self.upload("large.txt", b"CEO: Revenue grew beyond the configured limit.")
        finally:
            server.MAX_FULL_TRANSCRIPT_CHARS = original_limit
        self.assertEqual(raised.exception.status_code, 413)
        self.assertIn("absolute", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
