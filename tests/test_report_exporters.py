from __future__ import annotations

from app.report.exporters import markdown_to_docx_bytes, markdown_to_pdf_bytes


def test_markdown_to_docx_bytes_signature() -> None:
    payload = markdown_to_docx_bytes("# Title\n\n- one\n- two\n")
    # DOCX files are zip archives.
    assert payload[:2] == b"PK"
    assert len(payload) > 1000


def test_markdown_to_pdf_bytes_signature() -> None:
    payload = markdown_to_pdf_bytes("# Title\n\nSome paragraph text.\n")
    assert payload.startswith(b"%PDF-1.")
    assert payload.rstrip().endswith(b"%%EOF")
    assert len(payload) > 300
