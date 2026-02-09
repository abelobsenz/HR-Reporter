from __future__ import annotations

from app.ingest.chunker import chunk_documents
from app.models import RawDocument


def test_short_compliance_hotline_chunk_is_kept() -> None:
    doc = RawDocument(
        doc_id="doc-001",
        source_id="url:https://example.com/policy",
        source="https://example.com/policy",
        source_type="url",
        text=(
            "Complaint hotline: report concerns at ethics@example.com. "
            "Retaliation is prohibited and reports are investigated promptly."
        ),
    )

    chunks, _ = chunk_documents([doc], max_chars=220)
    assert chunks
    assert any("Complaint hotline" in chunk.text for chunk in chunks)
