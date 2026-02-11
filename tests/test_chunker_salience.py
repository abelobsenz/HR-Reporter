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


def test_short_non_salient_url_chunk_is_dropped() -> None:
    doc = RawDocument(
        doc_id="doc-002",
        source_id="url:https://example.com/nav",
        source="https://example.com/nav",
        source_type="url",
        text="Home\nAbout\nTeam\nCareers",
    )

    chunks, _ = chunk_documents([doc], max_chars=220)
    assert chunks == []


def test_short_non_salient_file_chunk_is_kept() -> None:
    doc = RawDocument(
        doc_id="doc-003",
        source_id="file:/tmp/handbook.md",
        source="/tmp/handbook.md",
        source_type="file",
        text="Home\nAbout\nTeam\nCareers",
    )

    chunks, _ = chunk_documents([doc], max_chars=220)
    assert chunks
    assert chunks[0].text == "Home\nAbout\nTeam\nCareers"


def test_short_non_salient_text_chunk_is_kept() -> None:
    doc = RawDocument(
        doc_id="doc-004",
        source_id="text:pasted",
        source="--text",
        source_type="text",
        text="Home\nAbout\nTeam\nCareers",
    )

    chunks, _ = chunk_documents([doc], max_chars=220)
    assert chunks
    assert chunks[0].text == "Home\nAbout\nTeam\nCareers"
