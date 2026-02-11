from __future__ import annotations

from app.ingest.chunker import chunk_documents, get_last_chunk_stats
from app.models import RawDocument


def test_semantic_chunking_adds_heading_paths_and_list_context(monkeypatch) -> None:
    monkeypatch.setenv("HR_REPORT_CHUNK_MODE", "semantic")
    bullets = "\n".join(
        [
            f"- Policy item {idx}: Immediate family bereavement leave provides {idx} day(s) paid leave with manager approval."
            for idx in range(1, 15)
        ]
    )
    doc = RawDocument(
        doc_id="doc-001-benefits",
        source_id="file:benefits.md",
        source="benefits.md",
        source_type="file",
        title="Benefits and Perks",
        text=(
            "# Benefits & Perks\n"
            "## Time Off\n"
            "Bereavement leave policy:\n"
            f"{bullets}\n"
            "## Insurance\n"
            "Medical coverage rules:\n"
            "- Medical plan is available to full-time employees after 30 days.\n"
            "- Dependents are eligible during annual enrollment.\n"
        ),
    )

    chunks, _ = chunk_documents([doc], max_chars=2600, overlap_chars=120)

    assert chunks
    assert any(chunk.section != "General" for chunk in chunks)
    assert any(chunk.heading_path == ["Benefits & Perks", "Time Off"] for chunk in chunks)
    assert any(chunk.heading_path == ["Benefits & Perks", "Insurance"] for chunk in chunks)

    time_off_chunks = [chunk for chunk in chunks if chunk.heading_path == ["Benefits & Perks", "Time Off"]]
    insurance_chunks = [chunk for chunk in chunks if chunk.heading_path == ["Benefits & Perks", "Insurance"]]
    assert time_off_chunks
    assert insurance_chunks
    assert all("medical plan is available" not in chunk.text.lower() for chunk in time_off_chunks)
    assert all("bereavement leave policy" not in chunk.text.lower() for chunk in insurance_chunks)
    assert any(chunk.is_list and "Bereavement leave policy:" in chunk.text for chunk in time_off_chunks)

    stats = get_last_chunk_stats()
    assert int(stats.get("sections_detected", 0)) >= 2
    assert isinstance(stats.get("chunks_by_type"), dict)
    assert set((stats.get("chunks_by_type") or {}).keys()) == {"list", "table", "prose"}
    assert float(stats.get("avg_chunk_chars", 0)) > 0
