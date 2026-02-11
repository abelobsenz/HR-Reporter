from __future__ import annotations

from pathlib import Path

from app.ingest.chunker import chunk_documents
from app.logic.evidence_collector import EvidenceCollector
from app.models import RawDocument
from app.profile_loader import load_consultant_profile


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_semantic_chunking_makes_leave_policy_bullets_retrievable(monkeypatch) -> None:
    monkeypatch.setenv("HR_REPORT_CHUNK_MODE", "semantic")
    profile = load_consultant_profile(REPO_ROOT / "tuning" / "profile.yaml")
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
            "- Bereavement leave: 10 days paid leave for immediate family.\n"
            "- Bereavement leave: 3 days paid leave for extended family.\n"
            "- Employees must notify their manager within 24 hours when possible.\n"
            "## Internal Tools\n"
            "- Dashboards report device and uptime metrics.\n"
        ),
    )

    chunks, _ = chunk_documents([doc], max_chars=2400, overlap_chars=100)
    result = EvidenceCollector(profile=profile, chunks=chunks, documents=[doc], client=None).collect()

    rows = result["expectation_evidence"].get("leave_overtime_coverage", [])
    assert rows
    assert any("bereavement leave policy" in str(row.get("snippet", "")).lower() for row in rows)
    assert any(
        "time off" in " > ".join(chunk.heading_path).lower()
        for chunk in chunks
        if chunk.chunk_id in {str(row.get("chunk_id", "")) for row in rows}
    )
