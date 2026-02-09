from __future__ import annotations

from typing import Any, Dict, List

from app.llm.extract_snapshot import _prepare_snapshot_chunks, extract_snapshot
from app.models import TextChunk


class _NullSnapshotClient:
    def structured_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Dict[str, Any],
        schema_name: str,
    ) -> Dict[str, Any]:
        return {
            "evidence_map": [
                {"field_path": "policies.leave_policy", "status": "not_provided_in_sources", "citations": []},
                {"field_path": "hris_data.hris_system", "status": "not_provided_in_sources", "citations": []},
            ]
        }


def _chunk(idx: int, text: str) -> TextChunk:
    return TextChunk(
        chunk_id=f"doc-001-c{idx:03d}",
        doc_id="doc-001",
        section="General",
        text=text,
    )


def test_extract_snapshot_supplements_explicit_leave_policy_and_hris() -> None:
    chunks = [
        _chunk(
            1,
            (
                "The sick time policy describes eligibility, accrual cadence, and manager review requirements. "
                "Employees are expected to follow the documented leave policy process and submit requests in advance."
            ),
        ),
        _chunk(
            2,
            (
                "Enter the time away request in Workday after manager approval; the HR operations team audits this log monthly. "
                "This workflow is the documented path for leave tracking and payroll alignment."
            ),
        ),
    ]

    snapshot = extract_snapshot(
        client=_NullSnapshotClient(),
        chunks=chunks,
        tracked_fields=["policies.leave_policy", "hris_data.hris_system"],
    )

    assert snapshot.policies.leave_policy is True
    assert snapshot.hris_data.hris_system == "Workday"
    assert snapshot.evidence_map["policies.leave_policy"].status == "present"
    assert snapshot.evidence_map["hris_data.hris_system"].status == "present"


def test_extract_snapshot_does_not_mark_present_from_menu_labels_only() -> None:
    chunks = [
        _chunk(
            1,
            "Time Off\nJob Families\nPeople Policies\nCompensation\nBenefits\nPerformance",
        )
    ]

    snapshot = extract_snapshot(
        client=_NullSnapshotClient(),
        chunks=chunks,
        tracked_fields=["policies.leave_policy"],
    )

    assert snapshot.policies.leave_policy is None
    assert snapshot.evidence_map["policies.leave_policy"].status == "not_provided_in_sources"


def test_prepare_snapshot_chunks_caps_for_speed() -> None:
    chunks: List[TextChunk] = []
    for idx in range(1, 11):
        chunks.append(
            _chunk(
                idx,
                (
                    f"Section {idx}\nTime Off\nJob Families\nBenefits\nCompensation\nPeople Group\n"
                    "Index page"
                ),
            )
        )
    chunks.append(
        _chunk(
            99,
            (
                "The anti-harassment policy is documented and reviewed with all managers each quarter. "
                "The policy also defines reporting, investigation, and remediation timelines."
            ),
        )
    )

    prepared = _prepare_snapshot_chunks(
        chunks=chunks,
        tracked_fields=["policies.anti_harassment_policy"],
        max_chunks=3,
    )

    assert len(prepared) <= 3
    assert any(chunk.chunk_id == "doc-001-c099" for chunk in prepared)
