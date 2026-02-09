from __future__ import annotations

import re
from typing import Any, Dict, List

from app.llm.extract_snapshot import extract_snapshot
from app.models import TextChunk


def _chunk(idx: int, text: str) -> TextChunk:
    return TextChunk(
        chunk_id=f"doc-001-c{idx:03d}",
        doc_id="doc-001",
        section="General",
        text=text,
    )


class _FakeSnapshotClient:
    def __init__(self, fail_if_multi_chunk: bool = False) -> None:
        self.fail_if_multi_chunk = fail_if_multi_chunk
        self.calls: List[List[str]] = []

    def structured_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Dict[str, Any],
        schema_name: str,
    ) -> Dict[str, Any]:
        chunk_ids = re.findall(r"\[chunk_id=([^\]]+)\]", user_prompt)
        self.calls.append(chunk_ids)

        if self.fail_if_multi_chunk and len(chunk_ids) > 1:
            raise RuntimeError("context_length_exceeded")

        payload: Dict[str, Any] = {"evidence_map": []}

        if "doc-001-c001" in chunk_ids:
            payload["headcount"] = 62
            payload["evidence_map"].append(
                {
                    "field_path": "headcount",
                    "status": "present",
                    "citations": [
                        {
                            "chunk_id": "doc-001-c001",
                            "snippet": "Headcount is 62 and the team has expanded steadily across two offices this year.",
                        }
                    ],
                }
            )

        if "doc-001-c002" in chunk_ids:
            payload["policies"] = {"anti_harassment_policy": True}
            payload["evidence_map"].append(
                {
                    "field_path": "policies.anti_harassment_policy",
                    "status": "present",
                    "citations": [
                        {
                            "chunk_id": "doc-001-c002",
                            "snippet": "The anti-harassment policy is in place and is reviewed with managers during onboarding sessions.",
                        }
                    ],
                }
            )

        return payload


def test_extract_snapshot_calls_once_when_context_fits() -> None:
    client = _FakeSnapshotClient()
    chunks = [
        _chunk(1, "Headcount is 62."),
        _chunk(2, "Anti-harassment policy is in place."),
        _chunk(3, "No other details."),
    ]

    snapshot = extract_snapshot(
        client=client,
        chunks=chunks,
        tracked_fields=[
            "headcount",
            "policies.anti_harassment_policy",
            "hris_data.mandatory_training_tracking",
        ],
    )

    assert len(client.calls) == 1
    assert snapshot.headcount == 62
    assert snapshot.policies.anti_harassment_policy is True
    assert snapshot.evidence_map["headcount"].status == "present"
    assert snapshot.evidence_map["policies.anti_harassment_policy"].status == "present"
    assert snapshot.evidence_map["hris_data.mandatory_training_tracking"].status == "not_provided_in_sources"


def test_extract_snapshot_retries_by_splitting_on_context_error() -> None:
    client = _FakeSnapshotClient(fail_if_multi_chunk=True)
    chunks = [
        _chunk(1, "Headcount is 62."),
        _chunk(2, "Anti-harassment policy is in place."),
    ]

    snapshot = extract_snapshot(
        client=client,
        chunks=chunks,
        tracked_fields=["headcount", "policies.anti_harassment_policy"],
    )

    # 1 combined call fails, then 2 single-chunk retries.
    assert len(client.calls) == 3
    assert snapshot.headcount == 62
    assert snapshot.policies.anti_harassment_policy is True


def test_extract_snapshot_uses_chunk_batches_when_prompt_too_large(monkeypatch) -> None:
    monkeypatch.setenv("HR_REPORT_SNAPSHOT_MAX_PROMPT_CHARS", "1")
    monkeypatch.setenv("HR_REPORT_SNAPSHOT_CHUNK_BATCH_SIZE", "1")
    client = _FakeSnapshotClient()
    chunks = [
        _chunk(1, "Headcount is 62."),
        _chunk(2, "Anti-harassment policy is in place."),
        _chunk(3, "No other details."),
    ]

    snapshot = extract_snapshot(
        client=client,
        chunks=chunks,
        tracked_fields=[
            "headcount",
            "policies.anti_harassment_policy",
            "hris_data.mandatory_training_tracking",
        ],
    )

    assert len(client.calls) == 3
    assert snapshot.headcount == 62
    assert snapshot.policies.anti_harassment_policy is True
