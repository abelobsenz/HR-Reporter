from __future__ import annotations

import json
import math

from app.llm.extract_snapshot import extract_snapshot_from_evidence
from app.models import CompanyPeopleSnapshot, TextChunk


class _FakeSnapshotClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def is_enabled(self) -> bool:
        return True

    def structured_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict,
        schema_name: str,
        max_output_tokens: int | None = None,
    ) -> dict:
        prefixes = [
            "Evidence analysis payload:\n",
            "Snapshot resolver payload:\n",
            "Evidence payload:\n",
        ]
        body = None
        for prefix in prefixes:
            if user_prompt.startswith(prefix):
                body = json.loads(user_prompt[len(prefix) :])
                break
        assert body is not None
        self.calls.append(
            {
                "schema_name": schema_name,
                "max_output_tokens": max_output_tokens,
                "body": body,
            }
        )

        if schema_name == "snapshot_field_analysis":
            tracked_fields = list(body.get("tracked_fields", []))
            evidence_payload = body.get("evidence_by_field", {}) or {}
            rows = []
            for field_path in tracked_fields:
                snippets = evidence_payload.get(field_path, []) or []
                rows.append(
                    {
                        "field_path": field_path,
                        "verdict": "present" if snippets else "not_found",
                        "support_strength": "strong" if snippets else "none",
                        "confidence": 0.92 if snippets else 0.1,
                        "value_text": "",
                        "hr_reasoning": "",
                        "needs_confirmation": False if snippets else True,
                        "citations": [
                            {
                                "chunk_id": str(snippets[0].get("chunk_id")),
                                "snippet": str(snippets[0].get("snippet")),
                            }
                        ]
                        if snippets
                        else [],
                    }
                )
            return {"field_assessments": rows}

        tracked_fields = list(body.get("tracked_fields", [])) or list(
            body.get("tracked_fields_for_evidence_map", [])
        )
        evidence_payload = body.get("evidence_by_field", {}) or {}

        payload = CompanyPeopleSnapshot().model_dump(mode="python")
        evidence_rows = []
        for field_path in tracked_fields:
            rows = evidence_payload.get(field_path, []) or []
            citations = []
            if rows:
                first = rows[0]
                citations = [
                    {
                        "chunk_id": str(first.get("chunk_id")),
                        "snippet": str(first.get("snippet")),
                    }
                ]
            evidence_rows.append(
                {
                    "field_path": field_path,
                    "status": "present" if citations else "not_provided_in_sources",
                    "citations": citations,
                }
            )
        payload["evidence_map"] = evidence_rows
        return payload


def test_snapshot_prompt_compacts_evidence_rows_but_keeps_anchor_metadata() -> None:
    client = _FakeSnapshotClient()
    tracked_fields = ["policies.leave_policy"]
    field_evidence = {
        "policies.leave_policy": [
            {
                "chunk_id": "doc-1-c001",
                "snippet": "Our time-off policy states all employees are eligible for paid leave after 30 days.",
                "source_id": "url:https://example.com/timeoff",
                "doc_id": "doc-1",
                "start_char": 10,
                "end_char": 120,
                "retrieval_score": 9.2,
                "kind": "explicit",
            }
        ]
    }

    snapshot = extract_snapshot_from_evidence(
        client=client,
        field_evidence=field_evidence,
        tracked_fields=tracked_fields,
        retrieval_statuses={},
        retrieval_queries={},
    )

    analysis_calls = [call for call in client.calls if call["schema_name"] == "snapshot_field_analysis"]
    resolver_calls = [call for call in client.calls if call["schema_name"] == "company_people_snapshot"]
    assert len(analysis_calls) == 1
    assert len(resolver_calls) == 1
    sent_row = analysis_calls[0]["body"]["evidence_by_field"]["policies.leave_policy"][0]
    assert set(sent_row.keys()) == {"chunk_id", "snippet", "kind"}
    assert sent_row["kind"] == "explicit"

    citations = snapshot.evidence_map["policies.leave_policy"].citations
    assert len(citations) == 1
    assert citations[0].source_id == "url:https://example.com/timeoff"
    assert citations[0].doc_id == "doc-1"


def test_snapshot_evidence_uses_field_batching_when_prompt_is_large(monkeypatch) -> None:
    client = _FakeSnapshotClient()
    tracked_fields = [
        "policies.employee_handbook",
        "policies.anti_harassment_policy",
        "policies.leave_policy",
        "policies.overtime_policy",
        "onboarding.onboarding_program",
        "manager_enablement.manager_training",
        "manager_enablement.one_on_one_cadence",
        "performance.review_cycle",
        "performance.goal_framework",
        "comp_leveling.leveling_framework",
        "comp_leveling.pay_bands",
        "hris_data.hris_system",
    ]

    long_snippet = " ".join(["policy"] * 350)
    field_evidence = {
        field: [
            {
                "chunk_id": f"{field}-c001",
                "snippet": long_snippet,
                "kind": "implicit",
            }
        ]
        for field in tracked_fields
    }

    monkeypatch.setenv("HR_REPORT_SNAPSHOT_EVIDENCE_MAX_PROMPT_CHARS", "1200")
    monkeypatch.setenv("HR_REPORT_SNAPSHOT_EVIDENCE_FIELD_BATCH_SIZE", "4")

    snapshot = extract_snapshot_from_evidence(
        client=client,
        field_evidence=field_evidence,
        tracked_fields=tracked_fields,
        retrieval_statuses={},
        retrieval_queries={},
    )

    expected_calls = math.ceil(len(tracked_fields) / 4)
    analysis_calls = [call for call in client.calls if call["schema_name"] == "snapshot_field_analysis"]
    assert len(analysis_calls) == expected_calls
    assert all(len(call["body"]["tracked_fields"]) <= 4 for call in analysis_calls)
    assert all(field in snapshot.evidence_map for field in tracked_fields)


def test_snapshot_evidence_overrides_headcount_from_explicit_chunk_signal() -> None:
    client = _FakeSnapshotClient()
    tracked_fields = ["headcount"]
    field_evidence = {"headcount": []}
    source_chunks = [
        TextChunk(
            chunk_id="doc-1-c001",
            doc_id="doc-1",
            section="About",
            text="We have 88 employees across the US and Canada.",
            start_char=0,
            end_char=46,
        )
    ]

    snapshot = extract_snapshot_from_evidence(
        client=client,
        field_evidence=field_evidence,
        tracked_fields=tracked_fields,
        retrieval_statuses={},
        retrieval_queries={},
        source_chunks=source_chunks,
    )

    assert snapshot.headcount == 88
    assert snapshot.evidence_map["headcount"].status == "present"
    assert snapshot.evidence_map["headcount"].retrieval_status == "MENTIONED_EXPLICIT"
    assert snapshot.evidence_map["headcount"].citations
