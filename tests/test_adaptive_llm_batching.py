from __future__ import annotations

import json
from pathlib import Path

from app.logic.discovery import discover_additional_observations
from app.models import Citation, Finding
from app.profile_loader import load_consultant_profile


REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeDiscoveryClient:
    def __init__(self) -> None:
        self.calls = 0

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
        _ = (system_prompt, json_schema, schema_name, max_output_tokens)
        self.calls += 1

        prefix = "Generate observation JSON:\n"
        assert user_prompt.startswith(prefix)
        payload = json.loads(user_prompt[len(prefix) :])
        catalog = payload.get("evidence_catalog", []) or []
        chunk_id = str(catalog[0].get("chunk_id")) if catalog else "not_found"
        return {
            "observations": [
                {
                    "id": f"obs_{self.calls}",
                    "title": f"Observation {self.calls}",
                    "area": "compliance",
                    "severity": "medium",
                    "rationale": "Potential HR risk worth validating.",
                    "evidence_chunk_ids": [chunk_id],
                    "hypothesis": False,
                    "question": "",
                }
            ]
        }


def _finding(idx: int) -> Finding:
    return Finding(
        check_id=f"check_{idx}",
        area="compliance",
        title=f"Finding {idx}",
        severity="high",
        evidence_status="present",
        retrieval_status="MENTIONED_EXPLICIT",
        needs_confirmation=False,
        stage_reason=("Evidence-backed risk signal. " * 30).strip(),
        evidence=[Citation(chunk_id=f"chunk-{idx}", snippet=("policy evidence " * 40).strip())],
        subchecks=[],
        actions=["Define owner and remediation timeline."],
        owner="TBD / assign (e.g., HR/People, Finance, Ops)",
        metrics=[],
        questions=[],
    )


def test_discovery_uses_batch_mode_when_catalog_prompt_is_large(monkeypatch) -> None:
    profile = load_consultant_profile(REPO_ROOT / "tuning" / "profile.yaml")
    client = _FakeDiscoveryClient()

    expectation_rows = []
    for idx in range(36):
        expectation_rows.append(
            {
                "chunk_id": f"chunk-{idx}",
                "snippet": ("Detailed compliance snippet " * 35).strip(),
                "source": "https://example.com/policy",
                "retrieval_score": 5.0 - (idx * 0.01),
            }
        )
    evidence_result = {"expectation_evidence": {"exp_policy": expectation_rows}}

    monkeypatch.setenv("HR_REPORT_DISCOVERY_MAX_PROMPT_CHARS", "1200")
    monkeypatch.setenv("HR_REPORT_DISCOVERY_CATALOG_BATCH_SIZE", "10")

    findings = discover_additional_observations(
        profile=profile,
        stage_label="101-300 employees",
        evidence_result=evidence_result,
        client=client,
    )

    assert client.calls >= 2
    assert len(findings) > 0
