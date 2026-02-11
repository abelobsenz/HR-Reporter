from __future__ import annotations

import pytest

from app.models import Citation, Finding, Subcheck


def _base_kwargs() -> dict:
    return {
        "check_id": "x",
        "area": "compliance_policies",
        "title": "X",
        "severity": "high",
        "stage_reason": "Reason",
    }


def test_present_requires_citations() -> None:
    with pytest.raises(ValueError):
        Finding(**_base_kwargs(), evidence_status="present", evidence=[])


def test_not_provided_cannot_have_citations() -> None:
    with pytest.raises(ValueError):
        Finding(
            **_base_kwargs(),
            evidence_status="not_provided_in_sources",
            evidence=[Citation(chunk_id="c1", snippet="some evidence")],
        )


def test_not_provided_rejects_citations_even_for_ambiguous_retrieval() -> None:
    with pytest.raises(ValueError):
        Finding(
            **_base_kwargs(),
            evidence_status="not_provided_in_sources",
            retrieval_status="MENTIONED_AMBIGUOUS",
            evidence=[Citation(chunk_id="c1", snippet="Policy mentions overtime context but no explicit eligibility rule.")],
            needs_confirmation=True,
            questions=["Can you confirm the overtime policy?"],
        )


def test_ambiguous_requires_citations() -> None:
    with pytest.raises(ValueError):
        Finding(
            **_base_kwargs(),
            evidence_status="ambiguous",
            retrieval_status="MENTIONED_AMBIGUOUS",
            evidence=[],
            needs_confirmation=True,
        )

    finding = Finding(
        **_base_kwargs(),
        evidence_status="ambiguous",
        retrieval_status="MENTIONED_AMBIGUOUS",
        evidence=[Citation(chunk_id="c1", snippet="Policy mentions overtime context but no explicit eligibility rule.")],
        needs_confirmation=True,
    )
    assert finding.evidence_status == "ambiguous"


def test_explicitly_missing_requires_absence_language() -> None:
    with pytest.raises(ValueError):
        Finding(
            **_base_kwargs(),
            evidence_status="explicitly_missing",
            evidence=[Citation(chunk_id="c1", snippet="policy exists")],
        )


def test_subcheck_constraints() -> None:
    with pytest.raises(ValueError):
        Subcheck(
            capability_key="policies.employee_handbook",
            evidence_status="not_provided_in_sources",
            citations=[Citation(chunk_id="c1", snippet="text")],
        )

    with pytest.raises(ValueError):
        Subcheck(
            capability_key="policies.employee_handbook",
            evidence_status="explicitly_missing",
            citations=[Citation(chunk_id="c1", snippet="policy mentioned")],
        )
