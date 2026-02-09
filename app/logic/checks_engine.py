from __future__ import annotations

import re
from typing import Dict, List, Tuple
import logging

from app.models import (
    Citation,
    CompanyPeopleSnapshot,
    ConsultantPack,
    Finding,
    MetricItem,
    RiskItem,
    Subcheck,
)
from app.utils import get_by_dotted_path, unique_preserve_order

logger = logging.getLogger(__name__)
SEVERITY_ORDER = ["low", "medium", "high", "critical"]
ABSENCE_PATTERNS = [
    r"\bno\b",
    r"\bnot in place\b",
    r"\bdoes not exist\b",
    r"\bmissing\b",
    r"\bnone\b",
    r"\bwithout\b",
    r"\blacking\b",
]
RETRIEVAL_EXPLICIT = {"MENTIONED_EXPLICIT"}
RETRIEVAL_IMPLICIT = {"MENTIONED_IMPLICIT"}
RETRIEVAL_AMBIGUOUS = {"MENTIONED_AMBIGUOUS"}
RETRIEVAL_NOT_FOUND = {"NOT_FOUND_IN_RETRIEVED"}
RETRIEVAL_NOT_RETRIEVED = {"NOT_RETRIEVED"}


def _is_known_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def _requirement_met(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def _severity_down_one(severity: str) -> str:
    idx = max(0, SEVERITY_ORDER.index(severity) - 1)
    return SEVERITY_ORDER[idx]


def _clip_snippet(snippet: str, max_words: int = 20) -> str:
    words = snippet.strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def _dedupe_citations(citations: List[Citation]) -> List[Citation]:
    out: List[Citation] = []
    seen = set()
    for citation in citations:
        normalized = Citation(chunk_id=citation.chunk_id, snippet=_clip_snippet(citation.snippet))
        key = (normalized.chunk_id, normalized.snippet)
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _owner_from_citations(citations: List[Citation]) -> str:
    text = " ".join(c.snippet.lower() for c in citations)
    if re.search(r"\bpeople team\b|\bpeople ops\b", text):
        return "People Team"
    if re.search(r"\bhr team\b|\bhuman resources\b", text):
        return "HR Team"
    if re.search(r"\bfinance\b", text):
        return "Finance"
    if re.search(r"\boperations\b|\bops\b", text):
        return "Ops"
    return "TBD / assign (e.g., HR/People, Finance, Ops)"


def _is_multistate(snapshot: CompanyPeopleSnapshot) -> bool:
    return len({loc.strip().lower() for loc in snapshot.primary_locations if loc.strip()}) >= 2


def _is_multicountry(snapshot: CompanyPeopleSnapshot) -> bool:
    locations = [loc.lower() for loc in snapshot.primary_locations]
    non_us_terms = [
        "canada",
        "uk",
        "united kingdom",
        "germany",
        "france",
        "india",
        "singapore",
        "australia",
        "japan",
        "mexico",
        "brazil",
        "europe",
        "apac",
        "emea",
        "latam",
    ]
    return any(any(term in loc for term in non_us_terms) for loc in locations)


def _is_global_signal(snapshot: CompanyPeopleSnapshot) -> bool:
    haystack = " ".join(
        [snapshot.company_name or ""] + snapshot.current_priorities + snapshot.key_risks
    ).lower()
    return any(
        token in haystack
        for token in [
            "global",
            "multi-country",
            "international",
            "board",
            "investor relations",
            " ir ",
            "workday",
            "global entities",
        ]
    )


def _is_regulated_signal(snapshot: CompanyPeopleSnapshot) -> bool:
    haystack = " ".join(
        [snapshot.company_name or ""] + snapshot.current_priorities + snapshot.key_risks
    ).lower()
    patterns = [
        r"health",
        r"hospital",
        r"pharma",
        r"biotech",
        r"bank",
        r"fintech",
        r"insurance",
        r"energy",
        r"education",
        r"government",
        r"defense",
    ]
    return any(re.search(pattern, haystack) for pattern in patterns)


def _is_threshold_prompt(snapshot: CompanyPeopleSnapshot, check_area: str, check_compliance: bool) -> bool:
    if not (check_compliance or check_area in {"compliance_policies", "er_retention"}):
        return False
    inferred_multi_country = _is_multicountry(snapshot)
    inferred_global = _is_global_signal(snapshot)
    inferred_multi_state = _is_multistate(snapshot)
    regulated = _is_regulated_signal(snapshot)
    headcount_threshold = snapshot.headcount is not None and snapshot.headcount >= 50
    return inferred_multi_country or inferred_global or inferred_multi_state or regulated or headcount_threshold


def _has_explicit_absence(citations: List[Citation]) -> bool:
    text = " ".join(c.snippet.lower() for c in citations)
    return any(re.search(pattern, text) for pattern in ABSENCE_PATTERNS)


def _format_capability(field_path: str) -> str:
    label = field_path.split(".")[-1].replace("_", " ").strip()
    return label.capitalize()


def run_checks(
    *,
    snapshot: CompanyPeopleSnapshot,
    pack: ConsultantPack,
    stage_id: str,
    stage_label: str,
) -> Tuple[List[Finding], List[RiskItem], List[str]]:
    snapshot_payload: Dict[str, object] = snapshot.model_dump(mode="python")

    findings: List[Finding] = []
    risks: List[RiskItem] = []
    unknowns: List[str] = []
    logger.info(
        "checks_start checks=%s stage_id=%s stage_label=%s",
        len(pack.checks),
        stage_id,
        stage_label,
    )

    for check in pack.checks:
        if check.applies_to_stages and stage_id not in check.applies_to_stages and stage_label not in check.applies_to_stages:
            continue

        required_fields = check.required_fields or [check.id]

        retrieval_statuses = [
            (snapshot.evidence_map.get(field_path).retrieval_status if snapshot.evidence_map.get(field_path) else None)
            for field_path in required_fields
        ]
        known_retrieval = [status for status in retrieval_statuses if status is not None]
        all_not_retrieved = (
            len(required_fields) > 0
            and len(known_retrieval) == len(required_fields)
            and all(status in RETRIEVAL_NOT_RETRIEVED for status in known_retrieval)
        )

        if all_not_retrieved:
            question_candidates = (
                list(check.followup_questions)
                or ([check.question_if_unknown] if check.question_if_unknown else [])
                or [f"Can you provide source coverage for {check.title.lower()}?"]
            )
            question = re.sub(r"\s+", " ", question_candidates[0]).strip()
            severity = _severity_down_one(check.severity)
            title = f"{check.title}: Coverage Gap"
            stage_reason = (
                f"At stage {stage_label}, expected source coverage for this domain was not retrieved."
            )
            finding = Finding(
                check_id=f"{check.id}:coverage_gap",
                area=check.area,
                title=title,
                severity=severity,
                evidence_status="not_assessed",
                retrieval_status="NOT_RETRIEVED",
                needs_confirmation=True,
                is_threshold_prompt=False,
                stage_reason=stage_reason,
                evidence=[],
                subchecks=[
                    Subcheck(
                        capability_key=required_fields[0],
                        evidence_status="not_assessed",
                        retrieval_status="NOT_RETRIEVED",
                        citations=[],
                        missing_reason="Coverage gap: expected domain/pages were not retrieved.",
                    )
                ],
                actions=[f"Expand retrieval coverage for {check.area} sources before concluding gaps."],
                owner="TBD / assign (e.g., HR/People, Finance, Ops)",
                metrics=[],
                questions=[question],
            )
            findings.append(finding)
            unknowns.append(required_fields[0])
            continue

        for field_path in required_fields:
            value = get_by_dotted_path(snapshot_payload, field_path)
            evidence = snapshot.evidence_map.get(field_path)
            retrieval_status = evidence.retrieval_status if evidence else None
            citations = _dedupe_citations(evidence.citations if evidence else [])

            if _requirement_met(value) and len(citations) > 0:
                # Capability is present and evidenced; do not produce a growth-area gap.
                continue

            if retrieval_status in RETRIEVAL_NOT_RETRIEVED:
                evidence_status = "not_assessed"
                missing_reason = "Coverage gap: expected domain/pages were not retrieved."
                unknowns.append(field_path)
            elif retrieval_status in RETRIEVAL_NOT_FOUND:
                evidence_status = "not_provided_in_sources"
                missing_reason = "Searched retrieved sources; no explicit match found."
                unknowns.append(field_path)
            elif retrieval_status in RETRIEVAL_AMBIGUOUS:
                evidence_status = "not_provided_in_sources"
                missing_reason = "Mentioned in retrieved sources but ambiguous."
                unknowns.append(field_path)
            elif retrieval_status in RETRIEVAL_IMPLICIT:
                evidence_status = "not_provided_in_sources"
                missing_reason = "Implicitly supported in retrieved sources; needs confirmation."
                unknowns.append(field_path)
            elif len(citations) > 0 and _has_explicit_absence(citations):
                evidence_status = "explicitly_missing"
                missing_reason = "Explicitly indicated as absent in reviewed sources."
            elif len(citations) > 0:
                evidence_status = "present"
                missing_reason = "Evidence exists but capability completeness is unclear."
            elif not _is_known_value(value):
                evidence_status = "not_provided_in_sources"
                missing_reason = "Not provided in sources; confirm whether it exists."
                unknowns.append(field_path)
            else:
                evidence_status = "not_assessed"
                missing_reason = "Domain was not fully assessed in reviewed sources."
                unknowns.append(field_path)

            is_threshold_prompt = _is_threshold_prompt(snapshot, check.area, check.compliance)
            needs_confirmation = evidence_status in {"not_provided_in_sources", "not_assessed"} or (
                retrieval_status in (RETRIEVAL_IMPLICIT | RETRIEVAL_AMBIGUOUS)
            )

            severity = check.severity
            if retrieval_status in RETRIEVAL_NOT_RETRIEVED and not is_threshold_prompt:
                severity = _severity_down_one(_severity_down_one(severity) if severity != "low" else severity)
            elif evidence_status == "not_provided_in_sources" and not is_threshold_prompt:
                severity = _severity_down_one(severity)

            if evidence_status == "explicitly_missing":
                stage_reason = (
                    f"At stage {stage_label}, this capability is explicitly missing based on reviewed sources."
                )
            elif retrieval_status in RETRIEVAL_NOT_RETRIEVED:
                stage_reason = (
                    f"At stage {stage_label}, this appears to be a coverage gap; expected sources were not retrieved."
                )
            elif retrieval_status in RETRIEVAL_NOT_FOUND:
                stage_reason = (
                    f"At stage {stage_label}, this was not observed in retrieved sources after targeted search."
                )
            elif evidence_status == "not_provided_in_sources":
                stage_reason = f"At stage {stage_label}, this capability was not provided in sources; confirm whether it exists."
            elif evidence_status == "not_assessed":
                stage_reason = f"At stage {stage_label}, this capability was not assessed in reviewed sources."
            else:
                stage_reason = f"At stage {stage_label}, available evidence suggests this capability needs confirmation."

            if is_threshold_prompt:
                stage_reason += " Confirm obligations with counsel/advisors."

            question_candidates = list(check.followup_questions)
            if check.question_if_unknown:
                question_candidates.append(check.question_if_unknown)
            if not question_candidates:
                question_candidates = [f"Can you confirm current state for {_format_capability(field_path)}?"]
            question = question_candidates[0]
            question = re.sub(r"\s+", " ", question).strip()

            metric_items: List[MetricItem] = []
            for metric_field in check.metrics:
                metric_value = get_by_dotted_path(snapshot_payload, metric_field)
                metric_evidence = snapshot.evidence_map.get(metric_field)
                metric_citations = _dedupe_citations(
                    metric_evidence.citations
                    if metric_evidence and metric_evidence.status == "present"
                    else []
                )
                if _is_known_value(metric_value) and metric_citations:
                    metric_items.append(
                        MetricItem(
                            metric=metric_field,
                            value=str(metric_value),
                            status="found",
                            evidence=metric_citations,
                        )
                    )
                else:
                    metric_items.append(
                        MetricItem(
                            metric=metric_field,
                            value=None,
                            status="unknown",
                            evidence=[],
                        )
                    )

            capability_label = _format_capability(field_path)
            title = check.title if len(required_fields) == 1 else f"{check.title}: {capability_label}"
            actions = list(check.actions)
            if needs_confirmation:
                actions = [f"Conditional: if {capability_label.lower()} is not in place, {action}" for action in actions]

            subcheck = Subcheck(
                capability_key=field_path,
                evidence_status=evidence_status,
                retrieval_status=retrieval_status,
                citations=(
                    citations
                    if (
                        evidence_status in {"present", "explicitly_missing"}
                        or retrieval_status in (RETRIEVAL_IMPLICIT | RETRIEVAL_AMBIGUOUS)
                    )
                    else []
                ),
                missing_reason=missing_reason,
            )

            finding = Finding(
                check_id=(f"{check.id}:{field_path}" if len(required_fields) > 1 else check.id),
                area=check.area,
                title=title,
                severity=severity,
                evidence_status=evidence_status,
                retrieval_status=retrieval_status,
                needs_confirmation=needs_confirmation,
                is_threshold_prompt=is_threshold_prompt,
                stage_reason=stage_reason,
                evidence=subcheck.citations,
                subchecks=[subcheck],
                actions=actions,
                owner=_owner_from_citations(citations),
                metrics=metric_items,
                questions=[question] if needs_confirmation else [],
            )

            grounded = finding.evidence_status in {"present", "explicitly_missing"} and len(finding.evidence) > 0
            confirmable = finding.needs_confirmation and len(finding.questions) > 0
            if not (grounded or confirmable):
                continue

            findings.append(finding)

            if check.compliance or check.area in {"compliance_policies", "er_retention"}:
                if retrieval_status in RETRIEVAL_NOT_RETRIEVED:
                    risk_statement = f"{title}: Coverage gap in retrieved sources; confirm domain-specific documents were ingested."
                elif retrieval_status in RETRIEVAL_NOT_FOUND:
                    risk_statement = f"{title}: Searched retrieved corpus; no explicit control evidence observed."
                elif evidence_status == "not_provided_in_sources":
                    risk_statement = f"{title}: Not provided in sources; confirm whether this control exists."
                elif evidence_status == "not_assessed":
                    risk_statement = f"{title}: Not assessed in reviewed sources."
                else:
                    risk_statement = f"{title}: {check.description}"
                risks.append(
                    RiskItem(
                        category=check.area,
                        severity=severity,
                        statement=risk_statement,
                        evidence=finding.evidence,
                        mitigation=finding.actions[:2] if finding.actions else ["Not provided in sources"],
                    )
                )

    unknowns = unique_preserve_order(unknowns)
    logger.info(
        "checks_done findings=%s risks=%s unknowns=%s",
        len(findings),
        len(risks),
        len(unknowns),
    )
    return findings, risks, unknowns
