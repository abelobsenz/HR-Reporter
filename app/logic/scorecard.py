from __future__ import annotations

from collections import defaultdict
import re
from typing import Dict, List

from app.models import Finding, FunctionalScorecardItem

AREA_LABELS = {
    "strategy": "HR Strategy and Alignment",
    "compliance": "Legal and Compliance",
    "talent_acquisition": "Recruitment and Selection",
    "onboarding": "Onboarding",
    "manager_capability": "Manager Capability",
    "performance_mgmt": "Performance Management",
    "talent_development": "Training and Career Development",
    "systems": "Employee Data Management and HR Technology",
    "retention": "Employee Experience",
    "compensation": "Compensation and Benefits Administration",
    "benefits": "Employee Benefits",
    "deib": "DEIB",
    "offboarding": "Offboarding",
    "change_management": "Change Management",
    "data_privacy": "HR Data Privacy",
}
AREA_ALIASES = {
    "diversity_equity_inclusion_belonging": "deib",
    "diversity_equity_inclusion_belonging_deib": "deib",
    "deib": "deib",
    "compensation_equity": "compensation",
    "compensation_and_equity": "compensation",
    "change_management_communication": "change_management",
    "workplace_conduct_incident_response": "compliance",
    "accommodations_legal_compliance": "compliance",
    "mental_health_and_wellness_support": "retention",
    "structured_interviewing": "talent_acquisition",
    "data_privacy_controls": "data_privacy",
}

IMPACT_ORDER = {"Significant": 3, "Moderate": 2, "Limited": 1}
MATURITY_RISK_ORDER = {"Under-Developed": 4, "Developing": 3, "Aligned": 2, "Strategic": 1}


def _maturity_from_findings(findings: List[Finding]) -> str:
    if not findings:
        return "Strategic"

    critical_or_high = [f for f in findings if f.severity in {"critical", "high"}]
    explicit_missing = [f for f in findings if f.evidence_status == "explicitly_missing"]
    unresolved = [f for f in findings if f.evidence_status in {"not_assessed", "not_provided_in_sources"}]

    if any(f.severity in {"critical", "high"} for f in explicit_missing):
        return "Under-Developed"
    if len(critical_or_high) >= 2 or any(f.severity == "critical" for f in unresolved):
        return "Developing"
    if len(findings) >= 1:
        return "Aligned"
    return "Strategic"


def _impact_from_findings(findings: List[Finding]) -> str:
    if not findings:
        return "Limited"
    severities = [finding.severity for finding in findings]
    if any(severity in {"critical", "high"} for severity in severities):
        return "Significant"
    if any(severity == "medium" for severity in severities):
        return "Moderate"
    return "Limited"


def _normalize_area(area: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", area.lower()).strip("_")
    if normalized in AREA_LABELS:
        return normalized
    return AREA_ALIASES.get(normalized, normalized)


def _label_for_unknown_area(area: str) -> str:
    cleaned = re.sub(r"[_/]+", " ", area).strip()
    return cleaned.title() if cleaned else "Other"


def build_functional_scorecard(
    *,
    findings: List[Finding],
) -> List[FunctionalScorecardItem]:
    findings_by_area: Dict[str, List[Finding]] = defaultdict(list)
    area_labels: Dict[str, str] = {}
    for finding in findings:
        area_key = _normalize_area(finding.area)
        findings_by_area[area_key].append(finding)
        if area_key not in area_labels:
            area_labels[area_key] = AREA_LABELS.get(area_key, _label_for_unknown_area(finding.area))

    area_keys = list(AREA_LABELS.keys())
    for area in findings_by_area:
        if area not in area_keys:
            area_keys.append(area)

    rows: List[FunctionalScorecardItem] = []
    for area in area_keys:
        area_findings = findings_by_area.get(area, [])

        impact = _impact_from_findings(area_findings)
        maturity = _maturity_from_findings(area_findings)

        rows.append(
            FunctionalScorecardItem(
                functional_area=area_labels.get(area, AREA_LABELS.get(area, area.replace("_", " ").title())),
                maturity_level=maturity,
                impact_level=impact,
                rationale=(
                    f"{len(area_findings)} concern(s) were flagged in this area."
                    if area_findings
                    else "No explicit gaps were flagged in reviewed evidence."
                ),
            )
        )

    rows.sort(
        key=lambda row: (
            IMPACT_ORDER.get(row.impact_level, 0),
            MATURITY_RISK_ORDER.get(row.maturity_level, 0),
            row.functional_area,
        ),
        reverse=True,
    )
    return rows
