from __future__ import annotations

from typing import List, Tuple
import logging

from app.models import ConsultantPack, Finding

logger = logging.getLogger(__name__)


def rank_findings_and_adjust_confidence(
    *,
    findings: List[Finding],
    pack: ConsultantPack,
    stage_confidence: float,
    unknown_count: int,
    tracked_field_count: int,
) -> Tuple[List[Finding], float]:
    check_map = {check.id: check for check in pack.checks}

    scored: List[Finding] = []
    for finding in findings:
        area_weight = pack.weights.area_weights.get(finding.area, 1.0)
        severity_weight = pack.weights.severity_weights.get(finding.severity, 1.0)
        base_check_id = finding.check_id.split(":", 1)[0]
        check = check_map.get(base_check_id)
        compliance_multiplier = pack.weights.compliance_multiplier if (check and check.compliance) else 1.0
        finding.score = round(area_weight * severity_weight * compliance_multiplier, 4)
        scored.append(finding)

    scored.sort(key=lambda f: (f.score or 0.0), reverse=True)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "scoring_ranked findings=%s top=%s",
            len(scored),
            [(f.check_id, f.score, f.severity) for f in scored[:5]],
        )

    denominator = max(tracked_field_count, 1)
    unknown_ratio = min(1.0, unknown_count / denominator)
    confidence_penalty = unknown_ratio * pack.weights.unknown_penalty
    confidence = max(0.05, min(1.0, stage_confidence - confidence_penalty))
    logger.info(
        "scoring_done findings=%s unknown_ratio=%.3f confidence=%.3f",
        len(scored),
        unknown_ratio,
        confidence,
    )
    return scored, round(confidence, 3)


def build_growth_drivers(initial_drivers: List[str], findings: List[Finding], top_n: int = 3) -> List[str]:
    drivers = list(initial_drivers)
    for finding in findings[:top_n]:
        drivers.append(f"{finding.title} flagged as {finding.severity} priority.")

    deduped = []
    seen = set()
    for driver in drivers:
        if driver in seen:
            continue
        seen.add(driver)
        deduped.append(driver)
    logger.debug("drivers_built count=%s", len(deduped))
    return deduped
