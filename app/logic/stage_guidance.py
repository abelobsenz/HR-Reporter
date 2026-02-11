from __future__ import annotations

from typing import Dict, List

from app.models import ConsultantProfile, GuidanceReference, StageBasedRecommendation, StartupGuidanceRule


def _rule_applicable(rule: StartupGuidanceRule, *, size_stage_id: str, funding_stage_id: str | None) -> bool:
    size_ok = not rule.size_stage_ids or size_stage_id in rule.size_stage_ids
    funding_ok = not rule.funding_stage_ids or (funding_stage_id is not None and funding_stage_id in rule.funding_stage_ids)
    return size_ok and funding_ok


def _rule_specificity(rule: StartupGuidanceRule) -> int:
    score = 0
    if rule.size_stage_ids:
        score += 2
    if rule.funding_stage_ids:
        score += 2
    return score


def _dedupe_keep_order(values: List[str], *, cap: int = 10) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        cleaned = value.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= cap:
            break
    return out


def _fallback_structure(size_stage_label: str) -> str:
    label = size_stage_label.lower()
    if "<20" in label or "under 20" in label:
        return (
            "Founder-led HR with fractional People leadership and outsourced payroll/benefits. "
            "Focus on minimum viable compliance and manager basics."
        )
    if "20" in label and "49" in label:
        return (
            "Dedicated HR generalist or People Ops manager with fractional specialist support "
            "for compliance and compensation."
        )
    if "50" in label and "99" in label:
        return (
            "People Ops lead with specialist support (TA, compliance, total rewards), plus manager enablement ownership."
        )
    if "100" in label and "250" in label:
        return (
            "Head of People plus core specialist pods (People Ops, Talent, Total Rewards/HRBP) with formal governance."
        )
    return (
        "Scaled People function with specialist teams and clear operating model, with strategic HR leadership "
        "tightly integrated into business planning."
    )


def build_stage_based_recommendation(
    *,
    profile: ConsultantProfile,
    size_stage_id: str,
    size_stage_label: str,
    funding_stage_id: str | None,
    funding_stage_label: str | None,
) -> StageBasedRecommendation:
    source_by_id: Dict[str, object] = {source.id: source for source in profile.startup_guidance_sources}

    applicable = [
        rule
        for rule in profile.startup_guidance_rules
        if _rule_applicable(rule, size_stage_id=size_stage_id, funding_stage_id=funding_stage_id)
    ]
    applicable.sort(key=lambda rule: (_rule_specificity(rule), len(rule.recommended_practices)), reverse=True)

    structure = ""
    practices: List[str] = []
    risks: List[str] = []
    source_ids: List[str] = []

    for rule in applicable:
        if not structure and rule.hr_structure_recommendation.strip():
            structure = rule.hr_structure_recommendation.strip()
        practices.extend(rule.recommended_practices)
        risks.extend(rule.potential_risks)
        source_ids.extend(rule.source_ids)

    if not structure:
        structure = _fallback_structure(size_stage_label)

    practices = _dedupe_keep_order(practices, cap=12)
    risks = _dedupe_keep_order(risks, cap=10)

    deduped_source_ids = _dedupe_keep_order(source_ids, cap=6)
    sources: List[GuidanceReference] = []
    for source_id in deduped_source_ids:
        source = source_by_id.get(source_id)
        if source is None:
            continue
        sources.append(
            GuidanceReference(
                title=source.title,
                url=source.url,
                published_date=source.published_date,
            )
        )

    if not sources:
        for source in profile.startup_guidance_sources[:3]:
            sources.append(
                GuidanceReference(
                    title=source.title,
                    url=source.url,
                    published_date=source.published_date,
                )
            )

    return StageBasedRecommendation(
        size_stage=size_stage_label,
        funding_stage=funding_stage_label,
        hr_structure_recommendation=structure,
        recommended_practices=practices,
        potential_risks=risks,
        sources=sources,
    )
