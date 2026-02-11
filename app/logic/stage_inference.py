from __future__ import annotations

import re
from typing import List, Optional, Tuple
import logging

from app.models import (
    Citation,
    CompanyPeopleSnapshot,
    ConsultantProfile,
    StageBand,
    StageCandidate,
    StageInferenceResult,
    TextChunk,
)

logger = logging.getLogger(__name__)
S5_PROXY_TERMS = [
    "global entities",
    "global",
    "multi-country",
    "international",
    "board",
    "investor relations",
    " ir ",
    "workday",
]

HEADCOUNT_CONTEXT_TERMS = {
    "headcount",
    "employees",
    "employee",
    "workforce",
    "staff",
    "fte",
    "full-time",
    "team members",
}

HEADCOUNT_RANGE_RE = re.compile(r"\b(\d{1,5})\s*(?:-|to)\s*(\d{1,5})\b", flags=re.IGNORECASE)
HEADCOUNT_PLUS_RE = re.compile(r"\b(\d{1,5})\s*\+")
HEADCOUNT_EXPLICIT_RE = re.compile(
    r"\bheadcount\s*(?:is|=|:|at)?\s*(?:about|around|approximately|~)?\s*(\d{2,5})\b",
    flags=re.IGNORECASE,
)
HEADCOUNT_EMPLOYEE_RE = re.compile(
    r"\b(\d{2,5})\s*(employees|employee|fte|team members)\b",
    flags=re.IGNORECASE,
)
HEADCOUNT_RANGE_CONTEXT_RE = re.compile(
    r"\b(?:headcount|employees|employee|fte|team members)\b.*?\b(\d{2,5})\s*(?:-|to)\s*(\d{2,5})\b",
    flags=re.IGNORECASE,
)
HEADCOUNT_PLUS_CONTEXT_RE = re.compile(
    r"\b(?:headcount|employees|employee|fte|team members)\b.*?\b(\d{2,5})\s*\+",
    flags=re.IGNORECASE,
)
NON_HEADCOUNT_NUMBER_PATTERNS = [
    re.compile(r"\b30\s*/\s*60\s*/\s*90\b"),
    re.compile(r"\b\d+\s*(day|days|week|weeks|month|months)\b"),
    re.compile(r"\bnotice\b"),
]

EXPECTED_STAGE_SIGNALS = [
    "explicit_headcount",
    "headcount_range",
    "funding_round",
    "primary_locations",
]


def _stage_for_headcount(headcount: int, stages: List[StageBand]) -> Optional[StageBand]:
    for stage in sorted(stages, key=lambda s: s.min_headcount):
        if stage.max_headcount is None:
            if headcount >= stage.min_headcount:
                return stage
        elif stage.min_headcount <= headcount <= stage.max_headcount:
            return stage
    return None


def _highest_stage(stages: List[StageBand]) -> StageBand:
    return sorted(stages, key=lambda s: s.min_headcount)[-1]


def _parse_headcount_range(value: str) -> Optional[Tuple[int, int | None]]:
    text = value.strip()
    if not text:
        return None

    range_match = re.search(r"(\d{1,5})\s*(?:-|to)\s*(\d{1,5})", text, flags=re.IGNORECASE)
    if range_match:
        lo = int(range_match.group(1))
        hi = int(range_match.group(2))
        return (min(lo, hi), max(lo, hi))

    plus_match = re.search(r"(\d{1,5})\s*\+", text)
    if plus_match:
        lo = int(plus_match.group(1))
        return (lo, None)

    single_match = re.search(r"(\d{1,5})", text)
    if single_match:
        num = int(single_match.group(1))
        return (num, num)

    return None


def _signals_from_snapshot_and_chunks(snapshot: CompanyPeopleSnapshot, chunks: List[TextChunk] | None) -> List[str]:
    source_text = " ".join(
        [snapshot.company_name or ""]
        + snapshot.current_priorities
        + snapshot.key_risks
        + [c.text for c in (chunks or [])]
    ).lower()

    signals: List[str] = []
    for term in S5_PROXY_TERMS:
        if term in source_text:
            signals.append(term)

    locations = [loc.lower() for loc in snapshot.primary_locations]
    if len({loc.strip() for loc in locations if loc.strip()}) >= 2:
        signals.append("multi-state")

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
        "emea",
        "apac",
        "latam",
    ]
    if any(any(term in loc for term in non_us_terms) for loc in locations):
        signals.append("multi-country")

    return sorted(set(signals))


def _signals_imply_s5(signals: List[str], drivers: List[str]) -> bool:
    signal_match = any(
        term in signals
        for term in [
            "global entities",
            "global",
            "multi-country",
            "board",
            "investor relations",
            "workday",
        ]
    )
    driver_match = any("500+ characteristics" in driver.lower() or "s5" in driver.lower() for driver in drivers)
    return signal_match or driver_match


def _candidate(stage: StageBand, confidence: float, drivers: List[str], signals: List[str]) -> StageCandidate:
    return StageCandidate(stage_id=stage.id, confidence=confidence, drivers=drivers, signals=signals)


def _split_sentences(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _funding_aliases(stage_id: str, label: str, aliases: List[str]) -> List[str]:
    expanded = {stage_id.lower().strip(), label.lower().strip()}
    expanded.update(alias.lower().strip() for alias in aliases if alias.strip())
    merged = " ".join(expanded)

    if "seed" in merged and "series" not in merged:
        expanded.update({"seed", "seed stage", "pre-seed", "pre seed"})
    if "series a" in merged or "series_a" in merged or "series-a" in merged:
        expanded.update({"series a", "series-a", "series_a", "post-seed"})
    if "series b" in merged or "series_b" in merged or "series-b" in merged:
        expanded.update({"series b", "series-b", "series_b"})
    if (
        "series c" in merged
        or "series_c" in merged
        or "series-c" in merged
        or "c+" in merged
        or "growth" in merged
    ):
        expanded.update(
            {
                "series c",
                "series-c",
                "series c+",
                "series d",
                "series e",
                "series f",
                "pre-ipo",
                "ipo",
                "public company",
            }
        )

    return sorted(value for value in expanded if len(value) >= 3)


def _infer_funding_stage(
    *,
    snapshot: CompanyPeopleSnapshot,
    chunks: List[TextChunk] | None,
    profile: ConsultantProfile | None,
) -> Tuple[str | None, str | None, float | None, List[Citation], List[str]]:
    if profile is None or not profile.funding_stages:
        return None, None, None, [], []

    alias_map = {
        stage.id: _funding_aliases(stage.id, stage.label, stage.aliases)
        for stage in profile.funding_stages
    }
    stage_lookup = {stage.id: stage for stage in profile.funding_stages}

    score_by_stage: dict[str, float] = {stage.id: 0.0 for stage in profile.funding_stages}
    citations_by_stage: dict[str, List[Citation]] = {stage.id: [] for stage in profile.funding_stages}

    chunk_pool = chunks or []
    for chunk in chunk_pool:
        for sentence in _split_sentences(chunk.text):
            lower = sentence.lower()
            for stage_id, aliases in alias_map.items():
                for alias in aliases:
                    if re.search(rf"\b{re.escape(alias)}\b", lower):
                        boost = 1.4 if alias.startswith("series ") else 1.0
                        score_by_stage[stage_id] += boost
                        if len(citations_by_stage[stage_id]) < 3:
                            citations_by_stage[stage_id].append(
                                Citation(chunk_id=chunk.chunk_id, snippet=sentence)
                            )

    if any(score_by_stage.values()):
        best_stage_id = sorted(score_by_stage.items(), key=lambda item: item[1], reverse=True)[0][0]
        best_stage = stage_lookup[best_stage_id]
        raw_score = score_by_stage[best_stage_id]
        confidence = round(min(0.9, 0.56 + (0.08 * raw_score)), 3)
        drivers = [f"Funding-stage signal found in source text: {best_stage.label}."]
        return best_stage_id, best_stage.label, confidence, citations_by_stage[best_stage_id], drivers

    return None, None, None, [], []


def _funding_stage_fallback_from_size(
    *,
    size_stage: StageBand,
    profile: ConsultantProfile | None,
) -> Tuple[str | None, str | None]:
    if profile is None or not profile.funding_stages:
        return None, None

    stage_map = {stage.id: stage for stage in profile.funding_stages}

    def _find_by_keywords(keywords: Tuple[str, ...]) -> str | None:
        for stage in profile.funding_stages:
            haystack = " ".join([stage.id.lower(), stage.label.lower()] + [alias.lower() for alias in stage.aliases])
            if any(keyword in haystack for keyword in keywords):
                return stage.id
        return None

    if size_stage.max_headcount is not None and size_stage.max_headcount < 20:
        stage_id = _find_by_keywords(("seed", "pre-seed"))
    elif size_stage.max_headcount is not None and size_stage.max_headcount < 50:
        stage_id = _find_by_keywords(("series a", "series-a", "series_a"))
    elif size_stage.max_headcount is not None and size_stage.max_headcount < 100:
        stage_id = _find_by_keywords(("series b", "series-b", "series_b"))
    else:
        stage_id = _find_by_keywords(("series c", "series-c", "series c+", "series d", "growth", "ipo"))

    if stage_id and stage_id in stage_map:
        stage = stage_map[stage_id]
        return stage.id, stage.label
    return None, None


def _company_stage_label(*, size_stage_label: str, funding_stage_label: str | None) -> str:
    if funding_stage_label:
        return f"{funding_stage_label} | {size_stage_label}"
    return size_stage_label


def _infer_from_chunks(
    *,
    chunks: List[TextChunk] | None,
    stages: List[StageBand],
) -> Tuple[StageBand | None, float, List[str], List[Citation], bool]:
    if not chunks:
        return None, 0.0, [], [], False

    best_stage: StageBand | None = None
    best_confidence = 0.0
    best_drivers: List[str] = []
    best_citations: List[Citation] = []
    explicit = False

    for chunk in chunks:
        for sentence in _split_sentences(chunk.text):
            lower = sentence.lower()
            if any(pattern.search(lower) for pattern in NON_HEADCOUNT_NUMBER_PATTERNS):
                continue
            if not any(term in lower for term in HEADCOUNT_CONTEXT_TERMS):
                continue

            range_match = HEADCOUNT_RANGE_CONTEXT_RE.search(lower) or HEADCOUNT_RANGE_RE.search(lower)
            if range_match:
                lo = int(range_match.group(1))
                hi = int(range_match.group(2))
                pivot = round((lo + hi) / 2)
                stage = _stage_for_headcount(pivot, stages)
                if stage and 0.74 > best_confidence:
                    best_stage = stage
                    best_confidence = 0.74
                    best_drivers = [f"Headcount range extracted from source text: {lo}-{hi}."]
                    best_citations = [Citation(chunk_id=chunk.chunk_id, snippet=sentence)]
                    explicit = True
                continue

            plus_match = HEADCOUNT_PLUS_CONTEXT_RE.search(lower) or HEADCOUNT_PLUS_RE.search(lower)
            if plus_match:
                lo = int(plus_match.group(1))
                stage = _stage_for_headcount(lo, stages)
                if stage and 0.7 > best_confidence:
                    best_stage = stage
                    best_confidence = 0.7
                    best_drivers = [f"Headcount lower-bound extracted from source text: {lo}+"]
                    best_citations = [Citation(chunk_id=chunk.chunk_id, snippet=sentence)]
                    explicit = True
                continue

            explicit_match = HEADCOUNT_EXPLICIT_RE.search(lower)
            employee_match = HEADCOUNT_EMPLOYEE_RE.search(lower)
            value_match = explicit_match or employee_match
            if value_match:
                value = int(value_match.group(1))
                stage = _stage_for_headcount(value, stages)
                if stage and 0.78 > best_confidence:
                    best_stage = stage
                    best_confidence = 0.78
                    best_drivers = [f"Headcount extracted from source text: {value}."]
                    best_citations = [Citation(chunk_id=chunk.chunk_id, snippet=sentence)]
                    explicit = True

    return best_stage, best_confidence, best_drivers, best_citations, explicit


def infer_stage(
    *,
    snapshot: CompanyPeopleSnapshot,
    stages: List[StageBand] | None = None,
    profile: ConsultantProfile | None = None,
    chunks: List[TextChunk] | None = None,
) -> StageInferenceResult:
    resolved_stages: List[StageBand]
    if stages is not None:
        resolved_stages = list(stages)
    elif profile is not None:
        resolved_stages = [
            StageBand(
                id=stage.id,
                label=stage.label,
                min_headcount=stage.min_headcount,
                max_headcount=stage.max_headcount,
                description=f"Profile stage {stage.label}",
            )
            for stage in profile.stages
        ]
    else:
        raise ValueError("infer_stage requires one of: stages or profile")

    stages_sorted = sorted(resolved_stages, key=lambda s: s.min_headcount)
    candidates: List[StageCandidate] = []
    signals = _signals_from_snapshot_and_chunks(snapshot, chunks)

    selected: StageBand | None = None
    confidence = 0.3
    drivers: List[str] = []
    source = "unknown"
    explicit_headcount_evidence = False
    stage_evidence = []

    if snapshot.headcount is not None:
        stage = _stage_for_headcount(snapshot.headcount, resolved_stages)
        if stage:
            selected = stage
            confidence = 1.0
            drivers = [f"Headcount explicitly stated as {snapshot.headcount}."]
            source = "rules"
            explicit_headcount_evidence = True
            candidates.append(_candidate(stage, confidence, drivers, signals))

    if selected is None and snapshot.headcount_range:
        parsed = _parse_headcount_range(snapshot.headcount_range)
        if parsed:
            lo, hi = parsed
            pivot = lo if hi is None else round((lo + hi) / 2)
            stage = _stage_for_headcount(pivot, resolved_stages)
            if stage:
                selected = stage
                confidence = 0.72
                drivers = [f"Headcount range signal found: {snapshot.headcount_range}."]
                source = "rules"
                explicit_headcount_evidence = True
                candidates.append(_candidate(stage, confidence, drivers, signals))

    if selected is None:
        (
            chunk_stage,
            chunk_confidence,
            chunk_drivers,
            chunk_evidence,
            chunk_explicit,
        ) = _infer_from_chunks(chunks=chunks, stages=resolved_stages)
        if chunk_stage is not None:
            selected = chunk_stage
            confidence = chunk_confidence
            drivers = chunk_drivers
            source = "rules"
            stage_evidence = chunk_evidence
            explicit_headcount_evidence = chunk_explicit
            candidates.append(_candidate(chunk_stage, confidence, drivers, signals))

    if selected is None:
        selected = stages_sorted[0]
        confidence = 0.3
        drivers = ["No reliable headcount signal found."]
        source = "unknown"
        candidates.append(_candidate(selected, confidence, drivers, signals))

    note: str | None = None
    stage_min = selected
    stage_max = selected
    highest = _highest_stage(resolved_stages)
    if not explicit_headcount_evidence and selected.id == highest.id:
        # Avoid definitive top-stage assignment from proxies/inference alone.
        if len(stages_sorted) > 1:
            stage_min = stages_sorted[-2]
            selected = stage_min
            drivers = drivers + ["Top-stage capped without explicit headcount evidence."]
            confidence = max(0.22, confidence - 0.1)
            note = "Stage inferred from proxies; explicit headcount not found."
        stage_max = highest

    if _signals_imply_s5(signals, drivers) and not explicit_headcount_evidence:
        stage_max = highest
        confidence = max(0.25, confidence - 0.08)
        drivers = drivers + ["Upper-bound stage expanded by proxy complexity signals."]
        note = "Stage inferred from proxies; explicit headcount not found."

    # Consistency validation against candidates/signals.
    for candidate in candidates:
        candidate_stage = next((s for s in resolved_stages if s.id == candidate.stage_id), None)
        if not candidate_stage:
            continue
        if candidate_stage.min_headcount > selected.min_headcount and source != "rules":
            confidence = max(0.2, confidence - 0.08)
            if note is None:
                note = "Stage inferred from proxies; explicit headcount not found."

    funding_stage_id: str | None = None
    funding_stage_label: str | None = None
    funding_stage_confidence: float | None = None
    funding_stage_evidence: List[Citation] = []
    funding_drivers: List[str] = []

    (
        funding_stage_id,
        funding_stage_label,
        funding_stage_confidence,
        funding_stage_evidence,
        funding_drivers,
    ) = _infer_funding_stage(snapshot=snapshot, chunks=chunks, profile=profile)

    if funding_stage_id is None:
        fallback_stage_id, fallback_stage_label = _funding_stage_fallback_from_size(
            size_stage=selected,
            profile=profile,
        )
        if fallback_stage_id:
            funding_stage_id = fallback_stage_id
            funding_stage_label = fallback_stage_label
            funding_stage_confidence = 0.34
            funding_drivers = ["Funding stage estimated from company-size band (no explicit funding-round signal)."]

    if not explicit_headcount_evidence:
        confidence = min(confidence, 0.68)
    if not funding_stage_evidence and funding_stage_confidence is not None:
        funding_stage_confidence = min(funding_stage_confidence, 0.45)

    selected_index = next((idx for idx, stage in enumerate(stages_sorted) if stage.id == selected.id), None)
    if selected_index is not None:
        for offset in (-1, 1):
            candidate_index = selected_index + offset
            if candidate_index < 0 or candidate_index >= len(stages_sorted):
                continue
            candidate_stage = stages_sorted[candidate_index]
            if any(existing.stage_id == candidate_stage.id for existing in candidates):
                continue
            candidate_confidence = round(max(0.1, confidence - 0.18), 3)
            candidates.append(
                _candidate(
                    candidate_stage,
                    candidate_confidence,
                    ["Adjacent size-band candidate retained because confidence is not absolute."],
                    signals,
                )
            )
    candidates = sorted(candidates, key=lambda row: row.confidence, reverse=True)

    signals_missing: List[str] = []
    if not explicit_headcount_evidence:
        signals_missing.append("explicit_headcount")
    if not snapshot.headcount_range:
        signals_missing.append("headcount_range")
    if not funding_stage_evidence:
        signals_missing.append("funding_round")
    if not snapshot.primary_locations:
        signals_missing.append("primary_locations")
    signals_missing = [signal for signal in EXPECTED_STAGE_SIGNALS if signal in set(signals_missing)]

    drivers = drivers + funding_drivers
    company_stage_label = _company_stage_label(
        size_stage_label=selected.label,
        funding_stage_label=funding_stage_label,
    )

    result = StageInferenceResult(
        stage_id=selected.id,
        stage_label=selected.label,
        confidence=round(min(1.0, max(0.05, confidence)), 3),
        stage_min=stage_min.id,
        stage_max=stage_max.id,
        stage_point_estimate=selected.id,
        stage_confidence=round(min(1.0, max(0.05, confidence)), 3),
        stage_evidence=stage_evidence,
        funding_stage_id=funding_stage_id,
        funding_stage_label=funding_stage_label,
        funding_stage_confidence=funding_stage_confidence,
        funding_stage_evidence=funding_stage_evidence,
        company_stage_label=company_stage_label,
        explicit_headcount_evidence=explicit_headcount_evidence,
        drivers=drivers,
        signals=signals,
        signals_missing=signals_missing,
        candidates=candidates,
        note=note,
        source=source,
    )
    logger.info(
        "stage_inference_done stage=%s funding_stage=%s confidence=%.3f explicit=%s signals=%s",
        result.stage_label,
        result.funding_stage_label or "unknown",
        result.confidence,
        result.explicit_headcount_evidence,
        ",".join(result.signals),
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("stage_candidates=%s", [c.model_dump() for c in result.candidates])
    return result
