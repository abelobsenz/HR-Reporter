from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from app.models import ConsultantPack, RawDocument, TextChunk

logger = logging.getLogger(__name__)


DEFAULT_EXPLICIT_PATTERNS: Dict[str, List[str]] = {
    "headcount": [r"\bheadcount\b", r"\b\d{1,5}\s*(employees|people|fte)\b"],
    "headcount_range": [r"\b\d{1,5}\s*-\s*\d{1,5}\b", r"\b\d{1,5}\+\b"],
    "hris_data.hris_system": [r"\b(workday|bamboohr|rippling|adp|ukg|successfactors|gusto)\b"],
    "hris_data.mandatory_training_tracking": [r"\b(track|tracking|completion)\b.*\b(training)\b"],
    "policies.overtime_policy": [r"\bovertime\b", r"\b(exempt|non-exempt)\b"],
    "er_retention.engagement_survey": [r"\bengagement survey\b", r"\bpulse survey\b"],
    "comp_leveling.pay_bands": [r"\bpay bands?\b", r"\bcompensation bands?\b"],
    "comp_leveling.leveling_framework": [r"\bleveling framework\b", r"\blevels?\b"],
}

DEFAULT_IMPLICIT_PATTERNS: Dict[str, List[str]] = {
    "policies.leave_policy": [r"\btime off\b", r"\bleave\b", r"\bsick time\b"],
    "benefits.benefits_eligibility_policy": [r"\beligibility\b", r"\bbenefits\b"],
}

FIELD_ANCHOR_TERMS: Dict[str, List[str]] = {
    "policies.employee_handbook": ["employee handbook", "handbook"],
    "policies.anti_harassment_policy": ["anti harassment", "harassment policy", "sexual harassment"],
    "hris_data.mandatory_training_tracking": ["training completion", "training tracking", "mandatory training"],
    "policies.leave_policy": ["leave policy", "time off policy", "sick time policy", "pto policy"],
    "policies.overtime_policy": ["overtime", "working time", "non exempt", "exempt"],
    "hiring.structured_interviews": ["structured interview", "interview rubric", "scorecard", "interview guide"],
    "hiring.offer_approval": ["offer approval", "offer signoff", "approval workflow"],
    "hiring.job_architecture": ["job architecture", "job family", "job level"],
    "onboarding.onboarding_program": ["onboarding", "new hire onboarding", "orientation"],
    "onboarding.role_plan_30_60_90": ["30 60 90", "30/60/90", "ramp plan"],
    "manager_enablement.manager_training": ["manager training", "manager essentials", "manager enablement"],
    "manager_enablement.one_on_one_cadence": ["1:1", "one on one", "one-on-one cadence"],
    "performance.review_cycle": ["performance review cycle", "review cadence", "calibration cycle"],
    "performance.goal_framework": ["goal framework", "okr", "kpi framework"],
    "comp_leveling.pay_bands": ["pay band", "salary band", "compensation band"],
    "comp_leveling.leveling_framework": ["leveling framework", "career level", "job level framework"],
    "hris_data.hris_system": ["hris", "system of record", "workday", "bamboohr", "rippling", "adp", "ukg"],
    "hris_data.data_audit_cadence": ["data audit", "audit cadence", "data quality review"],
    "er_retention.er_case_process": ["employee relations", "er case", "investigation process", "escalation process"],
    "er_retention.attrition_rate": ["attrition", "turnover", "retention rate"],
    "er_retention.engagement_survey": ["engagement survey", "pulse survey", "employee survey"],
    "benefits.benefits_eligibility_policy": ["benefits eligibility", "eligibility policy", "benefits policy"],
    "benefits.benefits_broker": ["benefits broker", "benefits vendor", "broker of record"],
    "headcount": ["headcount", "employees", "team members", "fte"],
    "headcount_range": ["headcount range", "employees", "fte", "500+"],
    "primary_locations": ["location", "country", "state", "region"],
    "current_priorities": ["priority", "initiative", "focus area", "roadmap"],
}

GENERIC_QUERY_STOPWORDS = {
    "find",
    "explicit",
    "evidence",
    "missing",
    "unknown",
    "confirm",
    "whether",
    "exists",
    "exist",
    "sources",
    "source",
    "reviewed",
    "coverage",
    "check",
    "checks",
    "field",
    "fields",
    "tracked",
    "risk",
    "risks",
    "create",
    "avoidable",
    "limits",
    "planning",
    "without",
    "what",
    "where",
    "when",
    "which",
    "should",
    "could",
    "would",
    "from",
    "into",
    "with",
    "that",
    "this",
    "there",
    "their",
    "they",
    "your",
    "you",
    "have",
    "has",
    "are",
    "use",
    "used",
    "using",
}

MIN_SCORE_BY_STRENGTH = {
    "explicit": 0.35,
    "implicit": 0.9,
    "ambiguous": 2.4,
}

MIN_MATCHED_TERMS_BY_STRENGTH = {
    "explicit": 1,
    "implicit": 1,
    "ambiguous": 3,
}

GENERIC_ANCHOR_STOPWORDS = {
    "policy",
    "policies",
    "process",
    "program",
    "framework",
    "system",
    "data",
    "current",
    "company",
    "people",
}


@dataclass
class FieldCandidate:
    field: str
    chunk_id: str
    doc_id: str
    source_id: str | None
    source: str
    score: float
    match_spans: List[Dict[str, int]]
    snippet: str
    start_char: int | None
    end_char: int | None
    nav_score: float | None
    evidence_strength_guess: str


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _query_terms_for_field(
    field: str,
    pack: ConsultantPack,
    checks_by_field: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    terms: List[str] = []

    # Top-level field hints file entries.
    field_hints = pack.field_query_hints.get(field, {})
    for hint in field_hints.get("retrieval_queries", []):
        terms.extend(_tokenize(str(hint)))

    for check_payload in checks_by_field.get(field, []):
        for hint in check_payload.get("retrieval_queries", []):
            terms.extend(_tokenize(str(hint)))
        for hint in check_payload.get("followup_questions", []):
            terms.extend(_tokenize(str(hint)))
        if check_payload.get("question_if_unknown"):
            terms.extend(_tokenize(str(check_payload["question_if_unknown"])))
        terms.extend(_tokenize(str(check_payload.get("title", ""))))

    # Canonical dotted-path tokens.
    terms.extend(_tokenize(field.replace(".", " ").replace("_", " ")))
    # Keep order, dedupe.
    seen = set()
    out = []
    for term in terms:
        if len(term) < 3:
            continue
        if term in seen:
            continue
        if term in GENERIC_QUERY_STOPWORDS:
            continue
        seen.add(term)
        out.append(term)
    return out


def _anchor_terms_for_field(field: str) -> List[str]:
    explicit = FIELD_ANCHOR_TERMS.get(field, [])
    derived = [
        token
        for token in _tokenize(field.replace(".", " ").replace("_", " "))
        if len(token) >= 3 and token not in GENERIC_ANCHOR_STOPWORDS
    ]
    terms: List[str] = []
    seen = set()
    for value in explicit + derived:
        cleaned = value.strip().lower()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        terms.append(cleaned)
    return terms


def _checks_by_field(pack: ConsultantPack) -> Dict[str, List[Dict[str, Any]]]:
    output: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for check in pack.checks:
        payload = check.model_dump(mode="python")
        for field in check.required_fields:
            output[field].append(payload)
    return output


def _build_bm25(chunks: List[TextChunk]) -> Tuple[List[Counter[str]], Dict[str, int], float]:
    docs = [Counter(_tokenize(chunk.text)) for chunk in chunks]
    df: Dict[str, int] = defaultdict(int)
    for doc in docs:
        for term in doc.keys():
            df[term] += 1
    avg_len = sum(sum(doc.values()) for doc in docs) / max(len(docs), 1)
    return docs, df, avg_len


def _bm25_score(
    query_terms: List[str],
    doc_tf: Counter[str],
    df: Dict[str, int],
    doc_count: int,
    avg_len: float,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    score = 0.0
    dl = max(sum(doc_tf.values()), 1)
    for term in query_terms:
        tf = doc_tf.get(term, 0)
        if tf == 0:
            continue
        n = df.get(term, 0)
        idf = math.log((doc_count - n + 0.5) / (n + 0.5) + 1.0)
        denom = tf + k1 * (1 - b + b * dl / max(avg_len, 1.0))
        score += idf * ((tf * (k1 + 1)) / denom)
    return score


def _match_spans(text: str, query_terms: List[str], max_spans: int = 6) -> List[Dict[str, int]]:
    spans: List[Dict[str, int]] = []
    lower = text.lower()
    for term in query_terms:
        for match in re.finditer(rf"\b{re.escape(term)}\b", lower):
            spans.append({"start": match.start(), "end": match.end()})
            if len(spans) >= max_spans:
                return spans
    return spans


def _matched_terms(text: str, query_terms: List[str], max_terms: int = 12) -> List[str]:
    lower = text.lower()
    hits: List[str] = []
    for term in query_terms:
        if re.search(rf"\b{re.escape(term)}\b", lower):
            hits.append(term)
            if len(hits) >= max_terms:
                break
    return hits


def _anchor_hits(text: str, anchor_terms: List[str], max_hits: int = 8) -> List[str]:
    lower = text.lower()
    hits: List[str] = []
    for term in anchor_terms:
        pattern = rf"\b{re.escape(term)}\b" if " " not in term else re.escape(term)
        if re.search(pattern, lower):
            hits.append(term)
            if len(hits) >= max_hits:
                break
    return hits


def _candidate_snippet(text: str, spans: List[Dict[str, int]]) -> str:
    if not text:
        return ""
    if not spans:
        return text[:220].strip()
    anchor = spans[0]["start"]
    start = max(0, anchor - 90)
    end = min(len(text), anchor + 170)
    return " ".join(text[start:end].split()).strip()


def _snippet_is_contextual(snippet: str) -> bool:
    cleaned = " ".join(snippet.split()).strip()
    if len(cleaned) < 70:
        return False
    if not any(token in cleaned for token in [".", ":", ";"]):
        return False
    return True


def _evidence_strength_guess(field: str, text: str) -> str:
    lower = text.lower()
    for pattern in DEFAULT_EXPLICIT_PATTERNS.get(field, []):
        if re.search(pattern, lower):
            return "explicit"
    for pattern in DEFAULT_IMPLICIT_PATTERNS.get(field, []):
        if re.search(pattern, lower):
            return "implicit"
    return "ambiguous"


def _coverage_for_field(field: str, documents: List[RawDocument], pack: ConsultantPack) -> Tuple[bool, str]:
    # Explicit expectations from checks or field hints.
    checks = _checks_by_field(pack).get(field, [])
    expected_patterns: List[str] = []
    for check in checks:
        expected_patterns.extend(check.get("coverage_expectations", {}).get("url_patterns", []))

    field_hints = pack.field_query_hints.get(field, {})
    expected_patterns.extend(field_hints.get("coverage_expectations", {}).get("url_patterns", []))

    if not expected_patterns:
        # fallback expectations based on domain token.
        token = field.split(".", 1)[0] if "." in field else field
        expected_patterns = [token.replace("_", "-"), token.replace("_", "")]

    sources = [doc.source.lower() for doc in documents if doc.source_type == "url"]
    if not sources:
        return False, "No URL/document coverage found for expected domain."
    for pattern in expected_patterns:
        pattern_lower = str(pattern).lower().strip()
        if not pattern_lower:
            continue
        if any(pattern_lower in source for source in sources):
            return True, f"Coverage matched expected pattern '{pattern_lower}'."
    return False, "Expected domain patterns not observed in retrieved sources."


def run_field_targeted_retrieval(
    *,
    chunks: List[TextChunk],
    documents: List[RawDocument],
    tracked_fields: List[str],
    pack: ConsultantPack,
    top_m: int = 6,
) -> Dict[str, Any]:
    doc_by_id = {doc.doc_id: doc for doc in documents}
    check_map = _checks_by_field(pack)

    doc_tfs, df, avg_len = _build_bm25(chunks)
    field_candidates: Dict[str, List[Dict[str, Any]]] = {}
    field_statuses: Dict[str, str] = {}
    coverage_notes: Dict[str, str] = {}
    queries_by_field: Dict[str, List[str]] = {}

    for field in tracked_fields:
        terms = _query_terms_for_field(field, pack, check_map)
        anchor_terms = _anchor_terms_for_field(field)
        queries_by_field[field] = terms[:20]

        scored: List[FieldCandidate] = []
        for idx, chunk in enumerate(chunks):
            strength_guess = _evidence_strength_guess(field, chunk.text)
            matched_terms = _matched_terms(chunk.text, terms)
            anchor_hits = _anchor_hits(chunk.text, anchor_terms)
            if not matched_terms and strength_guess == "ambiguous":
                continue
            bm25 = _bm25_score(terms, doc_tfs[idx], df, len(chunks), avg_len)
            spans = _match_spans(chunk.text, matched_terms or terms)
            keyword_score = float(len(matched_terms))
            nav_penalty = max(0.0, (chunk.nav_score or 0.0))
            score = (0.7 * bm25) + (0.3 * keyword_score) - (0.2 * nav_penalty)
            min_score = MIN_SCORE_BY_STRENGTH[strength_guess]
            min_terms = MIN_MATCHED_TERMS_BY_STRENGTH[strength_guess]
            if score < min_score:
                continue
            if len(matched_terms) < min_terms:
                continue
            if strength_guess in {"implicit", "ambiguous"} and not anchor_hits:
                continue
            if strength_guess == "ambiguous" and len(anchor_hits) < 2:
                continue
            doc = doc_by_id.get(chunk.doc_id)
            snippet = _candidate_snippet(chunk.text, spans)
            if strength_guess == "ambiguous" and not _snippet_is_contextual(snippet):
                continue
            scored.append(
                FieldCandidate(
                    field=field,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_id=(doc.source_id if doc else None),
                    source=(doc.source if doc else chunk.doc_id),
                    score=round(score, 4),
                    match_spans=spans,
                    snippet=snippet,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    nav_score=chunk.nav_score,
                    evidence_strength_guess=strength_guess,
                )
            )

        scored.sort(key=lambda c: (c.score, c.evidence_strength_guess == "explicit"), reverse=True)
        top_candidates = scored[:top_m]
        if top_candidates:
            best_score = top_candidates[0].score
            cutoff = max(0.9, best_score * 0.45)
            top_candidates = [candidate for candidate in top_candidates if candidate.score >= cutoff]
        field_candidates[field] = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "source_id": c.source_id,
                "source": c.source,
                "score": c.score,
                "match_spans": c.match_spans,
                "snippet": c.snippet,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "nav_score": c.nav_score,
                "evidence_strength_guess": c.evidence_strength_guess,
            }
            for c in top_candidates
        ]

        coverage_ok, coverage_note = _coverage_for_field(field, documents, pack)
        coverage_notes[field] = coverage_note
        if top_candidates:
            top_guess = top_candidates[0].evidence_strength_guess
            if top_guess == "explicit":
                field_statuses[field] = "MENTIONED_EXPLICIT"
            elif top_guess == "implicit":
                field_statuses[field] = "MENTIONED_IMPLICIT"
            else:
                field_statuses[field] = "MENTIONED_AMBIGUOUS"
        else:
            field_statuses[field] = "NOT_FOUND_IN_RETRIEVED" if coverage_ok else "NOT_RETRIEVED"
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "field_retrieval field=%s status=%s candidates=%s coverage_ok=%s",
                field,
                field_statuses[field],
                len(top_candidates),
                coverage_ok,
            )

    coverage_summary = {
        "retrieved_docs": len(documents),
        "retrieved_chunks": len(chunks),
        "fields_with_explicit": sum(1 for status in field_statuses.values() if status == "MENTIONED_EXPLICIT"),
        "fields_not_retrieved": sum(1 for status in field_statuses.values() if status == "NOT_RETRIEVED"),
        "fields_not_found": sum(
            1 for status in field_statuses.values() if status == "NOT_FOUND_IN_RETRIEVED"
        ),
    }
    logger.info(
        "field_retrieval_done fields=%s explicit=%s implicit=%s ambiguous=%s not_found=%s not_retrieved=%s",
        len(tracked_fields),
        coverage_summary["fields_with_explicit"],
        sum(1 for status in field_statuses.values() if status == "MENTIONED_IMPLICIT"),
        sum(1 for status in field_statuses.values() if status == "MENTIONED_AMBIGUOUS"),
        coverage_summary["fields_not_found"],
        coverage_summary["fields_not_retrieved"],
    )

    return {
        "field_candidates": field_candidates,
        "field_statuses": field_statuses,
        "coverage_notes": coverage_notes,
        "queries_by_field": queries_by_field,
        "coverage_summary": coverage_summary,
    }
