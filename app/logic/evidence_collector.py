from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence
import logging

from app.llm.client import OpenAIResponsesClient
from app.models import Citation, ConsultantProfile, RawDocument, TextChunk

logger = logging.getLogger(__name__)

RETRIEVAL_STATUS_ORDER = {
    "MENTIONED_EXPLICIT": 5,
    "MENTIONED_IMPLICIT": 4,
    "MENTIONED_AMBIGUOUS": 3,
    "NOT_FOUND_IN_RETRIEVED": 2,
    "NOT_RETRIEVED": 1,
}

EXPLICIT_POSITIVE_PATTERNS = [
    r"\b(has|have|uses|using|maintains?|maintained|implemented|documented|in place)\b.{0,40}\b(policy|process|program|training|framework|system|control|controls|workflow|cadence|review cycle)\b",
    r"\b(system of record|tracked|calibration|review cycle|escalation process|mandatory training)\b",
]
EXPLICIT_NEGATIVE_PATTERNS = [
    r"\b(no|none|missing|lacks?|lacking|absence of)\b.{0,40}\b(policy|process|program|training|framework|system|control|controls|workflow|cadence|review cycle)\b",
    r"\bdoes\s+not\s+(exist|have|include|provide|maintain)\b",
    r"\bnot\s+(in place|defined|documented|established|tracked|available)\b",
    r"\bwithout\b.{0,40}\b(policy|process|program|training|framework|system|control|controls)\b",
]
IRRELEVANT_TEXT_PATTERNS = [
    r"\bcookie\b",
    r"\bprivacy preference\b",
    r"\bsubscribe\b",
    r"\bjavascript\b",
]
LEGAL_PRIVACY_HEAVY_PATTERNS = [
    r"\bprivacy policy\b",
    r"\bpersonal information\b",
    r"\bdata controller\b",
    r"\bdata processor\b",
    r"\bdata retention\b",
    r"\brecord retention\b",
    r"\bgdpr\b",
    r"\bdpf\b",
    r"\bapi terms\b",
    r"\bterms of service\b",
    r"\brecourse and enforcement\b",
]
HR_FOUNDATIONAL_PATTERNS = [
    r"\bemployee handbook\b",
    r"\banti[- ]?harassment\b",
    r"\bharassment training\b",
    r"\bleave policy\b",
    r"\bovertime\b",
    r"\bnon[- ]?exempt\b",
    r"\bonboarding\b",
    r"\bmanager training\b",
    r"\b1:1\b|\bone[- ]on[- ]one\b",
    r"\bperformance review\b",
    r"\bgoal framework\b|\bokr\b",
    r"\bpay band\b|\bleveling\b",
    r"\bhris\b|\bsystem of record\b",
    r"\bemployee relations\b|\ber case\b",
    r"\bengagement survey\b|\battrition\b|\bturnover\b",
    r"\bbenefits eligibility\b|\bbenefits broker\b",
    r"\bmandatory training\b",
]
QUERY_NOISE_TOKENS = {
    "and",
    "are",
    "for",
    "from",
    "into",
    "that",
    "this",
    "with",
    "across",
    "clear",
    "defined",
    "documented",
    "exists",
    "identified",
    "is",
    "or",
    "regularly",
    "rules",
    "standard",
    "systems",
    "system",
    "reporting",
    "reports",
    "report",
    "channel",
    "channels",
    "clear",
    "tracked",
    "training",
    "use",
    "used",
    "using",
    "policy",
    "process",
    "program",
}
RETRIEVAL_NOISE_TOKENS = QUERY_NOISE_TOKENS | {
    "people",
    "person",
    "employee",
    "employees",
    "team",
    "members",
    "member",
    "control",
    "controls",
    "clear",
    "defined",
    "documented",
    "maintained",
    "regularly",
    "baseline",
    "receive",
    "receives",
    "required",
}
LOW_CONTEXT_PREFIXES = {"and", "or", "but", "to", "with", "without"}

HEADCOUNT_CONTEXT_TERMS = {
    "headcount",
    "employees",
    "employee",
    "people",
    "team members",
    "workforce",
    "staff",
    "fte",
}
HEADCOUNT_INEXACT_RE = re.compile(
    r"\b(about|around|approximately|approx\.?|~|over|under|nearly|almost|more than|less than|at least|at most)\b",
    flags=re.IGNORECASE,
)
HEADCOUNT_RANGE_RE = re.compile(
    r"\b(\d{1,5})\s*(?:-|–|to)\s*(\d{1,5})\s*(employees|employee|people|team members|fte)?\b",
    flags=re.IGNORECASE,
)
HEADCOUNT_PLUS_RE = re.compile(
    r"\b(\d{1,5})\s*\+\s*(employees|employee|people|team members|fte)?\b",
    flags=re.IGNORECASE,
)
HEADCOUNT_EXACT_RE = re.compile(
    r"\b(\d{1,5})\s*(employees|employee|people|team members|fte)\b",
    flags=re.IGNORECASE,
)

HR_POLICY_PROCESS_PATTERNS = [
    r"\bpolicy\b",
    r"\bprocess\b",
    r"\bprocedure\b",
    r"\btraining\b",
    r"\bhandbook\b",
    r"\bcomplaint\b",
    r"\breporting\b",
    r"\bhotline\b",
    r"\binvestigation\b",
    r"\bescalation\b",
]

EXPECTATION_DOMAIN_ANCHOR_PATTERNS: Dict[str, List[str]] = {
    "anti_harassment_policy": [
        r"\banti[- ]?harassment\b",
        r"\bharassment\b",
        r"\bretaliation\b",
        r"\bcomplaint\b",
        r"\bhotline\b",
    ],
    "i9_recordkeeping_controls": [
        r"\bi[- ]?9\b",
        r"\bemployment eligibility\b",
        r"\be[- ]?verify\b",
        r"\bwork authorization\b",
        r"\brecordkeeping\b",
    ],
    "hris_system": [
        r"\bhris\b",
        r"\bworkday\b",
        r"\bbamboohr\b",
        r"\brippling\b",
        r"\badp\b",
        r"\bukg\b",
    ],
}


@dataclass
class _QueryItem:
    item_id: str
    queries: List[str]
    snapshot_fields: List[str]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _phrase_pattern(tokens: List[str]) -> re.Pattern[str] | None:
    if not tokens:
        return None
    joined = r"(?:[\W_]+)".join(re.escape(token) for token in tokens)
    return re.compile(rf"\b{joined}\b", flags=re.IGNORECASE)


def _core_query_tokens(query: str) -> List[str]:
    return [
        token
        for token in _tokenize(query)
        if len(token) >= 3 and token not in QUERY_NOISE_TOKENS
    ]


def _query_windows(text: str) -> List[str]:
    windows = _split_paragraphs(text) + _split_sentences(text)
    out: List[str] = []
    seen = set()
    for window in windows:
        cleaned = " ".join(window.split()).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    if not out:
        fallback = " ".join(text.split()).strip()
        if fallback:
            out.append(fallback)
    return out


def _query_match_detail(query: str, *, chunk_text: str, windows: List[str]) -> Dict[str, Any]:
    query_tokens = _tokenize(query)
    core_tokens = _core_query_tokens(query)
    if not query_tokens:
        return {
            "qualified": False,
            "phrase_level": False,
            "matched_terms": [],
            "matched_query": query,
        }

    full_phrase_pattern = _phrase_pattern(query_tokens)
    phrase_level = bool(full_phrase_pattern and full_phrase_pattern.search(chunk_text))

    subphrase_match = False
    if not phrase_level and len(core_tokens) >= 2:
        for span in range(min(3, len(core_tokens)), 1, -1):
            for index in range(0, len(core_tokens) - span + 1):
                subphrase_tokens = core_tokens[index : index + span]
                subphrase_pattern = _phrase_pattern(subphrase_tokens)
                if subphrase_pattern and subphrase_pattern.search(chunk_text):
                    subphrase_match = True
                    break
            if subphrase_match:
                break

    two_core_tokens_in_window = False
    if len(core_tokens) >= 2:
        for window in windows:
            lower = window.lower()
            token_hits = sum(1 for token in set(core_tokens) if _term_matches_chunk(token, lower))
            if token_hits >= 2:
                two_core_tokens_in_window = True
                break

    phrase_level = phrase_level or subphrase_match
    qualified = phrase_level or two_core_tokens_in_window
    matched_terms = [
        token for token in core_tokens if _term_matches_chunk(token, chunk_text)
    ]
    return {
        "qualified": qualified,
        "phrase_level": phrase_level,
        "matched_terms": matched_terms,
        "matched_query": query,
    }


def _domain_anchor_patterns(item_id: str, queries: Sequence[str]) -> List[str]:
    from_map = EXPECTATION_DOMAIN_ANCHOR_PATTERNS.get(item_id, [])
    if from_map:
        return list(from_map)

    patterns: List[str] = []
    for query in queries:
        for token in _core_query_tokens(query):
            if token in {"planning", "hiring", "staffing", "review", "cycle"}:
                continue
            if len(token) < 4:
                continue
            pattern = rf"\b{re.escape(token)}\b"
            if pattern not in patterns:
                patterns.append(pattern)
            if len(patterns) >= 6:
                return patterns
    return patterns


def _term_matches_chunk(term: str, chunk_lower: str) -> bool:
    if not term:
        return False
    if re.search(rf"\b{re.escape(term)}\b", chunk_lower):
        return True
    # Light stemming for simple variants (track/tracked/tracking, policy/policies, etc.).
    if len(term) >= 5:
        stem = term[:-1]
        if stem and re.search(rf"\b{re.escape(stem)}[a-z]{{0,6}}\b", chunk_lower):
            return True
    return False


def _split_sentences(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _split_paragraphs(text: str) -> List[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return paragraphs


def _clip_text(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.split()).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars].rstrip()}..."


def _is_sentence_like(text: str) -> bool:
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return False
    words = re.findall(r"[a-z0-9']+", cleaned.lower())
    if len(words) < 6:
        return False
    if words and words[0] in LOW_CONTEXT_PREFIXES and len(words) < 14:
        return False
    if cleaned.endswith(":"):
        return False
    if not re.search(r"[.!?]$", cleaned) and len(words) < 12:
        return False
    return True


def _sentence_bounds(text: str, anchor_idx: int) -> tuple[int, int]:
    start_candidates = [text.rfind(token, 0, anchor_idx) for token in [".", "!", "?", "\n"]]
    sent_start = max(start_candidates)
    if sent_start < 0:
        sent_start = 0
    else:
        sent_start += 1

    end_candidates = [idx for token in [".", "!", "?", "\n"] if (idx := text.find(token, anchor_idx)) != -1]
    sent_end = min(end_candidates) + 1 if end_candidates else len(text)
    return sent_start, sent_end


def _expand_with_neighbor_sentence(
    text: str,
    *,
    sent_start: int,
    sent_end: int,
    max_chars: int,
) -> str:
    primary = " ".join(text[sent_start:sent_end].split()).strip()
    if not primary:
        return primary
    next_start, next_end = _sentence_bounds(text, min(len(text) - 1, sent_end))
    neighbor = " ".join(text[next_start:next_end].split()).strip()
    if not neighbor or neighbor == primary:
        return primary
    merged = " ".join(f"{primary} {neighbor}".split()).strip()
    if len(merged) <= max_chars and _is_sentence_like(merged):
        return merged
    return primary


def _paragraph_bounds(text: str, anchor_idx: int) -> tuple[int, int]:
    start = text.rfind("\n\n", 0, anchor_idx)
    if start < 0:
        start = 0
    else:
        start += 2
    end = text.find("\n\n", anchor_idx)
    if end < 0:
        end = len(text)
    return start, end


def _is_tabular_noise(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    pipe_lines = sum(1 for line in lines if "|" in line)
    short_lines = sum(1 for line in lines if len(line.split()) <= 5)
    return (pipe_lines / len(lines) >= 0.55) and (short_lines / len(lines) >= 0.45)


def _is_contextual_snippet(text: str) -> bool:
    cleaned = " ".join(text.split()).strip()
    if len(cleaned) < 80:
        return False
    if _is_tabular_noise(cleaned):
        return False
    sentence_hits = len(re.findall(r"[.!?](?:\s|$)", cleaned))
    if sentence_hits >= 2:
        return True
    words = re.findall(r"[a-z0-9']+", cleaned.lower())
    return len(words) >= 22


def _status_from_kind(kind: str) -> str:
    if kind in {"explicit", "explicit_absence"}:
        return "MENTIONED_EXPLICIT"
    if kind == "implicit":
        return "MENTIONED_IMPLICIT"
    if kind == "ambiguous":
        return "MENTIONED_AMBIGUOUS"
    return "NOT_FOUND_IN_RETRIEVED"


def _merge_status(existing: str | None, incoming: str | None) -> str | None:
    if incoming is None:
        return existing
    if existing is None:
        return incoming
    if RETRIEVAL_STATUS_ORDER.get(incoming, 0) >= RETRIEVAL_STATUS_ORDER.get(existing, 0):
        return incoming
    return existing


def _contains_any(patterns: Sequence[str], text: str) -> bool:
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in patterns)


def _match_count(patterns: Sequence[str], text: str) -> int:
    lower = text.lower()
    return sum(1 for pattern in patterns if re.search(pattern, lower))


def _chunk_signal_profile(text: str) -> tuple[int, int]:
    return _match_count(HR_FOUNDATIONAL_PATTERNS, text), _match_count(LEGAL_PRIVACY_HEAVY_PATTERNS, text)


def _is_legal_privacy_only(text: str) -> bool:
    hr_hits, legal_hits = _chunk_signal_profile(text)
    return legal_hits >= 2 and hr_hits == 0


class EvidenceCollector:
    """Single-pass evidence collection used by snapshot extraction and finding evaluation."""

    def __init__(
        self,
        *,
        profile: ConsultantProfile,
        chunks: List[TextChunk],
        documents: List[RawDocument],
        client: OpenAIResponsesClient | None = None,
    ) -> None:
        self.profile = profile
        self.chunks = chunks
        self.documents = documents
        self.client = client
        self._doc_by_id = {doc.doc_id: doc for doc in documents}

    def collect(self) -> Dict[str, Any]:
        if not self.chunks:
            return {
                "expectation_evidence": {},
                "expectation_statuses": {},
                "expectation_queries": {},
                "field_evidence": {},
                "field_statuses": {},
                "field_queries": {},
                "coverage_summary": {
                    "retrieved_docs": len(self.documents),
                    "retrieved_chunks": 0,
                    "fields_with_explicit": 0,
                    "fields_not_retrieved": 0,
                    "fields_not_found": 0,
                },
                "chunk_routes": {},
                "chunk_text_map": {},
            }

        items = self._build_query_items()
        query_item_map = {item.item_id: item for item in items}

        expectation_evidence: Dict[str, List[Dict[str, Any]]] = {}
        expectation_statuses: Dict[str, str] = {}
        expectation_queries: Dict[str, List[str]] = {}

        for item in items:
            evidence_entries, status = self._retrieve_for_item(item)
            expectation_evidence[item.item_id] = evidence_entries
            expectation_statuses[item.item_id] = status
            expectation_queries[item.item_id] = list(item.queries)

        field_evidence, field_statuses, field_queries = self._aggregate_fields(
            query_item_map,
            expectation_evidence,
            expectation_statuses,
        )
        self._inject_headcount_field_evidence(
            field_evidence=field_evidence,
            field_statuses=field_statuses,
            field_queries=field_queries,
        )

        coverage_summary = {
            "retrieved_docs": len(self.documents),
            "retrieved_chunks": len(self.chunks),
            "fields_with_explicit": sum(
                1 for status in field_statuses.values() if status == "MENTIONED_EXPLICIT"
            ),
            "fields_not_retrieved": sum(
                1 for status in field_statuses.values() if status == "NOT_RETRIEVED"
            ),
            "fields_not_found": sum(
                1 for status in field_statuses.values() if status == "NOT_FOUND_IN_RETRIEVED"
            ),
        }

        chunk_text_map = {chunk.chunk_id: _clip_text(chunk.text, 900) for chunk in self.chunks}

        logger.info(
            "evidence_collection_done expectations=%s explicit_fields=%s chunks=%s",
            len(expectation_evidence),
            coverage_summary["fields_with_explicit"],
            len(self.chunks),
        )
        return {
            "expectation_evidence": expectation_evidence,
            "expectation_statuses": expectation_statuses,
            "expectation_queries": expectation_queries,
            "field_evidence": field_evidence,
            "field_statuses": field_statuses,
            "field_queries": field_queries,
            "coverage_summary": coverage_summary,
            "chunk_routes": {},
            "chunk_text_map": chunk_text_map,
        }

    def _build_query_items(self) -> List[_QueryItem]:
        items: List[_QueryItem] = []
        for expectation in self.profile.expectations:
            queries = list(expectation.evidence_queries)
            if expectation.claim not in queries:
                queries.append(expectation.claim)
            items.append(
                _QueryItem(
                    item_id=expectation.id,
                    queries=queries,
                    snapshot_fields=list(expectation.snapshot_fields),
                )
            )
        return items

    def _retrieve_for_item(
        self,
        item: _QueryItem,
    ) -> tuple[List[Dict[str, Any]], str]:
        anchor_patterns = _domain_anchor_patterns(item.item_id, item.queries)

        candidates: List[Dict[str, Any]] = []
        max_snippets = self.profile.retrieval_policy.max_snippets_per_item
        max_chars = self.profile.retrieval_policy.snippet_max_chars
        searchable_chunks = 0

        for chunk in self.chunks:
            lower = chunk.text.lower()
            if _contains_any(IRRELEVANT_TEXT_PATTERNS, lower):
                continue
            if _is_legal_privacy_only(lower):
                continue
            searchable_chunks += 1

            windows = _query_windows(chunk.text)
            query_matches = [
                _query_match_detail(query, chunk_text=lower, windows=windows)
                for query in item.queries
            ]
            qualifying = [entry for entry in query_matches if entry["qualified"]]
            if not qualifying:
                continue

            matched_terms = list(
                dict.fromkeys(
                    [
                        str(term)
                        for entry in qualifying
                        for term in entry.get("matched_terms", [])
                        if str(term)
                    ]
                )
            )
            hr_hits, legal_hits = _chunk_signal_profile(lower)
            if not matched_terms:
                continue
            phrase_level_matches = sum(1 for entry in qualifying if entry["phrase_level"])
            policy_process_hits = _match_count(HR_POLICY_PROCESS_PATTERNS, lower)
            anchor_hits = sum(1 for pattern in anchor_patterns if re.search(pattern, lower))

            doc = self._doc_by_id.get(chunk.doc_id)
            snippet = self._best_snippet(
                chunk=chunk,
                text=chunk.text,
                terms=matched_terms,
                max_chars=max_chars,
                doc_text=(doc.text if doc is not None else None),
            )
            if len(snippet) < 40:
                continue

            kind = self._evidence_kind(snippet, matched_terms)
            nav_penalty = max(0.0, chunk.nav_score or 0.0)
            lexical = len(matched_terms)
            score = (
                (lexical * 2.1)
                + (0.72 * phrase_level_matches)
                + (0.25 * hr_hits)
                - (0.14 * legal_hits)
                - (0.18 * nav_penalty)
            )
            if policy_process_hits > 0 and anchor_hits > 0:
                score += 1.35 + (0.22 * min(anchor_hits, 3))

            citation = Citation(
                chunk_id=chunk.chunk_id,
                snippet=snippet,
                source_id=(doc.source_id if doc else None),
                doc_id=chunk.doc_id,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                evidence_recovered_by="targeted_retrieval",
                retrieval_score=round(score, 4),
            )
            candidates.append(
                {
                    "kind": kind,
                    "score": score,
                    "citation": citation,
                    "source": (doc.source if doc else chunk.doc_id),
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)

        kept: List[Dict[str, Any]] = []
        seen = set()
        for candidate in candidates:
            citation = candidate["citation"]
            dedupe_key = (citation.chunk_id, citation.snippet)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            kept.append(candidate)
            if len(kept) >= max_snippets:
                break

        if not kept:
            if searchable_chunks == 0:
                return [], "NOT_RETRIEVED"
            return [], "NOT_FOUND_IN_RETRIEVED"

        top_kind = kept[0]["kind"]
        status = _status_from_kind(top_kind)
        payload = []
        for row in kept:
            citation = row["citation"]
            payload.append(
                {
                    "chunk_id": citation.chunk_id,
                    "snippet": citation.snippet,
                    "source_id": citation.source_id,
                    "doc_id": citation.doc_id,
                    "start_char": citation.start_char,
                    "end_char": citation.end_char,
                    "evidence_recovered_by": citation.evidence_recovered_by,
                    "retrieval_score": citation.retrieval_score,
                    "source": row["source"],
                    "kind": row["kind"],
                }
            )
        return payload, status

    def _best_snippet(
        self,
        *,
        chunk: TextChunk,
        text: str,
        terms: List[str],
        max_chars: int,
        doc_text: str | None = None,
    ) -> str:
        if doc_text and chunk.start_char is not None and chunk.end_char is not None:
            window_start = max(0, chunk.start_char - 900)
            window_end = min(len(doc_text), chunk.end_char + 900)
            window = doc_text[window_start:window_end]
            if window.strip():
                anchor = max(0, min(len(window) - 1, chunk.start_char - window_start))
                for term in terms:
                    match = re.search(rf"\b{re.escape(term)}\b", window, flags=re.IGNORECASE)
                    if match:
                        anchor = match.start()
                        break
                para_start, para_end = _paragraph_bounds(window, anchor)
                paragraph = " ".join(window[para_start:para_end].split()).strip()
                if _is_contextual_snippet(paragraph):
                    return _clip_text(paragraph, max_chars)

                sent_start, sent_end = _sentence_bounds(window, anchor)
                sentence = " ".join(window[sent_start:sent_end].split()).strip()
                sentence = _expand_with_neighbor_sentence(
                    window,
                    sent_start=sent_start,
                    sent_end=sent_end,
                    max_chars=max_chars,
                )
                if _is_contextual_snippet(sentence):
                    return _clip_text(sentence, max_chars)

        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            paragraphs = [" ".join(text.split()).strip()]

        for idx, paragraph in enumerate(paragraphs):
            lower = paragraph.lower()
            if terms and not any(re.search(rf"\b{re.escape(term)}\b", lower) for term in terms):
                continue
            candidate = paragraph
            if not _is_contextual_snippet(candidate) and idx + 1 < len(paragraphs):
                merged = " ".join(f"{candidate} {paragraphs[idx + 1]}".split()).strip()
                if _is_contextual_snippet(merged):
                    candidate = merged
            if _is_contextual_snippet(candidate):
                return _clip_text(candidate, max_chars)

        sentences = _split_sentences(text)
        for idx, sentence in enumerate(sentences):
            candidate = sentence
            if not _is_sentence_like(candidate):
                continue
            if idx + 1 < len(sentences):
                merged = " ".join(f"{candidate} {sentences[idx + 1]}".split()).strip()
                if len(merged) <= max_chars and _is_sentence_like(merged):
                    candidate = merged
            if _is_contextual_snippet(candidate):
                return _clip_text(candidate, max_chars)

        fallback = " ".join(paragraphs[:2]) if paragraphs else text
        return _clip_text(fallback, max_chars)

    def _evidence_kind(self, snippet: str, terms: List[str]) -> str:
        lower = snippet.lower()
        if terms and _contains_any(EXPLICIT_NEGATIVE_PATTERNS, lower):
            return "explicit_absence"
        if terms and _contains_any(EXPLICIT_POSITIVE_PATTERNS, lower):
            return "explicit"
        if terms:
            return "implicit"
        return "ambiguous"

    def _inject_headcount_field_evidence(
        self,
        *,
        field_evidence: Dict[str, List[Dict[str, Any]]],
        field_statuses: Dict[str, str],
        field_queries: Dict[str, List[str]],
    ) -> None:
        max_snippets = self.profile.retrieval_policy.max_snippets_per_item
        headcount_rows, range_rows = self._collect_headcount_rows()

        def _ensure_queries(field_path: str, queries: List[str]) -> None:
            existing = list(field_queries.get(field_path, []))
            for query in queries:
                if query not in existing:
                    existing.append(query)
            field_queries[field_path] = existing

        if headcount_rows:
            field_evidence["headcount"] = headcount_rows[:max_snippets]
            field_statuses["headcount"] = _merge_status(
                field_statuses.get("headcount"),
                "MENTIONED_EXPLICIT",
            ) or "MENTIONED_EXPLICIT"
        elif "headcount" not in field_statuses:
            field_statuses["headcount"] = "NOT_FOUND_IN_RETRIEVED" if self.chunks else "NOT_RETRIEVED"

        if range_rows:
            field_evidence["headcount_range"] = range_rows[:max_snippets]
            field_statuses["headcount_range"] = _merge_status(
                field_statuses.get("headcount_range"),
                "MENTIONED_EXPLICIT",
            ) or "MENTIONED_EXPLICIT"
        elif "headcount_range" not in field_statuses:
            field_statuses["headcount_range"] = "NOT_FOUND_IN_RETRIEVED" if self.chunks else "NOT_RETRIEVED"

        _ensure_queries(
            "headcount",
            ["headcount", "employees", "team size", "people employed"],
        )
        _ensure_queries(
            "headcount_range",
            ["headcount range", "employees range", "about X-Y people", "X+ employees"],
        )

    def _collect_headcount_rows(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not self.chunks:
            return [], []

        exact_candidates: List[Dict[str, Any]] = []
        range_candidates: List[Dict[str, Any]] = []
        max_chars = self.profile.retrieval_policy.snippet_max_chars

        for chunk in self.chunks:
            doc = self._doc_by_id.get(chunk.doc_id)
            sentences = _split_sentences(chunk.text)
            if not sentences:
                sentences = [" ".join(chunk.text.split()).strip()]

            for sentence in sentences:
                snippet = _clip_text(sentence, max_chars)
                lower = snippet.lower()
                if len(snippet) < 30:
                    continue
                if not any(term in lower for term in HEADCOUNT_CONTEXT_TERMS):
                    continue

                range_match = HEADCOUNT_RANGE_RE.search(lower)
                plus_match = HEADCOUNT_PLUS_RE.search(lower)
                inexact = bool(HEADCOUNT_INEXACT_RE.search(lower))

                if range_match:
                    lo = int(range_match.group(1))
                    hi = int(range_match.group(2))
                    normalized = f"{min(lo, hi)}-{max(lo, hi)}"
                    score = 8.8 + (0.4 if inexact else 0.0)
                    range_candidates.append(
                        self._build_headcount_row(
                            chunk=chunk,
                            doc=doc,
                            snippet=snippet,
                            score=score,
                            kind="explicit",
                            normalized_value=normalized,
                        )
                    )
                    continue

                if plus_match:
                    value = int(plus_match.group(1))
                    normalized = f"{value}+"
                    score = 8.2 + (0.3 if inexact else 0.0)
                    range_candidates.append(
                        self._build_headcount_row(
                            chunk=chunk,
                            doc=doc,
                            snippet=snippet,
                            score=score,
                            kind="explicit",
                            normalized_value=normalized,
                        )
                    )
                    continue

                exact_match = HEADCOUNT_EXACT_RE.search(lower)
                if exact_match and not inexact:
                    score = 8.4
                    exact_candidates.append(
                        self._build_headcount_row(
                            chunk=chunk,
                            doc=doc,
                            snippet=snippet,
                            score=score,
                            kind="explicit",
                        )
                    )
                    continue

                if inexact:
                    fallback_exact = HEADCOUNT_EXACT_RE.search(lower)
                    if fallback_exact:
                        value = int(fallback_exact.group(1))
                        range_candidates.append(
                            self._build_headcount_row(
                                chunk=chunk,
                                doc=doc,
                                snippet=snippet,
                                score=7.4,
                                kind="ambiguous",
                                normalized_value=f"{value}+",
                            )
                        )

        return self._dedupe_rows(exact_candidates), self._dedupe_rows(range_candidates)

    def _build_headcount_row(
        self,
        *,
        chunk: TextChunk,
        doc: RawDocument | None,
        snippet: str,
        score: float,
        kind: str,
        normalized_value: str | None = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "chunk_id": chunk.chunk_id,
            "snippet": snippet,
            "source_id": (doc.source_id if doc else None),
            "doc_id": chunk.doc_id,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "evidence_recovered_by": "targeted_retrieval",
            "retrieval_score": round(float(score), 4),
            "source": (doc.source if doc else chunk.doc_id),
            "kind": kind,
        }
        if normalized_value:
            payload["normalized_value"] = normalized_value
        return payload

    def _dedupe_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for row in sorted(rows, key=lambda item: float(item.get("retrieval_score") or 0.0), reverse=True):
            key = (str(row.get("chunk_id")), str(row.get("snippet")))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped

    def _aggregate_fields(
        self,
        items_by_id: Dict[str, _QueryItem],
        expectation_evidence: Dict[str, List[Dict[str, Any]]],
        expectation_statuses: Dict[str, str],
    ) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str], Dict[str, List[str]]]:
        field_evidence: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        field_statuses: Dict[str, str] = {}
        field_queries: Dict[str, List[str]] = defaultdict(list)

        for expectation in self.profile.expectations:
            item = items_by_id[expectation.id]
            evidence = expectation_evidence.get(expectation.id, [])
            retrieval_status = expectation_statuses.get(expectation.id)
            for field_path in expectation.snapshot_fields:
                field_evidence[field_path].extend(evidence)
                field_statuses[field_path] = _merge_status(field_statuses.get(field_path), retrieval_status) or "NOT_RETRIEVED"
                for query in item.queries:
                    if query not in field_queries[field_path]:
                        field_queries[field_path].append(query)

        for field_path, evidence in field_evidence.items():
            deduped: List[Dict[str, Any]] = []
            seen = set()
            for row in sorted(evidence, key=lambda item: float(item.get("retrieval_score") or 0.0), reverse=True):
                key = (row.get("chunk_id"), row.get("snippet"))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(row)
            field_evidence[field_path] = deduped[: self.profile.retrieval_policy.max_snippets_per_item]

        return dict(field_evidence), field_statuses, dict(field_queries)
