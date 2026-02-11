from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Tuple


CHUNK_ID_RE = re.compile(r"\b([a-z0-9_-]+-c\d{3,5})\b", flags=re.IGNORECASE)
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
URL_RE = re.compile(r"https?://\S+", flags=re.IGNORECASE)
CONTROL_CHAR_RE = re.compile(r"[^\x09\x0A\x0D\x20-\x7E]")
LIST_MARKER_RE = re.compile(r"^\s*[-*+]\s+")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
VERB_HINT_RE = re.compile(
    r"\b(is|are|was|were|has|have|had|includes?|include|provides?|provide|requires?|require|"
    r"reviewed?|documented?|tracked?|offers?|offer|follows?|follow|maintains?|maintain|"
    r"receive|receives|using|use|supports?|support)\b",
    flags=re.IGNORECASE,
)
NAV_PHRASES = [
    "getting started",
    "benefits",
    "how we work",
    "making a career",
    "our rituals",
    "internal systems",
    "code of conduct",
    "state leave provisions",
]


def _chunk_ids_from_markdown(markdown: str) -> List[str]:
    found: List[str] = []
    for match in CHUNK_ID_RE.finditer(markdown or ""):
        chunk_id = match.group(1)
        if chunk_id not in found:
            found.append(chunk_id)
    return found


def _findings_without_citations(report_payload: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for section in ["profile_expectations", "additional_observations", "top_growth_areas"]:
        for finding in report_payload.get(section, []) or []:
            if not isinstance(finding, dict):
                continue
            evidence = finding.get("evidence", []) or []
            status = str(finding.get("evidence_status", "")).strip()
            if status == "present" and len(evidence) == 0:
                out.append(
                    {
                        "section": section,
                        "check_id": str(finding.get("check_id", "")),
                        "title": str(finding.get("title", "")),
                    }
                )
    return out


def _strip_markdown_links(text: str) -> str:
    return MD_LINK_RE.sub(lambda m: m.group(1).strip(), text or "")


def _normalize_snippet_text(text: str) -> str:
    cleaned = _strip_markdown_links(text or "")
    cleaned = URL_RE.sub(" ", cleaned)
    cleaned = (
        cleaned.replace("`", " ")
        .replace("|", " ")
        .replace("*", " ")
        .replace("#", " ")
        .replace("[", " ")
        .replace("]", " ")
        .replace("(", " ")
        .replace(")", " ")
    )
    cleaned = HEADING_RE.sub("", cleaned)
    cleaned = LIST_MARKER_RE.sub("", cleaned)
    cleaned = CONTROL_CHAR_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_noise_line(raw_line: str, cleaned_line: str) -> bool:
    if not cleaned_line:
        return True
    alpha = sum(1 for ch in cleaned_line if ch.isalpha())
    if alpha < 8:
        return True
    lower_raw = raw_line.lower()
    lower_clean = cleaned_line.lower()
    if "http://" in lower_raw or "https://" in lower_raw or ".md" in lower_raw:
        return True
    if raw_line.count("[") >= 2 or raw_line.count("](") >= 1 or raw_line.count("*") >= 4:
        return True
    if "-----" in raw_line:
        return True
    if sum(1 for token in ["/", "\\", "github.com", "blob/"] if token in lower_raw) >= 2:
        return True
    if len(lower_clean.split()) < 6:
        return True
    nav_terms = {
        "getting started",
        "benefits",
        "how we work",
        "our rituals",
        "internal systems",
        "readme",
        "table of contents",
    }
    if any(term in lower_clean for term in nav_terms) and "." not in cleaned_line:
        return True
    words = cleaned_line.split()
    capitalized = sum(1 for word in words if word[:1].isupper())
    capitalized_ratio = (capitalized / len(words)) if words else 0.0
    if capitalized_ratio >= 0.62 and not VERB_HINT_RE.search(cleaned_line):
        return True
    return False


def _first_sentence_like(text: str, max_chars: int = 420) -> str | None:
    cleaned = _normalize_snippet_text(text)
    if not cleaned:
        return None
    for sentence in SENTENCE_SPLIT_RE.split(cleaned):
        sentence = sentence.strip()
        words = sentence.split()
        if len(words) < 8 or not any(ch.isalpha() for ch in sentence):
            continue
        lower = sentence.lower()
        capitalized = sum(1 for word in words if word[:1].isupper())
        capitalized_ratio = capitalized / len(words) if words else 0.0
        has_verb = bool(VERB_HINT_RE.search(sentence))
        nav_hits = sum(1 for phrase in NAV_PHRASES if phrase in lower)
        if nav_hits >= 2 and not has_verb:
            continue
        if "-----" in sentence:
            continue
        if re.search(r"\bM\d\b", sentence) and not has_verb:
            continue
        if capitalized_ratio >= 0.62 and not has_verb:
            continue
        if len(sentence) > max_chars:
            sentence = sentence[:max_chars].rstrip()
        if sentence and sentence[-1] not in ".!?":
            sentence = f"{sentence}."
        return sentence
    if len(cleaned.split()) >= 8:
        lower = cleaned.lower()
        if sum(1 for phrase in NAV_PHRASES if phrase in lower) >= 2 and not VERB_HINT_RE.search(cleaned):
            return None
        clipped = cleaned[:max_chars].rstrip()
        if clipped and clipped[-1] not in ".!?":
            clipped = f"{clipped}."
        return clipped
    return None


def sanitize_citation_snippet(snippet: str, chunk_text: str | None = None, max_chars: int = 420) -> str | None:
    raw = snippet or ""
    lines = [line for line in re.split(r"\r?\n", raw) if line.strip()]
    kept: List[str] = []
    for line in lines:
        cleaned_line = _normalize_snippet_text(line)
        if _is_noise_line(line, cleaned_line):
            continue
        kept.append(cleaned_line)

    candidate = " ".join(kept).strip() if kept else _normalize_snippet_text(raw)
    sentence = _first_sentence_like(candidate, max_chars=max_chars)
    if sentence:
        return sentence

    if chunk_text:
        chunk_sentence = _first_sentence_like(chunk_text, max_chars=max_chars)
        if chunk_sentence:
            return chunk_sentence
    return None


def sanitize_report_citations(
    *,
    report_payload: Dict[str, Any],
    chunks_index: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cleaned = copy.deepcopy(report_payload)
    scanned = 0
    cleaned_count = 0
    dropped = 0
    recovered = 0
    unknown_chunk_ids: List[str] = []

    def _sanitize_citation_list(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        nonlocal scanned, cleaned_count, dropped, recovered
        out: List[Dict[str, Any]] = []
        seen = set()
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            scanned += 1
            chunk_id = str(citation.get("chunk_id", "")).strip()
            original = str(citation.get("snippet", "")).strip()
            chunk_row = chunks_index.get(chunk_id) if isinstance(chunks_index, dict) else None
            chunk_text = str(chunk_row.get("text", "")).strip() if isinstance(chunk_row, dict) else None
            if chunk_id and chunk_row is None and chunk_id not in unknown_chunk_ids:
                unknown_chunk_ids.append(chunk_id)
            rewritten = sanitize_citation_snippet(original, chunk_text)
            if not rewritten:
                dropped += 1
                continue
            if rewritten != original:
                cleaned_count += 1
                if not original.strip():
                    recovered += 1
            row = dict(citation)
            row["snippet"] = rewritten
            dedupe_key = (chunk_id, rewritten)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            out.append(row)
        return out

    for section in ["profile_expectations", "additional_observations", "top_growth_areas", "risks"]:
        rows = cleaned.get(section, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            evidence = row.get("evidence", [])
            if isinstance(evidence, list):
                row["evidence"] = _sanitize_citation_list(evidence)

    return (
        cleaned,
        {
            "scanned_citations": scanned,
            "cleaned_citations": cleaned_count,
            "dropped_citations": dropped,
            "recovered_citations": recovered,
            "unknown_chunk_ids": unknown_chunk_ids,
        },
    )


def sanitize_markdown_citations(
    *,
    markdown: str,
    chunks_index: Dict[str, Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    citation_line_re = re.compile(
        r"^(?P<prefix>\s*[-*]?\s*)`(?P<chunk_id>[A-Za-z0-9_-]+-c\d{3,5})`\s*(?P<snippet>.*)$"
    )
    lines = markdown.splitlines()
    rewritten = 0
    dropped = 0

    out_lines: List[str] = []
    for line in lines:
        line = CONTROL_CHAR_RE.sub("", line)
        match = citation_line_re.match(line)
        if not match:
            out_lines.append(line)
            continue
        prefix = match.group("prefix")
        chunk_id = match.group("chunk_id")
        snippet = match.group("snippet")
        chunk_row = chunks_index.get(chunk_id) if isinstance(chunks_index, dict) else None
        chunk_text = str(chunk_row.get("text", "")).strip() if isinstance(chunk_row, dict) else None
        cleaned = sanitize_citation_snippet(snippet, chunk_text)
        if not cleaned:
            dropped += 1
            out_lines.append(f"{prefix}`{chunk_id}`")
            continue
        if cleaned != snippet.strip():
            rewritten += 1
        out_lines.append(f"{prefix}`{chunk_id}` {cleaned}")

    return (
        "\n".join(out_lines),
        {
            "markdown_citation_lines_rewritten": rewritten,
            "markdown_citation_lines_dropped": dropped,
        },
    )


def audit_markdown_citations(
    *,
    markdown: str,
    chunks_index: Dict[str, Dict[str, Any]],
    report_payload: Dict[str, Any],
) -> Dict[str, Any]:
    cited_chunk_ids = _chunk_ids_from_markdown(markdown)
    known_ids = set(chunks_index.keys()) if isinstance(chunks_index, dict) else set()
    unknown_chunk_ids = [chunk_id for chunk_id in cited_chunk_ids if chunk_id not in known_ids]
    findings_without_citations = _findings_without_citations(report_payload)

    passed = (len(unknown_chunk_ids) == 0) and (len(findings_without_citations) == 0)
    return {
        "passed": passed,
        "summary": {
            "cited_chunk_ids": len(cited_chunk_ids),
            "unknown_chunk_ids": len(unknown_chunk_ids),
            "findings_without_citations": len(findings_without_citations),
        },
        "unknown_chunk_ids": unknown_chunk_ids,
        "findings_without_citations": findings_without_citations,
    }
