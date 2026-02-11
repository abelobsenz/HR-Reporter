from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from app.models import RawDocument, TextChunk

logger = logging.getLogger(__name__)

HEADING_PATTERNS = [
    re.compile(r"^\s{0,3}#{1,6}\s+(.+)$"),
    re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+)$"),
    re.compile(r"^\s*([A-Z][A-Z\s\-/]{3,80})\s*$"),
]
MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
NUMBERED_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]?\s+(.+?)\s*$")
UPPER_HEADING_RE = re.compile(r"^\s*([A-Z][A-Z\s\-/]{3,80})\s*$")
BULLET_LINE_RE = re.compile(r"^\s*([-*]|\d+[.)])\s+")
SALIENT_KEYWORDS = [
    "harassment",
    "complaint",
    "reporting",
    "retaliation",
    "hotline",
    "overtime",
    "exempt",
    "non-exempt",
    "jurisdiction",
    "required training",
    "eeo",
    "discrimination",
    "whistleblower",
    "eligibility",
    "benefits",
    "leave type",
    "leave policy",
]
_CHUNK_STATS: Dict[str, object] = {}


@dataclass
class _Section:
    heading_path: List[str]
    text: str
    start_offset: int


@dataclass
class _Block:
    mode: str
    text: str


def _parse_heading_line(line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    if not stripped:
        return None

    markdown_match = MARKDOWN_HEADING_RE.match(stripped)
    if markdown_match:
        level = len(markdown_match.group(1))
        label = " ".join(markdown_match.group(2).split()).strip()
        if label:
            return level, label

    numbered_match = NUMBERED_HEADING_RE.match(stripped)
    if numbered_match:
        level = min(6, len(numbered_match.group(1).split(".")))
        label = " ".join(numbered_match.group(2).split()).strip()
        if label:
            return max(1, level), label

    upper_match = UPPER_HEADING_RE.match(stripped)
    if upper_match:
        label = " ".join(upper_match.group(1).split()).strip()
        if label:
            return 1, label.title()

    return None


def _is_heading(line: str) -> str | None:
    parsed = _parse_heading_line(line)
    return parsed[1] if parsed else None


def _looks_structured_markdown(text: str) -> bool:
    markdown_hits = 0
    generic_hits = 0
    for raw_line in text.splitlines():
        parsed = _parse_heading_line(raw_line)
        if not parsed:
            continue
        if MARKDOWN_HEADING_RE.match(raw_line.strip()):
            markdown_hits += 1
        else:
            generic_hits += 1
    if markdown_hits >= 1:
        return True
    return generic_hits >= 2


def _split_sections(text: str) -> List[_Section]:
    if not text.strip():
        return []

    sections: List[_Section] = []
    heading_stack: List[tuple[int, str]] = []
    current_lines: List[str] = []
    current_start: int | None = None
    current_path: List[str] = ["General"]

    def _flush() -> None:
        nonlocal current_lines, current_start, current_path
        if not current_lines or current_start is None:
            current_lines = []
            current_start = None
            return
        section_text = "\n".join(current_lines).strip()
        if section_text:
            path = current_path[:] if current_path else ["General"]
            sections.append(_Section(heading_path=path, text=section_text, start_offset=current_start))
        current_lines = []
        current_start = None

    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        line_start = offset
        offset += len(raw_line)

        heading = _parse_heading_line(line)
        if heading is not None:
            _flush()
            level, label = heading
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, label))
            current_path = [item[1] for item in heading_stack] or ["General"]
            current_start = line_start
            current_lines = [line]
            continue

        if current_start is None:
            current_start = line_start
            current_path = [item[1] for item in heading_stack] if heading_stack else ["General"]
        current_lines.append(line)

    _flush()
    return sections


def _line_mode(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return "blank"
    if _is_heading(stripped):
        return "heading"
    if BULLET_LINE_RE.match(stripped):
        return "list"
    if "|" in stripped and len(stripped) >= 6:
        return "table"
    return "text"


def _split_blocks(text: str) -> List[_Block]:
    lines = text.splitlines()
    blocks: List[_Block] = []
    current: List[str] = []
    current_mode = "text"

    def _flush() -> None:
        nonlocal current
        if not current:
            return
        block = "\n".join(current).strip()
        if block:
            blocks.append(_Block(mode=current_mode, text=block))
        current = []

    for line in lines:
        mode = _line_mode(line)
        if mode == "blank":
            _flush()
            current_mode = "text"
            continue
        if not current:
            current = [line]
            current_mode = mode
            continue
        # Keep coherent runs intact.
        if current_mode == mode and mode in {"list", "table", "text"}:
            current.append(line)
            continue
        # Headings should stand alone.
        if mode == "heading" or current_mode == "heading":
            _flush()
            current = [line]
            current_mode = mode
            continue
        _flush()
        current = [line]
        current_mode = mode
    _flush()

    return blocks


def _split_list_entries(block: str) -> List[str]:
    lines = [line for line in block.splitlines() if line.strip()]
    if not lines:
        return []
    entries: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if BULLET_LINE_RE.match(line) and current:
            entries.append(current)
            current = [line]
            continue
        current.append(line)
    if current:
        entries.append(current)
    return ["\n".join(entry).strip() for entry in entries if any(part.strip() for part in entry)]


def _split_long_list_block(
    block: str,
    *,
    max_chars: int,
    min_bullets: int = 6,
    max_bullets: int = 12,
) -> List[str]:
    if len(block) <= max_chars:
        return [block]
    entries = _split_list_entries(block)
    if len(entries) <= 1:
        return _split_long_block(block, max_chars=max_chars, overlap_chars=0)

    chunks: List[str] = []
    current_entries: List[str] = []
    current_len = 0

    def _flush() -> None:
        nonlocal current_entries, current_len
        if not current_entries:
            return
        chunks.append("\n".join(current_entries).strip())
        current_entries = []
        current_len = 0

    for idx, entry in enumerate(entries):
        entry_len = len(entry) + (1 if current_entries else 0)
        remaining = len(entries) - idx
        should_flush_for_size = current_entries and (current_len + entry_len > max_chars)
        should_flush_for_count = len(current_entries) >= max_bullets
        if should_flush_for_count or (should_flush_for_size and len(current_entries) >= min_bullets):
            _flush()
        current_entries.append(entry)
        current_len += entry_len
        if len(current_entries) >= max_bullets and remaining > 0:
            _flush()

    _flush()

    normalized: List[str] = []
    for part in chunks:
        if len(part) <= max_chars:
            normalized.append(part)
            continue
        normalized.extend(_split_long_block(part, max_chars=max_chars, overlap_chars=0))
    return normalized


def _split_long_table_block(block: str, *, max_chars: int) -> List[str]:
    if len(block) <= max_chars:
        return [block]
    lines = [line for line in block.splitlines() if line.strip()]
    if len(lines) <= 2:
        return _split_long_block(block, max_chars=max_chars, overlap_chars=0)

    header = lines[:2] if len(lines) >= 2 and set(lines[1].replace("|", "").strip()) <= {"-", ":"} else lines[:1]
    rows = lines[len(header) :]
    chunks: List[str] = []
    current_rows: List[str] = []
    current_len = len("\n".join(header))

    def _flush() -> None:
        nonlocal current_rows, current_len
        if not current_rows:
            return
        rows_block = "\n".join(current_rows)
        chunks.append("\n".join(header + [rows_block]).strip())
        current_rows = []
        current_len = len("\n".join(header))

    for row in rows:
        row_len = len(row) + 1
        if current_rows and current_len + row_len > max_chars:
            _flush()
        current_rows.append(row)
        current_len += row_len
    _flush()
    return chunks or _split_long_block(block, max_chars=max_chars, overlap_chars=0)


def _context_header_from_previous_block(previous: _Block | None) -> str | None:
    if previous is None:
        return None
    lines = [line.strip() for line in previous.text.splitlines() if line.strip()]
    if not lines:
        return None
    candidate = lines[-1]
    if candidate.endswith(":"):
        return candidate
    if previous.mode == "heading":
        return candidate
    return None


def _split_long_block(block: str, max_chars: int, overlap_chars: int) -> List[str]:
    if len(block) <= max_chars:
        return [block]

    # Prefer sentence boundaries to avoid clipped evidence snippets.
    sentences = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+", block)
        if segment.strip()
    ]
    if len(sentences) <= 1:
        sentences = [part.strip() for part in block.split("\n") if part.strip()]
    if len(sentences) <= 1:
        sentences = [block]

    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(sentence) <= max_chars:
            current = sentence
            continue
        # Last fallback for a single long sentence/line.
        start = 0
        while start < len(sentence):
            end = min(start + max_chars, len(sentence))
            if end < len(sentence):
                boundary = sentence.rfind(" ", start + int(max_chars * 0.6), end)
                if boundary > start:
                    end = boundary
            piece = sentence[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(sentence):
                break
            next_start = max(0, end - overlap_chars)
            if next_start <= start:
                next_start = end
            start = next_start
        current = ""
    if current:
        chunks.append(current)
    return chunks


def _chunk_text(
    text: str,
    max_chars: int = 3200,
    overlap_chars: int = 320,
    *,
    list_table_max_chars: int | None = None,
) -> List[str]:
    stripped_text = text.strip()
    if len(stripped_text) <= max_chars:
        return [stripped_text]

    blocks = _split_blocks(stripped_text)
    if not blocks:
        blocks = [_Block(mode="text", text=stripped_text)]

    chunks: List[str] = []
    current = ""
    sep = "\n\n"
    previous_block: _Block | None = None

    for block in blocks:
        block_limit = max_chars
        if block.mode in {"list", "table"} and list_table_max_chars is not None:
            block_limit = max(500, min(max_chars, list_table_max_chars))

        if block.mode == "list":
            block_parts = _split_long_list_block(block.text, max_chars=block_limit)
        elif block.mode == "table":
            block_parts = _split_long_table_block(block.text, max_chars=block_limit)
        else:
            block_parts = _split_long_block(block.text, max_chars=block_limit, overlap_chars=overlap_chars)

        context_header = _context_header_from_previous_block(previous_block) if block.mode in {"list", "table"} else None

        for part in block_parts:
            enriched_part = part
            if context_header and context_header not in part:
                enriched_part = f"{context_header}\n{part}".strip()

            candidate = f"{current}{sep}{enriched_part}".strip() if current else enriched_part
            if len(candidate) <= max_chars:
                current = candidate
                continue

            if current:
                chunks.append(current)
            if len(enriched_part) <= max_chars:
                current = enriched_part
                continue

            for extra in _split_long_block(enriched_part, max_chars=max_chars, overlap_chars=overlap_chars):
                if len(extra) <= max_chars:
                    chunks.append(extra)
            current = ""
        previous_block = block

    if current:
        chunks.append(current)

    return chunks


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?](?:\s|$)", text))


def _nav_score(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    short_lines = sum(1 for line in lines if len(line.split()) <= 4)
    punct_lines = sum(1 for line in lines if any(token in line for token in ".:;!?"))
    short_ratio = short_lines / len(lines)
    punct_ratio = punct_lines / len(lines)
    return round(short_ratio - punct_ratio, 3)


def _is_list_chunk(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    list_lines = sum(1 for line in lines if re.match(r"^([-*]|\d+[.)])\s+", line))
    return list_lines >= 2


def _is_table_chunk(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    pipe_lines = sum(1 for line in lines if "|" in line)
    return pipe_lines >= 2


def _is_salient(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in SALIENT_KEYWORDS)


def _passes_quality_gate(text: str) -> bool:
    stripped = text.strip()
    low_length = len(stripped) < 300
    low_sentence_density = _sentence_count(stripped) < 2
    low_information = low_length or low_sentence_density
    return not (low_information and not _is_salient(stripped))


def _chunk_text_with_offsets(
    text: str,
    *,
    base_offset: int,
    max_chars: int = 3200,
    overlap_chars: int = 320,
    list_table_max_chars: int | None = None,
) -> List[Tuple[str, int, int]]:
    parts = _chunk_text(
        text,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        list_table_max_chars=list_table_max_chars,
    )
    out: List[Tuple[str, int, int]] = []
    cursor = 0
    for part in parts:
        if not part:
            continue
        idx = text.find(part, cursor)
        if idx < 0:
            idx = cursor
        start = base_offset + idx
        end = start + len(part)
        out.append((part, start, end))
        cursor = max(idx + len(part) - overlap_chars, idx + 1)
    return out


def chunk_documents(
    documents: List[RawDocument],
    max_chars: int = 3200,
    overlap_chars: int = 320,
    chunk_mode: str | None = None,
) -> tuple[List[TextChunk], Dict[str, Dict[str, Any]]]:
    chunks: List[TextChunk] = []
    index: Dict[str, Dict[str, Any]] = {}
    resolved_mode = (chunk_mode or os.getenv("HR_REPORT_CHUNK_MODE", "legacy")).strip().lower() or "legacy"
    if resolved_mode not in {"legacy", "semantic"}:
        resolved_mode = "legacy"
    semantic_mode = resolved_mode == "semantic"
    stats = {
        "candidate_chunks": 0,
        "kept_chunks": 0,
        "dropped_chunks": 0,
        "dropped_low_info": 0,
        "short_but_salient_kept": 0,
        "sections_detected": 0,
        "chunks_by_type": {"list": 0, "table": 0, "prose": 0},
        "avg_chunk_chars": 0.0,
    }

    logger.info(
        "chunker_start docs=%s max_chars=%s overlap=%s mode=%s",
        len(documents),
        max_chars,
        overlap_chars,
        resolved_mode,
    )
    for doc in documents:
        source_type = getattr(doc, "source_type", "text")
        apply_quality_gate = source_type == "url"
        split_by_headings = source_type in {"url", "file"} and _looks_structured_markdown(doc.text)
        if not semantic_mode and source_type == "file":
            split_by_headings = False

        list_table_max_chars: int | None = None
        if semantic_mode:
            list_table_max_chars = max(1400, min(1800, int(max_chars * 0.45)))
        logger.debug(
            "chunker_doc doc_id=%s source=%s source_type=%s text_chars=%s",
            doc.doc_id,
            doc.source,
            source_type,
            len(doc.text),
        )
        if split_by_headings:
            sections = _split_sections(doc.text)
            if not sections:
                sections = [_Section(heading_path=["General"], text=doc.text, start_offset=0)]
        else:
            # File/text inputs usually carry dense relevant context; keep larger contiguous chunks.
            sections = [_Section(heading_path=["General"], text=doc.text, start_offset=0)]

        seq = 1
        stats["sections_detected"] = int(stats["sections_detected"]) + len(sections)
        for section in sections:
            text = section.text
            section_start = int(section.start_offset)
            heading_path = section.heading_path[:] if section.heading_path else ["General"]
            section_label = heading_path[-1] if heading_path else "General"

            for part, start_char, end_char in _chunk_text_with_offsets(
                text,
                base_offset=section_start,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                list_table_max_chars=list_table_max_chars,
            ):
                stats["candidate_chunks"] = int(stats["candidate_chunks"]) + 1
                stripped_part = part.strip()
                short_or_sparse = (len(stripped_part) < 300) or (_sentence_count(stripped_part) < 2)
                salient = _is_salient(stripped_part)
                if apply_quality_gate and not _passes_quality_gate(stripped_part):
                    stats["dropped_chunks"] = int(stats["dropped_chunks"]) + 1
                    stats["dropped_low_info"] = int(stats["dropped_low_info"]) + 1
                    continue
                if apply_quality_gate and short_or_sparse and salient:
                    stats["short_but_salient_kept"] = int(stats["short_but_salient_kept"]) + 1
                chunk_id = f"{doc.doc_id}-c{seq:03d}"
                seq += 1
                chunk = TextChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    source=doc.source,
                    section=section_label,
                    heading_path=heading_path,
                    is_list=_is_list_chunk(part),
                    is_table=_is_table_chunk(part),
                    start_char=start_char,
                    end_char=end_char,
                    nav_score=_nav_score(part),
                    text=part,
                )
                chunks.append(chunk)
                if chunk.is_list:
                    stats["chunks_by_type"]["list"] = int(stats["chunks_by_type"]["list"]) + 1
                elif chunk.is_table:
                    stats["chunks_by_type"]["table"] = int(stats["chunks_by_type"]["table"]) + 1
                else:
                    stats["chunks_by_type"]["prose"] = int(stats["chunks_by_type"]["prose"]) + 1
                index[chunk_id] = {
                    "doc_id": doc.doc_id,
                    "doc_title": doc.title,
                    "section": section_label,
                    "heading_path": heading_path,
                    "is_list": chunk.is_list,
                    "is_table": chunk.is_table,
                    "start_char": start_char,
                    "end_char": end_char,
                    "nav_score": chunk.nav_score,
                    "text": part,
                    "source": doc.source,
                }
                stats["kept_chunks"] = int(stats["kept_chunks"]) + 1

    if chunks:
        stats["avg_chunk_chars"] = round(sum(len(chunk.text) for chunk in chunks) / len(chunks), 2)

    global _CHUNK_STATS
    _CHUNK_STATS = stats
    logger.info(
        "chunker_done candidate=%s kept=%s dropped=%s short_but_salient_kept=%s sections=%s avg_chunk_chars=%s",
        stats["candidate_chunks"],
        stats["kept_chunks"],
        stats["dropped_chunks"],
        stats["short_but_salient_kept"],
        stats["sections_detected"],
        stats["avg_chunk_chars"],
    )

    return chunks, index


def get_last_chunk_stats() -> Dict[str, object]:
    return dict(_CHUNK_STATS)
