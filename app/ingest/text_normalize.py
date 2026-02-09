from __future__ import annotations

import re
import unicodedata


_MOJIBAKE_MAP = {
    "â€™": "'",
    "â€TM": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€“": "-",
    "â€”": "-",
    "Â ": " ",
    "Â": "",
}


def _strip_control_chars(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if ch in {"\n", "\t"}:
            out.append(ch)
            continue
        if code < 32 or code == 127:
            continue
        out.append(ch)
    return "".join(out)


def normalize_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw or "")
    for bad, good in _MOJIBAKE_MAP.items():
        text = text.replace(bad, good)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _strip_control_chars(text)

    # Normalize horizontal spacing while preserving paragraph structure.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
