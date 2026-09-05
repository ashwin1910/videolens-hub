"""English validation gate — §5.10."""

from __future__ import annotations

import re
from pathlib import Path

DEVANAGARI = re.compile(r"[\u0900-\u097F]")

VERBATIM_OK = {
    "caption",
    "transcript",
    "quotes[].verbatim",
    "graphics[].verbatim",
    "ctaVerbatim",
    "provenance.subtitles[]",
}

ROMAN_HINDI = {
    "karo", "karlo", "kar", "lo", "banao", "banega", "hai", "hain", "nahi",
    "nahin", "bhi", "kya", "kyu", "kyun", "toh", "bhasad", "jitna", "aur",
    "mein", "mai", "apna", "apne", "bana", "khao", "kha", "paoge", "wala",
    "wali", "thoda", "zyada", "accha", "matlab", "yaar", "chalo", "dekho",
}

FRAME_IN_USAGE = re.compile(r"\bframe\s*\d", re.I)


def _walk(obj, prefix: str = ""):
    if isinstance(obj, str):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            yield from _walk(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[]" if prefix else "[]"
            yield from _walk(v, p)


def _path_ok(path: str) -> bool:
    for ok in VERBATIM_OK:
        if path == ok or path.endswith("." + ok.split(".")[-1]):
            if ok.endswith("[]"):
                base = ok[:-3]
                if path.startswith(base):
                    return True
            elif path == ok:
                return True
    if path in ("caption", "transcript"):
        return True
    if ".verbatim" in path:
        return True
    if path.startswith("provenance.subtitles"):
        return True
    return False


def roman_hindi_hits(text: str) -> int:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return sum(1 for w in words if w in ROMAN_HINDI)


def check_pass1_devanagari(data: dict) -> list[str]:
    """Warn paths with Devanagari outside verbatim slots."""
    hits = []
    verbatim_paths = (
        "notable_quotes", "graphics_text", "call_to_action_verbatim",
    )
    for path, s in _walk(data):
        if not s or DEVANAGARI.search(s) is None:
            continue
        if any(vp in path for vp in verbatim_paths) and path.endswith("verbatim"):
            continue
        if path in ("call_to_action_verbatim",):
            continue
        hits.append(path)
    return hits


def check_hub_export(reels: list[dict], review_path: Path | None = None) -> tuple[list[str], list[str]]:
    """Returns (devanagari_failures, roman_hindi_warnings)."""
    dev_fail: list[str] = []
    roman_warn: list[str] = []
    for reel in reels:
        rid = reel.get("id", "?")
        for path, s in _walk(reel):
            if not s or not isinstance(s, str):
                continue
            if _path_ok(path):
                continue
            if DEVANAGARI.search(s):
                dev_fail.append(f"{rid}: {path}")
                continue
            if roman_hindi_hits(s) >= 2:
                roman_warn.append(f"{rid}: {path}")
    if review_path and roman_warn:
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text("\n".join(roman_warn) + "\n", encoding="utf-8")
    elif review_path and review_path.is_file():
        review_path.write_text("", encoding="utf-8")
    return dev_fail, roman_warn


def _scrub_list(lst: list, path: str, rid: str, dropped: list[str]) -> None:
    list_path = f"{path}[]" if path else "[]"
    i = 0
    while i < len(lst):
        item = lst[i]
        if isinstance(item, str):
            if DEVANAGARI.search(item) and not _path_ok(list_path):
                dropped.append(f"{rid}: {list_path}")
                lst.pop(i)
                continue
        elif isinstance(item, dict):
            _scrub_dict(item, list_path, rid, dropped)
        elif isinstance(item, list):
            _scrub_list(item, list_path, rid, dropped)
        i += 1


def _scrub_dict(obj: dict, path: str, rid: str, dropped: list[str]) -> None:
    for key, val in list(obj.items()):
        child_path = f"{path}.{key}" if path else key
        if isinstance(val, list):
            _scrub_list(val, child_path, rid, dropped)
        elif isinstance(val, dict):
            _scrub_dict(val, child_path, rid, dropped)


def scrub_devanagari(reels: list[dict]) -> list[str]:
    """Drop Devanagari strings that appear as list items on non-verbatim paths."""
    dropped: list[str] = []
    for reel in reels:
        rid = reel.get("id", "?")
        _scrub_dict(reel, "", rid, dropped)
    return dropped
