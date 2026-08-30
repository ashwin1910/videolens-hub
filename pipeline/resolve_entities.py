"""Pass 2b — incremental entity resolution (§5.8)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from videolens import openai_chat, parse_json_blob, OPENAI_TEXT_MODELS, info, ok, warn

RESOLVE_SYSTEM = """You are given every distinct entity string extracted across
an archive of fitness and nutrition reels, with how many reels each appears in.
Some are the same thing spelled differently. Some are not entities at all — OCR
fragments, single letters, stray numbers, units, generic words.

Return ONE JSON object:

{
  "entities": [
    {"canonical": "The Func. Lab",
     "aliases": ["The Func Lab", "Func Lab", "THE FUNC.LAB", "Func. Lab"],
     "kind": "brand|product|ingredient|exercise|topic",
     "category": "footwear|apparel|supplement|food_drink|equipment|appliance|
                  wearable_tech|app_service|gym_studio|race_event|other",
     "keep": true}
  ]
}

Set keep:false for: single characters, bare numbers, units (g, ml, kcal), OCR
noise, and generic words that are not a specific entity ("protein", "food",
"gym", "workout"). Every input string must appear exactly once, either as a
canonical or inside exactly one aliases array. Prefer the creator's own most
complete spelling as the canonical, with normal capitalisation. Canonical names
are in English or are proper nouns; never return a canonical in Devanagari."""


def _slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "unknown"


def load_map(path: Path) -> dict:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "alias_lower" in data:
            return data["alias_lower"]
        if isinstance(data, dict):
            return data
    return {}


def save_map(path: Path, alias_map: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"alias_lower": alias_map}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def collect_strings(reels_reasoning: list[tuple[str, dict]]) -> dict[str, int]:
    """reels_reasoning: list of (reel_id, pass1 reasoning dict)."""
    counts: dict[str, int] = {}
    seen_per_reel: set[str] = set()

    def bump(s: str, reel_id: str) -> None:
        s = (s or "").strip()
        if not s:
            return
        key = reel_id + "::" + s.lower()
        if key in seen_per_reel:
            return
        seen_per_reel.add(key)
        counts[s] = counts.get(s, 0) + 1

    for reel_id, g in reels_reasoning:
        seen_per_reel.clear()
        for b in g.get("brands_or_products_visible") or []:
            if isinstance(b, dict):
                bump(b.get("name", ""), reel_id)
        for t in g.get("sub_topics") or []:
            bump(str(t), reel_id)
        payload = g.get("payload") or {}
        if isinstance(payload, dict):
            for pval in payload.values():
                if not isinstance(pval, dict):
                    continue
                for ing in pval.get("ingredients") or []:
                    if isinstance(ing, dict):
                        bump(ing.get("item", ""), reel_id)
        for ex in g.get("exercises_shown") or []:
            if isinstance(ex, dict):
                bump(ex.get("name", ""), reel_id)
            else:
                bump(str(ex), reel_id)
    return counts


def resolve_new_strings(
    unseen: list[str],
    existing_canonicals: list[str],
    *,
    force: bool = False,
) -> dict:
    if not unseen:
        return {"entities": []}
    user = (
        "Existing canonical entities (merge into one of these when appropriate):\n"
        + json.dumps(existing_canonicals[:200], ensure_ascii=False)
        + "\n\nNew strings to resolve (each must appear exactly once in output):\n"
        + json.dumps(unseen, ensure_ascii=False, indent=1)
    )
    raw = openai_chat(
        [{"role": "system", "content": RESOLVE_SYSTEM}, {"role": "user", "content": user}],
        json_mode=True,
        max_tokens=4000,
        temperature=0.1,
        models=OPENAI_TEXT_MODELS,
    )
    return parse_json_blob(raw)


def merge_entities(alias_map: dict, result: dict) -> None:
    for ent in result.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        canonical = (ent.get("canonical") or "").strip()
        if not canonical:
            continue
        kind = ent.get("kind") or "brand"
        category = ent.get("category") or "other"
        keep = ent.get("keep", True)
        entry = {
            "canonical": canonical,
            "kind": kind,
            "category": category,
            "keep": keep,
        }
        alias_map[canonical.lower()] = entry
        for alias in [canonical] + list(ent.get("aliases") or []):
            a = (alias or "").strip()
            if a:
                alias_map[a.lower()] = entry


def lookup(alias_map: dict, raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        return {"canonical": raw, "kind": "brand", "category": "other", "keep": True}
    hit = alias_map.get(raw.lower())
    if hit:
        return hit
    return {"canonical": raw, "kind": "brand", "category": "other", "keep": True}


def run_resolution(
    reels_reasoning: list[tuple[str, dict]],
    map_path: Path,
    *,
    reresolve: bool = False,
) -> dict:
    counts = collect_strings(reels_reasoning)
    alias_map = {} if reresolve else load_map(map_path)
    existing = sorted({v["canonical"] for v in alias_map.values() if v.get("canonical")})
    unseen = [s for s in counts if s.lower() not in alias_map]
    if unseen:
        info(f"resolving {len(unseen)} new entity string(s)…")
        result = resolve_new_strings(unseen, existing)
        merge_entities(alias_map, result)
        save_map(map_path, alias_map)
        ok(f"entity map now covers {len(alias_map)} alias(es)")
    else:
        info("entity map up to date")
    return alias_map
