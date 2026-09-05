#!/usr/bin/env python3
"""
export_hub.py — turn the pipeline's output/ folder into the data the website reads.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from language_gate import DEVANAGARI, check_hub_export, scrub_devanagari
from resolve_entities import lookup, run_resolution

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

FRAMES_PER_REEL = 6
NO_SPEECH_MAX = float(os.getenv("NO_SPEECH_MAX", "0.6"))
AVG_LOGPROB_MIN = float(os.getenv("AVG_LOGPROB_MIN", "-0.8"))
COMPRESSION_MAX = float(os.getenv("COMPRESSION_MAX", "2.4"))
MIN_COLLECTION_REELS = 6

TYPE_LABELS = {
    "recipe": "Recipe",
    "workout": "Workout",
    "session_log": "Session",
    "form_fix": "Form fix",
    "nutrition_note": "Nutrition",
    "product_rank": "Products",
    "mindset": "Mindset",
    "other": "Other",
}

CATEGORY_LABELS = {
    "footwear": "Shoes",
    "food_drink": "Food & drink",
    "app_service": "Apps",
    "supplement": "Supplements",
    "race_event": "Races",
    "appliance": "Kitchen kit",
    "wearable_tech": "Wearables",
    "gym_studio": "Gyms & studios",
    "apparel": "Clothing",
    "equipment": "Gym kit",
    "other": "Other",
}

COLLECTIONS = [
    ("cook-high-protein", "High-protein cooking",
     "Dinners rebuilt to land 25–35 g of protein without becoming a chore.",
     {"types": ["recipe"], "tags": ["high-protein"]}),
    ("copy-this-session", "Copy this session",
     "Sessions written out in full — exercises, sets, reps, kit needed.",
     {"types": ["workout"]}),
    ("fix-your-form", "Fix your form",
     "The mistake, then the correction, for lifts people get wrong.",
     {"types": ["form_fix"]}),
    ("inside-a-training-block", "Inside a training block",
     "Real sessions with real numbers, including the ones that went badly.",
     {"types": ["session_log"]}),
    ("what-to-actually-buy", "What to actually buy",
     "Verdicts, rankings and the products they pay for themselves.",
     {"types": ["product_rank"], "hasEndorsed": True}),
    ("eat-better-without-cooking", "Eat better without cooking",
     "Swaps, principles and label-reading — no recipe required.",
     {"types": ["nutrition_note"]}),
    ("quick-wins", "Under 15 minutes",
     "Everything here is done inside a quarter of an hour.",
     {"maxMinutes": 15}),
]

ROLE_RANK = {"endorsed": 3, "used": 2, "incidental": 1}

_DIM, _BLD, _GRN, _YEL, _RST = "\033[2m", "\033[1m", "\033[32m", "\033[33m", "\033[0m"


def info(m: str) -> None:
    print(f"  {m}")


def ok(m: str) -> None:
    print(f"  {_GRN}✓{_RST} {m}")


def warn(m: str) -> None:
    print(f"  {_YEL}!{_RST} {m}")


def load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def trusted_transcript(reel_dir: Path) -> str:
    data = load_json(reel_dir / "transcript.json")
    if not isinstance(data, dict):
        return ""
    segs = data.get("segments")
    if not isinstance(segs, list):
        return (data.get("text") or "").strip()
    kept = []
    for s in segs:
        if not isinstance(s, dict):
            continue
        if float(s.get("no_speech_prob", 0) or 0) > NO_SPEECH_MAX:
            continue
        if float(s.get("avg_logprob", 0) or 0) < AVG_LOGPROB_MIN:
            continue
        if float(s.get("compression_ratio", 0) or 0) > COMPRESSION_MAX:
            continue
        t = (s.get("text") or "").strip()
        if t:
            kept.append(t)
    return " ".join(kept).strip()


def pick_frames(frames_dir: Path, n: int = FRAMES_PER_REEL) -> list[Path]:
    if not frames_dir.is_dir():
        return []
    files = sorted(p for p in frames_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(files) <= n:
        return files
    step = (len(files) - 1) / (n - 1) if n > 1 else 0
    idx = sorted({int(round(i * step)) for i in range(n)})
    return [files[i] for i in idx]


def creator_display_name(creator_dir: Path, username: str) -> str:
    scrape = load_json(creator_dir / "_cache" / "scrape.json")
    if isinstance(scrape, list):
        for it in scrape:
            if isinstance(it, dict) and it.get("ownerFullName"):
                return str(it["ownerFullName"]).strip()
    return username


def _first_token(val: str) -> str:
    return (val or "").strip().split()[0] if val else ""


def _norm_claims(raw) -> list[dict]:
    out = []
    for c in raw or []:
        if isinstance(c, str):
            out.append({"text": c, "kind": "fact", "source": "frames", "confidence": 0.5})
        elif isinstance(c, dict) and c.get("text"):
            out.append(c)
    return out


def _sanitize_display_str(s: str | None) -> str | None:
    """Drop display strings that still contain Devanagari after pass 2."""
    if not s:
        return None
    if DEVANAGARI.search(s):
        return None
    return s


def _graphics_list(raw) -> list[dict]:
    out = []
    for g in raw or []:
        if isinstance(g, str):
            text_en = _sanitize_display_str(g)
            out.append({"text_en": text_en or g, "verbatim": g, "is_translation": False})
        elif isinstance(g, dict):
            text_en = _sanitize_display_str(g.get("text_en") or g.get("text") or "")
            out.append({
                "text_en": text_en or "",
                "verbatim": g.get("verbatim") or g.get("text_en") or "",
                "is_translation": bool(g.get("is_translation")),
            })
    return out


def _quotes_list(raw) -> list[dict]:
    out = []
    for q in raw or []:
        if isinstance(q, str):
            text_en = _sanitize_display_str(q)
            out.append({"text_en": text_en or q, "verbatim": q, "is_translation": False})
        elif isinstance(q, dict):
            text_en = _sanitize_display_str(q.get("text_en") or q.get("text") or "")
            out.append({
                "text_en": text_en or "",
                "verbatim": q.get("verbatim") or "",
                "is_translation": bool(q.get("is_translation")),
            })
    return out


def _payload_completeness(g: dict) -> str:
    ct = g.get("content_type") or "other"
    payload = g.get("payload") or {}
    if isinstance(payload, dict) and ct in payload:
        inner = payload[ct]
        if isinstance(inner, dict):
            return inner.get("completeness") or "partial"
    return "partial"


def _normalize_block_data(block: dict) -> dict:
    """Coerce pass-2 blocks when the model drifts from schema."""
    b = dict(block)
    kind = b.get("kind")
    data = b.get("data")
    if kind == "ingredients":
        if isinstance(data, list):
            b["data"] = {
                "items": [
                    {
                        "item": (x.get("item") or x.get("name") or ""),
                        "quantity": str(x.get("quantity") or ""),
                        "note": x.get("note") or x.get("notes") or "",
                    }
                    for x in data if isinstance(x, dict)
                ]
            }
        elif isinstance(data, dict) and "items" not in data:
            if "name" in data or "item" in data:
                b["data"] = {"items": [data]}
    elif kind in ("steps", "list"):
        if isinstance(data, list):
            b["data"] = {"items": data, "ordered": kind == "steps"}
    elif kind == "prose":
        if isinstance(data, str):
            b["data"] = {"text": data}
        elif isinstance(data, list):
            title = (b.get("title") or "").lower()
            if "instruction" in title or "step" in title or "method" in title:
                b["kind"] = "steps"
                b["data"] = {"items": [str(x) for x in data], "ordered": True}
            else:
                b["data"] = {"text": "\n".join(str(x) for x in data)}
    elif kind == "quote":
        if isinstance(data, str):
            b["data"] = {"text": data, "attribution": ""}
    elif kind in ("stats", "ranked", "pairs") and isinstance(data, list):
        b["data"] = {"items": data}
    elif kind == "table" and isinstance(data, dict) and "rows" not in data:
        b["data"] = {"columns": [], "rows": []}
    if not isinstance(b.get("data"), dict):
        b["data"] = {}
    return b


def _drop_empty_blocks(blocks: list) -> list:
    kept = []
    for b in blocks or []:
        if not isinstance(b, dict):
            continue
        data = b.get("data")
        if not isinstance(data, dict):
            continue
        kind = b.get("kind")
        empty = False
        if kind in ("steps", "list"):
            empty = not (data.get("items") or [])
        elif kind == "ingredients":
            empty = not (data.get("items") or [])
        elif kind == "table":
            empty = not (data.get("rows") or [])
        elif kind in ("pairs", "ranked", "stats"):
            empty = not (data.get("items") or [])
        elif kind == "prose":
            empty = not (data.get("text") or "").strip()
        elif kind == "quote":
            empty = not (data.get("text") or "").strip()
        if not empty:
            kept.append(b)
    return kept


def _takeaway(reader: dict, g: dict) -> str:
    for val in (reader.get("takeaway"), g.get("one_line_summary"), reader.get("title")):
        if val and str(val).strip():
            t = str(val).strip()
            if t != g.get("bucket_reason"):
                return t
    return ""


def build_reel(
    username: str,
    r: dict,
    creator_dir: Path,
    pub: Path,
    alias_map: dict,
    reader: dict | None,
) -> dict | None:
    if r.get("status") != "ok":
        return None
    g = r.get("reasoning") or {}
    if not g:
        return None
    reader = reader or r.get("reader") or {}

    code = r.get("shortcode") or r.get("id")
    if not code:
        return None

    reel_dir = creator_dir / "reels" / str(code)

    thumb_rel = ""
    src_thumb = creator_dir / "_cache" / "thumbs" / f"{code}.jpg"
    if src_thumb.is_file():
        dst = pub / "assets" / "thumbs" / f"{code}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_thumb, dst)
        thumb_rel = f"assets/thumbs/{code}.jpg"

    frame_rels: list[str] = []
    chosen = pick_frames(reel_dir / "frames")
    if chosen:
        fdst = pub / "assets" / "frames" / str(code)
        fdst.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(chosen, start=1):
            shutil.copyfile(src, fdst / f"f{i}.jpg")
            frame_rels.append(f"assets/frames/{code}/f{i}.jpg")

    cls = dict(r.get("classification") or {})
    shortlist = {k: cls.get(k) for k in ("reason", "confidence", "signal", "subject")
                 if cls.get(k) is not None}

    ct = g.get("content_type") or "other"
    claims = _norm_claims(g.get("key_claims"))
    advice = [c for c in claims if c.get("kind") == "advice"]
    facts = [c for c in claims if c.get("kind") == "fact"]
    personal = [c for c in claims if c.get("kind") == "personal"]

    brands_out = []
    reader_products = {p.get("name", "").lower(): p for p in reader.get("products") or [] if isinstance(p, dict)}
    for b in g.get("brands_or_products_visible") or []:
        if not isinstance(b, dict):
            continue
        raw_name = (b.get("name") or "").strip()
        if not raw_name:
            continue
        ent = lookup(alias_map, raw_name)
        if not ent.get("keep", True):
            continue
        canonical = ent.get("canonical") or raw_name
        cat = b.get("category") or ent.get("category") or "other"
        rp = reader_products.get(raw_name.lower()) or reader_products.get(canonical.lower()) or {}
        brands_out.append({
            "name": canonical,
            "raw": raw_name,
            "category": cat,
            "categoryLabel": CATEGORY_LABELS.get(cat, cat.replace("_", " ").title()),
            "role": b.get("role") or "incidental",
            "usage": rp.get("usage") or b.get("usage") or "",
            "verdict": rp.get("verdict"),
            "where": b.get("where"),
            "confidence": b.get("confidence", 0.5),
        })

    exercises = []
    for ex in g.get("exercises_shown") or []:
        if isinstance(ex, dict):
            exercises.append(ex)
        elif ex:
            exercises.append({"name": str(ex), "detail": ""})

    foods = []
    for f in g.get("food_or_supplements_shown") or []:
        if isinstance(f, dict):
            foods.append(f)
        elif f:
            foods.append({"name": str(f), "detail": ""})

    setting = _first_token(g.get("setting") or "")
    fmt = _first_token(g.get("content_format") or "")
    lang = (g.get("spoken_language") or "english").lower()
    raw_caption_en = reader.get("caption_en")
    raw_transcript_en = reader.get("transcript_en")
    caption_en = _sanitize_display_str(raw_caption_en)
    transcript_en = _sanitize_display_str(raw_transcript_en)
    if caption_en is None and raw_caption_en and DEVANAGARI.search(str(raw_caption_en)):
        warn(f"@{username}/{code}: dropped caption_en — still contains Devanagari")
    if transcript_en is None and raw_transcript_en and DEVANAGARI.search(str(raw_transcript_en)):
        warn(f"@{username}/{code}: dropped transcript_en — still contains Devanagari")
    needs_translation = lang not in ("english", "none") and (
        caption_en is not None or transcript_en is not None
    )

    blocks = _drop_empty_blocks([
        _normalize_block_data(b) for b in (reader.get("blocks") or []) if isinstance(b, dict)
    ])

    dur = r.get("duration_sec")
    quality = g.get("content_quality") or {}
    if isinstance(quality, dict):
        for k in ("production", "information_density", "watchability"):
            v = quality.get(k)
            if isinstance(v, str):
                quality[k] = {"low": 0.3, "medium": 0.6, "high": 0.9}.get(v.lower(), 0.5)

    return {
        "id": code,
        "creator": username,
        "url": r.get("url"),
        "caption": r.get("caption") or "",
        "hashtags": r.get("hashtags") or [],
        "timestamp": r.get("timestamp"),
        "views": r.get("views"),
        "likes": r.get("likes"),
        "comments": r.get("comments"),
        "duration": int(round(float(dur))) if isinstance(dur, (int, float)) else 0,
        "thumb": thumb_rel,
        "frames": frame_rels,
        "title": reader.get("title") or g.get("one_line_summary") or "",
        "takeaway": _takeaway(reader, g),
        "whyItMatters": reader.get("why_it_matters"),
        "effort": reader.get("effort") or {"label": "", "minutes": None},
        "blocks": blocks,
        "watchBecause": reader.get("watch_it_because"),
        "tags": reader.get("tags") or [],
        "searchText": reader.get("search_text") or "",
        "readerConfidence": reader.get("confidence") or "medium",
        "type": ct,
        "typeLabel": TYPE_LABELS.get(ct, ct.replace("_", " ").title()),
        "typeConfidence": g.get("content_type_confidence"),
        "payload": g.get("payload") or {},
        "completeness": _payload_completeness(g),
        "claims": claims,
        "advice": advice,
        "facts": facts,
        "personal": personal,
        "brands": brands_out,
        "exercises": exercises,
        "foods": foods,
        "equipment": g.get("equipment_visible") or [],
        "metrics": g.get("metrics") or [],
        "summary": g.get("one_line_summary"),
        "detail": g.get("detailed_summary"),
        "topic": g.get("primary_topic"),
        "subTopics": g.get("sub_topics") or [],
        "setting": setting or "other",
        "settingNote": g.get("setting_note"),
        "transcript": trusted_transcript(reel_dir),
        "graphics": _graphics_list(g.get("graphics_text")),
        "quotes": _quotes_list(g.get("notable_quotes")),
        "captionEn": caption_en,
        "transcriptEn": transcript_en,
        "needsTranslation": needs_translation,
        "language": lang,
        "audience": g.get("target_audience"),
        "cta": g.get("call_to_action"),
        "ctaVerbatim": g.get("call_to_action_verbatim"),
        "craft": {
            "hook": g.get("hook"),
            "hookTechnique": g.get("hook_technique"),
            "format": fmt or "other",
            "tone": g.get("tone") or [],
            "quality": quality,
            "views": r.get("views"),
            "likes": r.get("likes"),
        },
        "provenance": {
            "framesUsed": r.get("frames_used"),
            "engine": r.get("engine"),
            "evidence": g.get("evidence") or {},
            "uncertainties": g.get("uncertainties") or [],
            "shortlist": shortlist,
            "bucket": r.get("video_bucket") or cls.get("bucket"),
            "bucketReason": r.get("video_bucket_reason") or g.get("bucket_reason"),
            "subtitles": g.get("subtitle_text") or [],
        },
        # legacy flat fields for graceful degradation
        "hook": g.get("hook"),
        "format": fmt or "other",
        "tone": g.get("tone") or [],
        "framesUsed": r.get("frames_used"),
        "engine": r.get("engine"),
        "bucket": r.get("video_bucket") or cls.get("bucket"),
        "bucketReason": r.get("video_bucket_reason") or g.get("bucket_reason"),
        "evidence": g.get("evidence") or {},
        "uncertainties": g.get("uncertainties") or [],
        "shortlist": shortlist,
        "quality": quality,
    }


def _matches_collection(reel: dict, rule: dict) -> bool:
    if rule.get("types") and reel.get("type") not in rule["types"]:
        return False
    if rule.get("tags"):
        tags = set(reel.get("tags") or [])
        if not any(t in tags for t in rule["tags"]):
            return False
    if rule.get("hasEndorsed"):
        if not any(b.get("role") == "endorsed" for b in reel.get("brands") or []):
            if reel.get("type") != "product_rank":
                return False
    if rule.get("maxMinutes") is not None:
        mins = (reel.get("effort") or {}).get("minutes")
        if mins is None or mins > rule["maxMinutes"]:
            return False
    return True


def build_collections(reels: list[dict]) -> list[dict]:
    out = []
    for slug, title, blurb, rule in COLLECTIONS:
        ids = [r["id"] for r in reels if _matches_collection(r, rule)]
        if len(ids) < MIN_COLLECTION_REELS:
            continue
        out.append({
            "slug": slug,
            "title": title,
            "blurb": blurb,
            "rule": rule,
            "reelIds": ids,
            "count": len(ids),
        })
    return out


def build_playbook(reels: list[dict]) -> list[dict]:
    out = []
    for r in reels:
        reader_path_data = r.get("_reader_playbook")
        entries = reader_path_data if reader_path_data is not None else []
        for e in entries:
            if not isinstance(e, dict) or not e.get("text"):
                continue
            out.append({
                **e,
                "reelId": r["id"],
                "creator": r["creator"],
                "date": (r.get("timestamp") or "")[:10],
            })
    return out


def _product_rank_lookup(reels: list[dict]) -> dict[str, dict]:
    """Map product name (lower) → rank info from product_rank payloads."""
    out: dict[str, dict] = {}
    for r in reels:
        if r.get("type") != "product_rank":
            continue
        payload = (r.get("payload") or {}).get("product_rank") or {}
        metric = payload.get("metric") or ""
        for item in payload.get("items") or []:
            if not isinstance(item, dict):
                continue
            name = (item.get("name") or "").strip()
            if name:
                out[name.lower()] = {
                    "rank": item.get("rank"),
                    "value": item.get("value"),
                    "metric": metric,
                    "creator": r.get("creator"),
                }
    return out


def build_products_index(reels: list[dict]) -> list[dict]:
    rank_lookup = _product_rank_lookup(reels)
    products: dict[str, dict] = {}
    for r in reels:
        for b in r.get("brands") or []:
            name = b.get("name")
            if not name:
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
            ent = products.setdefault(slug, {
                "name": name,
                "slug": slug,
                "category": b.get("category", "other"),
                "categoryLabel": b.get("categoryLabel", "Other"),
                "aliases": set(),
                "count": 0,
                "creators": set(),
                "sightings": [],
                "verdictCount": 0,
                "topRole": "incidental",
            })
            if b.get("raw") and b["raw"] != name:
                ent["aliases"].add(b["raw"])
            ent["count"] += 1
            ent["creators"].add(r["creator"])
            role = b.get("role") or "incidental"
            if ROLE_RANK.get(role, 0) > ROLE_RANK.get(ent["topRole"], 0):
                ent["topRole"] = role
            if b.get("verdict"):
                ent["verdictCount"] += 1
            ent["sightings"].append({
                "reelId": r["id"],
                "creator": r["creator"],
                "date": (r.get("timestamp") or "")[:10],
                "role": role,
                "usage": b.get("usage") or "",
                "verdict": b.get("verdict"),
            })
            rk = rank_lookup.get(name.lower()) or rank_lookup.get((b.get("raw") or "").lower())
            if rk and not ent.get("rankNote"):
                ent["rankNote"] = (
                    f"#{rk['rank']} for {rk['metric']} — @{rk['creator']}"
                    if rk.get("rank") else ""
                )

    out = []
    for ent in products.values():
        ent["aliases"] = sorted(ent["aliases"])
        ent["creators"] = sorted(ent["creators"])
        out.append(ent)
    out.sort(key=lambda p: (-p["count"], p["name"]))
    return out


def generate_digest(creator_dir: Path, username: str, reels: list[dict]) -> dict:
    digest_path = creator_dir / "digest.json"
    reel_count = len(reels)
    cached = load_json(digest_path)
    if isinstance(cached, dict) and cached.get("_reel_count") == reel_count:
        return {k: v for k, v in cached.items() if not str(k).startswith("_")}

    from videolens import openai_chat, parse_json_blob, OPENAI_TEXT_MODELS

    type_counts = Counter(r.get("type") for r in reels)
    sample = "\n".join(
        f"- {r.get('title') or r.get('summary')}: {r.get('takeaway')}"
        for r in sorted(reels, key=lambda x: x.get("views") or 0, reverse=True)[:8]
    )
    prompt = (
        f"Creator @{username} ({reel_count} analysed reels).\n"
        f"Type mix: {dict(type_counts)}\n"
        f"Sample takeaways:\n{sample}\n\n"
        "Return JSON: good_for (<=120 chars), specialties (<=4 strings), "
        "not_for (<=100 chars or null), best_reel_ids (3 ids), "
        "cadence (posts most days | a few times a week | occasional)."
    )
    system = (
        "Describe what following this fitness/nutrition creator gets a reader. "
        "No hedging. English only. Return one JSON object."
    )
    try:
        raw = openai_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            json_mode=True,
            max_tokens=500,
            temperature=0.3,
            models=OPENAI_TEXT_MODELS,
        )
        digest = parse_json_blob(raw)
    except Exception as e:
        warn(f"digest for @{username} failed: {e}")
        digest = {
            "good_for": f"{reel_count} reels on training and nutrition.",
            "specialties": [],
            "not_for": None,
            "best_reel_ids": [r["id"] for r in reels[:3]],
            "cadence": "occasional",
        }
    digest["_reel_count"] = reel_count
    digest_path.write_text(json.dumps(digest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {k: v for k, v in digest.items() if not str(k).startswith("_")}


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild the website's data from pipeline output.")
    ap.add_argument("--out", default=str(ROOT / "output"))
    ap.add_argument("--public", default=str(ROOT / "public"))
    ap.add_argument("--reresolve", action="store_true", help="Re-resolve all entity strings")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    pub = Path(args.public).resolve()
    review_path = out / "_language_review.txt"

    print(f"\n{_BLD}Exporting the archive{_RST}  {_DIM}{out} → {pub}{_RST}\n")

    if not out.is_dir():
        warn(f"no output folder at {out}")
        return 1

    creator_dirs = sorted(d for d in out.iterdir()
                          if d.is_dir() and (d / "run.json").is_file())
    if not creator_dirs:
        warn("no creator folders with run.json")
        return 1

    # Collect reasoning for entity resolution
    reasoning_rows: list[tuple[str, dict]] = []
    for cdir in creator_dirs:
        run = load_json(cdir / "run.json")
        if not run:
            continue
        for r in run.get("reels") or []:
            if r.get("status") == "ok" and r.get("reasoning"):
                code = r.get("shortcode") or r.get("id")
                reasoning_rows.append((str(code), r["reasoning"]))

    alias_map = run_resolution(
        reasoning_rows,
        out / "_entities.json",
        reresolve=args.reresolve,
    )

    creators: list[dict] = []
    reels: list[dict] = []
    playbook_entries: list[dict] = []

    for cdir in creator_dirs:
        run = load_json(cdir / "run.json")
        if not run:
            continue

        username = run.get("username") or cdir.name
        counts = dict(run.get("counts") or {})
        settings = run.get("settings") or {}

        mine = []
        for r in run.get("reels") or []:
            code = r.get("shortcode") or r.get("id")
            reader = None
            if code:
                reader = load_json(cdir / "reels" / str(code) / "reader.json")
                if reader and isinstance(reader.get("playbook"), list):
                    r = dict(r)
                    r["_reader_playbook"] = reader["playbook"]
            built = build_reel(username, r, cdir, pub, alias_map, reader)
            if built:
                if reader and isinstance(reader.get("playbook"), list):
                    built["_reader_playbook"] = reader["playbook"]
                mine.append(built)

        scraped, dropped = counts.get("scraped"), counts.get("dropped")
        if isinstance(scraped, int) and isinstance(dropped, int):
            counts["kept"] = scraped - dropped
        else:
            counts.setdefault("kept", counts.get("analysed", len(mine)))
        counts["analysed"] = counts.get("analysed", len(mine))
        counts["published"] = len(mine)
        counts["byType"] = dict(Counter(r.get("type") for r in mine))

        digest = generate_digest(cdir, username, mine) if mine else {}

        creators.append({
            "username": username,
            "name": creator_display_name(cdir, username),
            "generatedAt": run.get("generated_at"),
            "engine": f"{settings.get('vision_provider', 'openai')} / "
                      f"{settings.get('vision_model', 'gpt-4.1-mini')}",
            "counts": counts,
            "buckets": settings.get("buckets") or ["fitness", "nutrition"],
            "mode": settings.get("mode", "top-n"),
            "droppedSample": [
                {"shortcode": d.get("shortcode"), "reason": d.get("reason"),
                 "confidence": d.get("confidence")}
                for d in (run.get("dropped_reels") or [])[:6]
            ],
            "digest": digest,
        })
        reels.extend(mine)
        for r in mine:
            for e in r.get("_reader_playbook") or []:
                if isinstance(e, dict) and e.get("text"):
                    playbook_entries.append({
                        **e,
                        "reelId": r["id"],
                        "creator": r["creator"],
                        "date": (r.get("timestamp") or "")[:10],
                    })

        ok(f"@{username}: {len(mine)} reel(s) published")

    reels.sort(key=lambda r: r.get("timestamp") or "", reverse=True)

    # Strip internal keys
    for r in reels:
        r.pop("_reader_playbook", None)

    collections = build_collections(reels)
    products = build_products_index(reels)
    playbook = playbook_entries

    payload = {
        "generatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "creators": creators,
        "reels": reels,
        "playbook": playbook,
        "products": products,
        "collections": collections,
    }

    dropped = scrub_devanagari(reels)
    if dropped:
        for line in dropped:
            warn(f"Dropped Devanagari: {line}")
        warn(f"dropped {len(dropped)} non-English evidence string(s)")

    dev_fail, roman_warn = check_hub_export(reels, review_path)
    if roman_warn:
        warn(f"{len(roman_warn)} romanised-Hindi warning(s) — see {review_path}")
    if dev_fail:
        for line in dev_fail[:20]:
            warn(f"Devanagari: {line}")
        warn(f"export failed: {len(dev_fail)} Devanagari string(s) outside allowed paths")
        return 1

    chat_rows = [{
        "id": r["id"],
        "creator": r["creator"],
        "date": (r.get("timestamp") or "")[:10],
        "type": r.get("type"),
        "title": r.get("title"),
        "takeaway": r.get("takeaway"),
        "searchText": r.get("searchText"),
        "tags": r.get("tags"),
        "url": r.get("url"),
    } for r in reels]

    live = {r["id"] for r in reels}
    for d in (pub / "assets" / "frames").glob("*"):
        if d.is_dir() and d.name not in live:
            shutil.rmtree(d, ignore_errors=True)
    for f in (pub / "assets" / "thumbs").glob("*.jpg"):
        if f.stem not in live:
            f.unlink(missing_ok=True)

    (pub / "data").mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, ensure_ascii=False, indent=1)
    (pub / "data" / "hub.json").write_text(blob, encoding="utf-8")
    (pub / "hub-data.js").write_text(f"window.HUB = {blob};\n", encoding="utf-8")
    (pub / "data" / "hub-chat.json").write_text(
        json.dumps(chat_rows, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    ok(f"{len(reels)} reels, {len(playbook)} playbook entries, "
       f"{len(products)} products, {len(collections)} collections")
    ok("wrote public/hub-data.js, hub.json, hub-chat.json")
    print(f"\n  {_DIM}Preview: npm run serve{_RST}\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
