#!/usr/bin/env python3
"""
export_hub.py — turn the pipeline's output/ folder into the data the website reads.

The pipeline writes one folder per creator under output/, full of run.json files,
frames, thumbnails and transcripts. The website wants a single flat payload plus a
small set of images. This script is the bridge, and it is the only place that
mapping lives.

    python3 pipeline/export_hub.py

It reads   output/<creator>/run.json          (and _cache/, reels/ beside it)
and writes public/hub-data.js                 (what index.html loads)
           public/data/hub.json               (same payload, for /api/chat)
           public/assets/thumbs/<code>.jpg
           public/assets/frames/<code>/f*.jpg

Safe to re-run: it rebuilds the payload from whatever is on disk and prunes images
belonging to reels that are no longer present.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# How many stills to publish per reel. The pipeline may sample up to 20 for the
# model; the page only ever shows a strip, so we publish an evenly spaced subset
# and keep the repository small.
FRAMES_PER_REEL = 6

# Same trust thresholds the pipeline uses before it shows Whisper's output to the
# reasoning model. Re-applied here so the transcript on the page is the same
# filtered text the model actually saw — never the raw guesses over music.
NO_SPEECH_MAX = float(os.getenv("NO_SPEECH_MAX", "0.6"))
AVG_LOGPROB_MIN = float(os.getenv("AVG_LOGPROB_MIN", "-0.8"))
COMPRESSION_MAX = float(os.getenv("COMPRESSION_MAX", "2.4"))

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
    """Whisper's text with its own low-confidence guesses removed."""
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
    """An evenly spaced sample across the reel, so the strip reads as a timeline."""
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
    """The human name, which only the raw Apify scrape carries."""
    scrape = load_json(creator_dir / "_cache" / "scrape.json")
    if isinstance(scrape, list):
        for it in scrape:
            if isinstance(it, dict) and it.get("ownerFullName"):
                return str(it["ownerFullName"]).strip()
    return username


def build_reel(username: str, r: dict, creator_dir: Path, pub: Path) -> dict | None:
    """One analysed reel, flattened into exactly the shape index.html reads."""
    if r.get("status") != "ok":
        return None
    g = r.get("reasoning") or {}
    if not g:
        return None

    code = r.get("shortcode") or r.get("id")
    if not code:
        return None

    reel_dir = creator_dir / "reels" / str(code)

    # --- images ---------------------------------------------------------
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

    # The classifier's verdict, minus the bookkeeping the page has no use for.
    cls = dict(r.get("classification") or {})
    shortlist = {k: cls.get(k) for k in ("reason", "confidence", "signal", "subject")
                 if cls.get(k) is not None}

    dur = r.get("duration_sec")
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
        "framesUsed": r.get("frames_used"),
        "engine": r.get("engine"),
        "shortlist": shortlist,
        "bucket": r.get("video_bucket") or cls.get("bucket"),
        "bucketReason": r.get("video_bucket_reason"),
        # --- the model's read of the video -------------------------------
        "hook": g.get("hook"),
        "summary": g.get("one_line_summary"),
        "detail": g.get("detailed_summary"),
        "topic": g.get("primary_topic"),
        "subTopics": g.get("sub_topics") or [],
        "format": g.get("content_format"),
        "setting": g.get("setting"),
        "people": g.get("people_on_screen") or {},
        "exercises": g.get("exercises_shown") or [],
        "foods": g.get("food_or_supplements_shown") or [],
        "equipment": g.get("equipment_visible") or [],
        "brands": g.get("brands_or_products_visible") or [],
        "graphics": g.get("graphics_text") or [],
        "subtitles": g.get("subtitle_text") or [],
        "claims": g.get("key_claims") or [],
        "cta": g.get("call_to_action"),
        "audience": g.get("target_audience"),
        "tone": g.get("tone") or [],
        "language": g.get("spoken_language"),
        "quality": g.get("content_quality") or {},
        "quotes": g.get("notable_quotes") or [],
        "evidence": g.get("evidence") or {},
        "uncertainties": g.get("uncertainties") or [],
        "transcript": trusted_transcript(reel_dir),
        "thumb": thumb_rel,
        "frames": frame_rels,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild the website's data from the pipeline's output folder.")
    ap.add_argument("--out", default=str(ROOT / "output"),
                    help="Pipeline output directory (default: ./output)")
    ap.add_argument("--public", default=str(ROOT / "public"),
                    help="Website directory to write into (default: ./public)")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    pub = Path(args.public).resolve()

    print(f"\n{_BLD}Exporting the archive{_RST}  {_DIM}{out} → {pub}{_RST}\n")

    if not out.is_dir():
        warn(f"no output folder at {out}")
        warn("run the backfill first:  ./pipeline/backfill.sh")
        return 1

    creator_dirs = sorted(d for d in out.iterdir()
                          if d.is_dir() and (d / "run.json").is_file())
    if not creator_dirs:
        warn(f"no creator folders with a run.json inside {out}")
        return 1

    creators: list[dict] = []
    reels: list[dict] = []
    skipped: list[str] = []

    for cdir in creator_dirs:
        run = load_json(cdir / "run.json")
        if not run:
            warn(f"{cdir.name}: run.json is unreadable — skipped")
            skipped.append(cdir.name)
            continue

        username = run.get("username") or cdir.name
        counts = dict(run.get("counts") or {})
        settings = run.get("settings") or {}

        mine = []
        for r in run.get("reels") or []:
            built = build_reel(username, r, cdir, pub)
            if built:
                mine.append(built)

        # The page reads counts.kept as "how many of the scraped posts were
        # actually on topic". Older run.json files spend that number under
        # "fitness" and use "kept" for something else, which made one creator
        # read as 3-of-40 when 30 posts had survived the filter. Derive it from
        # the two counts every version agrees on instead: whatever was scraped
        # and not dropped at the screening step is the shortlist.
        scraped, dropped = counts.get("scraped"), counts.get("dropped")
        if isinstance(scraped, int) and isinstance(dropped, int):
            counts["kept"] = scraped - dropped
        else:
            counts.setdefault("kept", counts.get("analysed", len(mine)))
        counts["analysed"] = counts.get("analysed", len(mine))
        counts["published"] = len(mine)

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
        })
        reels.extend(mine)

        flag = "" if mine else f"  {_YEL}(nothing analysed yet){_RST}"
        ok(f"@{username}: {len(mine)} reel(s) published "
           f"of {counts.get('kept', '?')} shortlisted{flag}")

    # Newest first — the feed leads with the most recent post.
    reels.sort(key=lambda r: r.get("timestamp") or "", reverse=True)

    payload = {
        "generatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "creators": creators,
        "reels": reels,
    }

    # --- prune images belonging to reels that no longer exist -------------
    live = {r["id"] for r in reels}
    for d in (pub / "assets" / "frames").glob("*"):
        if d.is_dir() and d.name not in live:
            shutil.rmtree(d, ignore_errors=True)
    for f in (pub / "assets" / "thumbs").glob("*.jpg"):
        if f.stem not in live:
            f.unlink(missing_ok=True)

    # --- write -----------------------------------------------------------
    (pub / "data").mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, ensure_ascii=False, indent=1)
    (pub / "data" / "hub.json").write_text(blob, encoding="utf-8")
    (pub / "hub-data.js").write_text(f"window.HUB = {blob};\n", encoding="utf-8")

    n_thumbs = len(list((pub / "assets" / "thumbs").glob("*.jpg")))
    n_frames = sum(len(list(d.glob("*.jpg")))
                   for d in (pub / "assets" / "frames").glob("*") if d.is_dir())

    print()
    ok(f"{len(reels)} reels across {len(creators)} creators")
    ok(f"{n_thumbs} thumbnails, {n_frames} frames")
    ok("wrote public/hub-data.js and public/data/hub.json")
    if skipped:
        warn(f"skipped: {', '.join(skipped)}")
    if not reels:
        warn("the site will render empty — no reel has completed analysis yet")
        return 1
    print(f"\n  {_DIM}Preview it with:  npm run dev{_RST}\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
