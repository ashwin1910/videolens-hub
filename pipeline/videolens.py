#!/usr/bin/env python3
"""
videolens.py — Creator reel → video-reasoning POC pipeline
==========================================================

ONE COMMAND, SIX STAGES:

  1. SCRAPE     Apify `apify/instagram-reel-scraper` → all recent reels for a
                creator, with captions, hashtags, stats and media URLs.
  2. CLASSIFY   OpenAI reads every caption + hashtag set AND looks at each
                reel's thumbnail, then labels it fitness / not-fitness with a
                topic, confidence and reason. The thumbnail matters: endurance
                creators post real training footage under captions like
                "1st session in the bag", which caption-only triage throws away.
                Comedy, filler, memes and promos get dropped here, before any
                video is downloaded.
  3. DOWNLOAD   The shortlisted reels' `videoUrl` (straight from the Apify
                dataset) is pulled down as an .mp4.
  4. SAMPLE     ffmpeg pulls N evenly-spaced key frames (default 12) and a
                small mono audio track. Cheap — no full-video upload.
  5. LISTEN     OpenAI Whisper transcribes the audio (Hinglish-friendly), with
                the classic Whisper repeat-loop collapsed.
  6. REASON     A vision model gets the frames + the transcript and returns a
                strict JSON read of the video. Two interchangeable backends:
                  --vision openai    (default) OpenAI vision, per the cookbook
                                     "video = an ordered list of base64 stills"
                  --vision seedance  BytePlus Ark, for when its balance is topped up

OUTPUT (all under ./output/<username>/):
    run.json                     one row per reel — the machine-readable index
    report.md                    human-readable write-up of every reel processed
    reels/<shortcode>/           per-reel working dir
        video.mp4                downloaded reel
        frames/frame_01.jpg …    the frames Seedance actually saw
        audio.mp3                extracted audio
        transcript.json          raw Whisper response
        reasoning.json           Seedance's structured read
        reasoning.md             the same, rendered readable
    _cache/scrape.json           raw Apify dataset (re-runs are free)
    _cache/classify.json         raw OpenAI classification

Everything is cached. Re-running costs nothing unless you pass --force-* flags.

USAGE
-----
    ./run.sh                              # anmolraina13, 3 reels, defaults
    ./run.sh --username someoneelse
    ./run.sh --top 5 --limit 60
    ./run.sh --reel-url https://www.instagram.com/reel/XXXX/
    ./run.sh --frames 15 --language hi
    ./run.sh --force-reason                # re-reason on cached frames
    ./run.sh --vision seedance             # use BytePlus instead of OpenAI
    ./run.sh --check                       # just verify keys + ffmpeg, no spend
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    sys.exit("ERROR: `requests` not installed. Run ./run.sh instead of calling python directly.")


HERE = Path(__file__).resolve().parent

# This script lives in pipeline/, but .env and output/ belong to the project root
# beside public/ and api/, so the website and the pipeline share one folder.
# Falls back to the script's own directory if it is ever run standalone.
PROJECT_ROOT = HERE.parent if (HERE.parent / "public").is_dir() else HERE


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_dotenv(path: Path) -> None:
    """Minimal .env loader — no extra dependency."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


load_dotenv(PROJECT_ROOT / ".env")

APIFY_TOKEN = os.environ.get("APIFY_TOKEN", "").strip()
APIFY_ACTOR = os.environ.get("APIFY_ACTOR", "apify~instagram-reel-scraper").strip()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
# Tried in order — if the first is retired on your account, the next is used.
OPENAI_TEXT_MODELS = [
    m.strip() for m in
    os.environ.get("OPENAI_TEXT_MODELS", "gpt-4o-mini,gpt-4.1-mini,gpt-4o").split(",")
    if m.strip()
]
WHISPER_MODELS = [
    m.strip() for m in
    os.environ.get("WHISPER_MODELS", "whisper-1,gpt-4o-transcribe").split(",")
    if m.strip()
]
# Vision models for the reasoning step when --vision openai (the default).
# Per the OpenAI cookbook: a video is just a list of base64 JPEG frames in one
# chat completion. Tried in order; first one your account accepts wins.
OPENAI_VISION_MODELS = [
    m.strip() for m in
    os.environ.get("OPENAI_VISION_MODELS", "gpt-4.1-mini,gpt-4o,gpt-4o-mini").split(",")
    if m.strip()
]

# "openai" or "seedance"
VISION_PROVIDER = os.environ.get("VISION_PROVIDER", "openai").strip().lower()

# Seedance / BytePlus Ark — same endpoint + model as your process_video.py sample.
SEEDANCE_API_KEY = (
    os.environ.get("SEEDANCE_API_KEY")
    or os.environ.get("BYTEPLUS_API_KEY")
    or os.environ.get("ARK_API_KEY")
    or ""
).strip()
SEEDANCE_BASE_URL = os.environ.get(
    "SEEDANCE_BASE_URL", "https://ark.ap-southeast.bytepluses.com/api/v3"
).rstrip("/")
SEEDANCE_MODEL = os.environ.get("SEEDANCE_MODEL", "seed-1-8-251228").strip()

REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "300"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", "720"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "4"))
PARALLEL = max(1, int(os.environ.get("PARALLEL", "3")))

LOOP_MIN_RUN = 3  # >=3 identical consecutive Whisper lines == a hallucinated loop

# Whisper self-reports its own uncertainty per segment. Anything past these
# thresholds is the model guessing over music or silence, so we delete it rather
# than let a made-up sentence reach the reasoning step as "evidence".
NO_SPEECH_MAX = float(os.environ.get("NO_SPEECH_MAX", "0.6"))
AVG_LOGPROB_MIN = float(os.environ.get("AVG_LOGPROB_MIN", "-0.8"))
COMPRESSION_MAX = float(os.environ.get("COMPRESSION_MAX", "2.4"))

# Frames scale with reel length: one still roughly every FRAME_EVERY_SEC seconds,
# clamped. A 17s reel does not need the same budget as a 60s one.
FRAME_EVERY_SEC = float(os.environ.get("FRAME_EVERY_SEC", "4"))
FRAMES_MIN = int(os.environ.get("FRAMES_MIN", "8"))
FRAMES_MAX = int(os.environ.get("FRAMES_MAX", "20"))


# ---------------------------------------------------------------------------
# Pretty logging (non-technical friendly)
# ---------------------------------------------------------------------------

_BOLD, _DIM, _GRN, _YEL, _RED, _CYN, _RST = (
    ("\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[0m")
    if sys.stdout.isatty() else ("", "", "", "", "", "", "")
)


def stage(n: int, total: int, title: str) -> None:
    print(f"\n{_BOLD}{_CYN}[{n}/{total}] {title}{_RST}", flush=True)


def info(msg: str) -> None:
    print(f"    {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"    {_GRN}✓{_RST} {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"    {_YEL}!{_RST} {msg}", flush=True)


def die(msg: str) -> None:
    print(f"\n{_RED}✗ {msg}{_RST}\n", file=sys.stderr, flush=True)
    sys.exit(1)


class CreatorFailed(Exception):
    """One creator's run couldn't finish. In a batch the others still run."""


def clean_handle(raw: str) -> str:
    """Turn whatever the user typed into a bare Instagram handle.

    People paste '@fit.khurana', a full profile URL, sometimes a URL with a
    trailing slash and a query string. All of those mean the same creator, and
    all of them must land in the same output folder — otherwise a second run
    typed slightly differently silently re-scrapes and re-pays for work that is
    already cached one directory over.
    """
    h = (raw or "").strip()
    m = re.search(r"instagram\.com/([^/?#\s]+)", h, re.I)
    if m:
        h = m.group(1)
    h = h.lstrip("@").strip().strip("/")
    return h.lower()


def human(n: Any) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "—"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def which_or_none(binary: str) -> str | None:
    return shutil.which(binary)


def preflight(provider: str, need_apify: bool = True) -> None:
    problems: list[str] = []
    if need_apify and not APIFY_TOKEN:
        problems.append("APIFY_TOKEN is missing from .env")
    if not OPENAI_API_KEY:
        problems.append("OPENAI_API_KEY is missing from .env")
    if provider == "seedance" and not SEEDANCE_API_KEY:
        problems.append("SEEDANCE_API_KEY is missing from .env (or use --vision openai)")
    if not which_or_none("ffmpeg"):
        problems.append("ffmpeg is not installed  →  run:  brew install ffmpeg")
    if not which_or_none("ffprobe"):
        problems.append("ffprobe is not installed (it ships with ffmpeg)")
    if problems:
        die("Setup problems found:\n\n      - " + "\n      - ".join(problems))
    keys = "Apify, OpenAI" + (", Seedance" if provider == "seedance" else "")
    ok(f"API keys present ({keys})")
    ok(f"ffmpeg found at {which_or_none('ffmpeg')}")
    if provider == "openai":
        ok(f"vision reasoning: OpenAI ({OPENAI_VISION_MODELS[0]})")
    else:
        ok(f"vision reasoning: Seedance ({SEEDANCE_MODEL})")


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# STAGE 1 — Apify scrape
# ---------------------------------------------------------------------------

def apify_scrape(username: str, limit: int, reel_urls: list[str] | None,
                 newer_than: str | None) -> list[dict]:
    """Start the Apify actor, poll until it finishes, return the dataset items."""
    if reel_urls:
        run_input: dict[str, Any] = {"username": reel_urls}
    else:
        run_input = {
            "username": [username],
            "resultsLimit": limit,
            "skipPinnedPosts": False,
            "skipTrialReels": True,
            "includeSharesCount": False,
            "includeTranscript": False,      # we do our own Whisper pass
            "includeDownloadedVideo": False,  # we fetch videoUrl ourselves
        }
        if newer_than:
            run_input["onlyPostsNewerThan"] = newer_than

    start_url = f"https://api.apify.com/v2/acts/{APIFY_ACTOR}/runs"
    info(f"actor: {APIFY_ACTOR.replace('~', '/')}")
    info(f"input: {json.dumps(run_input)}")

    r = requests.post(
        start_url,
        params={"token": APIFY_TOKEN},
        json=run_input,
        timeout=60,
    )
    if r.status_code not in (200, 201):
        die(f"Apify refused to start the run (HTTP {r.status_code}):\n      {r.text[:500]}")
    run = r.json()["data"]
    run_id = run["id"]
    dataset_id = run["defaultDatasetId"]
    info(f"run started: {run_id}")
    info(f"watch it live: https://console.apify.com/actors/runs/{run_id}")

    # Poll until terminal state.
    t0 = time.time()
    status = run.get("status", "RUNNING")
    last_print = 0.0
    while status in ("READY", "RUNNING"):
        time.sleep(4)
        s = requests.get(
            f"https://api.apify.com/v2/actor-runs/{run_id}",
            params={"token": APIFY_TOKEN},
            timeout=60,
        )
        if s.status_code != 200:
            warn(f"status poll returned HTTP {s.status_code}, retrying…")
            continue
        status = s.json()["data"]["status"]
        elapsed = time.time() - t0
        if elapsed - last_print >= 10:
            info(f"…{status.lower()} ({int(elapsed)}s)")
            last_print = elapsed

    if status != "SUCCEEDED":
        die(f"Apify run ended with status {status}. "
            f"Open https://console.apify.com/actors/runs/{run_id} to see why.")
    ok(f"scrape finished in {int(time.time() - t0)}s")

    items: list[dict] = []
    offset = 0
    while True:
        d = requests.get(
            f"https://api.apify.com/v2/datasets/{dataset_id}/items",
            params={"token": APIFY_TOKEN, "offset": offset, "limit": 1000, "clean": "true"},
            timeout=120,
        )
        if d.status_code != 200:
            die(f"Could not read the Apify dataset (HTTP {d.status_code}): {d.text[:300]}")
        batch = d.json()
        if not batch:
            break
        items.extend(batch)
        offset += len(batch)
        if len(batch) < 1000:
            break
    return items


# ---------------------------------------------------------------------------
# STAGE 2 — OpenAI caption classification
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM = """You are triaging an Indian fitness/endurance creator's Instagram \
Reels. For each reel you get its caption, its hashtags, its duration, and usually its \
THUMBNAIL IMAGE. You sort each reel into EXACTLY ONE of three buckets.

THERE ARE ONLY TWO KEEP BUCKETS. Everything else is dropped. Do not invent a third \
keep bucket, do not keep something because it is "fitness-adjacent" or "on-brand".

  bucket = "fitness"
    The reel's SUBJECT is physical training or the body's performance. Workouts, form,
    technique, programming, splits, PRs, drills, mobility, running, cycling, swimming,
    triathlon/Ironman, Hyrox, actual race or competition performance footage, injury,
    physio, rehab, stretching, deloads, recovery protocols, body-composition and
    transformation updates, coaching advice, training myth-busting.

  bucket = "nutrition"
    The reel's SUBJECT is what goes into the body. Meals, diet, macros, protein, calories,
    supplements, powders, hydration, race fuelling, grocery or kitchen content, eating
    for a goal. Use this bucket only when nutrition is the MAIN subject — a passing sip
    of a shake inside a training vlog is bucket "fitness", not "nutrition".

  bucket = "drop"
    EVERYTHING ELSE. Including, and especially:
      - comedy, skits, memes, trends, reactions, lip-syncs, dances, relatable-humour
        posts — EVEN IF filmed in a gym, EVEN IF the creator is shirtless, EVEN IF the
        thumbnail shows a track or a physique. A joke set in a gym is a joke, not a
        training reel.
      - relationship, dating, "POV", motivational-quote and mindset posts with no
        training or nutrition substance
      - event attendance with no performance: expo floors, posing with people, meet-ups,
        panels, "we're coming to Mumbai", registration announcements, finish-line selfies
        with no actual race footage
      - travel, nightlife, restaurant outings, day-in-my-life with no training or eating-
        for-a-goal angle
      - podcast trailers, collab shoutouts, giveaways, pure brand promos, "link in bio"
      - gear hauls, unboxings, shoe reviews, product plugs — these are commerce, not
        training or nutrition
      - anything where you cannot name the training or the food it is about

HOW TO DECIDE
1. Read the thumbnail AND the caption. Captions here are throwaway — "1st session in the
   bag", "it's 🏃 season", an emoji, nothing at all — so a blank caption is not evidence
   either way. The thumbnail usually settles it.
2. Then ask one question: WHAT IS THIS REEL ABOUT? Not what is in the frame — what is it
   about. A shirtless torso in the thumbnail tells you the creator's niche, not the
   reel's subject. If the caption is a punchline ("Fk it we back", "Some might say... 😢",
   "You wanna talk about it?") and the thumbnail is a pose or a text overlay rather than
   someone mid-movement, it is comedy. Drop it.
3. If you cannot name the specific training session or the specific food/nutrition point
   the reel is making, the bucket is "drop". Do not keep it "just in case" — a wrong keep
   costs a full video analysis and pollutes the results.

Captions may be Hinglish (Hindi in Latin script) — judge meaning, not language.

CONFIDENCE — use the whole scale honestly. Most reels are not 0.9.
  0.9-1.0  unmistakable: visible training/food AND a caption that names it
  0.7-0.85 clear from one strong signal, the other is neutral
  0.5-0.65 plausible but you are inferring the subject rather than seeing it
  0.3-0.45 a coin flip; you are guessing
Do not compress everything into 0.8-1.0. If you are guessing, say so with a low number.

Return STRICT JSON, no prose, in this exact shape:
{"results": [{"index": <int, the index you were given>,
              "bucket": "<fitness|nutrition|drop>",
              "subject": "<name the actual session or nutrition point in <=8 words, or 'none'>",
              "confidence": <0.0-1.0>,
              "signal": "<thumbnail|caption|both>",
              "reason": "<one short sentence — say what decided it, and for a drop say which drop category>"}]}
Return one object for EVERY index you were given, in the same order."""

KEEP_BUCKETS = ("fitness", "nutrition")


def normalise_verdict(cls: dict) -> dict:
    """Fold the classifier's answer into a bucket + is_fitness pair.

    Tolerates the older `is_fitness`/`topic` shape so a stale classify.json cache
    doesn't blow up, and treats anything unrecognised as a drop.
    """
    cls = dict(cls or {})
    bucket = str(cls.get("bucket") or "").strip().lower()
    if bucket not in KEEP_BUCKETS and bucket != "drop":
        # legacy cache: derive a bucket from the old fields
        if cls.get("is_fitness"):
            bucket = "nutrition" if cls.get("topic") == "nutrition" else "fitness"
        else:
            bucket = "drop"
    cls["bucket"] = bucket
    cls["is_fitness"] = bucket in KEEP_BUCKETS
    cls.setdefault("subject", cls.get("topic") or "")
    try:
        cls["confidence"] = float(cls.get("confidence") or 0.0)
    except (TypeError, ValueError):
        cls["confidence"] = 0.0
    return cls


def openai_chat(messages: list[dict], *, json_mode: bool = False,
                max_tokens: int = 4096, temperature: float = 0.1,
                models: list[str] | None = None) -> str:
    """Chat completion with model fallback + retry."""
    models = models or OPENAI_TEXT_MODELS
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    last_err = ""
    for model in models:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = requests.post(f"{OPENAI_BASE_URL}/chat/completions",
                                  headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code in (400, 404) and "model" in r.text.lower():
                    warn(f"model {model} unavailable, trying next…")
                    break  # try next model
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(30, 2 ** attempt))
                    continue
                break
            except Exception as e:  # network hiccup
                last_err = str(e)
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"OpenAI chat failed: {last_err}")


def fetch_thumbnail(url: str, dest: Path) -> Path | None:
    """Grab a reel's thumbnail and shrink it — captions alone are a weak signal,
    but one glance at the cover frame settles most reels instantly."""
    if dest.exists() and dest.stat().st_size > 500:
        return dest
    try:
        r = requests.get(url, timeout=45)
        if r.status_code != 200:
            return None
        raw = dest.with_suffix(".orig.jpg")
        raw.write_bytes(r.content)
        p = run_cmd(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(raw),
                     "-vf", "scale=-2:512", "-q:v", "5", str(dest)])
        raw.unlink(missing_ok=True)
        if dest.exists() and dest.stat().st_size > 500:
            return dest
        del p
    except Exception:
        return None
    return None


def classify_posts(items: list[dict], cache_dir: Path, use_thumbnails: bool) -> list[dict]:
    """Label every reel fitness / not-fitness from caption + hashtags + thumbnail."""
    thumbs: dict[int, Path] = {}
    if use_thumbnails:
        thumb_dir = cache_dir / "thumbs"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        info(f"fetching {len(items)} thumbnails…")
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(fetch_thumbnail, it.get("displayUrl") or "",
                            thumb_dir / f"{it.get('shortCode') or i}.jpg"): i
                for i, it in enumerate(items) if it.get("displayUrl")
            }
            for fut in as_completed(futures):
                path = fut.result()
                if path:
                    thumbs[futures[fut]] = path
        ok(f"{len(thumbs)}/{len(items)} thumbnails ready")
        if len(thumbs) < len(items):
            warn(f"{len(items) - len(thumbs)} reels will be judged on caption alone")

    def describe(i: int, it: dict) -> dict:
        caption = (it.get("caption") or "").strip()
        return {
            "index": i,
            "caption": caption[:700] if caption else "(no caption)",
            "hashtags": it.get("hashtags") or [],
            "duration_sec": round(float(it.get("videoDuration") or 0), 1),
            "has_thumbnail": i in thumbs,
        }

    results: list[dict] = []
    BATCH = 12 if thumbs else 25
    for start in range(0, len(items), BATCH):
        idxs = list(range(start, min(start + BATCH, len(items))))
        info(f"classifying reels {idxs[0] + 1}–{idxs[-1] + 1} of {len(items)}…")

        content: list[dict] = [{
            "type": "text",
            "text": ("Reels to triage (thumbnails follow, each labelled with its index):\n"
                     + json.dumps([describe(i, items[i]) for i in idxs], ensure_ascii=False)),
        }]
        for i in idxs:
            if i in thumbs:
                content.append({"type": "text", "text": f"Thumbnail for index {i}:"})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": file_to_data_url(thumbs[i]), "detail": "low"},
                })

        models = OPENAI_VISION_MODELS if thumbs else OPENAI_TEXT_MODELS
        raw = openai_chat(
            [
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user", "content": content},
            ],
            json_mode=True,
            max_tokens=4096,
            models=models,
        )
        try:
            parsed = json.loads(raw)
            results.extend(parsed.get("results") or [])
        except json.JSONDecodeError:
            warn("a classification batch came back unparseable; skipping it")
    return results


# ---------------------------------------------------------------------------
# STAGE 3 — download the reel
# ---------------------------------------------------------------------------

def download_video(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 10_000:
        info(f"video already downloaded ({dest.stat().st_size/1_048_576:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    }
    last_err = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=REQUEST_TIMEOUT) as r:
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}"
                    time.sleep(2 ** attempt)
                    continue
                tmp = dest.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for block in r.iter_content(chunk_size=1 << 20):
                        f.write(block)
                tmp.rename(dest)
            ok(f"downloaded {dest.stat().st_size/1_048_576:.1f} MB")
            return
        except Exception as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
    raise RuntimeError(
        f"Could not download the video ({last_err}). Instagram CDN links expire — "
        f"re-run with --force-scrape to get fresh URLs."
    )


# ---------------------------------------------------------------------------
# STAGE 4 — ffmpeg: frames + audio
# ---------------------------------------------------------------------------

def probe_duration(video: Path) -> float:
    p = run_cmd(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(video)])
    try:
        return float(p.stdout.strip())
    except ValueError:
        return 0.0


def frames_for_duration(duration: float) -> int:
    """One still roughly every FRAME_EVERY_SEC seconds, clamped.

    A fixed budget wastes calls on a 17s reel and under-samples a 60s one, where
    12 frames means a five-second blind spot between stills.
    """
    if duration <= 0:
        return FRAMES_MIN
    return max(FRAMES_MIN, min(FRAMES_MAX, round(duration / FRAME_EVERY_SEC)))


def extract_frames(video: Path, out_dir: Path, n_frames: int | None, force: bool) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = probe_duration(video)
    if duration <= 0:
        raise RuntimeError("ffprobe could not read the video duration — file may be corrupt")
    if not n_frames:  # --frames auto
        n_frames = frames_for_duration(duration)
        info(f"{duration:.0f}s reel → sampling {n_frames} frames "
             f"(one every ~{duration / n_frames:.1f}s)")

    existing = sorted(out_dir.glob("frame_*.jpg"))
    if existing and len(existing) == n_frames and not force:
        info(f"reusing {len(existing)} cached frames")
        return existing
    for f in existing:
        f.unlink()

    frames: list[Path] = []
    for i in range(n_frames):
        # Sample at the midpoint of each of n equal slices — avoids black
        # first/last frames and gives even coverage of the reel.
        t = duration * (i + 0.5) / n_frames
        dest = out_dir / f"frame_{i + 1:02d}.jpg"
        p = run_cmd([
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
            "-ss", f"{t:.3f}", "-i", str(video),
            "-frames:v", "1",
            "-vf", f"scale=-2:{FRAME_HEIGHT}",
            "-q:v", str(JPEG_QUALITY),
            str(dest),
        ])
        if dest.exists() and dest.stat().st_size > 0:
            frames.append(dest)
        else:
            warn(f"frame at {t:.1f}s failed: {p.stderr.strip()[:160]}")
    if not frames:
        raise RuntimeError("ffmpeg produced no frames")
    ok(f"{len(frames)} frames sampled across {duration:.0f}s")
    return frames


def extract_audio(video: Path, dest: Path, force: bool) -> Path | None:
    if dest.exists() and dest.stat().st_size > 1000 and not force:
        info("reusing cached audio")
        return dest
    p = run_cmd([
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
        # strip long internal silences — these are what make Whisper loop
        "-af", "silenceremove=stop_periods=-1:stop_duration=1.5:stop_threshold=-40dB",
        str(dest),
    ])
    if not dest.exists() or dest.stat().st_size < 1000:
        warn(f"no usable audio track ({p.stderr.strip()[:160]}) — continuing frames-only")
        return None
    ok(f"audio extracted ({dest.stat().st_size/1024:.0f} KB)")
    return dest


# ---------------------------------------------------------------------------
# STAGE 5 — Whisper
# ---------------------------------------------------------------------------

def whisper_transcribe(audio: Path, language: str | None) -> dict:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    last_err = ""
    for model in WHISPER_MODELS:
        data: dict[str, Any] = {"model": model, "response_format": "verbose_json"}
        if language:
            data["language"] = language
        if model == "whisper-1":
            data["timestamp_granularities[]"] = "segment"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with open(audio, "rb") as fh:
                    r = requests.post(
                        f"{OPENAI_BASE_URL}/audio/transcriptions",
                        headers=headers,
                        files={"file": (audio.name, fh, "audio/mpeg")},
                        data=data,
                        timeout=REQUEST_TIMEOUT,
                    )
                if r.status_code == 200:
                    return r.json()
                last_err = f"HTTP {r.status_code}: {r.text[:250]}"
                if r.status_code in (400, 404) and "model" in r.text.lower():
                    break
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(30, 2 ** attempt))
                    continue
                break
            except Exception as e:
                last_err = str(e)
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Whisper failed: {last_err}")


def drop_unreliable_segments(segments: list[dict]) -> tuple[list[dict], list[str]]:
    """Throw away segments Whisper itself flagged as guesses.

    Whisper hallucinates confident-sounding sentences over music and silence, and
    it tells you when it is doing so: `no_speech_prob` climbs, `avg_logprob` sinks,
    `compression_ratio` explodes on repetition loops. A music-only reel produced the
    line "प्रस्तुति पर करते हैं" at no_speech_prob 0.78 / avg_logprob -0.98, which the
    reasoning model then cited as evidence of a spoken introduction. Cheapest possible
    fix: believe Whisper's own uncertainty and delete the segment.
    """
    kept: list[dict] = []
    notes: list[str] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        nsp = seg.get("no_speech_prob")
        alp = seg.get("avg_logprob")
        cr = seg.get("compression_ratio")
        why = None
        if isinstance(nsp, (int, float)) and nsp > NO_SPEECH_MAX:
            why = f"no_speech_prob {nsp:.2f}"
        elif isinstance(alp, (int, float)) and alp < AVG_LOGPROB_MIN:
            why = f"avg_logprob {alp:.2f}"
        elif isinstance(cr, (int, float)) and cr > COMPRESSION_MAX:
            why = f"compression_ratio {cr:.2f}"
        if why:
            notes.append(f"[{fmt_ts(float(seg.get('start') or 0))}] {text[:60]} ({why})")
            continue
        kept.append(seg)
    return kept, notes


def collapse_loops(segments: list[dict]) -> tuple[list[dict], int]:
    """Collapse >=3 identical consecutive segments — the classic Whisper
    hallucination on silence. Returns (cleaned, n_dropped)."""
    cleaned: list[dict] = []
    dropped = 0
    i = 0
    while i < len(segments):
        text = (segments[i].get("text") or "").strip()
        j = i + 1
        while j < len(segments) and (segments[j].get("text") or "").strip() == text:
            j += 1
        run = j - i
        seg = dict(segments[i])
        if run >= LOOP_MIN_RUN:
            seg["_loop_n"] = run
            dropped += run - 1
        cleaned.append(seg)
        i = j
    return cleaned, dropped


def fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def render_transcript(segments: list[dict]) -> str:
    if not segments:
        return ""
    lines = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        t0 = float(seg.get("start") or 0.0)
        loop_n = seg.get("_loop_n")
        suffix = f"  ⟲ ×{loop_n} (repeat loop collapsed — treat as silence)" if loop_n else ""
        lines.append(f"[{fmt_ts(t0)}] {text}{suffix}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# STAGE 6 — vision reasoning + reader compose
# ---------------------------------------------------------------------------

from reader_layer import REASON_SYSTEM, COMPOSE_SYSTEM, validate_pass1  # noqa: E402


NO_SPEECH_NOTICE = (
    "(NO RELIABLE SPEECH — this reel is music or ambient sound only. Every field must "
    "come from the frames. Return an empty evidence.from_audio, empty notable_quotes, "
    "and spoken_language \"none\".)"
)


def file_to_data_url(path: Path, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def build_reason_messages(meta: dict, frames: list[Path], transcript_md: str) -> list[dict]:
    """The frames-as-images payload. Identical shape for OpenAI and Seedance —
    both speak the OpenAI chat-completions dialect."""
    caption = (meta.get("caption") or "").strip()
    user_text = (
        f"REEL: {meta.get('url') or meta.get('shortCode')}\n"
        f"Creator: @{meta.get('ownerUsername')}\n"
        f"Duration: {round(float(meta.get('videoDuration') or 0), 1)}s\n"
        f"Posted: {meta.get('timestamp')}\n"
        f"Views: {meta.get('videoPlayCount') or meta.get('videoViewCount')}  "
        f"Likes: {meta.get('likesCount')}  Comments: {meta.get('commentsCount')}\n\n"
        f"CAPTION (author's own words):\n{caption[:4000] or '(none)'}\n\n"
        f"HASHTAGS: {', '.join(meta.get('hashtags') or []) or '(none)'}\n\n"
        f"FRAMES: {len(frames)} stills sampled evenly across the reel, chronological order.\n\n"
        "AUDIO TRANSCRIPT (timestamped; low-confidence segments already deleted):\n"
        "----------------------------------------------------\n"
        f"{transcript_md or NO_SPEECH_NOTICE}\n"
        "----------------------------------------------------\n\n"
        "Now return the single JSON object described in the system prompt."
    )

    content: list[dict] = [{"type": "text", "text": user_text}]
    for fp in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": file_to_data_url(fp), "detail": "high"},
        })

    return [
        {"role": "system", "content": REASON_SYSTEM},
        {"role": "user", "content": content},
    ]


def build_compose_messages(analysis: dict, meta: dict, transcript_md: str) -> list[dict]:
    trimmed = {k: v for k, v in analysis.items() if not str(k).startswith("_")}
    caption = (meta.get("caption") or "")[:4000]
    user = (
        f"## Vision analysis\n```json\n"
        f"{json.dumps(trimmed, ensure_ascii=False, indent=1)}\n```\n\n"
        f"## Full caption as posted\n{caption or '(no caption)'}\n\n"
        f"## Trusted transcript\n"
        f"{transcript_md or '(no reliable speech — read from frames and caption only)'}\n"
    )
    return [{"role": "system", "content": COMPOSE_SYSTEM},
            {"role": "user", "content": user}]


def seedance_reason(meta: dict, frames: list[Path], transcript_md: str) -> str:
    if not SEEDANCE_API_KEY:
        raise RuntimeError("SEEDANCE_API_KEY is not set")

    payload = {
        "model": SEEDANCE_MODEL,
        "messages": build_reason_messages(meta, frames, transcript_md),
        "temperature": 0.2,
        "max_tokens": 4096,
    }
    headers = {"Authorization": f"Bearer {SEEDANCE_API_KEY}", "Content-Type": "application/json"}
    url = f"{SEEDANCE_BASE_URL}/chat/completions"

    last_err = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            last_err = f"HTTP {r.status_code}: {r.text[:300]}"
            # Billing / auth problems will never succeed on retry — fail fast
            # with a message that says what to actually do about it.
            if r.status_code == 403 and "Overdue" in r.text:
                raise RuntimeError(
                    "Seedance rejected the request: the BytePlus account has an overdue "
                    "balance. Either top it up, or switch providers with:  "
                    "./run.sh --vision openai --force-reason")
            if r.status_code in (401, 403):
                raise RuntimeError(
                    f"Seedance rejected the key ({last_err}). Try:  "
                    f"./run.sh --vision openai --force-reason")
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt))
                continue
            break
        except RuntimeError:
            raise
        except Exception as e:
            last_err = str(e)
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"Seedance failed: {last_err}")


def openai_vision_reason(meta: dict, frames: list[Path], transcript_md: str) -> str:
    """Same frames, same prompt, OpenAI's vision endpoint instead.

    This is the OpenAI cookbook pattern ("Processing and narrating a video with
    GPT's visual capabilities"): a video is handed to the model as an ordered
    list of base64 JPEG stills inside one chat completion. We additionally pin
    the reply to JSON mode, so no code-fence stripping is needed.
    """
    return openai_chat(
        build_reason_messages(meta, frames, transcript_md),
        json_mode=True,
        max_tokens=4096,
        temperature=0.2,
        models=OPENAI_VISION_MODELS,
    )


def reason_about_video(provider: str, meta: dict, frames: list[Path],
                       transcript_md: str) -> str:
    if provider == "seedance":
        return seedance_reason(meta, frames, transcript_md)
    return openai_vision_reason(meta, frames, transcript_md)


def parse_json_blob(raw: str) -> dict:
    """Models sometimes wrap JSON in ``` fences or add a stray sentence."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {"_parse_failed": True, "_raw": raw}


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _bullets(items: Any, empty: str = "_none identified_") -> str:
    if not items:
        return empty
    if isinstance(items, str):
        return items
    return "\n".join(f"- {x}" for x in items)


def render_reel_md(meta: dict, cls: dict, data: dict, transcript_md: str,
                   n_frames: int) -> str:
    if data.get("_parse_failed"):
        return (f"# {meta.get('shortCode')}\n\n"
                f"The vision model returned something that wasn't valid JSON:\n\n"
                f"```\n{data.get('_raw', '')[:4000]}\n```\n")

    views = meta.get("videoPlayCount") or meta.get("videoViewCount")
    brands = data.get("brands_or_products_visible") or []
    brand_lines = "\n".join(
        f"- **{b.get('name')}** — {b.get('where')} _(confidence {b.get('confidence')})_"
        for b in brands if isinstance(b, dict)
    ) or "_none identified_"
    q = data.get("content_quality") or {}
    ev = data.get("evidence") or {}
    ppl = data.get("people_on_screen") or {}
    if isinstance(ppl, dict):
        people = (f"{ppl.get('max_in_any_frame', '—')} at most in one frame"
                  f"{' · creator present' if ppl.get('creator_present') else ''}"
                  + (f" — {ppl['note']}" if ppl.get("note") else ""))
    else:  # legacy `people_visible`
        people = str(data.get("people_visible", ppl or "—"))

    v_bucket = str(data.get("bucket") or "").lower()
    bucket_line = f"`{v_bucket or '—'}`"
    if v_bucket and v_bucket not in KEEP_BUCKETS:
        bucket_line = f"⚠️ `{v_bucket}` — OFF BUCKET after watching the full video"

    return f"""# {data.get('one_line_summary', '(no summary)')}

**Reel** {meta.get('url')}
**Posted** {meta.get('timestamp')} · **{human(views)} views** · {human(meta.get('likesCount'))} likes · {human(meta.get('commentsCount'))} comments · {round(float(meta.get('videoDuration') or 0), 1)}s
**Shortlisted because** {cls.get('reason', '—')} _(bucket: {cls.get('bucket', '—')}, confidence {cls.get('confidence')}, decided on {cls.get('signal', 'caption')})_
**Bucket after watching** {bucket_line} — {data.get('bucket_reason', '—')}
**Analysed from** {n_frames} frames + {'transcript' if transcript_md else 'no usable audio'} by {data.get('_engine', '—')}

## What happens

{data.get('detailed_summary', '—')}

**Hook** — {data.get('hook', '—')}

## Classification

| | |
|---|---|
| Primary topic | {data.get('primary_topic', '—')} |
| Sub-topics | {', '.join(data.get('sub_topics') or []) or '—'} |
| Format | {data.get('content_format', '—')} |
| Setting | {data.get('setting', '—')} |
| People on screen | {people} |
| Spoken language | {data.get('spoken_language', '—')} |
| Tone | {', '.join(data.get('tone') or []) or '—'} |
| Target audience | {data.get('target_audience', '—')} |

## What's shown

**Exercises**
{_bullets(data.get('exercises_shown'))}

**Food / supplements**
{_bullets(data.get('food_or_supplements_shown'))}

**Equipment**
{_bullets(data.get('equipment_visible'))}

**Brands & products visible**
{brand_lines}

**On-screen graphics** _(title cards, labels, stats — not the subtitle track)_
{_bullets(data.get('graphics_text') or data.get('on_screen_text'), '_none read_')}

**Burned-in subtitles** _(sample only — the full text is in the transcript below)_
{_bullets(data.get('subtitle_text'), '_none_')}

## What's claimed

{_bullets(data.get('key_claims'), '_no explicit claims_')}

**Call to action** — {data.get('call_to_action') or '_none_'}

## Notable quotes

{_bullets(data.get('notable_quotes'), '_none_')}

## Content quality

- **Production** — {q.get('production', '—')}
- **Information density** — {q.get('information_density', '—')}
- **Watchability** — {q.get('watchability', '—')}

## Evidence the model used

**From frames**
{_bullets(ev.get('from_frames'), '_—_')}

**From audio**
{_bullets(ev.get('from_audio'), '_no usable speech in this reel_')}

**Uncertainties**
{_bullets(data.get('uncertainties'), '_none flagged_')}

## Caption (as posted)

> {(meta.get('caption') or '(no caption)').replace(chr(10), chr(10) + '> ')}

## Transcript

```
{transcript_md or '(no reliable speech — music or ambient audio only)'}
```
"""


# ---------------------------------------------------------------------------
# Per-reel pipeline
# ---------------------------------------------------------------------------

def process_reel(meta: dict, cls: dict, reel_dir: Path, args) -> dict:
    shortcode = meta.get("shortCode") or meta.get("id")
    reel_dir.mkdir(parents=True, exist_ok=True)

    video = reel_dir / "video.mp4"
    frames_dir = reel_dir / "frames"
    audio = reel_dir / "audio.mp3"
    transcript_json = reel_dir / "transcript.json"
    reasoning_json = reel_dir / "reasoning.json"
    reasoning_md = reel_dir / "reasoning.md"

    result: dict[str, Any] = {
        "shortcode": shortcode,
        "url": meta.get("url"),
        "caption": meta.get("caption"),
        "hashtags": meta.get("hashtags"),
        "timestamp": meta.get("timestamp"),
        "views": meta.get("videoPlayCount") or meta.get("videoViewCount"),
        "likes": meta.get("likesCount"),
        "comments": meta.get("commentsCount"),
        "duration_sec": meta.get("videoDuration"),
        "classification": cls,
        "status": "pending",
    }

    # --- 3. download -------------------------------------------------------
    video_url = meta.get("videoUrl")
    if not video_url:
        result["status"] = "no_video_url"
        result["error"] = "Apify returned no videoUrl for this reel"
        return result
    download_video(video_url, video)

    # --- 4. frames + audio -------------------------------------------------
    frames = extract_frames(video, frames_dir, args.frames, args.force_frames)
    audio_path = extract_audio(video, audio, args.force_frames)

    # --- 5. whisper --------------------------------------------------------
    transcript_md = ""
    if audio_path and not args.no_audio:
        if transcript_json.exists() and not args.force_transcribe:
            info("reusing cached transcript")
            tr = json.loads(transcript_json.read_text())
        else:
            info("transcribing with Whisper…")
            tr = whisper_transcribe(audio_path, args.language)
            transcript_json.write_text(json.dumps(tr, ensure_ascii=False, indent=2))
        segs = tr.get("segments") or []
        if not segs and tr.get("text"):
            segs = [{"start": 0.0, "text": tr["text"]}]
        n_raw = len(segs)
        segs, junk = drop_unreliable_segments(segs)
        if junk:
            warn(f"dropped {len(junk)} low-confidence transcript line(s) Whisper "
                 f"was guessing at:")
            for line in junk[:3]:
                info(f"{_DIM}  {line}{_RST}")
        cleaned, dropped = collapse_loops(segs)
        if dropped:
            warn(f"collapsed {dropped} looped transcript lines")
        transcript_md = render_transcript(cleaned)
        result["transcript_segments_raw"] = n_raw
        result["transcript_segments_dropped"] = len(junk)
        if transcript_md:
            ok(f"transcript: {len(transcript_md.splitlines())} usable lines")
        else:
            ok("no reliable speech — this reel will be read from the frames alone")
    elif args.no_audio:
        info("audio skipped (--no-audio)")

    # --- 6. seedance -------------------------------------------------------
    engine = ("Seedance " + SEEDANCE_MODEL if args.vision == "seedance"
              else "OpenAI " + OPENAI_VISION_MODELS[0])
    if reasoning_json.exists() and not args.force_reason:
        info("reusing cached reasoning")
        data = validate_pass1(json.loads(reasoning_json.read_text()))
    else:
        info(f"sending {len(frames)} frames + transcript to {engine}…")
        t0 = time.time()
        raw = reason_about_video(args.vision, meta, frames, transcript_md)
        data = validate_pass1(parse_json_blob(raw))
        data["_engine"] = engine
        reasoning_json.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        ok(f"{engine} replied in {time.time() - t0:.0f}s")

    # --- 6b. compose the reader layer -------------------------------------
    reader_json = reel_dir / "reader.json"
    if reader_json.exists() and not args.force_compose:
        info("reusing cached reader layer")
        reader = json.loads(reader_json.read_text())
    elif data.get("_parse_failed"):
        reader = {}
    else:
        info("composing the reader layer…")
        raw2 = openai_chat(
            build_compose_messages(data, meta, transcript_md),
            json_mode=True, max_tokens=3000, temperature=0.3,
            models=OPENAI_TEXT_MODELS,
        )
        reader = parse_json_blob(raw2)
        reader_json.write_text(json.dumps(reader, ensure_ascii=False, indent=2))
        ok("reader layer composed")
    result["reader"] = reader

    md = render_reel_md(meta, cls, data, transcript_md, len(frames))
    reasoning_md.write_text(md, encoding="utf-8")

    # --- second gate -------------------------------------------------------
    # The thumbnail filter only ever saw one still. Now that a model has watched
    # the whole reel, ask it again which bucket this belongs in. A comedy skit
    # shot in a gym gets past a thumbnail; it does not get past this.
    video_bucket = str(data.get("bucket") or "").strip().lower()
    result["video_bucket"] = video_bucket or None
    result["video_bucket_reason"] = data.get("bucket_reason")
    result["on_bucket"] = video_bucket in KEEP_BUCKETS if video_bucket else True

    result["status"] = "parse_failed" if data.get("_parse_failed") else "ok"
    result["engine"] = data.get("_engine", engine)
    result["frames_used"] = len(frames)
    result["transcript_lines"] = len(transcript_md.splitlines()) if transcript_md else 0
    result["reasoning"] = data
    result["files"] = {
        "video": str(video),
        "frames_dir": str(frames_dir),
        "transcript": str(transcript_json) if transcript_json.exists() else None,
        "reasoning_json": str(reasoning_json),
        "reasoning_md": str(reasoning_md),
    }
    result["_markdown"] = md
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_creator(args, username: str) -> dict:
    """Everything for one creator, start to finish, into its own folder.

    Raises CreatorFailed rather than exiting, so one bad handle in a batch of
    four doesn't throw away the three that worked.
    """
    label = username if not args.reel_url else "direct-urls"
    root = Path(args.out) / re.sub(r"[^A-Za-z0-9_.-]", "_", label)
    cache = root / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    total_stages = 4

    # --- 1. scrape ---------------------------------------------------------
    stage(1, total_stages, f"Scraping reels for @{label} via Apify")
    scrape_file = cache / "scrape.json"
    if scrape_file.exists() and not args.force_scrape:
        items = json.loads(scrape_file.read_text())
        info(f"reusing cached scrape ({len(items)} reels) — pass --force-scrape for fresh data")
        warn("cached Instagram CDN links expire after ~a day; if downloads fail, --force-scrape")
    else:
        items = apify_scrape(username, args.limit, args.reel_url or None, args.newer_than)
        scrape_file.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    items = [it for it in items if it.get("videoUrl") or it.get("shortCode")]
    if not items:
        raise CreatorFailed(
            f"Apify returned no reels for @{label}. Is the profile public, does it "
            f"have reels, and is the handle spelled right?")
    ok(f"{len(items)} reels available")

    # --- 2. classify -------------------------------------------------------
    signal = "captions only" if args.no_thumbnails else "captions + thumbnails"
    stage(2, total_stages, f"Keeping only fitness + nutrition ({signal})")
    classify_file = cache / "classify.json"
    results = None
    if classify_file.exists() and not args.force_classify and not args.force_scrape:
        cached = json.loads(classify_file.read_text())
        if cached and all("bucket" in r for r in cached):
            results = cached
            info(f"reusing cached classification ({len(results)} labels)")
        else:
            warn("cached verdicts predate the fitness/nutrition buckets — reclassifying")
    if results is None:
        results = classify_posts(items, cache, use_thumbnails=not args.no_thumbnails)
        classify_file.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    by_index = {int(r["index"]): normalise_verdict(r)
                for r in results if isinstance(r.get("index"), (int, float))}
    shortlist: list[tuple[dict, dict]] = []
    rejected: list[tuple[dict, dict]] = []
    for i, it in enumerate(items):
        cls = by_index.get(i)
        if not cls:
            continue
        keep = cls["is_fitness"] and cls["confidence"] >= args.min_confidence
        if cls["is_fitness"] and not keep:
            cls = dict(cls, reason=f"{cls.get('reason')} — below the "
                                   f"{args.min_confidence:g} confidence floor")
        (shortlist if keep else rejected).append((it, cls))

    n_fit = sum(1 for _, c in shortlist if c["bucket"] == "fitness")
    n_nut = sum(1 for _, c in shortlist if c["bucket"] == "nutrition")
    ok(f"kept {len(shortlist)} ({n_fit} fitness, {n_nut} nutrition), "
       f"dropped {len(rejected)}")
    for it, cls in rejected[:5]:
        info(f"{_DIM}dropped {it.get('shortCode')}: {cls.get('reason')}{_RST}")
    if len(rejected) > 5:
        info(f"{_DIM}…and {len(rejected) - 5} more{_RST}")

    if not shortlist:
        raise CreatorFailed(
            f"Nothing from @{label} survived the fitness/nutrition filter — this may "
            f"simply not be a fitness creator. Try --limit 80, --newer-than "
            f"'12 months', or --min-confidence 0.4.")

    if args.sort == "confidence":
        # Confidence is coarse and ties constantly, so views is the real tie-break.
        shortlist.sort(key=lambda p: (round(p[1]["confidence"], 1),
                                      int(p[0].get("videoPlayCount")
                                          or p[0].get("videoViewCount") or 0)), reverse=True)
    elif args.sort == "views":
        shortlist.sort(key=lambda p: int(p[0].get("videoPlayCount")
                                         or p[0].get("videoViewCount") or 0), reverse=True)
    else:
        shortlist.sort(key=lambda p: p[0].get("timestamp") or "", reverse=True)

    # `--top all` means "everything that survived the filter" — resolved here,
    # per creator, because each creator's shortlist is a different length.
    target = len(shortlist) if args.top is None else min(args.top, len(shortlist))
    analyse_all = args.top is None

    print()
    info(f"{_BOLD}Queue ({'all ' if analyse_all else 'top '}{target} "
         f"of {len(shortlist)} shortlisted):{_RST}")
    for it, cls in shortlist[:target]:
        info(f"  • {it.get('shortCode')}  [{cls['bucket']}]  "
             f"{human(it.get('videoPlayCount') or it.get('videoViewCount'))} views — "
             f"{cls.get('subject') or cls.get('reason')}")

    # --- 3. per reel -------------------------------------------------------
    stage(3, total_stages, "Downloading, sampling frames, transcribing, reasoning")
    rows: list[dict] = []
    off_bucket: list[dict] = []
    attempts = 0
    # When analysing everything there is nothing to "backfill from" — the queue
    # is the whole shortlist either way, so we simply walk it to the end.
    max_attempts = len(shortlist) if (args.backfill or analyse_all) else target
    queue = list(shortlist)

    while sum(1 for r in rows if r.get("status") == "ok") < target and attempts < max_attempts:
        it, cls = queue[attempts]
        attempts += 1
        shortcode = it.get("shortCode") or it.get("id")
        n = attempts if analyse_all else len(rows) + 1
        print(f"\n  {_BOLD}── reel {n}/{target}: {shortcode}{_RST}  {it.get('url')}")
        try:
            row = process_reel(it, cls, root / "reels" / str(shortcode), args)
        except Exception as e:
            warn(f"{_RED}failed:{_RST} {e}")
            rows.append({"shortcode": shortcode, "url": it.get("url"),
                         "status": "error", "error": str(e), "classification": cls})
            continue
        # Having now watched the whole reel, the model gets a veto on the
        # thumbnail filter's guess.
        if row.get("status") == "ok" and not row.get("on_bucket", True):
            warn(f"rejected after watching: {row.get('video_bucket_reason') or 'off bucket'}")
            off_bucket.append(row)
            if analyse_all:
                info("dropped from the archive; continuing through the shortlist")
            elif args.backfill:
                info("pulling the next reel from the shortlist instead")
            continue
        rows.append(row)

    # --- write outputs -----------------------------------------------------
    stage(4, total_stages, "Writing results")
    run_json = root / "run.json"
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "username": username,
        "settings": {
            "scrape_limit": args.limit,
            "top": "all" if analyse_all else args.top,
            "analysed_target": target,
            "mode": "backfill-all" if analyse_all else "top-n",
            "frames": args.frames or "auto",
            "language": args.language, "sort": args.sort, "no_audio": args.no_audio,
            "min_confidence": args.min_confidence,
            "backfill": args.backfill,
            "vision_provider": args.vision,
            "vision_model": (SEEDANCE_MODEL if args.vision == "seedance"
                             else OPENAI_VISION_MODELS[0]),
            "filter_used_thumbnails": not args.no_thumbnails,
            "buckets": list(KEEP_BUCKETS),
        },
        "counts": {
            "scraped": len(items),
            "kept": len(shortlist),
            "kept_fitness": n_fit,
            "kept_nutrition": n_nut,
            "dropped": len(rejected),
            "analysed": len(rows),
            "succeeded": sum(1 for r in rows if r.get("status") == "ok"),
            "rejected_after_watching": len(off_bucket),
        },
        "dropped_reels": [
            {"shortcode": it.get("shortCode"), "url": it.get("url"),
             "reason": cls.get("reason"), "bucket": cls.get("bucket"),
             "confidence": cls.get("confidence")}
            for it, cls in rejected
        ],
        "rejected_after_watching": [
            {"shortcode": r.get("shortcode"), "url": r.get("url"),
             "video_bucket": r.get("video_bucket"),
             "reason": r.get("video_bucket_reason"),
             "thumbnail_verdict": (r.get("classification") or {}).get("reason")}
            for r in off_bucket
        ],
        "reels": [{k: v for k, v in r.items() if k != "_markdown"} for r in rows],
    }
    run_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        f"# VideoLens POC — @{username}",
        "",
        f"Run {payload['generated_at']} · reasoning by "
        f"`{payload['settings']['vision_provider']}` / "
        f"`{payload['settings']['vision_model']}` · "
        f"{args.frames or 'auto'} frames per reel · filter used {signal}",
        "",
        f"Scraped **{len(items)}** reels → kept **{len(shortlist)}** "
        f"(**{n_fit}** fitness, **{n_nut}** nutrition) at confidence ≥ "
        f"{args.min_confidence:g} → dropped **{len(rejected)}** → "
        + (f"rejected **{len(off_bucket)}** more after watching the full video → "
           if off_bucket else "")
        + f"analysed **{len(rows)}**.",
        "",
        "---",
        "",
    ]
    for r in rows:
        if r.get("_markdown"):
            report.append(r["_markdown"])
        else:
            report.append(f"# {r.get('shortcode')} — FAILED\n\n{r.get('error')}\n")
        report.append("\n---\n")

    if off_bucket:
        report.append("## Rejected after watching the full video\n")
        report.append("The thumbnail filter let these through; the model watched the "
                      "whole reel and put them outside fitness/nutrition.\n")
        for r in off_bucket:
            report.append(f"- **{r.get('shortcode')}** → `{r.get('video_bucket')}` — "
                          f"{r.get('video_bucket_reason')} ({r.get('url')})")
        report.append("")

    report.append("## Reels dropped by the fitness / nutrition filter\n")
    for it, cls in rejected:
        report.append(f"- **{it.get('shortCode')}** — {cls.get('reason')} "
                      f"_(conf {cls.get('confidence')})_ ({it.get('url')})")
    report_md = root / "report.md"
    report_md.write_text("\n".join(report), encoding="utf-8")

    succeeded = payload["counts"]["succeeded"]
    print()
    ok(f"{succeeded}/{len(rows)} reels fully analysed")
    ok(f"machine-readable: {run_json}")
    ok(f"readable report:  {report_md}")

    payload["_paths"] = {"root": str(root), "run_json": str(run_json),
                         "report_md": str(report_md)}
    return payload


def write_index(out_dir: Path, runs: list[dict], failures: list[tuple[str, str]]) -> Path:
    """One page across every creator analysed so far, so a batch of four is
    readable without opening four folders.

    Deliberately rebuilt from the run.json files on disk rather than from this
    session's results only — the user runs one creator per command, so the index
    has to accumulate across separate invocations or it would show one row.
    """
    seen = {r["username"]: r for r in runs}
    for rj in sorted(out_dir.glob("*/run.json")):
        try:
            data = json.loads(rj.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        name = data.get("username")
        if name and name not in seen:
            data.setdefault("_paths", {})["report_md"] = str(rj.parent / "report.md")
            seen[name] = data

    lines = [
        "# VideoLens — all creators",
        "",
        f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')}. "
        f"One row per creator; each links to that creator's full report.",
        "",
        "| Creator | Scraped | Kept | Fitness | Nutrition | Dropped | Analysed | Report |",
        "|---|--:|--:|--:|--:|--:|--:|---|",
    ]
    for name in sorted(seen):
        d = seen[name]
        c = d.get("counts", {})
        folder = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        # A run.json written before the two-bucket split has no per-bucket counts.
        cell = lambda k: "—" if c.get(k) is None else c[k]
        lines.append(
            f"| **@{name}** | {cell('scraped')} | {cell('kept')} | {cell('kept_fitness')} | "
            f"{cell('kept_nutrition')} | {cell('dropped')} | "
            f"{cell('succeeded')} | [report]({folder}/report.md) |")

    if failures:
        lines += ["", "## Didn't finish", ""]
        for name, why in failures:
            lines.append(f"- **@{name}** — {why}")

    lines += ["", "---", "",
              "Re-run any single creator with `./run.sh <handle>`; "
              "this index refreshes itself each time."]
    index = out_dir / "index.md"
    index.write_text("\n".join(lines), encoding="utf-8")
    return index


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Scrape a creator's reels, keep only the fitness and nutrition "
                    "ones, and run video reasoning on the top few.",
        epilog="Example:  ./run.sh fit.khurana        "
               "or several at once:  ./run.sh fit.khurana overlydaa hustlerani")
    ap.add_argument("usernames", nargs="*", metavar="HANDLE",
                    help="Instagram handle(s) to analyse. '@name' and full profile "
                         "URLs work too. Each gets its own output folder.")
    ap.add_argument("--username", dest="username_flag", default=None,
                    help="Same as passing the handle directly (kept for older commands)")
    ap.add_argument("--reel-url", action="append", default=[],
                    help="Analyse specific reel URL(s) instead of scraping a profile. Repeatable.")
    ap.add_argument("--limit", type=int, default=40,
                    help="How many recent reels to scrape before filtering (default: 40)")
    ap.add_argument("--top", default="all",
                    help="How many shortlisted fitness reels to actually analyse: a "
                         "number, or 'all' for every reel that survives the filter "
                         "(default: all). The POC used 3.")
    ap.add_argument("--frames", default="auto",
                    help="Frames per reel: a number, or 'auto' to scale with reel length "
                         f"(~1 every {FRAME_EVERY_SEC:g}s, {FRAMES_MIN}-{FRAMES_MAX}). Default: auto")
    ap.add_argument("--min-confidence", type=float, default=0.6,
                    help="Discard filter verdicts below this confidence (default: 0.6)")
    ap.add_argument("--no-backfill", dest="backfill", action="store_false",
                    help="Don't replace reels the full-video check rejects — just skip them")
    ap.add_argument("--language", default="hi",
                    help="Whisper language hint: hi, en, or '' to auto-detect (default: hi)")
    ap.add_argument("--newer-than", default=None,
                    help="Only reels newer than this, e.g. '6 months' or '2026-01-01'")
    ap.add_argument("--sort", choices=["confidence", "views", "recent"], default="confidence",
                    help="How to pick the top reels from the shortlist (default: confidence)")
    ap.add_argument("--vision", choices=["openai", "seedance"], default=VISION_PROVIDER,
                    help=f"Which model does the video reasoning (default: {VISION_PROVIDER})")
    ap.add_argument("--no-thumbnails", action="store_true",
                    help="Filter on captions alone — don't show the model each reel's thumbnail")
    ap.add_argument("--no-audio", action="store_true",
                    help="Skip Whisper entirely — frames-only reasoning")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output"), help="Output directory")
    ap.add_argument("--force-scrape", action="store_true", help="Re-run the Apify scrape")
    ap.add_argument("--force-classify", action="store_true", help="Re-run caption classification")
    ap.add_argument("--force-frames", action="store_true", help="Re-extract frames and audio")
    ap.add_argument("--force-transcribe", action="store_true", help="Re-run Whisper")
    ap.add_argument("--force-reason", action="store_true", help="Re-run vision reasoning")
    ap.add_argument("--force-compose", action="store_true",
                    help="re-run pass 2 only, against cached pass-1 output")
    ap.add_argument("--check", action="store_true", help="Verify setup and exit — spends nothing")
    args = ap.parse_args()

    if args.language in ("", "auto", "none"):
        args.language = None

    # --top all  → analyse every reel that survives the filter. This is the
    # backfill default; the POC's fixed 3 was only ever a cost guard.
    if str(args.top).strip().lower() in ("all", "*", "0", ""):
        args.top = None          # resolved per creator, once the shortlist is known
    else:
        try:
            args.top = max(1, int(args.top))
        except ValueError:
            die(f"--top must be a whole number or 'all', not {args.top!r}")

    if str(args.frames).strip().lower() in ("auto", "", "0"):
        args.frames = None  # decided per reel from its duration
    else:
        try:
            args.frames = max(1, min(FRAMES_MAX, int(args.frames)))
        except ValueError:
            die(f"--frames must be a number or 'auto', not {args.frames!r}")

    print(f"\n{_BOLD}VideoLens POC{_RST}  {_DIM}creator reels → fitness filter → video reasoning{_RST}")

    print(f"\n{_BOLD}{_CYN}Checking setup{_RST}")
    preflight(args.vision, need_apify=True)
    if args.check:
        print(f"\n{_GRN}{_BOLD}All good — you're ready to run.{_RST}\n")
        return 0

    args.usernames = [clean_handle(u) for u in args.usernames]
    if args.username_flag:
        args.usernames.append(clean_handle(args.username_flag))
    if not args.usernames:
        default = clean_handle(os.environ.get("DEFAULT_USERNAME", "anmolraina13"))
        args.usernames = ["direct-urls"] if args.reel_url else [default]
    # Same handle typed twice in one command shouldn't run twice.
    args.usernames = list(dict.fromkeys(args.usernames))

    out_dir = Path(args.out)
    runs: list[dict] = []
    failures: list[tuple[str, str]] = []

    for n, username in enumerate(args.usernames, 1):
        if len(args.usernames) > 1:
            print(f"\n{_BOLD}{_CYN}{'=' * 62}{_RST}")
            print(f"{_BOLD}  @{username}{_RST}  {_DIM}creator {n} of {len(args.usernames)}{_RST}")
            print(f"{_BOLD}{_CYN}{'=' * 62}{_RST}")
        try:
            runs.append(run_creator(args, username))
        except CreatorFailed as e:
            # One creator's problem is not the batch's problem.
            print(f"\n{_RED}✗ @{username}: {e}{_RST}", file=sys.stderr, flush=True)
            failures.append((username, str(e)))
        except Exception as e:
            print(f"\n{_RED}✗ @{username} failed unexpectedly: {e}{_RST}",
                  file=sys.stderr, flush=True)
            failures.append((username, f"unexpected error: {e}"))

    index = write_index(out_dir, runs, failures)

    print()
    if len(args.usernames) > 1 or len(runs) != 1:
        for r in runs:
            c = r["counts"]
            ok(f"@{r['username']}: {c['succeeded']} reel(s) analysed "
               f"({c['kept_fitness']} fitness / {c['kept_nutrition']} nutrition kept "
               f"of {c['scraped']} scraped)")
        for name, why in failures:
            warn(f"@{name}: {why}")
    ok(f"all creators:     {index}")
    if runs:
        print(f"\n{_BOLD}Open the latest report:{_RST}  "
              f"open \"{runs[-1]['_paths']['report_md']}\"")
    print(f"{_BOLD}Open the index:{_RST}          open \"{index}\"\n")

    return 0 if runs else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
