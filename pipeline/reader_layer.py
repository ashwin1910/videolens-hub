"""Pass 1 validation and reader-layer prompts."""

from __future__ import annotations

import logging
import re
from typing import Any

from language_gate import DEVANAGARI, FRAME_IN_USAGE, check_pass1_devanagari

log = logging.getLogger(__name__)

SETTING_ENUMS = frozenset(
    "gym home outdoor kitchen studio street other".split()
)
FORMAT_ENUMS = frozenset(
    "talking_head exercise_demo tutorial voiceover_broll listicle "
    "myth_bust transformation vlog other".split()
)
CONTENT_TYPES = frozenset(
    "recipe workout session_log form_fix nutrition_note product_rank mindset other".split()
)
BRAND_CATEGORIES = frozenset(
    "footwear apparel supplement food_drink equipment appliance "
    "wearable_tech app_service gym_studio race_event other".split()
)
CLAIM_KINDS = frozenset("advice fact personal sentiment".split())


def _first_token(val: str, allowed: frozenset[str]) -> str:
    if not val:
        return "other"
    token = str(val).strip().split()[0].lower()
    token = token.split("-")[0] if "-" in token and token not in allowed else token
    if token in allowed:
        return token
    for a in allowed:
        if str(val).lower().startswith(a):
            return a
    return "other"


def validate_pass1(data: dict) -> dict:
    """Coerce pass-1 output; log every fix."""
    if data.get("_parse_failed"):
        return data

    before_setting = data.get("setting")
    data["setting"] = _first_token(str(before_setting or ""), SETTING_ENUMS)
    if data["setting"] != (before_setting or "").strip().split()[0].lower() if before_setting else True:
        if before_setting and data["setting"] != str(before_setting).strip():
            log.warning("coerced setting %r → %r", before_setting, data["setting"])

    before_fmt = data.get("content_format")
    data["content_format"] = _first_token(str(before_fmt or ""), FORMAT_ENUMS)
    if before_fmt and data["content_format"] != str(before_fmt).strip():
        log.warning("coerced content_format %r → %r", before_fmt, data["content_format"])

    ct = str(data.get("content_type") or "other").strip().lower()
    if ct not in CONTENT_TYPES:
        log.warning("unknown content_type %r → other", ct)
        ct = "other"
        data["content_type"] = ct

    # Motivational montages often land as other — nudge to mindset when obvious.
    if data.get("content_type") == "other":
        tones = " ".join(str(t) for t in (data.get("tone") or [])).lower()
        summary = (data.get("one_line_summary") or "").lower()
        if "motivat" in tones or "motivat" in summary or "inspir" in summary:
            data["content_type"] = "mindset"
            if "mindset" not in (data.get("payload") or {}):
                data["payload"] = {"mindset": {"message": data.get("one_line_summary") or "",
                                                "context": "", "for_whom": ""}}

    payload = data.get("payload")
    if not isinstance(payload, dict) or len(payload) != 1:
        log.warning("payload shape wrong — coercing content_type to other")
        data["content_type"] = "other"
    else:
        key = next(iter(payload))
        if key != data.get("content_type"):
            log.warning("payload key %r != content_type %r — coercing to other", key, data.get("content_type"))
            data["content_type"] = "other"

    claims = []
    for c in data.get("key_claims") or []:
        if isinstance(c, str):
            claims.append({"text": c, "kind": "fact", "source": "frames", "confidence": 0.5})
        elif isinstance(c, dict) and c.get("text") and c.get("kind") in CLAIM_KINDS:
            claims.append(c)
        elif isinstance(c, dict) and c.get("text"):
            log.warning("dropping claim without valid kind: %r", c.get("text", "")[:60])
    data["key_claims"] = claims

    brands = []
    for b in data.get("brands_or_products_visible") or []:
        if not isinstance(b, dict):
            continue
        name = (b.get("name") or "").strip()
        if not name or len(name) <= 1 or name.isdigit():
            continue
        cat = str(b.get("category") or "other").strip().lower()
        if cat not in BRAND_CATEGORIES:
            log.warning("brand %r category %r → other", name, cat)
            cat = "other"
        usage = (b.get("usage") or "").strip()
        if FRAME_IN_USAGE.search(usage):
            log.warning("dropping frame reference in usage for %r", name)
            usage = re.sub(r"\bframe\s*\d+\b", "", usage, flags=re.I).strip(" ,;")
        brands.append({**b, "category": cat, "usage": usage})
    data["brands_or_products_visible"] = brands

    for path in check_pass1_devanagari(data):
        log.warning("Devanagari in pass-1 field %s (will fail at export)", path)

    return data


REASON_SYSTEM = r"""You are a video-understanding engine analysing a single \
short-form fitness or nutrition reel. You will be given, in this order: the \
post's metadata and full caption, a trusted transcript of the spoken audio, and \
an ordered sequence of still frames sampled evenly across the video.

Return ONE JSON object and nothing else. No prose, no code fences.

## The two jobs

1. PERCEIVE. Report what is actually in this reel — objects, text, exercises, \
foods, brands, numbers. Never infer beyond the evidence.
2. TYPE AND EXTRACT. Decide what KIND of thing this reel is, and pull its \
substance into the matching typed payload so a reader can act on it without \
watching the video.

## Evidence rules — these override everything else

- The caption is first-class evidence, equal to audio and frames. Creators \
routinely put the full recipe, the full macros or the full session in the \
caption and only gesture at it on camera. If the caption has quantities and \
the audio does not, use the caption's quantities.
- If the transcript section says there is no reliable audio, then there is no \
audio. Do not quote anyone, do not describe what was said, do not infer a \
voiceover. Read the reel from frames and caption alone.
- Never invent a number. If a quantity is not stated in the caption, the audio \
or on screen, leave it null. A null is correct; a plausible guess is a bug.
- Distinguish burnt-in auto-captions from real on-screen information. \
graphics_text is for title cards, day labels, macro overlays, pace and \
distance readouts, ranked lists, exercise names with set/rep counts — the \
things that carry information. subtitle_text is a word-for-word echo of the \
speech; return at most 3 sample lines from it and never more.

## Language — every output field is English

The reader speaks English. These creators speak Hindi, Hinglish and English, \
often within one sentence. Your job includes translating.

- Write EVERY field in natural English. Not transliteration — translation.
- Romanised Hindi is not English. Translate it.
- Three fields keep the original: notable_quotes[].verbatim, \
graphics_text[].verbatim and call_to_action_verbatim. Fill both text_en and \
verbatim; set is_translation true when they differ.
- Keep proper nouns as they are.

## Choosing content_type

Ask: what does a viewer walk away able to DO?
recipe, workout, session_log, form_fix, nutrition_note, product_rank, mindset, other

## The payload

Emit exactly one key inside "payload" matching content_type. Set completeness: \
full, partial, or mentioned_only.

## Claims

Each key_claims entry needs kind: advice, fact, personal, or sentiment. \
Maximum 8. No "the creator says" framing.

## Fields that must be a single enum token

setting and content_format: exactly one token, description in setting_note.

## Products and brands

category, role (endorsed|used|incidental), usage (no frame numbers).

## Schema

{
  "bucket": "fitness|nutrition|drop",
  "bucket_reason": "one sentence",
  "content_type": "recipe|workout|session_log|form_fix|nutrition_note|product_rank|mindset|other",
  "content_type_confidence": 0.0,
  "content_type_reason": "one sentence",
  "payload": { "<content_type>": { ... } },
  "primary_topic": "training|endurance|recovery|body_composition|fitness_science|nutrition|other",
  "sub_topics": ["<=5, lowercase, 1-3 words"],
  "one_line_summary": "<=140 chars, plain English",
  "detailed_summary": "2-4 sentences",
  "setting": "gym|home|outdoor|kitchen|studio|street|other",
  "setting_note": "free text or null",
  "people_on_screen": {"creator_present": true, "max_in_any_frame": 0, "note": ""},
  "exercises_shown": [{"name": "", "detail": ""}],
  "food_or_supplements_shown": [{"name": "", "detail": ""}],
  "equipment_visible": [""],
  "brands_or_products_visible": [
    {"name": "", "category": "footwear|apparel|supplement|food_drink|equipment|appliance|wearable_tech|app_service|gym_studio|race_event|other",
     "role": "endorsed|used|incidental", "usage": "<=90 chars, no frame numbers",
     "where": "frames|audio|caption|graphics", "confidence": 0.0}
  ],
  "graphics_text": [{"text_en": "", "verbatim": "", "is_translation": false}],
  "subtitle_text": ["<=3, in English"],
  "key_claims": [{"text": "", "kind": "advice|fact|personal|sentiment",
                  "source": "audio|graphics|caption|frames", "confidence": 0.0}],
  "metrics": [{"label": "", "value": 0.0, "unit": "", "context": "", "source": ""}],
  "target_audience": "",
  "call_to_action": "in English or null",
  "call_to_action_verbatim": "as said or written, or null",
  "hook": "in English",
  "hook_technique": "question|claim|shock|demo|promise|story|none",
  "content_format": "talking_head|exercise_demo|tutorial|voiceover_broll|listicle|myth_bust|transformation|vlog|other",
  "tone": ["<=3"],
  "spoken_language": "english|hindi|hinglish|none|other",
  "content_quality": {"production": 0, "information_density": 0, "watchability": 0},
  "notable_quotes": [{"text_en": "", "verbatim": "", "is_translation": false}],
  "evidence": {"from_frames": ["in English"], "from_audio": ["in English"]},
  "uncertainties": [""]
}

Use null for anything unknown. Never use the string "unknown", "N/A" or "".
"""


COMPOSE_SYSTEM = r"""You turn a machine analysis of one Instagram reel into the \
page a person reads instead of watching it.

You will be given: the structured analysis from the vision pass, the post's full \
caption, and the full trusted transcript. Return ONE JSON object, nothing else.

## What you are for

Someone follows this creator and wants the information without losing forty \
minutes to the feed. Your output IS the thing they read.

## Rules of writing

- Write about the SUBJECT, never about the reel. Never write "this reel", \
"the video", "the creator explains".
- takeaway is the single most important string — specific, useful out of context.
- title says what the thing IS, under 60 characters.
- Use the caption when it has numbers the analysis missed.
- Never invent quantities.

## Blocks

Return blocks in reading order per content_type. Omit empty blocks.

## The playbook

Lift 0-4 standalone fragments: cue, swap, principle, benchmark, myth, verdict.

## Products

usage clause per product, no frame numbers, verdict only when judged.

## Language

Everything in English. caption_en and transcript_en for non-English reels; null if already English.

## Schema

{
  "title": "<=60 chars",
  "takeaway": "<=180 chars",
  "why_it_matters": "<=200 chars or null",
  "effort": {"label": "", "minutes": 0},
  "blocks": [{"kind": "steps|ingredients|table|pairs|ranked|stats|prose|quote|list",
              "title": "<=40 chars", "note": "or null", "data": {}}],
  "playbook": [{"kind": "cue|swap|principle|benchmark|myth|verdict",
                "text": "<=140 chars", "context": "",
                "strength": "stated|demonstrated|opinion"}],
  "products": [{"name": "", "usage": "<=90 chars", "verdict": "or null"}],
  "tags": ["<=6 lowercase"],
  "search_text": "<=400 chars, English only",
  "caption_en": "or null if already English",
  "transcript_en": "or null if already English",
  "watch_it_because": "one sentence or null",
  "confidence": "high|medium|low"
}
"""
