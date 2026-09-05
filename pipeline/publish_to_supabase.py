#!/usr/bin/env python3
"""
publish_to_supabase.py — push one creator's export_hub output to Supabase.

Upserts the creator row, upserts each reel (full export object in payload),
uploads assets to the public Storage bucket, and rewrites image paths to
Storage URLs. Safe to run twice on the same creator.
"""

from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

BUCKET = "assets"
ASSET_PREFIX = "assets/"


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _require_env(name: str) -> str:
    val = (os.environ.get(name) or "").strip()
    if not val:
        sys.exit(f"ERROR: {name} is not set.")
    return val


class SupabasePublisher:
    def __init__(self, base_url: str, secret_key: str) -> None:
        self.base = base_url.rstrip("/")
        self.secret_key = secret_key
        self.rest_headers = {
            "apikey": secret_key,
            "Authorization": f"Bearer {secret_key}",
            "Content-Type": "application/json",
        }
        self._upload_cache: dict[str, str] = {}

    def public_asset_url(self, storage_path: str) -> str:
        return f"{self.base}/storage/v1/object/public/{BUCKET}/{storage_path}"

    def upload_asset(self, local_path: Path, storage_path: str) -> str:
        if storage_path in self._upload_cache:
            return self._upload_cache[storage_path]

        if not local_path.is_file():
            raise FileNotFoundError(f"missing asset: {local_path}")

        content_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
        url = f"{self.base}/storage/v1/object/{BUCKET}/{storage_path}"
        headers = {
            "apikey": self.secret_key,
            "Authorization": f"Bearer {self.secret_key}",
            "Content-Type": content_type,
            "x-upsert": "true",
        }
        data = local_path.read_bytes()
        resp = requests.post(url, headers=headers, data=data, timeout=120)
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"storage upload failed for {storage_path}: "
                f"{resp.status_code} {resp.text[:300]}"
            )

        public_url = self.public_asset_url(storage_path)
        self._upload_cache[storage_path] = public_url
        return public_url

    def upsert_creator(self, creator: dict) -> str:
        handle = creator["username"]
        row = {
            "handle": handle,
            "display_name": creator.get("name"),
            "digest": creator.get("digest"),
            "counts": creator.get("counts"),
            "status": "ready",
            "last_run_at": creator.get("generatedAt") or datetime.now(timezone.utc).isoformat(),
        }
        url = f"{self.base}/rest/v1/creators?on_conflict=handle"
        headers = {
            **self.rest_headers,
            "Prefer": "resolution=merge-duplicates,return=representation",
        }
        resp = requests.post(url, headers=headers, json=row, timeout=60)
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"creator upsert failed for @{handle}: {resp.status_code} {resp.text[:300]}"
            )
        rows = resp.json()
        if not rows:
            raise RuntimeError(f"creator upsert returned no row for @{handle}")
        return rows[0]["id"]

    def upsert_reel(self, creator_id: str, reel: dict) -> None:
        row = {
            "id": reel["id"],
            "creator_id": creator_id,
            "posted_at": reel.get("timestamp"),
            "payload": reel,
        }
        url = f"{self.base}/rest/v1/reels?on_conflict=id"
        headers = {
            **self.rest_headers,
            "Prefer": "resolution=merge-duplicates",
        }
        resp = requests.post(url, headers=headers, json=row, timeout=60)
        if resp.status_code not in (200, 201, 204):
            raise RuntimeError(
                f"reel upsert failed for {reel['id']}: {resp.status_code} {resp.text[:300]}"
            )


def _rel_to_storage_path(rel_path: str) -> str:
    rel = rel_path.lstrip("/")
    if rel.startswith(ASSET_PREFIX):
        return rel[len(ASSET_PREFIX):]
    return rel


def _rewrite_reel_assets(reel: dict, public_dir: Path, publisher: SupabasePublisher) -> dict:
    out = copy.deepcopy(reel)

    thumb = out.get("thumb")
    if isinstance(thumb, str) and thumb.strip():
        rel = thumb.strip()
        storage_path = _rel_to_storage_path(rel)
        local_path = public_dir / rel
        out["thumb"] = publisher.upload_asset(local_path, storage_path)

    frames = out.get("frames")
    if isinstance(frames, list):
        rewritten: list[str] = []
        for rel in frames:
            if not isinstance(rel, str) or not rel.strip():
                continue
            rel = rel.strip()
            storage_path = _rel_to_storage_path(rel)
            local_path = public_dir / rel
            rewritten.append(publisher.upload_asset(local_path, storage_path))
        out["frames"] = rewritten

    return out


def publish_creator(
    publisher: SupabasePublisher,
    public_dir: Path,
    creator: dict,
    reels: list[dict],
) -> dict:
    handle = creator["username"]
    creator_id = publisher.upsert_creator(creator)

    uploaded_assets = 0
    for reel in reels:
        payload = _rewrite_reel_assets(reel, public_dir, publisher)
        if payload.get("thumb"):
            uploaded_assets += 1
        uploaded_assets += len(payload.get("frames") or [])
        publisher.upsert_reel(creator_id, payload)

    return {
        "handle": handle,
        "creator_id": creator_id,
        "reels": len(reels),
        "assets": uploaded_assets,
    }


def load_hub_data(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^\s*window\.HUB\s*=\s*", "", text.strip())
    text = text.rstrip().removesuffix(";").strip()
    return json.loads(text)


def publish_from_hub_data(path: Path, public_dir: Path, publisher: SupabasePublisher) -> list[dict]:
    hub = load_hub_data(path)
    creators = hub.get("creators") or []
    reels = hub.get("reels") or []
    by_creator: dict[str, list[dict]] = {}
    for reel in reels:
        handle = reel.get("creator")
        if handle:
            by_creator.setdefault(handle, []).append(reel)

    results = []
    for creator in creators:
        handle = creator.get("username")
        if not handle:
            continue
        mine = by_creator.get(handle, [])
        results.append(publish_creator(publisher, public_dir, creator, mine))
    return results


def build_creator_from_output(handle: str, out_dir: Path, public_dir: Path) -> tuple[dict, list[dict]]:
    sys.path.insert(0, str(HERE))
    import export_hub as eh  # noqa: WPS433 — local pipeline module

    cdir = out_dir / handle
    if not cdir.is_dir():
        matches = [d for d in out_dir.iterdir() if d.is_dir() and d.name.lower() == handle.lower()]
        if len(matches) == 1:
            cdir = matches[0]
        else:
            raise FileNotFoundError(f"no output folder for @{handle} in {out_dir}")

    run = eh.load_json(cdir / "run.json")
    if not run:
        raise RuntimeError(f"missing or invalid run.json for @{handle}")

    username = run.get("username") or cdir.name
    alias_map = eh.load_json(out_dir / "_entities.json") or {}

    mine: list[dict] = []
    for r in run.get("reels") or []:
        code = r.get("shortcode") or r.get("id")
        reader = eh.load_json(cdir / "reels" / str(code) / "reader.json") if code else None
        if reader and isinstance(reader.get("playbook"), list):
            r = dict(r)
            r["_reader_playbook"] = reader["playbook"]
        built = eh.build_reel(username, r, cdir, public_dir, alias_map, reader)
        if built:
            built.pop("_reader_playbook", None)
            mine.append(built)

    counts = dict(run.get("counts") or {})
    scraped, dropped = counts.get("scraped"), counts.get("dropped")
    if isinstance(scraped, int) and isinstance(dropped, int):
        counts["kept"] = scraped - dropped
    else:
        counts.setdefault("kept", counts.get("analysed", len(mine)))
    counts["analysed"] = counts.get("analysed", len(mine))
    counts["published"] = len(mine)
    counts["byType"] = dict(__import__("collections").Counter(r.get("type") for r in mine))

    settings = run.get("settings") or {}
    digest = eh.generate_digest(cdir, username, mine) if mine else {}

    creator = {
        "username": username,
        "name": eh.creator_display_name(cdir, username),
        "generatedAt": run.get("generated_at"),
        "engine": f"{settings.get('vision_provider', 'openai')} / "
                  f"{settings.get('vision_model', 'gpt-4.1-mini')}",
        "counts": counts,
        "buckets": settings.get("buckets") or ["fitness", "nutrition"],
        "mode": settings.get("mode", "top-n"),
        "droppedSample": [
            {
                "shortcode": d.get("shortcode"),
                "reason": d.get("reason"),
                "confidence": d.get("confidence"),
            }
            for d in (run.get("dropped_reels") or [])[:6]
        ],
        "digest": digest,
    }
    return creator, mine


def main() -> int:
    ap = argparse.ArgumentParser(description="Publish export_hub output to Supabase.")
    ap.add_argument("--handle", help="Publish one creator from pipeline output/")
    ap.add_argument("--hub-data", help="Publish all creators from public/hub-data.js")
    ap.add_argument("--out", default=str(ROOT / "output"), help="Pipeline output directory")
    ap.add_argument("--public", default=str(ROOT / "public"), help="Public assets directory")
    args = ap.parse_args()

    if not args.handle and not args.hub_data:
        ap.error("pass --handle USERNAME or --hub-data PATH")

    _load_dotenv(ROOT / ".env")
    _load_dotenv(ROOT.parent / ".env")
    base_url = _require_env("SUPABASE_URL")
    secret_key = _require_env("SUPABASE_SECRET_KEY")
    if secret_key.startswith("eyJ"):
        sys.exit(
            "ERROR: SUPABASE_SECRET_KEY looks like a legacy JWT. "
            "Use an sb_secret_… key from Settings → API Keys."
        )

    public_dir = Path(args.public).resolve()
    publisher = SupabasePublisher(base_url, secret_key)

    if args.hub_data:
        results = publish_from_hub_data(Path(args.hub_data).resolve(), public_dir, publisher)
    else:
        creator, reels = build_creator_from_output(args.handle, Path(args.out).resolve(), public_dir)
        results = [publish_creator(publisher, public_dir, creator, reels)]

    total_reels = sum(r["reels"] for r in results)
    print(f"\nPublished {len(results)} creator(s), {total_reels} reel(s):\n")
    for row in results:
        print(
            f"  @{row['handle']}  id={row['creator_id']}  "
            f"reels={row['reels']}  assets={row['assets']}"
        )
    print()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
