#!/usr/bin/env python3
"""
worker.py — GitHub Actions job runner for SignalFeed.

Claims queued jobs from Supabase, runs videolens → export_hub → publish_to_supabase
for one creator at a time, and loops until the queue is empty or five hours elapse.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from publish_to_supabase import (  # noqa: E402
    SupabasePublisher,
    _load_dotenv,
    _require_env,
    build_creator_from_output,
    publish_creator,
)

MAX_RUNTIME_SEC = 5 * 60 * 60
PROGRESS_POLL_SEC = 15
MAX_DAILY_BACKFILLS = 30
MAX_GLOBAL_CREATORS = 60


class WorkerClient(SupabasePublisher):
    def _patch(self, table: str, match: dict, body: dict) -> list[dict]:
        params = {key: f"eq.{value}" for key, value in match.items()}
        headers = {
            **self.rest_headers,
            "Prefer": "return=representation",
        }
        resp = requests.patch(
            f"{self.base}/rest/v1/{table}",
            headers=headers,
            params=params,
            json=body,
            timeout=60,
            verify=self.verify,
        )
        if resp.status_code not in (200, 204):
            raise RuntimeError(f"{table} patch failed: {resp.status_code} {resp.text[:300]}")
        if resp.status_code == 204 or not resp.text.strip():
            return []
        return resp.json()

    def _insert(self, table: str, body: dict) -> None:
        resp = requests.post(
            f"{self.base}/rest/v1/{table}",
            headers=self.rest_headers,
            json=body,
            timeout=60,
            verify=self.verify,
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"{table} insert failed: {resp.status_code} {resp.text[:300]}")

    def enqueue_refresh_jobs(self) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        resp = requests.get(
            f"{self.base}/rest/v1/creators",
            headers=self.rest_headers,
            params={
                "select": "id,handle,last_run_at",
                "status": "eq.ready",
                "last_run_at": f"lt.{cutoff}",
            },
            timeout=60,
            verify=self.verify,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"creators list failed: {resp.status_code} {resp.text[:300]}")

        inserted = 0
        for creator in resp.json():
            job_resp = requests.post(
                f"{self.base}/rest/v1/jobs",
                headers=self.rest_headers,
                json={
                    "creator_id": creator["id"],
                    "kind": "refresh",
                    "status": "queued",
                },
                timeout=60,
                verify=self.verify,
            )
            if job_resp.status_code in (200, 201):
                inserted += 1
                print(f"  queued refresh for @{creator['handle']}")
            elif job_resp.status_code == 409:
                continue
            else:
                print(
                    f"  ! refresh for @{creator['handle']}: "
                    f"{job_resp.status_code} {job_resp.text[:200]}"
                )
        return inserted

    def count_backfills_started_24h(self) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        resp = requests.get(
            f"{self.base}/rest/v1/jobs",
            headers=self.rest_headers,
            params={
                "select": "id",
                "kind": "eq.backfill",
                "status": "in.(running,done,failed)",
                "created_at": f"gte.{cutoff}",
            },
            timeout=60,
            verify=self.verify,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"jobs count failed: {resp.status_code} {resp.text[:300]}")
        return len(resp.json())

    def claim_job(self) -> dict | None:
        resp = requests.get(
            f"{self.base}/rest/v1/jobs",
            headers=self.rest_headers,
            params={
                "select": "id,creator_id,kind,status,created_at,creators(handle)",
                "status": "eq.queued",
                "order": "created_at.asc",
                "limit": "50",
            },
            timeout=60,
            verify=self.verify,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"jobs list failed: {resp.status_code} {resp.text[:300]}")
        rows = resp.json()
        if not rows:
            return None

        backfill_cap = self.count_backfills_started_24h() >= MAX_DAILY_BACKFILLS
        for job in rows:
            if job.get("kind") == "backfill" and backfill_cap:
                try:
                    self.update_job_progress(job["id"], {"phase": "daily_cap"})
                except Exception as exc:
                    print(f"  ! daily-cap notice failed for job {job['id']}: {exc}")
                continue

            claimed = self._patch("jobs", {"id": job["id"], "status": "queued"}, {"status": "running"})
            if not claimed:
                continue
            result = claimed[0]
            if job.get("creators"):
                result["creators"] = job["creators"]
            return result
        return None

    def update_job_progress(self, job_id: int, progress: dict) -> None:
        self._patch("jobs", {"id": job_id}, {"progress": progress})

    def complete_job(self, job_id: int, creator_id: str, generated_at: str | None = None) -> None:
        now = generated_at or datetime.now(timezone.utc).isoformat()
        self._patch(
            "jobs",
            {"id": job_id},
            {
                "status": "done",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "progress": None,
                "error": None,
            },
        )
        self._patch(
            "creators",
            {"id": creator_id},
            {
                "status": "ready",
                "status_detail": None,
                "last_run_at": now,
            },
        )

    def fail_job(self, job_id: int, creator_id: str, message: str) -> None:
        msg = (message or "unknown error")[:2000]
        self._patch(
            "jobs",
            {"id": job_id},
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": msg,
            },
        )
        self._patch(
            "creators",
            {"id": creator_id},
            {"status": "failed", "status_detail": msg[:500]},
        )

    def set_creator_processing(self, creator_id: str) -> None:
        self._patch(
            "creators",
            {"id": creator_id},
            {"status": "processing", "status_detail": None},
        )


def estimate_shortlist_total(out_dir: Path) -> int:
    classify_file = out_dir / "_cache" / "classify.json"
    scrape_file = out_dir / "_cache" / "scrape.json"
    if not classify_file.exists() or not scrape_file.exists():
        return 0
    try:
        results = json.loads(classify_file.read_text(encoding="utf-8"))
        items = json.loads(scrape_file.read_text(encoding="utf-8"))
    except Exception:
        return 0

    by_index = {
        int(r["index"]): r
        for r in results
        if isinstance(r, dict) and isinstance(r.get("index"), (int, float))
    }
    kept = 0
    for i, _item in enumerate(items):
        cls = by_index.get(i)
        if not cls:
            continue
        if cls.get("is_fitness") and float(cls.get("confidence") or 0) >= 0.6:
            kept += 1
    return kept


def count_finished_reels(out_dir: Path) -> int:
    reels_dir = out_dir / "reels"
    if not reels_dir.is_dir():
        return 0
    return sum(1 for p in reels_dir.glob("*/reasoning.json") if p.is_file())


def delete_creator_videos(out_dir: Path) -> int:
    removed = 0
    reels_dir = out_dir / "reels"
    if not reels_dir.is_dir():
        return 0
    for mp4 in reels_dir.glob("*/video.mp4"):
        if mp4.is_file():
            mp4.unlink(missing_ok=True)
            removed += 1
    return removed


class ProgressPoller:
    def __init__(self, client: WorkerClient, job_id: int, out_dir: Path) -> None:
        self.client = client
        self.job_id = job_id
        self.out_dir = out_dir
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(PROGRESS_POLL_SEC):
            current = count_finished_reels(self.out_dir)
            total = estimate_shortlist_total(self.out_dir) or max(current, 1)
            try:
                self.client.update_job_progress(
                    self.job_id,
                    {"current": current, "total": total},
                )
            except Exception as exc:
                print(f"  ! progress update failed: {exc}")


def run_subprocess(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def process_job(client: WorkerClient, job: dict) -> None:
    creator = job.get("creators") or {}
    handle = creator.get("handle")
    creator_id = job["creator_id"]
    job_id = job["id"]
    if not handle:
        raise RuntimeError(f"job {job_id} has no creator handle")

    print(f"\n=== Job {job_id} ({job.get('kind')}) → @{handle} ===")
    client.set_creator_processing(creator_id)
    client.update_job_progress(job_id, {"current": 0, "total": 0, "phase": "starting"})

    out_dir = ROOT / "output" / handle
    public_dir = ROOT / "public"
    py = sys.executable

    poller = ProgressPoller(client, job_id, out_dir)
    poller.start()
    try:
        run_subprocess([
            py, str(HERE / "videolens.py"), handle,
            "--top", "all", "--limit", "40",
            "--out", str(ROOT / "output"),
        ])
    finally:
        poller.stop()

    removed = delete_creator_videos(out_dir)
    print(f"  deleted {removed} video file(s)")

    client.update_job_progress(job_id, {"current": count_finished_reels(out_dir), "phase": "exporting"})
    run_subprocess([py, str(HERE / "export_hub.py"), "--out", str(ROOT / "output"), "--public", str(public_dir)])

    client.update_job_progress(job_id, {"current": count_finished_reels(out_dir), "phase": "publishing"})
    built_creator, reels = build_creator_from_output(handle, ROOT / "output", public_dir)
    publish_creator(client, public_dir, built_creator, reels)

    client.complete_job(job_id, creator_id, built_creator.get("generatedAt"))
    print(f"  ✓ job {job_id} done — {len(reels)} reel(s) published for @{handle}")


def main() -> int:
    _load_dotenv(ROOT / ".env")
    _load_dotenv(ROOT.parent / ".env")
    _require_env("OPENAI_API_KEY")
    _require_env("APIFY_TOKEN")
    base_url = _require_env("SUPABASE_URL")
    secret_key = _require_env("SUPABASE_SECRET_KEY")

    client = WorkerClient(base_url, secret_key)
    deadline = time.time() + MAX_RUNTIME_SEC
    processed = 0
    failed = 0

    print("SignalFeed worker")
    print(f"  runtime limit: {MAX_RUNTIME_SEC // 3600}h")
    refresh_added = client.enqueue_refresh_jobs()
    print(f"  refresh jobs queued: {refresh_added}")

    while time.time() < deadline:
        job = client.claim_job()
        if not job:
            print("\nNo queued jobs left.")
            break
        try:
            process_job(client, job)
            processed += 1
        except subprocess.CalledProcessError as exc:
            msg = f"pipeline command failed (exit {exc.returncode})"
            print(f"  ✗ {msg}", flush=True)
            client.fail_job(job["id"], job["creator_id"], msg)
            failed += 1
        except Exception as exc:
            msg = str(exc) or exc.__class__.__name__
            print(f"  ✗ {msg}", flush=True)
            client.fail_job(job["id"], job["creator_id"], msg)
            failed += 1

    print(f"\nFinished — processed {processed} job(s), {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
