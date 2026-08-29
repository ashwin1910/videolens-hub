# Lens

An archive of what a set of Instagram fitness creators actually posted — every
on-topic reel read frame by frame, written up, and made searchable — plus the
pipeline that builds it.

Two halves, one folder:

- **The website** (`public/`, `api/`) — static, deploys to Vercel, no build step.
- **The pipeline** (`pipeline/`) — Python, runs on your machine, writes the data
  the website serves.

They meet at exactly one file: `public/hub-data.js`, which the exporter writes
and the page reads.

---

## Deploying the site

The site in this repo is ready to go. Nothing is compiled and there is no build
step — Vercel serves `public/` and turns `api/*.js` into serverless functions.

```
git push
```

Then, on Vercel: **New Project → import this repository → Deploy.** Take every
default; `vercel.json` already says what to do.

One setting matters. For **Ask the archive** to work, add the key:

**Project → Settings → Environment Variables**

| Name | Value | Notes |
|---|---|---|
| `OPENAI_API_KEY` | `sk-proj-…` | Required for chat. Server-side only — it never reaches the browser. |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Optional. |

Redeploy after adding it. Everything except the chat works without it.

Check a deployment with `https://<your-site>/api/health` — it reports whether
the key is configured and how many reels shipped.

---

## Rebuilding the archive

The site reads whatever the last export produced. To refresh it:

```
cp .env.example .env        # once — paste your APIFY_TOKEN and OPENAI_API_KEY in
./pipeline/backfill.sh      # scrape, filter, analyse, export
git add -A && git commit -m "Refresh archive" && git push
```

That runs every creator in `pipeline/creators.txt`. One creator only:

```
./pipeline/backfill.sh fit.khurana
```

You need `python3` and `ffmpeg` (`brew install ffmpeg`). The script builds its
own Python environment on first run.

**Every stage caches, per creator.** Stopping the backfill and re-running it
costs nothing for work already done, so a failure halfway through is not a
setback — re-run the handles that didn't finish.

### What it does

```
Apify instagram-reel-scraper — the last 40 posts
        │
        ▼
screen each one on caption + thumbnail  →  fitness | nutrition | drop
        │  everything else is dropped here, before anything is downloaded
        ▼
for EVERY reel that survives:
   download → ffmpeg frames + 16kHz audio → Whisper → OpenAI vision
        │
        ▼
second gate: having watched the whole reel, is it still on topic?
        │
        ▼
output/<creator>/run.json
        │
        ▼
pipeline/export_hub.py  →  public/hub-data.js + public/assets/
```

The POC analysed only the top 3 reels per creator. That was a cost guard, not a
design choice, and it is gone: `--top` now defaults to `all`, so the backfill
analyses **every reel that survives the filter** — roughly 7 to 35 per creator
depending on how much of their page is actually about training or food.

The filter is what makes this affordable. Screening happens on the caption and
the thumbnail, so the expensive work — download, frames, Whisper, vision — only
ever runs on reels that are already known to be on topic.

### Useful variations

```
./pipeline/backfill.sh --limit 80              # look further back than 40 posts
./pipeline/backfill.sh --newer-than "6 months" # only recent reels
./pipeline/backfill.sh --min-confidence 0.4    # loosen the filter
./pipeline/backfill.sh --top 3                 # the old POC behaviour
./pipeline/backfill.sh --force-reason          # re-run reasoning only, after a prompt edit
./pipeline/backfill.sh --force-scrape          # fresh scrape — Instagram links expire in ~a day
```

`python3 pipeline/export_hub.py` re-exports from whatever is already on disk,
without touching an API.

---

## Layout

```
public/            the website, exactly as designed
  index.html       the whole app — template and logic
  hub-data.js      GENERATED — the archive the page reads
  data/hub.json    GENERATED — the same payload, for the API
  assets/          GENERATED — thumbnails and frame strips
  support.js       the rendering runtime
  organic.css      the design system
  vendor/          React and Babel, served from this deployment
api/
  chat.js          POST — "Ask the archive", the only user of the OpenAI key
  health.js        GET  — is this deployment wired up correctly
pipeline/
  videolens.py     the pipeline
  backfill.sh      the one command that runs it
  export_hub.py    output/ → public/
  creators.txt     which creators the archive tracks
scripts/serve.js   local preview without the Vercel CLI
output/            pipeline working files — gitignored, reproducible
```

**`output/` is deliberately not committed.** It holds every downloaded video and
sampled frame — hundreds of megabytes, all reproducible from a re-run. What the
site serves is exported into `public/` and *is* committed, so a clone of this
repo deploys with the archive intact and needs no pipeline run to work.

---

## Running it locally

```
npm run serve       # http://localhost:3000
```

Serves `public/` and routes `/api/*` through the same handlers Vercel uses, so
the chat behaves locally as it does in production. `npm run dev` uses the Vercel
CLI instead, if you have it.

---

## Two things worth knowing

**The transcript you see is not raw Whisper.** Whisper invents confident
sentences over music, and one such hallucination was once quoted back as
evidence of a spoken introduction. Every segment it flags as uncertain
(`no_speech_prob`, `avg_logprob`, `compression_ratio`) is deleted before the
reasoning model ever sees it, and the exporter re-applies the same thresholds —
so the transcript on the page is exactly what the model was allowed to read. A
reel with nothing left is explicitly told it has no audio and must not quote
anyone. Empty transcripts are normal and correct.

**On-screen text is split in two.** These creators burn auto-captions into every
frame, which used to flood the analysis with the spoken words a second time.
`graphics` holds the real information — title cards, day labels, pace and
distance — and `subtitles` holds a few sample lines and nothing more.

---

## Not built yet

This repo covers past data only: a one-time backfill, exported to static files,
deployed. Deliberately still to come:

- **Supabase** as the store, instead of committed JSON.
- **The daily job** — pick up each creator's new posts, run the same pipeline
  over just those, and write the daily brief the site already has a page for.

The `Daily brief` tab renders from the same archive today. When the daily job
lands, it fills that page from real per-day output rather than the backfill.
