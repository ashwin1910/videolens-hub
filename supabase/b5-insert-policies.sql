-- B5 — run once in Supabase SQL Editor (after B1 schema).
-- Lets signed-in users queue new creators and backfill jobs from the browser.

create policy "queue new creator" on creators
  for insert to authenticated
  with check (status = 'queued');

create policy "request backfill" on jobs
  for insert to authenticated
  with check (kind = 'backfill' and status = 'queued');
