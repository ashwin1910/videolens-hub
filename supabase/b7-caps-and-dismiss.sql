-- B7 — run once in Supabase SQL Editor (after B1 schema + b5 insert policies).
-- Caps (C5), failed-job dismiss, and supporting RLS.

-- Failed jobs the user dismissed no longer drive the waiting screen.
alter table jobs add column if not exists dismissed_at timestamptz;

create policy "dismiss failed job" on jobs
  for update to authenticated
  using (
    status = 'failed'
    and dismissed_at is null
    and exists (
      select 1 from user_creators uc
      where uc.creator_id = jobs.creator_id and uc.user_id = auth.uid()
    )
  )
  with check (dismissed_at is not null);

-- C5: max 4 creators per user (DB-enforced, not UI-only).
create or replace function check_user_creator_limit()
returns trigger as $$
begin
  if (select count(*) from user_creators where user_id = new.user_id) >= 4 then
    raise exception 'creator limit reached';
  end if;
  return new;
end;
$$ language plpgsql;

drop trigger if exists user_creators_limit on user_creators;
create trigger user_creators_limit
  before insert on user_creators
  for each row execute function check_user_creator_limit();

-- C5: max 60 creators in the shared library.
create or replace function check_global_creator_limit()
returns trigger as $$
begin
  if (select count(*) from creators) >= 60 then
    raise exception 'library full';
  end if;
  return new;
end;
$$ language plpgsql;

drop trigger if exists creators_global_limit on creators;
create trigger creators_global_limit
  before insert on creators
  for each row execute function check_global_creator_limit();
