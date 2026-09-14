-- RouteMind location links — Supabase (Postgres) migration, 2026-09-14
-- Run in the Supabase SQL editor as the project owner (docs/deep-links-setup.md §5).
--
-- The backend writes with the SERVICE ROLE key (bypasses RLS). RLS is enabled with
-- no policies so the anon key can read nothing directly — every read goes through
-- the FastAPI service, which enforces expiry/revocation and never returns tokens.

create table if not exists public.shared_links (
  id              text primary key,                          -- 8 chars, [A-Za-z0-9], 48 bits
  kind            text not null check (kind in ('place', 'route')),
  link            jsonb not null,                            -- normalised LocationLink JSON
  label           text,
  creator         text,                                      -- 'android' | 'ios' | null (no PII)
  edit_token_hash text not null,                             -- sha256 of the creator's revoke token
  created_at      timestamptz not null default now(),
  expires_at      timestamptz,                               -- null = never
  revoked         boolean not null default false
);

create index if not exists shared_links_expires_idx on public.shared_links (expires_at);

create table if not exists public.live_trips (
  id            text primary key,                            -- 10 chars
  label         text,
  destination   jsonb not null,                              -- normalised LocationLink JSON (has a point)
  token_hash    text not null,                               -- sha256 of the sharer's progress token
  position      jsonb,                                       -- {lat,lng,heading,updated_at}
  eta_seconds   integer,
  remaining_m   integer,
  ended         boolean not null default false,
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now(),
  expires_at    timestamptz not null
);

create index if not exists live_trips_expires_idx on public.live_trips (expires_at);

alter table public.shared_links enable row level security;
alter table public.live_trips   enable row level security;
-- (no policies on purpose: service role only)

-- Housekeeping: drop expired live trips after a day, expired links after 30 days.
-- Schedule with pg_cron if enabled:  select cron.schedule('routemind-links-gc', '17 3 * * *', $$
--   delete from public.live_trips   where expires_at < now() - interval '1 day';
--   delete from public.shared_links where expires_at is not null and expires_at < now() - interval '30 days';
-- $$);
