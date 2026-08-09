-- TimescaleDB init script for the palaestrAI result store.
-- Runs automatically on first container start (docker-entrypoint-initdb.d).
-- palaestrai's `database-create` command will create the schema tables and
-- hypertables; this script just ensures the extensions are available.

-- NOTE: only timescaledb here. The timescale/timescaledb image does not ship
-- PostGIS (only the -ha variant does), and the Postgres entrypoint runs these
-- scripts with ON_ERROR_STOP=1 -- a missing extension aborts container init.
-- The palaestrAI store schema has no geometry columns; GIS layers live in the
-- separate `postgis` service on port 5432.
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
