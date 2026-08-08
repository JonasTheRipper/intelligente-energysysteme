-- TimescaleDB init script for the palaestrAI result store.
-- Runs automatically on first container start (docker-entrypoint-initdb.d).
-- palaestrai's `database-create` command will create the schema tables and
-- hypertables; this script just ensures the extensions are available.

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS postgis CASCADE;
