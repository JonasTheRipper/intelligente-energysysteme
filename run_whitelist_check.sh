#!/usr/bin/env bash
# =============================================================================
# Whitelist equivalence check: does the ARL sensor whitelist change the
# simulation?  (Expected answer: no -- sensors are read-only reports.)
#
# Runs three short experiments and makes two comparisons:
#
#   OFF vs OFF2 -> determinism control. If these differ, STOP: the scenario is
#                  not reproducible and no conclusion about the whitelist is
#                  valid.
#   OFF vs ON   -> the whitelist's effect. Expect "ALL PHASES IDENTICAL".
#
# Each run uses its own database and executor port, so this is safe to run
# alongside nothing else -- but do NOT run it during a training run: 15 extra
# worker processes on a busy box can trip palaestrAI's simulation_timeout and
# kill both.
#
# Requires: docker compose up -d timescale;  ~15 min total.
#
# Artefacts (logs, per-run runtime configs) go to _outputs/whitelist_check/,
# which is git-ignored -- they are per-machine. The run files themselves live
# in palaestrai_socal/experiment_whitelist_check_*.yml and are versioned.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PGURI="${PGURI:-postgresql://palaestrai:socal_local@127.0.0.1:5433}"
OUT=_outputs/whitelist_check
mkdir -p "$OUT"

run_one () {                       # $1 = arm, $2 = bus port
  local arm="$1" port="$2"
  local db="wl_${arm}" yml="palaestrai_socal/experiment_whitelist_check_${arm}.yml"

  .venv/bin/python - <<EOF
import psycopg2
c = psycopg2.connect("$PGURI/postgres"); c.autocommit = True
c.cursor().execute("DROP DATABASE IF EXISTS $db")
c.cursor().execute("CREATE DATABASE $db")
EOF
  cat > "$OUT/rt_${arm}.conf.yaml" <<EOF
store_uri: "$PGURI/$db"
executor_bus_port: $port
logger_port: 0
public_bind: False
store_buffer_size: 100
EOF
  env PYTHONPATH="$PWD" .venv/bin/palaestrai \
    -c "$OUT/rt_${arm}.conf.yaml" database-create >/dev/null 2>&1

  echo "=== running ${arm} ==="
  local t0 t1
  t0=$(date +%s)
  env PYTHONPATH="$PWD" .venv/bin/palaestrai \
    -c "$OUT/rt_${arm}.conf.yaml" start "$yml" > "$OUT/${arm}.log" 2>&1
  t1=$(date +%s)
  echo "    wall $((t1 - t0)) s | criticals: $(grep -icE 'CRITICAL|Traceback' "$OUT/${arm}.log")"
}

run_one off  4251
run_one off2 4252
run_one on   4253

echo
echo "############### 1. determinism control (OFF vs OFF2) ###############"
if .venv/bin/python analysis/compare_runs.py \
     --a "$PGURI/wl_off" --b "$PGURI/wl_off2" --label-a OFF --label-b OFF2; then
  echo
  echo "############### 2. whitelist effect (OFF vs ON) ####################"
  .venv/bin/python analysis/compare_runs.py \
    --a "$PGURI/wl_off" --b "$PGURI/wl_on" --label-a OFF --label-b ON
else
  echo
  echo "CONTROL FAILED: two identical configurations diverged."
  echo "The scenario is not reproducible, so OFF vs ON cannot be interpreted."
  echo "Find the residual nondeterminism before drawing any conclusion."
  exit 1
fi
