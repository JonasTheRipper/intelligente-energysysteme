#!/usr/bin/env bash
# =============================================================================
# WHITELIST VALIDATION -- the full-length version of run_whitelist_check.sh.
#
# Five arms (see analysis/make_whitelist_configs.py), three comparisons:
#
#   OFF  vs OFF2   determinism  -- must be IDENTICAL, else nothing below means
#                                  anything and the script stops.
#   OFF  vs ON     the treatment -- expected IDENTICAL.
#   NEGB vs NEG    sensitivity  -- must DIVERGE. Without this, "IDENTICAL"
#                                  twice is equally consistent with a
#                                  comparator that cannot see anything at all.
#
# Runtime ~2 h: the two un-whitelisted 60-step arms cost ~55 min each, the
# whitelisted one ~12 min, the two 8-step sensitivity arms ~2 min each.
#
# Do NOT run this alongside anything else. Sixteen extra worker processes can
# trip palaestrAI's simulation_timeout and kill both jobs, and the wall-time
# numbers this collects are meaningless on a contended box.
#
# Artefacts -> _outputs/whitelist_validation/ (git-ignored, per-machine).
# Timings -> _outputs/whitelist_validation/timings.json, read by
# analysis/whitelist_figure.py.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PGURI="${PGURI:-postgresql://palaestrai:socal_local@127.0.0.1:5433}"
OUT=_outputs/whitelist_validation
mkdir -p "$OUT"

if pgrep -f "palaestrAI\[Executor\]" >/dev/null; then
  echo "ERROR: a palaestrAI run is already active on this machine." >&2
  echo "Wait for it to finish -- concurrent runs invalidate the timings." >&2
  exit 1
fi

.venv/bin/python analysis/make_whitelist_configs.py --check >/dev/null || {
  echo "ERROR: experiment_wlval_*.yml drifted from the generator." >&2
  echo "Run: python analysis/make_whitelist_configs.py" >&2
  exit 1
}

echo '{' > "$OUT/timings.json"
first=1

run_one () {                        # $1 = arm, $2 = bus port
  local arm="$1" port="$2" db="wlval_$1"
  local yml="palaestrai_socal/experiment_wlval_${arm}.yml"

  .venv/bin/python - <<EOF
import psycopg2
c = psycopg2.connect("$PGURI/postgres"); c.autocommit = True
cur = c.cursor()
cur.execute("select pg_terminate_backend(pid) from pg_stat_activity "
            "where datname = '$db' and pid <> pg_backend_pid()")
cur.execute("DROP DATABASE IF EXISTS $db")
cur.execute("CREATE DATABASE $db")
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
  local t0 t1 secs
  t0=$(date +%s)
  env PYTHONPATH="$PWD" .venv/bin/palaestrai \
    -c "$OUT/rt_${arm}.conf.yaml" start "$yml" > "$OUT/${arm}.log" 2>&1
  t1=$(date +%s); secs=$((t1 - t0))
  echo "    wall ${secs} s | criticals: $(grep -icE 'CRITICAL|Traceback' "$OUT/${arm}.log")"

  [ $first -eq 1 ] || echo ',' >> "$OUT/timings.json"
  printf '  "%s": %d' "$arm" "$secs" >> "$OUT/timings.json"
  first=0
}

run_one off  4271
run_one off2 4272
run_one on   4273
run_one onb  4274
run_one negb 4275
run_one neg  4276
echo >> "$OUT/timings.json"; echo '}' >> "$OUT/timings.json"

cmp_arms () {                       # $1=a $2=b $3=label-a $4=label-b
  .venv/bin/python analysis/compare_runs.py \
    --a "$PGURI/wlval_$1" --b "$PGURI/wlval_$2" --label-a "$3" --label-b "$4"
}

echo
echo "############### 1. determinism control (OFF vs OFF2) ###############"
if ! cmp_arms off off2 OFF OFF2; then
  echo
  echo "CONTROL FAILED: two identical configurations diverged."
  echo "Nothing below can be interpreted. Find the nondeterminism first."
  exit 1
fi

echo
echo "############### 2. sensitivity control (NEGB vs NEG) ##############"
echo "(this one MUST report DIVERGENCE FOUND)"
if cmp_arms negb neg NEGB NEG; then
  echo
  echo "SENSITIVITY CONTROL FAILED: a deliberately different scenario compared"
  echo "as IDENTICAL. The comparator is blind; the whitelist result below is"
  echo "worthless until this is fixed."
  exit 1
fi

echo
echo "############### 3. whitelist effect (OFF vs ON) ####################"
cmp_arms off on OFF ON

echo
echo "Timings: $OUT/timings.json"
echo "Figure:  .venv/bin/python analysis/whitelist_figure.py"
