#!/usr/bin/env bash
# Run the SoCal MIDAS scenario (NOAA-weather-enriched).
#
# Usage:
#   midas_socal/run_sim.sh            # full scenario  (socal_midas)
#   midas_socal/run_sim.sh --smoke    # short 8-step smoke run (CI / quick check)
#
# Resolves paths relative to this script so it works locally and in CI.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
# socal_midas.yml refers to the grid as a repo-relative path, so the working
# directory must be the repo root. Do not remove.
cd "$ROOT"

CONFIG="$HERE/socal_midas.yml"
SCENARIO="socal_midas"
if [[ "${1:-}" == "--smoke" ]]; then
  # socal_midas_smoke lives in the SAME config file and inherits from
  # socal_midas via `parent:`. It used to be looked up as a separate
  # socal_midas_smoke.yml that was never committed, so --smoke silently ran the
  # full one-day scenario instead of a smoke test.
  SCENARIO="socal_midas_smoke"
fi

if ! command -v midasctl >/dev/null 2>&1; then
  echo "ERROR: midasctl not on PATH. It ships with midas-mosaik, which is" >&2
  echo "       pulled in transitively by midas-powergrid / midas-palaestrai." >&2
  exit 127
fi

LOG="$HERE/run.log"
echo "RUN_AT=$(date)" > "$LOG"
echo "SCENARIO=$SCENARIO CONFIG=$CONFIG CWD=$PWD" >> "$LOG"

# tee, not '>>': the previous version sent everything to run.log only, and
# run.log was not among the CI artifacts. A failing run therefore produced a
# completely silent job with no way to see why.
set +e
midasctl run "$SCENARIO" -c "$CONFIG" --skip-download 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e

# Under `set -e` the old `RC=$?` after the command was dead code: a non-zero
# exit terminated the script before the assignment, so EXIT_CODE was only ever
# written as 0. PIPESTATUS is also required because of the tee pipeline.
echo "EXIT_CODE=$RC" >> "$LOG"
echo "DONE_AT=$(date)" >> "$LOG"
exit "$RC"
