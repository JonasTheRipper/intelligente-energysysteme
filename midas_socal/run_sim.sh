#!/usr/bin/env bash
# Run the SoCal MIDAS scenario (NOAA-weather-enriched).
#
# Usage:
#   midas_socal/run_sim.sh            # full scenario from socal_midas.yml
#   midas_socal/run_sim.sh --smoke    # short smoke config (CI / quick check)
#
# Resolves paths relative to this script so it works locally and in CI.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$ROOT"

CONFIG="$HERE/socal_midas.yml"
SCENARIO="socal_midas"
if [[ "${1:-}" == "--smoke" ]]; then
  # Prefer a dedicated smoke config if present; otherwise fall back to the
  # full scenario (still --skip-download so CI does not hit the network).
  if [[ -f "$HERE/socal_midas_smoke.yml" ]]; then
    CONFIG="$HERE/socal_midas_smoke.yml"
    SCENARIO="socal_midas_smoke"
  fi
fi

LOG="$HERE/run.log"
echo "RUN_AT=$(date)" > "$LOG"
echo "SCENARIO=$SCENARIO CONFIG=$CONFIG" >> "$LOG"

midasctl run "$SCENARIO" -c "$CONFIG" --skip-download >> "$LOG" 2>&1
RC=$?
echo "EXIT_CODE=$RC" >> "$LOG"
echo "DONE_AT=$(date)" >> "$LOG"
exit $RC
