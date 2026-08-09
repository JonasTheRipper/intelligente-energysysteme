#!/usr/bin/env bash
# =============================================================================
#  run_experiment.sh -- end-to-end driver for the SoCal wildfire / grid testbed
# =============================================================================
#
#  WHAT THIS SCRIPT IS FOR
#  -----------------------
#  This is the single entry point for a full experiment. It walks through the
#  five things that have to happen, in order, and explains each one as it goes:
#
#      stage  docker    start the PostGIS + TimescaleDB containers
#      stage  gis       load the GIS layers into PostGIS (one-shot, idempotent)
#      stage  schema    create the palaestrAI result schema in TimescaleDB
#      stage  simulate  run the experiment (the long part: ~30-45 min)
#      stage  analysis  render figures and reports from the stored results
#
#  It is written to be read. If you are new to the testbed, run it once with
#  --dry-run and read the commands it prints; that is the whole workflow.
#
#
#  WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
#  -----------------------------------------
#  It does NOT install Python packages. That is intentional, so the script can
#  be pointed at whatever checkout you are working on:
#
#      python -m venv .venv && source .venv/bin/activate
#      pip install -e /path/to/palaestrai            # e.g. an MR branch
#      pip install -e /path/to/midas-powergrid       # e.g. an MR branch
#      pip install -e .                              # this repo
#
#  The preflight stage prints exactly which files each import resolves to, so
#  you can confirm at a glance that you are testing the branch you think you
#  are and not a stale wheel from PyPI.
#
#
#  QUICK START
#  -----------
#      ./run_experiment.sh --dry-run          # print the plan, change nothing
#      ./run_experiment.sh                    # full run, default experiment
#      ./run_experiment.sh --only analysis    # re-render figures from old data
#
#
#  TEACHING NOTES
#  --------------
#  * Every stage is separately runnable (--only / --skip). The interesting
#    failure modes are stage-local, so students can break one stage and rerun
#    just that one instead of paying 45 minutes for each iteration.
#  * The simulation writes to a *shared* TimescaleDB database. Runs are kept
#    apart by their experiment run uid and phase uids, not by separate
#    databases. --tag only names the log files and the figure output folder.
#  * The phase list handed to the analysis scripts is derived from the
#    experiment YAML itself (see build_phase_spec below), so the figures always
#    describe the phases that actually ran.
#
# =============================================================================

set -euo pipefail

# Resolve the repository root from the script's own location, so the script
# works no matter which directory it is called from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
#  Defaults. Every one of these can be overridden from the command line.
# -----------------------------------------------------------------------------

# The experiment to run. experiment_eaton_firefighting.yml is the reference
# scenario: the January 2025 Eaton fire, six phases, from "no firefighters at
# all" through to a trained DRL triage agent.
EXPERIMENT="palaestrai_socal/experiment_eaton_firefighting.yml"

# The palaestrAI runtime config. This is what selects the *store*: the default
# runtime.conf.yaml points at TimescaleDB on 127.0.0.1:5433, which is the port
# docker-compose.yml publishes. runtime_sqlite.conf.yaml is the fallback for
# machines without Docker; it is slower and cannot be queried while running.
RUNTIME="runtime.conf.yaml"

# Names the log files and the analysis output directory. Not a database name.
TAG=""

# Stage selection. STAGES is the authoritative list; --only/--skip are checked
# against it so that a typo (`--only analsis`) fails immediately instead of
# silently running nothing and looking like a success.
STAGES=(docker gis schema simulate analysis)
ONLY=""
SKIP=""

DRY_RUN=0     # print commands instead of running them
DETACH=0      # run the simulation with setsid and return immediately
FOLLOW=0      # tail the run log after starting a detached run
ASSUME_YES=0  # skip the confirmation prompt before the long stage

# -----------------------------------------------------------------------------
#  Small output helpers. Colour only when stdout is a terminal, so that piping
#  the output into a file or into CI produces clean text.
# -----------------------------------------------------------------------------
if [[ -t 1 ]]; then
    C_HEAD=$'\033[1;36m'; C_OK=$'\033[1;32m'; C_WARN=$'\033[1;33m'
    C_ERR=$'\033[1;31m';  C_DIM=$'\033[2m';   C_OFF=$'\033[0m'
else
    C_HEAD=""; C_OK=""; C_WARN=""; C_ERR=""; C_DIM=""; C_OFF=""
fi

stage_banner() {
    printf '\n%s========================================================================%s\n' "$C_HEAD" "$C_OFF"
    printf '%s  STAGE: %s%s\n' "$C_HEAD" "$1" "$C_OFF"
    printf '%s  %s%s\n' "$C_DIM" "$2" "$C_OFF"
    printf '%s========================================================================%s\n' "$C_HEAD" "$C_OFF"
}
info()  { if [[ -z "$*" ]]; then printf '\n'; else printf '  %s\n' "$*"; fi; }
ok()    { printf '  %s+%s %s\n' "$C_OK" "$C_OFF" "$*"; }
warn()  { printf '  %s!%s %s\n' "$C_WARN" "$C_OFF" "$*" >&2; }
die()   { printf '\n  %sFAILED:%s %s\n\n' "$C_ERR" "$C_OFF" "$*" >&2; exit 1; }

# run: echo a command, then execute it (or not, under --dry-run).
# Every state-changing command in this script goes through here, which is what
# makes --dry-run a trustworthy description of the real run.
#
# The echo uses %q so arguments are printed *shell-quoted*. That matters more
# than it looks: the --phases labels contain spaces and --cities contains
# semicolons, so an unquoted echo would print a line that breaks when pasted
# into a terminal. What --dry-run prints is meant to be copy-pasteable.
run() {
    printf '  %s$' "$C_DIM"
    printf ' %q' "$@"
    printf '%s\n' "$C_OFF"
    [[ $DRY_RUN -eq 1 ]] && return 0
    "$@"
}

usage() {
    sed -n '2,60p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

OPTIONS
    -e, --experiment PATH   Experiment YAML.       [experiment_eaton_firefighting.yml]
    -r, --runtime PATH      palaestrAI runtime conf.            [runtime.conf.yaml]
    -t, --tag NAME          Label for logs and figures.   [derived from experiment]
        --only STAGE[,...]  Run only these stages.
        --skip STAGE[,...]  Run everything except these stages.
        --detach            Start the simulation detached and return.
        --follow            With --detach, tail the log afterwards.
    -n, --dry-run           Print every command without running anything.
    -y, --yes               Do not prompt before the long simulation stage.
    -h, --help              This text.

    Stages: docker, gis, schema, simulate, analysis

EXAMPLES
    ./run_experiment.sh --dry-run
    ./run_experiment.sh --experiment palaestrai_socal/experiment_m1_smoke.yml --tag smoke
    ./run_experiment.sh --only analysis --tag eaton_v07
    ./run_experiment.sh --detach --follow -y
USAGE
}

# -----------------------------------------------------------------------------
#  Argument parsing.
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--experiment) EXPERIMENT="$2"; shift 2 ;;
        -r|--runtime)    RUNTIME="$2";    shift 2 ;;
        -t|--tag)        TAG="$2";        shift 2 ;;
        --only)          ONLY="$2";       shift 2 ;;
        --skip)          SKIP="$2";       shift 2 ;;
        --detach)        DETACH=1;        shift ;;
        --follow)        FOLLOW=1;        shift ;;
        -n|--dry-run)    DRY_RUN=1;       shift ;;
        -y|--yes)        ASSUME_YES=1;    shift ;;
        -h|--help)       usage; exit 0 ;;
        *) die "unknown option: $1  (try --help)" ;;
    esac
done

# Default tag: the experiment filename without the experiment_ prefix and the
# extension. experiment_eaton_firefighting.yml -> eaton_firefighting
if [[ -z "$TAG" ]]; then
    TAG="$(basename "$EXPERIMENT" .yml)"
    TAG="${TAG#experiment_}"
fi

# Reject unknown stage names before doing any work.
validate_stages() {
    local list="$1" flag="$2" name known
    IFS=',' read -r -a _requested <<<"$list"
    for name in "${_requested[@]}"; do
        [[ -z "$name" ]] && continue
        known=0
        for s in "${STAGES[@]}"; do [[ "$s" == "$name" ]] && known=1; done
        [[ $known -eq 1 ]] || die "$flag: unknown stage '$name'
       Valid stages: ${STAGES[*]}"
    done
}
[[ -n "$ONLY" ]] && validate_stages "$ONLY" "--only"
[[ -n "$SKIP" ]] && validate_stages "$SKIP" "--skip"

want_stage() {
    local s="$1"
    if [[ -n "$ONLY" ]]; then [[ ",$ONLY," == *",$s,"* ]]; return; fi
    if [[ -n "$SKIP" && ",$SKIP," == *",$s,"* ]]; then return 1; fi
    return 0
}

LOG_DIR="$REPO_ROOT/_outputs"
RUN_LOG="$LOG_DIR/${TAG}_run.log"
SAMPLER_LOG="$LOG_DIR/${TAG}_sampler.log"
FIG_DIR="$REPO_ROOT/analysis/_${TAG}"

# =============================================================================
#  PREFLIGHT -- always runs. Cheap checks that fail before the expensive stages.
# =============================================================================
stage_banner "preflight" "Check the toolchain and show which code will actually be imported."

[[ -f "$EXPERIMENT" ]] || die "experiment YAML not found: $EXPERIMENT"
[[ -f "$RUNTIME"    ]] || die "runtime config not found: $RUNTIME"
ok "experiment: $EXPERIMENT"
ok "runtime:    $RUNTIME"
ok "tag:        $TAG"

command -v python >/dev/null 2>&1 || die "no 'python' on PATH -- activate your venv first"
info "python:     $(python -VV | head -1)  ($(command -v python))"

# The provenance dump. This is the check that matters when you are running from
# MR branches: an editable install shows a path inside your checkout, a release
# wheel shows a path inside site-packages. If you meant to test a branch and see
# site-packages here, your pip install -e did not take effect.
info ""
info "Imports resolve to:"
python - <<'PY' || die "a required package is missing -- install the project first (see the header of this script)"
import importlib, sys
REQUIRED = ["palaestrai", "midas_powergrid", "pandapower", "palaestrai_socal"]
OPTIONAL = ["midas", "midas_palaestrai", "palaestrai_mosaik", "mosaik"]
missing = []
def show(name, required):
    try:
        m = importlib.import_module(name)
    except Exception as exc:
        (missing.append(name) if required else None)
        print(f"      {name:<20} -- NOT IMPORTABLE ({type(exc).__name__})")
        return
    ver = getattr(m, "__version__", "-")
    # Namespace packages have __file__ == None; fall back to their search path.
    path = getattr(m, "__file__", None)
    if not path:
        path = next(iter(getattr(m, "__path__", []) or []), "?")
    origin = "site-packages" if "/site-packages/" in path else "editable"
    print(f"      {name:<20} {ver:<12} [{origin}] {path}")
for n in REQUIRED:
    show(n, True)
for n in OPTIONAL:
    show(n, False)
sys.exit(1 if missing else 0)
PY

# Docker is only needed for the container-backed stages. A SQLite runtime does
# not need it at all, so this is a warning rather than a hard failure.
NEED_DOCKER=0
want_stage docker && NEED_DOCKER=1
want_stage gis    && NEED_DOCKER=1
if [[ $NEED_DOCKER -eq 1 ]]; then
    if ! command -v docker >/dev/null 2>&1; then
        die "docker is required for the docker/gis stages.
       Either install Docker, or skip them and use the SQLite fallback:
           ./run_experiment.sh --skip docker,gis --runtime runtime_sqlite.conf.yaml"
    fi
    docker compose version >/dev/null 2>&1 || die "'docker compose' (v2) not available"
    ok "docker:     $(docker --version)"
fi

# Pull the store URI out of the runtime config rather than hard-coding it, so
# switching --runtime automatically switches every downstream psql/analysis call.
STORE_URI="$(python - "$RUNTIME" <<'PY'
import sys, yaml
conf = yaml.safe_load(open(sys.argv[1])) or {}
print(conf.get("store_uri", ""))
PY
)"
[[ -n "$STORE_URI" ]] || die "no store_uri in $RUNTIME"
# Mask the password when printing.
info "store:      $(sed -E 's#://([^:]+):[^@]*@#://\1:***@#' <<<"$STORE_URI")"

mkdir -p "$LOG_DIR"

# =============================================================================
#  STAGE 1 -- docker
# =============================================================================
if want_stage docker; then
    stage_banner "docker" "Start PostGIS (input GIS data) and TimescaleDB (simulation results)."

    # Two databases, two jobs:
    #   postgis   :5432  the *input* side -- fuel, terrain, WUI, grid geometry
    #   timescale :5433  the *output* side -- palaestrAI's result store
    # They are separate because the input layers are static reference data you
    # load once, while the result store is rewritten by every run.
    info "postgis   -> localhost:5432   input GIS layers"
    info "timescale -> localhost:5433   palaestrAI result store"
    info ""
    run docker compose up -d postgis timescale

    # docker-compose.yml defines healthchecks for both services. Waiting on the
    # healthcheck rather than sleeping avoids the classic race where the schema
    # stage connects before Postgres has finished its own init scripts.
    if [[ $DRY_RUN -eq 0 ]]; then
        info ""
        info "Waiting for both containers to report healthy ..."
        for svc in socal_wildfire_postgis socal_wildfire_timescale; do
            for _ in $(seq 1 60); do
                state="$(docker inspect -f '{{.State.Health.Status}}' "$svc" 2>/dev/null || echo missing)"
                [[ "$state" == "healthy" ]] && break
                sleep 2
            done
            [[ "$state" == "healthy" ]] || die "$svc did not become healthy (state: $state)
       Look at the container log:  docker logs $svc"
            ok "$svc healthy"
        done
    fi
fi

# =============================================================================
#  STAGE 2 -- gis
# =============================================================================
if want_stage gis; then
    stage_banner "gis" "Load fuel/terrain/WUI rasters into PostGIS. Idempotent; safe to repeat."

    # The gis-loader service is a one-shot container: it installs its own deps,
    # runs wildfire_cma.postgis_load, and exits. `--exit-code-from` makes its
    # exit status our exit status, so a failed load stops the pipeline here
    # instead of surfacing as a confusing empty-raster error 40 minutes later.
    info "This reads the source rasters and writes them into the 'wildfire' database."
    info "It takes a few minutes the first time and is a no-op afterwards."
    info ""
    run docker compose up --exit-code-from gis-loader gis-loader
    ok "GIS layers loaded"
fi

# =============================================================================
#  STAGE 3 -- schema
# =============================================================================
if want_stage schema; then
    stage_banner "schema" "Create palaestrAI's result tables and TimescaleDB hypertables."

    # database-create is idempotent in the sense that it will not clobber an
    # existing schema, but it will complain if the schema is already there.
    # For a teaching script that is noise, not an error, so it is tolerated.
    info "Creates the tables palaestrAI writes into, and turns the large"
    info "time-series tables (world_states, muscle_actions, ...) into hypertables."
    info ""
    if [[ $DRY_RUN -eq 1 ]]; then
        run palaestrai -c "$RUNTIME" database-create
    else
        printf '  %s$ palaestrai -c %s database-create%s\n' "$C_DIM" "$RUNTIME" "$C_OFF"
        if palaestrai -c "$RUNTIME" database-create 2>&1 | sed 's/^/      /'; then
            ok "schema ready"
        else
            warn "database-create reported a problem -- this is expected and harmless"
            warn "if the schema already exists. Verify with:"
            warn "    psql \"\$STORE_URI\" -c '\\dt'"
        fi
    fi
fi

# =============================================================================
#  STAGE 4 -- simulate    (the long one)
# =============================================================================
if want_stage simulate; then
    stage_banner "simulate" "Run the experiment. This is the ~30-45 minute stage."

    # What the student should expect to see happen:
    info "Each phase of the experiment is a separate counterfactual, and phases run"
    info "sequentially -- so peak memory is one phase's worth (~5 GB), not all of them."
    info "Every phase steps two environments in lockstep:"
    info "    gis_world   the fire: spread, suppression, damage"
    info "    socal_grid  the power system: MIDAS + pandapower power flow"
    info ""
    info "Phases in this experiment:"
    python - "$EXPERIMENT" <<'PY'
import sys, yaml
doc = yaml.safe_load(open(sys.argv[1])) or {}
for entry in doc.get("schedule", []):
    for uid, body in entry.items():
        eps = (body.get("phase_config") or {}).get("episodes", "?")
        print(f"      {uid:<24} episodes: {eps}")
PY
    info ""
    info "Logs:"
    info "    run log     $RUN_LOG"
    info "    resources   $SAMPLER_LOG"
    info ""

    if [[ $DRY_RUN -eq 0 && $ASSUME_YES -eq 0 && $DETACH -eq 0 ]]; then
        read -r -p "  Start the simulation now? [y/N] " reply
        [[ "$reply" =~ ^[Yy]$ ]] || die "aborted by user"
    fi

    # A tiny background sampler records RSS and disk every 30 s. This is what
    # turns "it died" into "it died right after phase 3 at 7.8 GB", which is the
    # difference between a diagnosable run and a mystery.
    start_sampler() {
        (
            while true; do
                printf '%s  rss_mb=%s  store_mb=%s\n' \
                    "$(date -Is)" \
                    "$(ps -eo rss,comm | awk '/palaestr|python/ {s+=$1} END {printf "%.0f", s/1024}')" \
                    "$(du -sm "$LOG_DIR" 2>/dev/null | cut -f1)"
                sleep 30
            done
        ) >>"$SAMPLER_LOG" 2>&1 &
        echo $!
    }

    # The run itself. The EXIT_RC marker at the end of the log is the contract
    # the docs and the monitoring commands rely on -- grep for it to find out
    # how a detached run ended.
    #
    # IMPORTANT, and a good teaching point: palaestrAI can exit 0 even when a
    # subprocess logged CRITICAL and died. Do not trust the return code alone;
    # the analysis stage below is the real proof that usable data was produced.
    SIM_CMD=(palaestrai -c "$RUNTIME" start "$EXPERIMENT")

    if [[ $DRY_RUN -eq 1 ]]; then
        run "${SIM_CMD[@]}"
        if [[ $DETACH -eq 1 ]]; then
            # Keep the dry run honest: a real detached run returns here and
            # never reaches the analysis stage, because there is nothing to
            # analyse yet.
            info ""
            info "(--detach: the real run would stop here and leave the"
            info " simulation in the background; any later stage is skipped.)"
            exit 0
        fi
    elif [[ $DETACH -eq 1 ]]; then
        printf '  %s$ setsid %s%s\n' "$C_DIM" "${SIM_CMD[*]}" "$C_OFF"
        SAMPLER_PID="$(start_sampler)"
        setsid bash -c '
            "$@" >>"'"$RUN_LOG"'" 2>&1
            echo "'"$TAG"'_EXIT_RC=$?" >>"'"$RUN_LOG"'"
        ' _ "${SIM_CMD[@]}" </dev/null >/dev/null 2>&1 &
        ok "detached (sampler pid $SAMPLER_PID)"
        info ""
        info "Monitor with:"
        info "    grep EXIT_RC $RUN_LOG     # finished? want ${TAG}_EXIT_RC=0"
        info "    tail -f $RUN_LOG"
        info "    pgrep -fc 'palaestrAI\\['      # still alive?"
        if [[ $FOLLOW -eq 1 ]]; then
            info ""
            info "Following the log (Ctrl-C stops watching, not the run) ..."
            tail -f "$RUN_LOG"
        fi
        # A detached run has not produced results yet, so analysis is skipped.
        info ""
        info "Run the analysis once it finishes:"
        info "    ./run_experiment.sh --only analysis --tag $TAG"
        exit 0
    else
        printf '  %s$ %s%s\n' "$C_DIM" "${SIM_CMD[*]}" "$C_OFF"
        SAMPLER_PID="$(start_sampler)"
        # shellcheck disable=SC2064
        trap "kill $SAMPLER_PID 2>/dev/null || true" EXIT
        set +e
        "${SIM_CMD[@]}" 2>&1 | tee "$RUN_LOG"
        SIM_RC=${PIPESTATUS[0]}
        set -e
        echo "${TAG}_EXIT_RC=${SIM_RC}" >>"$RUN_LOG"
        kill "$SAMPLER_PID" 2>/dev/null || true
        if [[ $SIM_RC -ne 0 ]]; then
            die "simulation exited with rc=$SIM_RC -- see $RUN_LOG"
        fi
        ok "simulation finished (rc=0)"
        warn "rc=0 is necessary but not sufficient; the analysis stage is the real check."
    fi
fi

# =============================================================================
#  STAGE 5 -- analysis
# =============================================================================
if want_stage analysis; then
    stage_banner "analysis" "Turn the stored time series into figures and reports."

    mkdir -p "$FIG_DIR"

    # The renderers need to know which phases to compare and how to label them.
    # Deriving that from the experiment YAML (uid, n_planes, a readable label)
    # keeps the figures honest: they describe the phases that actually ran,
    # even if someone edits the experiment.
    PHASE_SPEC="$(python - "$EXPERIMENT" <<'PY'
import json, re, sys, yaml
doc = yaml.safe_load(open(sys.argv[1])) or {}
parts = []
for entry in doc.get("schedule", []):
    for uid, body in entry.items():
        # n_planes lives inside the environment params; find it wherever it is.
        found = re.search(r'"n_planes":\s*(\d+)', json.dumps(body))
        planes = found.group(1) if found else "0"
        label = uid.split("_", 2)[-1].replace("_", " ")
        parts.append(f"{uid}:{planes}:{label}")
print(",".join(parts))
PY
)"
    [[ -n "$PHASE_SPEC" ]] || die "could not derive the phase list from $EXPERIMENT"
    info "Phase spec (uid:planes:label), derived from the experiment YAML:"
    printf '      %s\n' "${PHASE_SPEC//,/$'\n      '}"
    info ""

    # Sanity check before rendering: an empty store produces empty figures and
    # a very confusing debugging session. Fail loudly instead.
    if [[ $DRY_RUN -eq 0 ]] && command -v psql >/dev/null 2>&1; then
        rows="$(psql "$STORE_URI" -tAc \
            "SELECT count(*) FROM world_states;" 2>/dev/null || echo "")"
        if [[ -z "$rows" ]]; then
            warn "could not query the store -- is the container up?"
        elif [[ "$rows" == "0" ]]; then
            die "the store holds 0 world_states rows: the simulation produced no data.
       Check $RUN_LOG for a CRITICAL message."
        else
            ok "store holds $rows world_state snapshots"
        fi
    fi

    # 5a. Grid metrics: per-phase voltage, loading and served-load curves. This
    #     is the figure that shows what the fire did to the power system.
    info ""
    info "-- grid metrics report"
    run python analysis/grid_metrics_report.py \
        --store "$STORE_URI" \
        --out "$FIG_DIR/grid_metrics_${TAG}.png" \
        --phases "$PHASE_SPEC" \
        --title "SoCal testbed -- $TAG"

    # 5b. Firefighter report: how much suppression each phase bought, keyed by
    #     the number of aircraft. --run is repeatable as n_planes=store_uri, so
    #     the same call can compare several stores side by side.
    info ""
    info "-- firefighter effectiveness report"
    run python analysis/firefighter_report.py \
        --run "3=$STORE_URI" \
        --gis-uid gis_world \
        --grid-uid socal_grid

    # 5c. The timelapse. Slowest of the three: it renders one map frame per
    #     simulated step per phase, over a satellite basemap, then encodes video.
    #     --stride 2 halves the frame count if you only want a quick look.
    #
    #     Two optional overlays make the map readable: the official fire
    #     perimeter (cyan dashed, so you can see how the simulated fire compares
    #     to what actually burned) and labelled place markers. Both are
    #     fire-specific, so they are chosen from the experiment name rather than
    #     hard-coded -- pointing the Eaton perimeter at a Palisades run would
    #     silently produce a misleading figure.
    case "$EXPERIMENT" in
        *palisades*)
            PERIMETER="data/perimeters/palisades_perimeter.geojson"
            CITIES="Pacific Palisades,-118.526,34.048;Malibu,-118.667,34.032;Santa Monica,-118.491,34.020;Topanga,-118.601,34.094"
            ;;
        *)
            PERIMETER="data/perimeters/eaton_perimeter.geojson"
            CITIES="Altadena,-118.131,34.190;Pasadena,-118.145,34.156;Sierra Madre,-118.053,34.162;La Canada,-118.201,34.199"
            ;;
    esac

    TIMELAPSE_ARGS=(
        --store "$STORE_URI"
        --stride 1 --fps 10
        --outdir "$FIG_DIR"
        --phases "$PHASE_SPEC"
        --cities "$CITIES"
    )
    if [[ -f "$PERIMETER" ]]; then
        TIMELAPSE_ARGS+=(--perimeter "$PERIMETER")
    else
        warn "no perimeter file at $PERIMETER -- rendering without the overlay"
    fi

    info ""
    info "-- comparison timelapse (slow: renders every step as a map frame)"
    info "   overlay: $PERIMETER"
    run python analysis/make_comparison_timelapse.py "${TIMELAPSE_ARGS[@]}"

    ok "figures written to $FIG_DIR"
fi

# =============================================================================
printf '\n%s========================================================================%s\n' "$C_OK" "$C_OFF"
if [[ $DRY_RUN -eq 1 ]]; then
    printf '%s  DRY RUN complete -- nothing was executed.%s\n' "$C_OK" "$C_OFF"
else
    printf '%s  Done.%s\n' "$C_OK" "$C_OFF"
    printf '    run log   %s\n' "$RUN_LOG"
    printf '    figures   %s\n' "$FIG_DIR"
fi
printf '%s========================================================================%s\n\n' "$C_OK" "$C_OFF"

# -----------------------------------------------------------------------------
#  WHERE TO GO NEXT
#  ----------------
#  docs/RUNNING_THE_EXPERIMENT.md   the long-form version of this workflow,
#                                   including the Palisades fire, concurrent
#                                   runs on separate bus ports, and calibration
#  palaestrai_socal/                environments, agents and experiment YAMLs
#  analysis/                        the renderers invoked above
#  tests/                           pytest -m "not slow" is the fast gate
#
#  Shutting down:  docker compose down          (keeps the data volumes)
#                  docker compose down -v       (deletes results and GIS data)
# -----------------------------------------------------------------------------
