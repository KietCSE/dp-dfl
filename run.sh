#!/bin/bash
# Run multiple experiments sequentially. Logs to ./logs/<NN>_<algo>_<slug>.log.
# Tail a specific run live:
#   tail -f logs/01_dp-fedavg_mnist_dfl_fedavg_ldp_20_sign.log
#
# Ctrl+C kills the current python run AND any leftover workers, then exits.
#
# Usage:
#   ./run.sh                  # uses FOLDER constant below — edit it manually
#                             # to switch between utility / robustness / etc.
#
# Compatible with bash 3.2 (macOS default) — no associative arrays.

# Make this script the leader of its own process group so kill -- -$$ targets
# every descendant (python + multiprocessing workers) on Ctrl+C.
set -m

cleanup() {
  echo ""
  echo "Interrupted. Killing all child processes..."
  kill -TERM -- -$$ 2>/dev/null
  sleep 1
  kill -KILL -- -$$ 2>/dev/null
  exit 130
}
trap cleanup SIGINT SIGTERM

mkdir -p logs

# ============== CUSTOMIZE BELOW ==================================
# 1) Folder constant. Edit this manually to switch experiment scope.
#    Valid values: any subdir of config/experiments/ (e.g.
#    robustness | utility | noise_game).
FOLDER="robustness"

# 2) Per-algorithm config lists. Paths are relative to
#    config/experiments/$FOLDER/ — comment a line with `#` to skip it.
#    Var name pattern: RUNS_<ALGO_UPPER_WITH_UNDERSCORES>
#      e.g. dp-fedavg → RUNS_DP_FEDAVG
#           trimmed-mean → RUNS_TRIMMED_MEAN
#    Algorithms supported (see `python run.py --help`):
#      dpsgd-kurtosis | fedavg | krum | trust-aware | noise-game
#      dp-fedavg | cfl-fedavg | trimmed-mean | fltrust | flame
#      adaptive-noise | balance

RUNS_DP_FEDAVG="
  mnist/dfl_fedavg_ldp_scale_-5.yaml
  mnist/dfl_fedavg_ldp_scale_-10.yaml
  mnist/dfl_fedavg_ldp_scale_-20.yaml
  mnist/dfl_fedavg_ldp_scale_40.yaml
  mnist/dfl_fedavg_ldp_scale_60.yaml
  fashion_mnist/dfl_fedavg_ldp_scale_-5.yaml
  fashion_mnist/dfl_fedavg_ldp_scale_-10.yaml
  fashion_mnist/dfl_fedavg_ldp_scale_-20.yaml
  fashion_mnist/dfl_fedavg_ldp_scale_40.yaml
  fashion_mnist/dfl_fedavg_ldp_scale_60.yaml
"

RUNS_CFL_FEDAVG="
  mnist/cfl_fedavg_ldp_scale_-5.yaml
  mnist/cfl_fedavg_ldp_scale_-10.yaml
  mnist/cfl_fedavg_ldp_scale_-20.yaml
  mnist/cfl_fedavg_ldp_scale_40.yaml
  mnist/cfl_fedavg_ldp_scale_60.yaml
  fashion_mnist/cfl_fedavg_ldp_scale_-5.yaml
  fashion_mnist/cfl_fedavg_ldp_scale_-10.yaml
  fashion_mnist/cfl_fedavg_ldp_scale_-20.yaml
  fashion_mnist/cfl_fedavg_ldp_scale_40.yaml
  fashion_mnist/cfl_fedavg_ldp_scale_60.yaml
"

RUNS_KRUM="
  # mnist/krum_dp.yaml
  # fashion_mnist/krum_dp.yaml
"

RUNS_TRIMMED_MEAN="
  # mnist/trimmed_mean_dp.yaml
  # fashion_mnist/trimmed_mean_dp.yaml
"

RUNS_BALANCE="
  # mnist/balance.yaml
"

RUNS_TRUST_AWARE="
  # mnist/trust_aware.yaml
  # fashion_mnist/trust_aware.yaml
"

RUNS_NOISE_GAME="
  # ── Note: noise_game has its own folder hierarchy. To run these,
  #    change FOLDER above to \"noise_game\". Paths below are relative
  # #    to config/experiments/noise_game/.
  # robustness/mnist/20/noise_game_scale_1.yaml
  # robustness/mnist/20/noise_game_scale_2.yaml
  # robustness/mnist/20/noise_game_scale_3.yaml
  # robustness/mnist/40/noise_game_scale_1.yaml
  # robustness/mnist/40/noise_game_scale_2.yaml
  # robustness/mnist/40/noise_game_scale_3.yaml
  # robustness/mnist/60/noise_game_scale_1.yaml
  # robustness/mnist/60/noise_game_scale_2.yaml
  # robustness/mnist/60/noise_game_scale_3.yaml
  # robustness/fashion_mnist/20/noise_game_scale_1.yaml
  # robustness/fashion_mnist/20/noise_game_scale_2.yaml
  # robustness/fashion_mnist/20/noise_game_scale_3.yaml
  # robustness/fashion_mnist/40/noise_game_scale_1.yaml
  # robustness/fashion_mnist/40/noise_game_scale_2.yaml
  # robustness/fashion_mnist/40/noise_game_scale_3.yaml
  # robustness/fashion_mnist/60/noise_game_scale_1.yaml
  # robustness/fashion_mnist/60/noise_game_scale_2.yaml
  # robustness/fashion_mnist/60/noise_game_scale_3.yaml
  # utility/mnist/noise_game_8_1.yaml
  # utility/fashion_mnist/noise_game_8_1.yaml
"

RUNS_FLTRUST=""
RUNS_FLAME=""
RUNS_ADAPTIVE_NOISE=""
RUNS_DPSGD_KURTOSIS=""
RUNS_FEDAVG=""

# 3) Execution order. Algorithms not in this list are skipped even if
#    their RUNS_* variable has configs.
ALGO_ORDER=(
  dp-fedavg
  cfl-fedavg
  krum
  trimmed-mean
  balance
  trust-aware
  noise-game
  fltrust
  flame
  adaptive-noise
  dpsgd-kurtosis
  fedavg
)
# ============== END CUSTOMIZE ====================================

ROOT="config/experiments/$FOLDER"
if [ ! -d "$ROOT" ]; then
  echo "Folder not found: $ROOT"
  echo "Available: $(ls config/experiments/ 2>/dev/null | tr '\n' ' ')"
  exit 1
fi

# Map algo name (lowercase with dashes) → RUNS_* var name
algo_to_var() {
  echo "RUNS_$(echo "$1" | tr 'a-z-' 'A-Z_')"
}

# Flatten (algo, config) pairs in ALGO_ORDER, skipping empty/comment lines.
PAIRS=()
for algo in "${ALGO_ORDER[@]}"; do
  varname="$(algo_to_var "$algo")"
  list="${!varname}"
  [ -z "$list" ] && continue
  while IFS= read -r line; do
    cfg="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [ -z "$cfg" ] && continue
    case "$cfg" in \#*) continue ;; esac
    PAIRS+=("$algo|$cfg")
  done <<< "$list"
done

TOTAL=${#PAIRS[@]}
if [ "$TOTAL" -eq 0 ]; then
  echo "No configs to run (all lists empty/commented)."
  exit 0
fi

START_TS=$(date +%s)
PASS=0
FAIL=0
FAILED_LIST=()

echo "Folder: $ROOT"
echo "Running $TOTAL scenarios. Logs in ./logs/"
echo "Tip: tail -f logs/<file>.log to watch a specific run."
echo "----------------------------------------------------------------"

for i in "${!PAIRS[@]}"; do
  pair="${PAIRS[$i]}"
  algo="${pair%%|*}"
  cfg="${pair#*|}"
  num=$((i + 1))
  slug=$(echo "$cfg" | tr '/' '_' | sed 's/\.yaml$//')
  log="logs/$(printf '%02d' "$num")_${algo}_${slug}.log"

  printf "[%2d/%d] %s  RUN  [%-14s] %s\n" \
    "$num" "$TOTAL" "$(date +%H:%M:%S)" "$algo" "$cfg"
  scenario_start=$(date +%s)

  if python run.py -a "$algo" "$ROOT/$cfg" > "$log" 2>&1; then
    elapsed=$(( $(date +%s) - scenario_start ))
    printf "[%2d/%d] %s  OK   [%-14s] %s  (%ds)  -> %s\n" \
      "$num" "$TOTAL" "$(date +%H:%M:%S)" "$algo" "$cfg" "$elapsed" "$log"
    PASS=$((PASS + 1))
  else
    elapsed=$(( $(date +%s) - scenario_start ))
    printf "[%2d/%d] %s  FAIL [%-14s] %s  (%ds)  -> %s\n" \
      "$num" "$TOTAL" "$(date +%H:%M:%S)" "$algo" "$cfg" "$elapsed" "$log"
    FAIL=$((FAIL + 1))
    FAILED_LIST+=("$algo: $cfg")
  fi
done

TOTAL_ELAPSED=$(( $(date +%s) - START_TS ))
echo "----------------------------------------------------------------"
printf "Done in %ds. Pass: %d/%d  Fail: %d/%d\n" \
  "$TOTAL_ELAPSED" "$PASS" "$TOTAL" "$FAIL" "$TOTAL"
if [ "$FAIL" -gt 0 ]; then
  echo "Failed scenarios:"
  for f in "${FAILED_LIST[@]}"; do echo "  - $f"; done
  exit 1
fi
