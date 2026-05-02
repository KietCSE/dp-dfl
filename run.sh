#!/bin/bash
# Run all robustness scenarios sequentially. Each scenario logs to its own file
# in logs/ so the terminal stays clean. Tail a specific run live:
#   tail -f logs/01_mnist_dfl_fedavg_ldp_20_sign.log
#
# Ctrl+C kills the current python run AND any leftover workers, then exits.

# Make this script the leader of its own process group so kill -- -$$ targets
# every descendant (python + multiprocessing workers) on Ctrl+C.
set -m

cleanup() {
  echo ""
  echo "Interrupted. Killing all child processes..."
  # Send SIGTERM to the entire process group of this script
  kill -TERM -- -$$ 2>/dev/null
  # Give children a moment to clean up, then SIGKILL anything still alive
  sleep 1
  kill -KILL -- -$$ 2>/dev/null
  exit 130
}
trap cleanup SIGINT SIGTERM

mkdir -p logs

CONFIGS=(
  "mnist/dfl_fedavg_ldp_20_sign.yaml"
  "mnist/dfl_fedavg_ldp_20_scale.yaml"
  "mnist/dfl_fedavg_ldp_20_label.yaml"
  "mnist/dfl_fedavg_ldp_30_sign.yaml"
  "mnist/dfl_fedavg_ldp_30_scale.yaml"
  "mnist/dfl_fedavg_ldp_30_label.yaml"
  "fashion_mnist/dfl_fedavg_ldp_20_sign.yaml"
  "fashion_mnist/dfl_fedavg_ldp_20_scale.yaml"
  "fashion_mnist/dfl_fedavg_ldp_20_label.yaml"
  "fashion_mnist/dfl_fedavg_ldp_30_sign.yaml"
  "fashion_mnist/dfl_fedavg_ldp_30_scale.yaml"
  "fashion_mnist/dfl_fedavg_ldp_30_label.yaml"
)

TOTAL=${#CONFIGS[@]}
START_TS=$(date +%s)
PASS=0
FAIL=0
FAILED_LIST=()

echo "Running $TOTAL scenarios. Logs in ./logs/"
echo "Tip: tail -f logs/<file>.log  to watch a specific run."
echo "----------------------------------------------------------------"

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  num=$((i + 1))
  slug=$(echo "$cfg" | tr '/' '_' | sed 's/\.yaml$//')
  log="logs/$(printf '%02d' "$num")_${slug}.log"

  printf "[%2d/%d] %s  RUN  %s\n" "$num" "$TOTAL" "$(date +%H:%M:%S)" "$cfg"
  scenario_start=$(date +%s)

  if python run.py -a dp-fedavg "config/experiments/robustness/$cfg" > "$log" 2>&1; then
    elapsed=$(( $(date +%s) - scenario_start ))
    printf "[%2d/%d] %s  OK   %s  (%ds)  -> %s\n" "$num" "$TOTAL" "$(date +%H:%M:%S)" "$cfg" "$elapsed" "$log"
    PASS=$((PASS + 1))
  else
    elapsed=$(( $(date +%s) - scenario_start ))
    printf "[%2d/%d] %s  FAIL %s  (%ds)  -> %s\n" "$num" "$TOTAL" "$(date +%H:%M:%S)" "$cfg" "$elapsed" "$log"
    FAIL=$((FAIL + 1))
    FAILED_LIST+=("$cfg")
  fi
done

TOTAL_ELAPSED=$(( $(date +%s) - START_TS ))
echo "----------------------------------------------------------------"
printf "Done in %ds. Pass: %d/%d  Fail: %d/%d\n" "$TOTAL_ELAPSED" "$PASS" "$TOTAL" "$FAIL" "$TOTAL"
if [ "$FAIL" -gt 0 ]; then
  echo "Failed scenarios:"
  for f in "${FAILED_LIST[@]}"; do echo "  - $f"; done
  exit 1
fi
