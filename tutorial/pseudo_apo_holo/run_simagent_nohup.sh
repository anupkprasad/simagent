#!/usr/bin/env bash
# Apo/holo tutorial, started in the background.
#
#   conda activate SimAgentEnv
#   bash tutorial/pseudo_apo_holo/run_simagent_nohup.sh
#
# Edit the SimAgent options in the command below. The goal text is goal.txt
# in this same folder.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
LOG="$HERE/simagent_nohup.log"
PID_FILE="$HERE/simagent_nohup.pid"

cd "$REPO"

nohup python SimAgent.py \
  --goal "$(cat "$HERE/goal.txt")" \
  --working-dir "$HERE" \
  --force-field amber99sb-ildn \
  --water-model tip3p \
  --llm-model gpt-oss:20b \
  --llm-base-url http://127.0.0.1:11434 \
  --subtask preprocess simsetup hpcjob analysis reporter \
  --hpc-check-interval 10m \
  --allowed-hpc-jobs 4 \
  >"$LOG" 2>&1 &

echo $! >"$PID_FILE"
echo "Started SimAgent pid=$(cat "$PID_FILE")"
echo "  log: $LOG"
echo "  tail -f $LOG"
