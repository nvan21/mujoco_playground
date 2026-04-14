#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "Usage: $0 --env_name ENV --checkpoint PATH [options]"
  echo ""
  echo "Required:"
  echo "  --env_name ENV           Environment name (e.g. LeapCubeReorient)"
  echo ""
  echo "Optional (auto-detected if omitted):"
  echo "  --checkpoint PATH        Path to checkpoint directory (default: most recent logs/<ENV>-*/checkpoints)"
  echo ""
  echo "Optional:"
  echo "  --num_videos N           Number of rollout videos to record (default: 1)"
  echo "  --seed N                 Random seed (default: 1)"
  echo "  --logdir DIR             Output directory for videos (default: logs/)"
  echo "  --suffix STR             Suffix appended to experiment name"
  echo "  --config_overrides JSON  JSON string of playground env config overrides"
  exit 1
}

ENV_NAME=""
CHECKPOINT=""
NUM_VIDEOS=1
SEED=1
LOGDIR=""
SUFFIX=""
CONFIG_OVERRIDES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env_name)         ENV_NAME="$2";        shift 2 ;;
    --checkpoint)       CHECKPOINT="$2";      shift 2 ;;
    --num_videos)       NUM_VIDEOS="$2";      shift 2 ;;
    --seed)             SEED="$2";            shift 2 ;;
    --logdir)           LOGDIR="$2";          shift 2 ;;
    --suffix)           SUFFIX="$2";          shift 2 ;;
    --config_overrides) CONFIG_OVERRIDES="$2"; shift 2 ;;
    -h|--help)          usage ;;
    *) echo "Unknown flag: $1"; usage ;;
  esac
done

if [[ -z "$ENV_NAME" ]]; then
  echo "Error: --env_name is required."
  usage
fi

if [[ -z "$CHECKPOINT" ]]; then
  SEARCH_BASE="${LOGDIR:-$SCRIPT_DIR/logs}"
  CHECKPOINT=$(ls -d "$SEARCH_BASE/${ENV_NAME}"-*/checkpoints 2>/dev/null | sort | tail -1)
  if [[ -z "$CHECKPOINT" ]]; then
    echo "Error: no checkpoint found under $SEARCH_BASE for env '$ENV_NAME'."
    exit 1
  fi
  echo "Auto-selected checkpoint: $CHECKPOINT"
fi

CMD=(
  python "$SCRIPT_DIR/train_jax_ppo.py"
  --play_only
  --run_evals=false
  --env_name="$ENV_NAME"
  --load_checkpoint_path="$CHECKPOINT"
  --num_videos="$NUM_VIDEOS"
  --seed="$SEED"
)

[[ -n "$LOGDIR" ]]           && CMD+=(--logdir="$LOGDIR")
[[ -n "$SUFFIX" ]]           && CMD+=(--suffix="$SUFFIX")
[[ -n "$CONFIG_OVERRIDES" ]] && CMD+=(--playground_config_overrides="$CONFIG_OVERRIDES")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
