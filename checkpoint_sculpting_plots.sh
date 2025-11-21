#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Usage: bash checkpoint_sculpting_plots.sh 0-80

RANGE="$1"

LO="${RANGE%-*}"
HI="${RANGE#*-}"

# Epoch list (increments of 10 from 10 to 80)
# EPOCHS=({$LO..$HI..10})
EPOCHS=( $(seq "$LO" 10 "$HI") )
# Or explicitly: EPOCHS=(10 20 30 40 50 60 70 80)

BASE="Version_20250524_MVAID_forPreApp"
CKPT_DIR="$BASE/after_random_search_best1/checkpoints"
WORK_CKPT="$BASE/after_random_search_best1/mlp.pth"
CONF_DIR="config/Version_20250524_MVAID_forPreApp"
OUT_DIR="$BASE/optuna_categorization"

for EPOCH in "${EPOCHS[@]}"; do
  echo "Starting epoch $EPOCH"

  src="$CKPT_DIR/epoch${EPOCH}/mlp.pth"
  cp -f "$src" "$WORK_CKPT"

  python3 -m utils.predictions \
    --input_dir "$BASE" \
    --checkpoint_path "$WORK_CKPT" \
    --training_config_path "$CONF_DIR/training_config.yaml"

  python3 run_multiclass_strategy.py \
    --config_path "$CONF_DIR/" \
    --out_path "$BASE" \
    --get_predictions \
    --perform_categorisation

  dest="$OUT_DIR/epoch${EPOCH}"
  mkdir -p "$dest"

  # Move summary image if it exists
  [[ -f "$OUT_DIR/category_summary_new.png" ]] && mv -f "$OUT_DIR/category_summary_new.png" "$dest/"

  # Move mass sculpting plots if present
  if [[ -d "$OUT_DIR/mass_sculpting_plots" ]]; then
    mkdir -p "$dest/mass_sculpting_plots"
    pngs=("$OUT_DIR"/mass_sculpting_plots/*.png)
    if ((${#pngs[@]})); then
      mv -f "${pngs[@]}" "$dest/mass_sculpting_plots/"
    fi
  fi

  echo "Finished epoch $EPOCH"
done

