#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Usage: bash checkpoint_sculpting_plots.sh 0-80

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 RANGE (e.g., 0-80)"
  exit 1
fi
RANGE="$1"

LO="${RANGE%-*}"
HI="${RANGE#*-}"

# Epoch list (increments of 10 from 10 to 80)
# EPOCHS=({$LO..$HI..10})
EPOCHS=( $(seq "$LO" 10 "$HI") )
# for epoch in "${EPOCHS[@]}"; do
#     echo $epoch
# done
# exit
# Or explicitly: EPOCHS=(10 20 30 40 50 60 70 80)

BASE="Version_20250524_MVAID_forPreApp"
CKPT_DIR="$BASE/after_random_search_best1/checkpoints"
WORK_CKPT="$BASE/after_random_search_best1/mlp.pth"
CONF_DIR="config/Version_20250524_MVAID_forPreApp"
OUT_DIR="$BASE/optuna_categorization"

for EPOCH in "${EPOCHS[@]}"; do
  src="$CKPT_DIR/epoch${EPOCH}/mlp.pth"
  if [[ ! -f "$src" ]]; then
    echo "Error: Checkpoint not found: $src"
    exit 1
  fi

  cp -f "$src" "$WORK_CKPT"

  python3 run_multiclass_strategy.py \
    --config_path "$CONF_DIR/" \
    --out_path "$BASE" \
    --get_predictions \
    --perform_categorisation \
    || exit 1

  dest="$OUT_DIR/epoch${EPOCH}"
  mkdir -p "$dest"

  # Move summary image if it exists
  [[ -f "$OUT_DIR/category_summary_new.png" ]] && mv -f "$OUT_DIR/category_summary_new.png" "$dest/"

  # Move summary csv and parquets if they exist
  [[ -f "$OUT_DIR/category_summary_data.parquet" ]] && mv -f "$OUT_DIR/category_summary_data.parquet" "$dest/"
  [[ -f "$OUT_DIR/category_summary_data.csv" ]] && mv -f "$OUT_DIR/category_summary_data.csv" "$dest/"

  # Move event yields pdf if it exists
  [[ -f "$OUT_DIR/event_yields_100_180.pdf" ]] && mv -f "$OUT_DIR/event_yields_100_180.pdf" "$dest/"
  
  # Move categorization yields text file if it exists
  [[ -f "$OUT_DIR/categorization_yields.txt" ]] && mv -f "$OUT_DIR/categorization_yields.txt" "$dest/"

  # Move best cut params files if they exist
  [[ -f "$OUT_DIR/best_cut_params.json" ]] && mv -f "$OUT_DIR/best_cut_params.json" "$dest/"
  [[ -f "$OUT_DIR/best_cut_params.txt" ]] && mv -f "$OUT_DIR/best_cut_params.txt" "$dest/"



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

