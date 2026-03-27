#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SCENE_DIR="${SCENE_DIR:-data/360_v2}"
SCENES_STR="${SCENES:-bicycle stump bonsai counter kitchen room}"
read -r -a SCENES <<< "$SCENES_STR"
MAX_STEPS="${MAX_STEPS:-15000}"
MID_EVAL_STEP="${MID_EVAL_STEP:-7000}"
FINAL_STEP="$MAX_STEPS"
FINAL_STATS_STEP="$((MAX_STEPS - 1))"

BASE_RESULT_ROOT="${BASE_RESULT_ROOT:-results/benchmark_15000_all}"
RC_RESULT_ROOT="${RC_RESULT_ROOT:-results/benchmark_rc_15000_all}"
SUMMARY_JSON="${SUMMARY_JSON:-results/benchmark_compare_15000_all_scenes.json}"

echo "[INFO] Scenes: ${SCENES[*]}"
echo "[INFO] Max steps: ${MAX_STEPS}"

run_one() {
  local method="$1"
  local scene="$2"
  local result_root="$3"
  local data_factor="$4"

  local result_dir="${result_root}/${scene}"
  mkdir -p "$result_dir"

  echo "[RUN] method=${method} scene=${scene} data_factor=${data_factor}"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" simple_trainer.py "$method" \
    --disable-viewer \
    --data-dir "${SCENE_DIR}/${scene}/" \
    --data-factor "$data_factor" \
    --result-dir "${result_dir}/" \
    --batch-size 1 \
    --max-steps "$MAX_STEPS" \
    --eval-steps "$MID_EVAL_STEP" "$FINAL_STEP" \
    --save-steps "$MID_EVAL_STEP" "$FINAL_STEP" \
    --test-every 8 \
    --init-type sfm \
    --init-num-pts 100000 \
    --init-extent 3.0 \
    --init-opa 0.1 \
    --init-scale 1.0 \
    --sh-degree 3 \
    --sh-degree-interval 1000 \
    --ssim-lambda 0.2 \
    --near-plane 0.01 \
    --far-plane 1e10 \
    --no-packed \
    --no-sparse-grad \
    --no-visible-adam \
    --no-antialiased \
    --no-random-bkgd \
    --means-lr 1.6e-4 \
    --scales-lr 5e-3 \
    --opacities-lr 5e-2 \
    --quats-lr 1e-3 \
    --sh0-lr 2.5e-3 \
    --shN-lr 1.25e-4 \
    --strategy.prune-opa 0.005 \
    --strategy.grow-scale3d 0.01 \
    --strategy.grow-scale2d 0.05 \
    --strategy.prune-scale3d 0.1 \
    --strategy.prune-scale2d 0.15 \
    --strategy.refine-start-iter 500 \
    --strategy.refine-stop-iter 15000 \
    --strategy.refine-every 100 \
    --strategy.reset-every 3000 \
    --strategy.no-absgrad \
    --strategy.no-revised-opacity \
    --strategy.key-for-gradient means2d
}

run_rc() {
  local scene="$1"
  local data_factor="$2"
  local result_dir="${RC_RESULT_ROOT}/${scene}"
  mkdir -p "$result_dir"

  echo "[RUN] method=residual_coverage scene=${scene} data_factor=${data_factor}"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" simple_trainer.py residual_coverage \
    --disable-viewer \
    --data-dir "${SCENE_DIR}/${scene}/" \
    --data-factor "$data_factor" \
    --result-dir "${result_dir}/" \
    --batch-size 1 \
    --max-steps "$MAX_STEPS" \
    --eval-steps "$MID_EVAL_STEP" "$FINAL_STEP" \
    --save-steps "$MID_EVAL_STEP" "$FINAL_STEP" \
    --test-every 8 \
    --init-type sfm \
    --init-num-pts 100000 \
    --init-extent 3.0 \
    --init-opa 0.1 \
    --init-scale 1.0 \
    --sh-degree 3 \
    --sh-degree-interval 1000 \
    --ssim-lambda 0.2 \
    --near-plane 0.01 \
    --far-plane 1e10 \
    --no-packed \
    --no-sparse-grad \
    --no-visible-adam \
    --no-antialiased \
    --no-random-bkgd \
    --means-lr 1.6e-4 \
    --scales-lr 5e-3 \
    --opacities-lr 5e-2 \
    --quats-lr 1e-3 \
    --sh0-lr 2.5e-3 \
    --shN-lr 1.25e-4 \
    --strategy.prune-opa 0.005 \
    --strategy.grow-scale3d 0.01 \
    --strategy.grow-scale2d 0.05 \
    --strategy.prune-scale3d 0.1 \
    --strategy.prune-scale2d 0.15 \
    --strategy.refine-start-iter 500 \
    --strategy.refine-stop-iter 15000 \
    --strategy.refine-every 100 \
    --strategy.reset-every 3000 \
    --strategy.no-absgrad \
    --strategy.no-revised-opacity \
    --strategy.key-for-gradient means2d \
    --strategy.growth-topk-ratio 0.20 \
    --strategy.residual-threshold 0.20 \
    --strategy.coverage-min 0.05 \
    --strategy.target-coverage 0.15 \
    --strategy.residual-ema-decay 0.9 \
    --strategy.coverage-ema-decay 0.99 \
    --strategy.max-new-gs 32000
}

for scene in "${SCENES[@]}"; do
  if [[ "$scene" == "bonsai" || "$scene" == "counter" || "$scene" == "kitchen" || "$scene" == "room" ]]; then
    data_factor=2
  else
    data_factor=4
  fi

  run_one default "$scene" "$BASE_RESULT_ROOT" "$data_factor"
  run_rc "$scene" "$data_factor"
done

echo "[INFO] Aggregating results to ${SUMMARY_JSON}"
"$PYTHON_BIN" benchmarks/summarize_compare_30000.py \
  --baseline-root "$BASE_RESULT_ROOT" \
  --rc-root "$RC_RESULT_ROOT" \
  --final-step "$FINAL_STATS_STEP" \
  --output "$SUMMARY_JSON"

echo "[DONE] Summary written to ${SUMMARY_JSON}"