#!/usr/bin/env bash
set -euo pipefail

PY=/usr/local/bin/python
SCENE=garden
SCENE_DIR=data/360_v2
CUDA_DEVICE=0
COMMON_ARGS=(
  --disable-viewer
  --data-dir ${SCENE_DIR}/${SCENE}/
  --data-factor 4
  --batch-size 1
  --max-steps 15000
  --eval-steps 7000 15000
  --save-steps 7000 15000
  --test-every 8
  --init-type sfm
  --init-num-pts 100000
  --init-extent 3.0
  --init-opa 0.1
  --init-scale 1.0
  --sh-degree 3
  --sh-degree-interval 1000
  --ssim-lambda 0.2
  --near-plane 0.01
  --far-plane 1e10
  --no-packed
  --no-sparse-grad
  --no-visible-adam
  --no-antialiased
  --no-random-bkgd
  --means-lr 1.6e-4
  --scales-lr 5e-3
  --opacities-lr 5e-2
  --quats-lr 1e-3
  --sh0-lr 2.5e-3
  --shN-lr 1.25e-4
  --strategy.prune-opa 0.005
  --strategy.grow-scale3d 0.01
  --strategy.grow-scale2d 0.05
  --strategy.prune-scale3d 0.1
  --strategy.prune-scale2d 0.15
  --strategy.refine-start-iter 500
  --strategy.refine-stop-iter 15000
  --strategy.refine-every 100
  --strategy.reset-every 3000
  --strategy.no-absgrad
  --strategy.no-revised-opacity
  --strategy.key-for-gradient means2d
  --strategy.residual-ema-decay 0.9
  --strategy.coverage-ema-decay 0.99
  --strategy.residual-quantile 0.8
  --strategy.coverage-score-power 0.75
  --strategy.stage-transition-iter 4000
  --strategy.stage-end-iter 8000
  --strategy.gate-score-early 0.08
  --strategy.coverage-min-early 0.02
  --strategy.growth-budget-min-scale 1.0
  --strategy.growth-budget-max-scale 1.0
  --strategy.replace-guard-residual-scale 0.85
  --strategy.replace-guard-steps 5
)

run_case() {
  local name="$1"; shift
  local result_dir="results/${name}/${SCENE}"
  local log_file="results/${name}.log"
  mkdir -p "$result_dir"
  echo "[RUN] ${name} start $(date '+%F %T')" | tee -a "$log_file"
  CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} "$PY" simple_trainer.py residual_coverage \
    --result-dir "${result_dir}/" \
    "${COMMON_ARGS[@]}" \
    "$@" 2>&1 | tee -a "$log_file"
  echo "[RUN] ${name} done $(date '+%F %T')" | tee -a "$log_file"
}

# G-A
run_case benchmark_rc_15000_planGA_garden \
  --strategy.growth-topk-ratio 0.24 \
  --strategy.residual-threshold 0.155 \
  --strategy.coverage-min 0.03 \
  --strategy.target-coverage 0.15 \
  --strategy.max-new-gs 40000 \
  --strategy.cap-max 3600000 \
  --strategy.coverage-gate-min-score 0.08 \
  --strategy.gate-score-late 0.06 \
  --strategy.coverage-min-late 0.025 \
  --strategy.prune-warmup-iter 5000 \
  --strategy.prune-opa-warmup-scale 0.35 \
  --strategy.replace-start-iter 5000 \
  --strategy.replace-budget-ratio 0.03 \
  --strategy.min-replace-budget 64 \
  --strategy.max-replace-budget 4096

# G-B
run_case benchmark_rc_15000_planGB_garden \
  --strategy.growth-topk-ratio 0.26 \
  --strategy.residual-threshold 0.145 \
  --strategy.coverage-min 0.03 \
  --strategy.target-coverage 0.15 \
  --strategy.max-new-gs 48000 \
  --strategy.cap-max 3500000 \
  --strategy.coverage-gate-min-score 0.08 \
  --strategy.gate-score-late 0.05 \
  --strategy.coverage-min-late 0.02 \
  --strategy.prune-warmup-iter 6000 \
  --strategy.prune-opa-warmup-scale 0.30 \
  --strategy.replace-start-iter 5500 \
  --strategy.replace-budget-ratio 0.02 \
  --strategy.min-replace-budget 64 \
  --strategy.max-replace-budget 4096

# G-C
run_case benchmark_rc_15000_planGC_garden \
  --strategy.growth-topk-ratio 0.24 \
  --strategy.residual-threshold 0.155 \
  --strategy.coverage-min 0.03 \
  --strategy.target-coverage 0.15 \
  --strategy.max-new-gs 40000 \
  --strategy.cap-max 3400000 \
  --strategy.coverage-gate-min-score 0.08 \
  --strategy.gate-score-late 0.06 \
  --strategy.coverage-min-late 0.025 \
  --strategy.prune-warmup-iter 5000 \
  --strategy.prune-opa-warmup-scale 0.35 \
  --strategy.replace-start-iter 5000 \
  --strategy.replace-budget-ratio 0.03 \
  --strategy.min-replace-budget 64 \
  --strategy.max-replace-budget 4096 \
  --strategy.refine-every 120 \
  --strategy.reset-every 3500

