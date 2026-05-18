#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PY=/usr/local/bin/python
SCENE=garden
SCENE_DIR=data/360_v2
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RESULT=results/benchmark_rc_15000_planGM_garden/${SCENE}
LOG=results/benchmark_rc_15000_planGM_garden.log

mkdir -p "$RESULT"
echo "[RUN] planGM start $(date '+%F %T')" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} "$PY" simple_trainer.py residual_coverage \
  --disable-viewer \
  --data-dir ${SCENE_DIR}/${SCENE}/ \
  --data-factor 4 \
  --result-dir ${RESULT}/ \
  --batch-size 1 \
  --max-steps 15000 \
  --eval-steps 7000 15000 \
  --save-steps 7000 15000 \
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
  --strategy.prune-opa 0.002 \
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
  --strategy.growth-topk-ratio 0.55 \
  --strategy.residual-threshold 0.09 \
  --strategy.coverage-min 0.025 \
  --strategy.target-coverage 0.15 \
  --strategy.residual-ema-decay 0.9 \
  --strategy.coverage-ema-decay 0.99 \
  --strategy.max-new-gs 90000 \
  --strategy.cap-max 5200000 \
  --strategy.residual-quantile 0.8 \
  --strategy.coverage-gate-min-score 0.05 \
  --strategy.coverage-score-power 0.72 \
  --strategy.prune-warmup-iter 9500 \
  --strategy.prune-opa-warmup-scale 0.20 \
  --strategy.replace-start-iter 9000 \
  --strategy.replace-budget-ratio 0.001 \
  --strategy.min-replace-budget 32 \
  --strategy.max-replace-budget 256 \
  --strategy.stage-transition-iter 4000 \
  --strategy.stage-end-iter 10000 \
  --strategy.gate-score-early 0.08 \
  --strategy.gate-score-late 0.03 \
  --strategy.coverage-min-early 0.02 \
  --strategy.coverage-min-late 0.018 \
  --strategy.residual-threshold-early-scale 0.90 \
  --strategy.residual-threshold-late-scale 0.98 \
  --strategy.contribution-ema-decay 0.965 \
  --strategy.prune-contribution-weight 0.15 \
  --strategy.prune-opacity-weight 0.45 \
  --strategy.prune-coverage-weight 0.22 \
  --strategy.prune-residual-weight 0.18 \
  --strategy.growth-spike-guard \
  --strategy.growth-spike-ratio-limit 2.5 \
  --strategy.growth-spike-warmup-iter 2500 \
  --strategy.growth-budget-min-scale 1.0 \
  --strategy.growth-budget-max-scale 1.0 \
  --strategy.replace-guard-residual-scale 0.80 \
  --strategy.replace-guard-steps 8 \
  2>&1 | tee -a "$LOG"

echo "[RUN] planGM done $(date '+%F %T')" | tee -a "$LOG"
