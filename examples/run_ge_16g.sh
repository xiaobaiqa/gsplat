#!/usr/bin/env bash
set -euo pipefail

PY=/usr/local/bin/python
SCENE=garden
SCENE_DIR=data/360_v2
CUDA_DEVICE=0
RESULT=results/benchmark_rc_15000_planGE_garden/${SCENE}
LOG=results/benchmark_rc_15000_planGE_garden.log

mkdir -p "$RESULT"
echo "[RUN] planGE start $(date '+%F %T')" | tee -a "$LOG"

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
  --strategy.growth-topk-ratio 0.29 \
  --strategy.residual-threshold 0.135 \
  --strategy.coverage-min 0.028 \
  --strategy.target-coverage 0.16 \
  --strategy.residual-ema-decay 0.9 \
  --strategy.coverage-ema-decay 0.99 \
  --strategy.max-new-gs 52000 \
  --strategy.cap-max 3800000 \
  --strategy.residual-quantile 0.8 \
  --strategy.coverage-gate-min-score 0.06 \
  --strategy.coverage-score-power 0.72 \
  --strategy.prune-warmup-iter 7000 \
  --strategy.prune-opa-warmup-scale 0.28 \
  --strategy.replace-start-iter 6200 \
  --strategy.replace-budget-ratio 0.010 \
  --strategy.min-replace-budget 64 \
  --strategy.max-replace-budget 1536 \
  --strategy.stage-transition-iter 4000 \
  --strategy.stage-end-iter 9000 \
  --strategy.gate-score-early 0.08 \
  --strategy.gate-score-late 0.045 \
  --strategy.coverage-min-early 0.02 \
  --strategy.coverage-min-late 0.018 \
  --strategy.residual-threshold-early-scale 0.97 \
  --strategy.residual-threshold-late-scale 1.05 \
  --strategy.contribution-ema-decay 0.965 \
  --strategy.prune-contribution-weight 0.16 \
  --strategy.prune-opacity-weight 0.50 \
  --strategy.prune-coverage-weight 0.30 \
  --strategy.prune-residual-weight 0.20 \
  --strategy.growth-budget-min-scale 1.0 \
  --strategy.growth-budget-max-scale 1.0 \
  --strategy.replace-guard-residual-scale 0.85 \
  --strategy.replace-guard-steps 5 \
  2>&1 | tee -a "$LOG"

echo "[RUN] planGE done $(date '+%F %T')" | tee -a "$LOG"
