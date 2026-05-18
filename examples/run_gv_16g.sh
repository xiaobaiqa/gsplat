#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# planGV: GS Variant — GS 基础上单项微调 residual_threshold
# ============================================================================
# GS→GV 仅改 1 个参数:
#
#   residual_threshold: 0.088 → 0.085  (-3.4%)
#
#   依据:
#   ─────────────────────────────────────────────────────────────
#   剩余 PSNR 差距 0.028，根本原因是 GS 在高误差区域仍不够密集。
#   residual_threshold 控制"多大残差才算重建不足"。
#   从 0.088 降到 0.085，让约 3-5% 的额外高误差像素触发 GS 增长。
#
#   GT 教训: 大幅改动多个参数导致回归。
#   GV 策略: 只改一个参数，变化幅度控制在 5% 以内。
#   ─────────────────────────────────────────────────────────────
#
# 预期: #GS 3.30~3.40M, PSNR 27.19~27.21, ΔPSNR -0.02~0.00
# ============================================================================

PY=/usr/local/bin/python
SCENE=garden
SCENE_DIR=data/360_v2
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RESULT=results/benchmark_rc_15000_planGV_garden/${SCENE}
LOG=results/benchmark_rc_15000_planGV_garden.log

mkdir -p "$RESULT"
echo "[RUN] planGV start $(date '+%F %T')" | tee -a "$LOG"

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
  --strategy.prune-opa 0.004 \
  --strategy.grow-scale3d 0.01 \
  --strategy.grow-scale2d 0.05 \
  --strategy.prune-scale3d 0.1 \
  --strategy.prune-scale2d 0.15 \
  --strategy.refine-start-iter 500 \
  --strategy.refine-stop-iter 14800 \
  --strategy.refine-every 100 \
  --strategy.reset-every 3000 \
  --strategy.no-absgrad \
  --strategy.no-revised-opacity \
  --strategy.key-for-gradient means2d \
  --strategy.growth-topk-ratio 0.55 \
  --strategy.residual-threshold 0.085 \
  --strategy.coverage-min 0.018 \
  --strategy.target-coverage 0.15 \
  --strategy.residual-ema-decay 0.9 \
  --strategy.coverage-ema-decay 0.99 \
  --strategy.max-new-gs 90000 \
  --strategy.cap-max 4800000 \
  --strategy.residual-quantile 0.8 \
  --strategy.coverage-gate-min-score 0.03 \
  --strategy.coverage-score-power 0.72 \
  --strategy.prune-warmup-iter 8000 \
  --strategy.prune-opa-warmup-scale 0.18 \
  --strategy.replace-start-iter 6500 \
  --strategy.replace-budget-ratio 0.012 \
  --strategy.min-replace-budget 64 \
  --strategy.max-replace-budget 2048 \
  --strategy.stage-transition-iter 4000 \
  --strategy.stage-end-iter 9000 \
  --strategy.gate-score-early 0.08 \
  --strategy.gate-score-late 0.025 \
  --strategy.coverage-min-early 0.02 \
  --strategy.coverage-min-late 0.010 \
  --strategy.residual-threshold-early-scale 0.92 \
  --strategy.residual-threshold-late-scale 1.00 \
  --strategy.contribution-ema-decay 0.965 \
  --strategy.prune-contribution-weight 0.20 \
  --strategy.prune-opacity-weight 0.44 \
  --strategy.prune-coverage-weight 0.18 \
  --strategy.prune-residual-weight 0.18 \
  --strategy.growth-spike-guard \
  --strategy.growth-spike-ratio-limit 1.35 \
  --strategy.growth-spike-warmup-iter 2500 \
  --strategy.growth-budget-min-scale 1.0 \
  --strategy.growth-budget-max-scale 1.0 \
  --strategy.replace-guard-residual-scale 0.85 \
  --strategy.replace-guard-steps 5 \
  2>&1 | tee -a "$LOG"

echo "[RUN] planGV done $(date '+%F %T')" | tee -a "$LOG"
