#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# planGR: GF Refined — 以 planGF 为基础微调，目标是在保持 PSNR 的同时提升 SSIM/LPIPS
# ============================================================================
# 微调思路（详见下方注释）:
#   1. 提升增长容量           → 让模型有更多 GS 捕捉细节 (提升 SSIM)
#   2. 降低残差检测阈值        → 更敏感地发现重建不足区域 (提升 LPIPS)
#   3. 延长增长阶段            → 多 1000 步用于结构增长
#   4. 降低透明度剪枝力度      → 保留更多半透明 GS (GP 的 SSIM 优势来源)
#   5. Prune 权重向 GP 靠拢   → 更重视 opacity，减少对 residual 的依赖
#   6. 提升替换预算            → 更积极的"腾笼换鸟"
#   7. 后期门槛略微放松        → 避免过早锁死增长
#   8. 启用增长尖峰保护        → 防止 GS 数量突变 (GG/GH 家族的安全机制)
# ============================================================================

PY=/usr/local/bin/python
SCENE=garden
SCENE_DIR=data/360_v2
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RESULT=results/benchmark_rc_15000_planGR_garden/${SCENE}
LOG=results/benchmark_rc_15000_planGR_garden.log

mkdir -p "$RESULT"
echo "[RUN] planGR start $(date '+%F %T')" | tee -a "$LOG"

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
  --strategy.prune-opa 0.0045 \
  --strategy.grow-scale3d 0.01 \
  --strategy.grow-scale2d 0.05 \
  --strategy.prune-scale3d 0.1 \
  --strategy.prune-scale2d 0.15 \
  --strategy.refine-start-iter 500 \
  --strategy.refine-stop-iter 14000 \
  --strategy.refine-every 100 \
  --strategy.reset-every 3000 \
  --strategy.no-absgrad \
  --strategy.no-revised-opacity \
  --strategy.key-for-gradient means2d \
  --strategy.growth-topk-ratio 0.48 \
  --strategy.residual-threshold 0.095 \
  --strategy.coverage-min 0.022 \
  --strategy.target-coverage 0.15 \
  --strategy.residual-ema-decay 0.9 \
  --strategy.coverage-ema-decay 0.99 \
  --strategy.max-new-gs 70000 \
  --strategy.cap-max 4200000 \
  --strategy.residual-quantile 0.8 \
  --strategy.coverage-gate-min-score 0.04 \
  --strategy.coverage-score-power 0.72 \
  --strategy.prune-warmup-iter 8000 \
  --strategy.prune-opa-warmup-scale 0.22 \
  --strategy.replace-start-iter 6500 \
  --strategy.replace-budget-ratio 0.008 \
  --strategy.min-replace-budget 64 \
  --strategy.max-replace-budget 1536 \
  --strategy.stage-transition-iter 4000 \
  --strategy.stage-end-iter 9000 \
  --strategy.gate-score-early 0.08 \
  --strategy.gate-score-late 0.030 \
  --strategy.coverage-min-early 0.02 \
  --strategy.coverage-min-late 0.012 \
  --strategy.residual-threshold-early-scale 0.92 \
  --strategy.residual-threshold-late-scale 1.01 \
  --strategy.contribution-ema-decay 0.965 \
  --strategy.prune-contribution-weight 0.24 \
  --strategy.prune-opacity-weight 0.40 \
  --strategy.prune-coverage-weight 0.22 \
  --strategy.prune-residual-weight 0.22 \
  --strategy.growth-spike-guard \
  --strategy.growth-spike-ratio-limit 1.25 \
  --strategy.growth-spike-warmup-iter 2500 \
  --strategy.growth-budget-min-scale 1.0 \
  --strategy.growth-budget-max-scale 1.0 \
  --strategy.replace-guard-residual-scale 0.85 \
  --strategy.replace-guard-steps 5 \
  2>&1 | tee -a "$LOG"

echo "[RUN] planGR done $(date '+%F %T')" | tee -a "$LOG"
