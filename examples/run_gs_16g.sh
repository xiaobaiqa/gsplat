#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# planGS: GR Second — 在 GR 基础上继续微调，目标追上 Baseline 质量
# ============================================================================
# GR→GS 微调思路（9 组改动，按影响力排序）:
#
#   🔴 增长侧 — 给模型更多 GS 去逼近 baseline 的 3.94M
#   1. growth_topk_ratio: 0.48→0.55  更包容的增长候选 (GR 的 1% 额外 GS 已证明有效)
#   2. max_new_gs: 70000→90000        每轮多 2 万新 GS 配额
#   3. cap_max: 4.2M→4.8M             硬上限提高到 baseline 以上，不再成为瓶颈
#   4. residual_threshold: 0.095→0.088 更敏感的残差检测 (捕捉更多重建不足区域)
#   5. coverage_min: 0.022→0.018      降低覆盖度门槛 (低可见区域也能生长)
#   6. refine_stop_iter: 14000→14800   几乎全程开放增长 (仅留 200 步纯优化)
#
#   🟠 剪枝侧 — 向 GP 的 SSIM/LPIPS 优势进一步靠拢
#   7. prune_opa: 0.0045→0.004        透明度阈值与 GP 持平 (保留更多半透明 GS)
#   8. prune_opa_warmup: 0.22→0.18    warmup 期更温和 (与 GP 一致)
#   9. prune_opacity_weight: 0.40→0.44 更强调 opacity 信号 (GP SSIM 优势的核心)
#  10. prune_contribution: 0.24→0.20  降低 contribution 权重
#  11. prune_coverage: 0.22→0.18      降低 coverage 权重
#  12. prune_residual: 0.22→0.18      降低 residual 权重
#
#   🟡 替换侧 — 更积极的"新陈代谢"
#  13. replace_budget: 0.008→0.012    1.2% 替换率 (原 0.8%)
#  14. max_replace_budget: 1536→2048  单轮最多替换 2048 个
#
#   🟢 后期阶段 — 全面放松，让后期仍有微增长空间
#  15. gate_score_late: 0.030→0.025   后期门控更宽松
#  16. coverage_min_late: 0.012→0.010 后期覆盖要求降低
#  17. residual_thr_late: 1.01→1.00   后期不再提高残差阈值 (与早期一致)
#
#   🔵 安全机制 — 适应更激进的增长
#  18. growth_spike_ratio: 1.25→1.35  略微放宽尖峰限制
#
# 预期: #GS 3.4~3.6M, PSNR 27.19~27.22, SSIM 0.852~0.854, LPIPS 0.093~0.095
# ============================================================================

PY=/usr/local/bin/python
SCENE=garden
SCENE_DIR=data/360_v2
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RESULT=results/benchmark_rc_15000_planGS_garden/${SCENE}
LOG=results/benchmark_rc_15000_planGS_garden.log

mkdir -p "$RESULT"
echo "[RUN] planGS start $(date '+%F %T')" | tee -a "$LOG"

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
  --strategy.residual-threshold 0.088 \
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

echo "[RUN] planGS done $(date '+%F %T')" | tee -a "$LOG"
