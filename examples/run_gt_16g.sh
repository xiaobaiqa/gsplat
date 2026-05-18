#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# planGT: GS Turbo — GS 基础上最后冲刺，目标追上 Baseline (ΔPSNR < 0.01)
# ============================================================================
# GS→GT 微调思路（8 组改动，22 个参数）:
#
#   🎯 核心诊断:
#   GS 每次 reset (3000步周期) 会大规模剪枝 116k~190k GS，
#   后期 (9000步后) 增长几乎停滞，最终 GS=3.25M vs Baseline 3.94M。
#   → 需要: 减少 reset 损失、延长增长窗口、更激进的增长策略。
#
#   🔴 增长侧 — 全力冲刺，给模型足够 GS 去追上 baseline
#   1. growth_topk_ratio: 0.55→0.62   候选池再扩大 13%
#   2. max_new_gs: 90000→110000        每轮 +2 万配额
#   3. cap_max: 4.8M→5.5M              进一步解除上限
#   4. residual_threshold: 0.088→0.080  更敏感残差检测 (捕捉最细微的重建误差)
#   5. coverage_min: 0.018→0.014       降低覆盖门槛 (GS 的 2.99M→3.25M 增长太慢)
#   6. coverage_gate_min: 0.03→0.022   更宽松的门控准入
#   7. refine_stop_iter: 14800→15000   全程开放增长 (不留纯优化步)
#
#   🟠 剪枝侧 — 核心改动: 减少 reset 周期的大规模剪枝
#   8. prune_opa: 0.004→0.0035        透明度阈值大幅降低 (GS 的 0.004 仍偏激进)
#   9. prune_opa_warmup: 0.18→0.14    warmup 期更温和 (实际阈值=0.0035×0.14=0.0005)
#  10. prune_warmup_iter: 8000→10000  延迟剪枝启动 2000 步 (给 GS 更多生长时间)
#  11. prune_opacity_weight: 0.44→0.50 opacity 占剪枝决策的一半
#  12. prune_contribution: 0.20→0.14   进一步降
#  13. prune_coverage: 0.18→0.14       进一步降
#  14. prune_residual: 0.18→0.14       进一步降
#      → 权重分布: opacity 50% + contribution 14% + coverage 14% + residual 14% = 92%（剩余 8% 为其他因素）
#
#   🟡 替换侧 — 更积极的"新陈代谢"
#  15. replace_budget: 0.012→0.018     1.8% 替换率
#  16. max_replace_budget: 2048→3072   单轮上限 +50%
#  17. replace_start_iter: 6500→7000   稍晚启动替换 (等 prune_warmup 结束前 3000 步)
#
#   🟢 后期阶段 — 让后期 (9000步后) 恢复增长活力
#  18. gate_score_late: 0.025→0.018    后期门控大幅放宽
#  19. coverage_min_late: 0.010→0.006  后期覆盖要求几乎取消
#  20. residual_threshold_late_scale: 1.00→0.96  后期反而降低阈值 (actively encourage late growth)
#  21. stage_end_iter: 9000→10000      延长早期阶段 1000 步
#
#   🔵 安全机制
#  22. growth_spike_ratio: 1.35→1.60   更宽松的尖峰容忍
#
# 预期: #GS 3.5~3.7M, PSNR 27.20~27.23, SSIM 0.853~0.855, LPIPS 0.092~0.094
# ============================================================================

PY=/usr/local/bin/python
SCENE=garden
SCENE_DIR=data/360_v2
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RESULT=results/benchmark_rc_15000_planGT_garden/${SCENE}
LOG=results/benchmark_rc_15000_planGT_garden.log

mkdir -p "$RESULT"
echo "[RUN] planGT start $(date '+%F %T')" | tee -a "$LOG"

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
  --strategy.prune-opa 0.0035 \
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
  --strategy.growth-topk-ratio 0.62 \
  --strategy.residual-threshold 0.080 \
  --strategy.coverage-min 0.014 \
  --strategy.target-coverage 0.15 \
  --strategy.residual-ema-decay 0.9 \
  --strategy.coverage-ema-decay 0.99 \
  --strategy.max-new-gs 110000 \
  --strategy.cap-max 5500000 \
  --strategy.residual-quantile 0.8 \
  --strategy.coverage-gate-min-score 0.022 \
  --strategy.coverage-score-power 0.72 \
  --strategy.prune-warmup-iter 10000 \
  --strategy.prune-opa-warmup-scale 0.14 \
  --strategy.replace-start-iter 7000 \
  --strategy.replace-budget-ratio 0.018 \
  --strategy.min-replace-budget 64 \
  --strategy.max-replace-budget 3072 \
  --strategy.stage-transition-iter 4000 \
  --strategy.stage-end-iter 10000 \
  --strategy.gate-score-early 0.08 \
  --strategy.gate-score-late 0.018 \
  --strategy.coverage-min-early 0.02 \
  --strategy.coverage-min-late 0.006 \
  --strategy.residual-threshold-early-scale 0.92 \
  --strategy.residual-threshold-late-scale 0.96 \
  --strategy.contribution-ema-decay 0.965 \
  --strategy.prune-contribution-weight 0.14 \
  --strategy.prune-opacity-weight 0.50 \
  --strategy.prune-coverage-weight 0.14 \
  --strategy.prune-residual-weight 0.14 \
  --strategy.growth-spike-guard \
  --strategy.growth-spike-ratio-limit 1.60 \
  --strategy.growth-spike-warmup-iter 2500 \
  --strategy.growth-budget-min-scale 1.0 \
  --strategy.growth-budget-max-scale 1.0 \
  --strategy.replace-guard-residual-scale 0.85 \
  --strategy.replace-guard-steps 5 \
  2>&1 | tee -a "$LOG"

echo "[RUN] planGT done $(date '+%F %T')" | tee -a "$LOG"
