#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BASELINE_ROOT="results/benchmark_15000"
COMPARE_OUT="results/garden_compare/summary_14999.json"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

ensure_run() {
  local result_root="$1"
  local run_script="$2"
  local train_stats="${result_root}/garden/stats/train_step14999_rank0.json"
  local val_stats="${result_root}/garden/stats/val_step14999.json"
  if [[ -f "${train_stats}" && -f "${val_stats}" ]]; then
    echo "[SKIP] ${result_root} already has final stats."
    return
  fi

  echo "[RUN] missing final stats for ${result_root}, launching ${run_script}"
  CUDA_DEVICE="${CUDA_DEVICE}" bash "${run_script}"
}

ensure_run "${BASELINE_ROOT}" "benchmarks/baseline_garden_15000.sh"
ensure_run "results/benchmark_rc_15000_planGF_garden" "run_gf_16g.sh"
ensure_run "results/benchmark_rc_15000_planGJ_garden" "run_gj_16g.sh"
ensure_run "results/benchmark_rc_15000_planGK_garden" "run_gk_16g.sh"
ensure_run "results/benchmark_rc_15000_planGL_garden" "run_gl_16g.sh"

python benchmarks/summarize_garden_compare.py \
  --baseline-root "${BASELINE_ROOT}" \
  --candidate-root "results/benchmark_rc_15000_planGF_garden" \
  --candidate-root "results/benchmark_rc_15000_planGJ_garden" \
  --candidate-root "results/benchmark_rc_15000_planGK_garden" \
  --candidate-root "results/benchmark_rc_15000_planGL_garden" \
  --output "${COMPARE_OUT}"

echo "[DONE] Summary written to ${COMPARE_OUT}"
