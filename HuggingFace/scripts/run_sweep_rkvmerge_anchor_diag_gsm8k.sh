#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MODEL_PATH="${MODEL_PATH:-/prj/corp/crd/morpheus/lasvegas/sit/jinwooa/models/Qwen/Qwen3-4B}"
export DATASETS="gsm8k"
export METHODS="rkv_merge_anchor_diag"

# ---- identical to the rkv baseline this is being ablated against -------------------------------
export FRACTION="${FRACTION:-1.0}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
export NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-1}"
export TEMPERATURE="${TEMPERATURE:-0.6}"
export TOP_P="${TOP_P:-0.95}"
export SEED="${SEED:-42}"
export USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-1}"
export ENABLE_THINKING="${ENABLE_THINKING:-1}"
export DIVIDE_METHOD="${DIVIDE_METHOD:-generated_length}"
export DIVIDE_LENGTH="${DIVIDE_LENGTH:-32}"
export COMPRESSION_CONTENT="${COMPRESSION_CONTENT:-all}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"

# R-KV scorer hyperparameters -- these MUST match the rkv baseline or the ablation is confounded.
export WINDOW_SIZE="${WINDOW_SIZE:-8}"
export MIX_LAMBDA="${MIX_LAMBDA:-0.1}"
export RETAIN_RATIO="${RETAIN_RATIO:-0.2}"
export RETAIN_DIRECTION="${RETAIN_DIRECTION:-last}"
# Unlike rkv_merge, first_tokens is LOAD-BEARING here: it defines the sink prefix that is excluded
# from being a merge target. It still does not protect those slots from eviction, so the source
# selection stays identical to R-KV's and the scorer ablation is unconfounded.
export FIRST_TOKENS="${FIRST_TOKENS:-4}"

# ---- eager, for the same reason as covariance_merge -------------------------------------------
# The merge stores a per-slot additive logit bias that must be added before the softmax;
# rkv/modeling.py forces `_attn_implementation = "eager"` for any press with uses_merge_metadata.
# Naming it here keeps the run tag honest. The rkv baseline ran sdpa, which computes the same
# function, so comparability is unaffected.
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"

# ---- future-query model: only the merge metric reads this, not the scorer ----------------------
export EMA_HALF_LIFE="${EMA_HALF_LIFE:-64}"
export FUTURE_HORIZON="${FUTURE_HORIZON:-128}"
export FUTURE_DECAY="${FUTURE_DECAY:-1.0}"

read -r -a BUDGET_LIST <<<"${BUDGETS:-128 1024}"
read -r -a THRESHOLD_LIST <<<"${THRESHOLDS:-1.0 1.5}"

echo "==================== rkv_merge_anchor ablation sweep ===================="
echo "  scorer          : R-KV (attn*${MIX_LAMBDA} - redundancy*$(awk "BEGIN{print 1-${MIX_LAMBDA}}"))"
echo "  kv_budget       : ${BUDGET_LIST[*]}"
echo "  merge_threshold : ${THRESHOLD_LIST[*]}"
echo "  cells           : $(( ${#BUDGET_LIST[@]} * ${#THRESHOLD_LIST[@]} ))"
echo "  batch size      : ${EVAL_BATCH_SIZE}   attn: ${ATTN_IMPLEMENTATION}"
echo "  merge metric    : H=${EMA_HALF_LIFE} P=${FUTURE_HORIZON} decay=${FUTURE_DECAY}"
echo "  sink (no-target): first ${FIRST_TOKENS};  recent (no-target): last ${WINDOW_SIZE}"
echo "  compare against : runs/*/gsm8k/{rkv,rkv_merge}/  (same budget, bs, clock, seed)"
echo "================================================================="

n_fail=0
for budget in "${BUDGET_LIST[@]}"; do
    for threshold in "${THRESHOLD_LIST[@]}"; do
        export KV_BUDGET="${budget}"
        export MERGE_THRESHOLD="${threshold}"
        echo
        echo ">>> cell: kv_budget=${budget}  merge_threshold=${threshold}"
        "${SCRIPT_DIR}/run_sweep.sh" "$@" || n_fail=$((n_fail + 1))
    done
done

echo
echo "==================== sweep complete ============================="
echo "cells with failures: ${n_fail}"
echo "summary: ${SCRIPT_DIR}/collect_results.py --runs-root ${RUNS_ROOT:-<HuggingFace>/runs}"
[[ ${n_fail} -gt 0 ]] && exit 1
exit 0
