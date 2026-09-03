#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=sweep_config.sh
source "${SCRIPT_DIR}/sweep_config.sh"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# DATASETS/METHODS may arrive as a space-separated string from the environment.
read -r -a DATASETS <<<"${DATASETS[*]}"
read -r -a METHODS <<<"${METHODS[*]}"

SWEEP_START_EPOCH="$(date +%s)"
SWEEP_STAMP="$(date +%Y%m%d-%H%M%S)"
SWEEP_LOG_DIR="${RUNS_ROOT}/_sweeps"
SWEEP_LOG="${SWEEP_LOG_DIR}/sweep-${SWEEP_STAMP}.log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}"
    [[ ${DRY_RUN} -eq 0 ]] && echo "${msg}" >>"${SWEEP_LOG}"
    return 0
}

hms() { printf '%dh%02dm%02ds' $(($1 / 3600)) $((($1 % 3600) / 60)) $(($1 % 60)); }

if [[ ${DRY_RUN} -eq 0 ]]; then
    mkdir -p "${SWEEP_LOG_DIR}"
    : >"${SWEEP_LOG}"
fi

# --------------------------------------------------------------- sanity checks
cd "${HF_ROOT}" || exit 1

if [[ ! -d "${MODEL_PATH}" ]]; then
    log "FATAL: model path does not exist: ${MODEL_PATH}"
    exit 1
fi
for dataset in "${DATASETS[@]}"; do
    if [[ ! -f "${HF_ROOT}/data/${dataset}.jsonl" ]]; then
        log "FATAL: dataset file not found: ${HF_ROOT}/data/${dataset}.jsonl"
        exit 1
    fi
done
for method in "${METHODS[@]}"; do
    case "${method}" in
        fullkv | rkv | snapkv | h2o | streamingllm | covariance_merge | rkv_merge | rkv_merge_anchor | rkv_merge_anchor_diag | rkv_merge_anchor_id) ;;
        *)
            log "FATAL: unknown method '${method}' (expected: fullkv rkv snapkv h2o streamingllm covariance_merge rkv_merge rkv_merge_anchor)"
            exit 1
            ;;
    esac
done

# ------------------------------------------------------------------ the plan
log "================ sweep plan ================"
log "model            : ${MODEL_PATH}"
log "datasets         : ${DATASETS[*]}"
log "methods          : ${METHODS[*]}"
log "kv_budget        : ${KV_BUDGET} (compressed methods only)"
log "fraction         : ${FRACTION}"
log "max_new_tokens   : ${MAX_NEW_TOKENS}"
log "samples/prompt   : ${NUM_RETURN_SEQUENCES}  (do_sample, T=${TEMPERATURE}, top_p=${TOP_P})"
log "eval_batch_size  : ${EVAL_BATCH_SIZE}  (>1 supported; use DIVIDE_METHOD=generated_length)"
log "chat template    : ${USE_CHAT_TEMPLATE}  (enable_thinking=${ENABLE_THINKING})"
if [[ " ${METHODS[*]} " == *" covariance_merge "* || " ${METHODS[*]} " == *" rkv_merge "* || " ${METHODS[*]} " == *" rkv_merge_anchor "* ]]; then
    log "merge_threshold  : ${MERGE_THRESHOLD}  (squared logits; sqrt is a gap std-dev in nats)"
    log "future-query EMA : H=${EMA_HALF_LIFE} P=${FUTURE_HORIZON} decay=${FUTURE_DECAY}"
fi
log "runs root        : ${RUNS_ROOT}"
log "CUDA_VISIBLE_DEV : ${CUDA_VISIBLE_DEVICES}"
for dataset in "${DATASETS[@]}"; do
    total="$(wc -l <"${HF_ROOT}/data/${dataset}.jsonl")"
    used="$(${PYTHON} -c "print(int(${FRACTION} * ${total}))")"
    log "  ${dataset}: ${total} examples, ${used} used at fraction ${FRACTION}, x${NUM_RETURN_SEQUENCES} samples"
done
log "============================================"

if [[ ${DRY_RUN} -eq 1 ]]; then
    for dataset in "${DATASETS[@]}"; do
        for method in "${METHODS[@]}"; do
            d="$(run_dir "${dataset}" "${method}")"
            status="PENDING"
            [[ -f "${d}/DONE" ]] && status="SKIP (already complete)"
            log "[plan] ${dataset}/${method} -> ${status}"
            log "       ${d}"
        done
    done
    log "dry run: nothing executed."
    exit 0
fi

# --------------------------------------------------------------- run one config
# Returns 0 on success, 1 on generation failure, 2 on eval failure, 3 if skipped.
run_one() {
    local dataset="$1" method="$2"
    local dir
    dir="$(run_dir "${dataset}" "${method}")"

    if [[ -f "${dir}/DONE" ]]; then
        log "SKIP ${dataset}/${method}: already complete -> ${dir}"
        return 3
    fi

    # An existing-but-unfinished directory is archived, never overwritten.
    if [[ -d "${dir}" ]]; then
        local n=1
        while [[ -e "${dir}.attempt-${n}" ]]; do n=$((n + 1)); done
        mv "${dir}" "${dir}.attempt-${n}"
        log "NOTE ${dataset}/${method}: archived incomplete previous attempt -> ${dir}.attempt-${n}"
    fi

    local gen_dir="${dir}/generation"   # holds ONLY generation.jsonl: eval_math.py
                                        # scans this whole directory for *.jsonl
    local eval_dir="${dir}/eval"
    local log_dir="${dir}/logs"
    mkdir -p "${gen_dir}" "${eval_dir}" "${log_dir}"

    local gen_file="${gen_dir}/generation.jsonl"
    echo "RUNNING" >"${dir}/STATUS"

    # ---- record the exact configuration next to the results -----------------
    cat >"${dir}/config.json" <<EOF
{
  "model_path": "${MODEL_PATH}",
  "model_name": "${MODEL_NAME}",
  "dataset": "${dataset}",
  "method": "${method}",
  "kv_budget": $(if [[ "${method}" == "fullkv" ]]; then echo null; else echo "${KV_BUDGET}"; fi),
  "fraction": ${FRACTION},
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "eval_batch_size": ${EVAL_BATCH_SIZE},
  "do_sample": true,
  "num_return_sequences": ${NUM_RETURN_SEQUENCES},
  "temperature": ${TEMPERATURE},
  "top_p": ${TOP_P},
  "seed": ${SEED},
  "use_chat_template": $([[ "${USE_CHAT_TEMPLATE}" == "1" ]] && echo true || echo false),
  "enable_thinking": $([[ "${ENABLE_THINKING}" == "1" ]] && echo true || echo false),
  "window_size": ${WINDOW_SIZE},
  "first_tokens": ${FIRST_TOKENS},
  "mix_lambda": ${MIX_LAMBDA},
  "retain_ratio": ${RETAIN_RATIO},
  "retain_direction": "${RETAIN_DIRECTION}",
  "divide_method": "${DIVIDE_METHOD}",
  "divide_length": ${DIVIDE_LENGTH},
  "compression_content": "${COMPRESSION_CONTENT}",
  "attn_implementation": "${ATTN_IMPLEMENTATION}",
  "stop_on_repetition": $([[ "${STOP_ON_REPETITION}" == "1" ]] && echo true || echo false),
  "repetition_repeats": ${REPETITION_REPEATS},
  "repetition_min_period": ${REPETITION_MIN_PERIOD},
  "repetition_max_period": ${REPETITION_MAX_PERIOD},
  "trim_incomplete_boxed": $([[ "${TRIM_INCOMPLETE_BOXED}" == "1" ]] && echo true || echo false),
  "merge_threshold": $(if [[ "${method}" == "covariance_merge" || "${method}" == "rkv_merge" || "${method}" == "rkv_merge_anchor" || "${method}" == "rkv_merge_anchor_diag" || "${method}" == "rkv_merge_anchor_id" ]]; then echo "${MERGE_THRESHOLD}"; else echo null; fi),
  "ema_half_life": $(if [[ "${method}" == "covariance_merge" || "${method}" == "rkv_merge" || "${method}" == "rkv_merge_anchor" || "${method}" == "rkv_merge_anchor_diag" || "${method}" == "rkv_merge_anchor_id" ]]; then echo "${EMA_HALF_LIFE}"; else echo null; fi),
  "future_horizon": $(if [[ "${method}" == "covariance_merge" || "${method}" == "rkv_merge" || "${method}" == "rkv_merge_anchor" || "${method}" == "rkv_merge_anchor_diag" || "${method}" == "rkv_merge_anchor_id" ]]; then echo "${FUTURE_HORIZON}"; else echo null; fi),
  "future_decay": $(if [[ "${method}" == "covariance_merge" || "${method}" == "rkv_merge" || "${method}" == "rkv_merge_anchor" || "${method}" == "rkv_merge_anchor_diag" || "${method}" == "rkv_merge_anchor_id" ]]; then echo "${FUTURE_DECAY}"; else echo null; fi),
  "tag": "$(run_tag "${method}")",
  "started_at": "$(date -Is)"
}
EOF

    # ---- build the generation command --------------------------------------
    local -a cmd=(
        "${PYTHON}" "${HF_ROOT}/run_math.py"
        --seed "${SEED}"
        --dataset_path "${HF_ROOT}/data/${dataset}.jsonl"
        --save_path "${gen_file}"
        --model_path "${MODEL_PATH}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --eval_batch_size "${EVAL_BATCH_SIZE}"
        --attn_implementation "${ATTN_IMPLEMENTATION}"
        --method "${method}"
        --fraction "${FRACTION}"
        --do_sample
        --num_return_sequences "${NUM_RETURN_SEQUENCES}"
        --temperature "${TEMPERATURE}"
        --top_p "${TOP_P}"
        --divide_method "${DIVIDE_METHOD}"
        --divide_length "${DIVIDE_LENGTH}"
        --compression_content "${COMPRESSION_CONTENT}"
    )
    # run_math.py's flag is --apply_chat_template; --enable_thinking is a
    # separate store_true flag it only reads inside that branch (passed as
    # tokenizer.apply_chat_template's enable_thinking= kwarg), so it's a
    # no-op unless apply_chat_template is also on.
    [[ "${USE_CHAT_TEMPLATE}" == "1" ]] && cmd+=(--apply_chat_template)
    [[ "${USE_CHAT_TEMPLATE}" == "1" && "${ENABLE_THINKING}" == "1" ]] && cmd+=(--enable_thinking)
    if [[ "${STOP_ON_REPETITION}" == "1" ]]; then
        cmd+=(
            --stop_on_repetition
            --repetition_repeats "${REPETITION_REPEATS}"
            --repetition_min_period "${REPETITION_MIN_PERIOD}"
            --repetition_max_period "${REPETITION_MAX_PERIOD}"
        )
    fi
    [[ "${TRIM_INCOMPLETE_BOXED}" == "1" ]] && cmd+=(--trim_incomplete_boxed)
    if [[ "${method}" == "covariance_merge" || "${method}" == "rkv_merge" || "${method}" == "rkv_merge_anchor" || "${method}" == "rkv_merge_anchor_diag" || "${method}" == "rkv_merge_anchor_id" ]]; then
        cmd+=(
            --merge_threshold "${MERGE_THRESHOLD}"
            --ema_half_life "${EMA_HALF_LIFE}"
            --future_horizon "${FUTURE_HORIZON}"
            --future_decay "${FUTURE_DECAY}"
        )
    fi
    # kv_budget and the eviction hyperparameters are meaningless for fullkv and
    # would trip the `budget - window_size > 0` asserts if budget were unset.
    if [[ "${method}" != "fullkv" ]]; then
        cmd+=(
            --kv_budget "${KV_BUDGET}"
            --window_size "${WINDOW_SIZE}"
            --first_tokens "${FIRST_TOKENS}"
            --mix_lambda "${MIX_LAMBDA}"
            --retain_ratio "${RETAIN_RATIO}"
            --retain_direction "${RETAIN_DIRECTION}"
        )
    fi

    printf '%q ' "${cmd[@]}" >"${dir}/command.txt"
    echo >>"${dir}/command.txt"

    # ---- generate -----------------------------------------------------------
    log "RUN  ${dataset}/${method} -> ${dir}"
    local t0 t1
    t0="$(date +%s)"
    "${cmd[@]}" >"${log_dir}/generate.log" 2>&1
    local gen_rc=$?
    t1="$(date +%s)"
    local gen_secs=$((t1 - t0))

    if [[ ${gen_rc} -ne 0 ]]; then
        echo "FAILED_GENERATION rc=${gen_rc}" >"${dir}/STATUS"
        log "FAIL ${dataset}/${method}: generation exited ${gen_rc} after $(hms ${gen_secs}); see ${log_dir}/generate.log"
        return 1
    fi
    if [[ ! -s "${gen_file}" ]]; then
        echo "FAILED_GENERATION empty_output" >"${dir}/STATUS"
        log "FAIL ${dataset}/${method}: generation produced no output; see ${log_dir}/generate.log"
        return 1
    fi

    local n_rows
    n_rows="$(wc -l <"${gen_file}")"
    echo "GEN_DONE" >"${dir}/STATUS"
    log "     ${dataset}/${method}: generated ${n_rows} rows in $(hms ${gen_secs})"

    # ---- score --------------------------------------------------------------
    # eval_math.py resolves relative output_dir against outputs/ when the path
    # does not exist, so eval_dir is created above and passed absolute.
    t0="$(date +%s)"
    "${PYTHON}" "${HF_ROOT}/evaluation/eval_math.py" \
        --exp_name eval \
        --output_dir "${eval_dir}" \
        --base_dir "${gen_dir}" \
        --dataset "${dataset}" \
        >"${log_dir}/eval.log" 2>&1
    local eval_rc=$?
    t1="$(date +%s)"
    local eval_secs=$((t1 - t0))

    if [[ ${eval_rc} -ne 0 ]]; then
        echo "FAILED_EVAL rc=${eval_rc}" >"${dir}/STATUS"
        log "FAIL ${dataset}/${method}: eval exited ${eval_rc}; generation IS kept at ${gen_file}"
        return 2
    fi

    # Surface the metrics at the top of the run dir for easy collection.
    local metrics_src
    metrics_src="$(find "${eval_dir}" -name '*_metrics.json' -print -quit 2>/dev/null)"
    if [[ -n "${metrics_src}" ]]; then
        cp "${metrics_src}" "${dir}/metrics.json"
    fi

    "${PYTHON}" - "${dir}" "${gen_secs}" "${eval_secs}" "${n_rows}" <<'PY'
import json, os, sys
run_dir, gen_secs, eval_secs, n_rows = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
timing = {"generate_seconds": gen_secs, "eval_seconds": eval_secs, "generation_rows": n_rows}
with open(os.path.join(run_dir, "timing.json"), "w") as f:
    json.dump(timing, f, indent=2)
m = os.path.join(run_dir, "metrics.json")
if os.path.exists(m):
    with open(m) as f:
        print("     acc = %s" % json.load(f).get("acc"))
PY

    echo "DONE" >"${dir}/STATUS"
    date -Is >"${dir}/DONE"
    log "OK   ${dataset}/${method}: eval in $(hms ${eval_secs}) -> ${dir}/metrics.json"
    return 0
}

# ------------------------------------------------------------------- main loop
trap 'log "INTERRUPTED: stopping sweep. Completed runs keep their DONE marker; re-run this script to resume."; exit 130' INT TERM

n_ok=0 n_skip=0 n_fail=0
for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        run_one "${dataset}" "${method}"
        case $? in
            0) n_ok=$((n_ok + 1)) ;;
            3) n_skip=$((n_skip + 1)) ;;
            *) n_fail=$((n_fail + 1)) ;;  # keep going: one bad config must not
                                          # cost the whole night
        esac
    done
done

log "================ sweep finished ================"
log "ok=${n_ok}  skipped=${n_skip}  failed=${n_fail}  elapsed=$(hms $(($(date +%s) - SWEEP_START_EPOCH)))"
log "summary: ${SCRIPT_DIR}/collect_results.py --runs-root ${RUNS_ROOT}"

"${PYTHON}" "${SCRIPT_DIR}/collect_results.py" --runs-root "${RUNS_ROOT}" 2>&1 | tee -a "${SWEEP_LOG}"

[[ ${n_fail} -gt 0 ]] && exit 1
exit 0
