#!/bin/bash
# ============================================================
# Paper 4 Ablation Experiments — 10 seeds (0–9) (2 ontology experiments excluded)
# ============================================================
# Run order: 1(Base) → 2(State) → 3(Reward) → 4(Full NeSy)
# Ontology and Full Ontology experiments are excluded.
# ============================================================

set -euo pipefail

# ===== Settings (same for all ablations) =====
PY_SCRIPT="rllib_train_cyborg_with_metrics_paper_ablation.py"
STOP_ITERS=50
NUM_WORKERS=2
TRAIN_BATCH_SIZE=4000
ROLLOUT_FRAGMENT_LENGTH=200
NESY_LAM=1.0
MAX_EPISODE_STEPS=800

# 10 seeds (0–9), for paper
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# Experiment timestamp and log directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="ray_results/paper_4_ablations_10seeds_${TIMESTAMP}"

# Python path (prefer virtualenv)
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
else
    PYTHON_CMD="python"
fi

# ===== Ablation order: 4 only (2 ontology excluded) 1→2→3→4 =====
# Order | Experiment  | --ablation | Description
# ------|-------------|------------|------------------------------------------
#  1    | Base        | base       | Raw obs, no shaping
#  2    | State       | state      | 52-dim state only
#  3    | Reward      | reward     | Multi-objective reward only
#  4    | Full NeSy   | full       | State + multi-objective (2+3)
ABLATION_ORDER=(base state reward full)

# ===== Ray cleanup =====
cleanup_ray() {
    echo "🧹 Cleaning up Ray processes..."
    ray stop --force 2>/dev/null || true
    pkill -f "ray::" 2>/dev/null || true
    sleep 3
    echo "✅ Cleanup complete."
}

# ===== Single experiment run (same conditions; only --ablation/--seed/--exp-name differ) =====
run_experiment() {
    local ablation=$1
    local seed=$2
    local exp_name="paper_nesy_${ablation}_seed${seed}_${TIMESTAMP}"
    local exp_log_dir="${LOG_DIR}/${exp_name}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 [START] ${exp_name}"
    echo "   Ablation: ${ablation} | Seed: ${seed} | NESY_LAM: ${NESY_LAM} | Stop: ${STOP_ITERS}"
    echo "   Log: ${exp_log_dir}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ -f "${exp_log_dir}/final_summary.json" ]; then
        echo "⏭️  Skipping (already completed)."
        return 0
    fi
    mkdir -p "${exp_log_dir}"

    "${PYTHON_CMD}" -u "${PY_SCRIPT}" \
        --exp-name "${exp_name}" \
        --seed "${seed}" \
        --ablation "${ablation}" \
        --nesy-lam "${NESY_LAM}" \
        --stop-iters "${STOP_ITERS}" \
        --max-episode-steps "${MAX_EPISODE_STEPS}" \
        --rollout-fragment-length "${ROLLOUT_FRAGMENT_LENGTH}" \
        --train-batch-size "${TRAIN_BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --num-gpus 0 \
        --logdir "${LOG_DIR}" \
        2>&1 | tee "${exp_log_dir}/experiment.log"

    sleep 2
    echo "✅ [DONE] ${exp_name}"
}

# ===== Main =====
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Paper 4 Ablation Experiments — 10 seeds (0–9) (ontology excluded)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Run order: 1 Base → 2 State → 3 Reward → 4 Full NeSy"
echo "Ablation list: ${ABLATION_ORDER[*]}"
echo "Seeds: ${SEEDS[*]} (${#SEEDS[@]} total)"
echo ""
echo "Same conditions:"
echo "  STOP_ITERS=${STOP_ITERS} NESY_LAM=${NESY_LAM} NUM_WORKERS=${NUM_WORKERS}"
echo "  TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} ROLLOUT_FRAGMENT_LENGTH=${ROLLOUT_FRAGMENT_LENGTH}"
echo "  MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS}"
echo ""
echo "Log directory: ${LOG_DIR}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cleanup_ray
mkdir -p "${LOG_DIR}"

# In order: 1 → 2 → 3 → 4
phase=1
for ablation in "${ABLATION_ORDER[@]}"; do
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════════════"
    printf "📊 PHASE %d: %s\n" "$phase" "$ablation"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    for seed in "${SEEDS[@]}"; do
        run_experiment "${ablation}" "${seed}"
        cleanup_ray
    done
    phase=$((phase + 1))
done

cleanup_ray

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 4 Ablations × ${#SEEDS[@]} seeds completed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results:"
echo "  Logs: ${LOG_DIR}"
echo "  Summary: find ${LOG_DIR} -name 'final_summary.json' -exec echo {} \\; -exec cat {} \\;"
echo "  Analysis: python analyze_6_ablations.py --logdir ${LOG_DIR}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
