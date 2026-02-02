#!/bin/bash
# ============================================================
# 논문 4개 Ablation 실험 — 시드 10개 (0–9) (ontology 2개 제외)
# ============================================================
# 실행 순서: 1(Base) → 2(State) → 3(Reward) → 4(Full NeSy)
# Ontology, Full Ontology 실험은 제외.
# ============================================================

set -euo pipefail

# ===== 설정 (모든 ablation 동일) =====
PY_SCRIPT="rllib_train_cyborg_with_metrics_paper_ablation.py"
STOP_ITERS=50
NUM_WORKERS=2
TRAIN_BATCH_SIZE=4000
ROLLOUT_FRAGMENT_LENGTH=200
NESY_LAM=1.0
MAX_EPISODE_STEPS=800

# 시드 10개 (0–9), 논문용
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# 실험 타임스탬프 및 로그 디렉터리
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="ray_results/paper_4_ablations_10seeds_${TIMESTAMP}"

# Python 경로 (가상환경 우선)
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
else
    PYTHON_CMD="python"
fi

# ===== Ablation 순서: 4개만 (ontology 2개 제외) 1→2→3→4 =====
# 순서 | 실험명     | --ablation | 설명
# -----|------------|------------|------------------------------------------
#  1   | Base       | base       | Raw obs, no shaping
#  2   | State      | state      | 52차원 state만
#  3   | Reward     | reward     | Multi-objective reward만
#  4   | Full NeSy  | full       | State + multi-objective (2+3)
ABLATION_ORDER=(base state reward full)

# ===== Ray 정리 =====
cleanup_ray() {
    echo "🧹 Cleaning up Ray processes..."
    ray stop --force 2>/dev/null || true
    pkill -f "ray::" 2>/dev/null || true
    sleep 3
    echo "✅ Cleanup complete."
}

# ===== 단일 실험 실행 (동일 조건 유지: 동일 인자만 --ablation/--seed/--exp-name 차이) =====
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

# ===== 메인 =====
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 논문 4개 Ablation 실험 — 시드 10개 (0–9) (ontology 제외)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "실험 순서: 1 Base → 2 State → 3 Reward → 4 Full NeSy"
echo "Ablation 목록: ${ABLATION_ORDER[*]}"
echo "시드: ${SEEDS[*]} (총 ${#SEEDS[@]}개)"
echo ""
echo "동일 조건:"
echo "  STOP_ITERS=${STOP_ITERS} NESY_LAM=${NESY_LAM} NUM_WORKERS=${NUM_WORKERS}"
echo "  TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} ROLLOUT_FRAGMENT_LENGTH=${ROLLOUT_FRAGMENT_LENGTH}"
echo "  MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS}"
echo ""
echo "로그 디렉터리: ${LOG_DIR}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cleanup_ray
mkdir -p "${LOG_DIR}"

# 순서대로: 1 → 2 → 3 → 4
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
echo "🎉 4개 Ablation × ${#SEEDS[@]} seeds 실험 완료"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "결과 확인:"
echo "  로그: ${LOG_DIR}"
echo "  요약: find ${LOG_DIR} -name 'final_summary.json' -exec echo {} \\; -exec cat {} \\;"
echo "  분석: python analyze_6_ablations.py --logdir ${LOG_DIR}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
