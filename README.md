# 논문용 실험 패키 (2025-01-30)

가장 최근 **10-seed** 4개 ablation 결과와 관련 소스·스크립트·문서를 모은 디렉토리입니다.

## 포함 내용

### 실험 결과 (ontology 제외)
- **ray_results/paper_6_ablations_10seeds_20260128_181447/**
- **base**, **state**, **reward**, **full** 4개 ablation × 시드 0–9 (총 40개 실험 디렉토리)
- 분석 문서: `ANALYSIS_10seeds.md`, `ANALYSIS_DIM.md`, `ANALYSIS_Uptime_EpLen.md`

### 소스코드
- `rllib_train_cyborg_with_metrics_paper_ablation.py` — 논문 ablation 학습 스크립트
- `cyborg_env_maker_paper_ablation.py` — 논문용 CybORG 환경
- `analyze_6_ablations.py` — **4개 ablation** 결과 분석 (ontology 제외). 출력: `analysis_4_ablations.txt`, `analysis_4_ablations.json`
- `analyze_ablation_results.py`, `analyze_paper_results.py`, `analyze_experiment_results.py` — 결과 분석

### 스크립트 (.sh)  
- `run_paper_6_ablations_10seeds.sh` — 10-seed 4개 ablation (base, state, reward, full). 로그: `ray_results/paper_4_ablations_10seeds_*`
- `run_paper_6_ablations.sh` — 5-seed 4개 ablation. 로그: `ray_results/paper_4_ablations_*`
- `verify_6_ablations_1seed.sh` — 1-seed 4개 ablation 검증. 로그: `ray_results/verify_4_ablations_*`
- `run_ontology_5pct_improvement.sh` — 5-seed 4개 ablation (동일 조건, ontology 실험 제외)

### 문서 (.md)
- 논문 설정·검증: `paper_revision_6_experiments_IEEE_Access.md`, `VERIFICATION_6_ablations.md`
- 실험·분석 관련 루트의 기타 .md 파일들

## 재실행 시
- **실험 조건**: STOP_ITERS=50, NESY_LAM=1.0, NUM_WORKERS=2, TRAIN_BATCH_SIZE=4000, ROLLOUT_FRAGMENT_LENGTH=200, MAX_EPISODE_STEPS=800 (이전과 동일).
- **실행 순서**: 1 Base → 2 State → 3 Reward → 4 Full NeSy 
- **분석**: `python analyze_6_ablations.py --logdir ray_results/paper_4_ablations_10seeds_<TIMESTAMP>`

