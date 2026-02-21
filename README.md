# Paper Experiment Package (2025-01-30)

This directory contains the most recent **10-seed** 4-ablation results along with related source code, scripts, and documentation.

## Contents

### Experiment Results (ontology excluded)
- **ray_results/paper_4_ablations_10seeds_20260128_181447/**
- **base**, **state**, **reward**, **full** — 4 ablations × seeds 0–9 (40 experiment directories in total)
- Analysis docs: `ANALYSIS_10seeds.md`, `ANALYSIS_DIM.md`, `ANALYSIS_Uptime_EpLen.md`

### Source Code
- `rllib_train_cyborg_with_metrics_paper_ablation.py` — Paper ablation training script
- `cyborg_env_maker_paper_ablation.py` — CybORG environment for the paper
- `analyze_4_ablations.py` — 4-ablation result analysis (ontology excluded). Outputs: `analysis_4_ablations.txt`, `analysis_4_ablations.json`
- `analyze_ablation_results.py`, `analyze_paper_results.py`, `analyze_experiment_results.py` — Result analysis

### Scripts (.sh)
- `run_paper_4_ablations_10seeds.sh` — 10-seed 4 ablations (base, state, reward, full). Logs: `ray_results/paper_4_ablations_10seeds_*`
- `run_paper_4_ablations.sh` — 5-seed 4 ablations. Logs: `ray_results/paper_4_ablations_*`
- `verify_4_ablations_1seed.sh` — 1-seed 4-ablation verification. Logs: `ray_results/verify_4_ablations_*`
- `run_ontology_5pct_improvement.sh` — 5-seed 4 ablations (same settings, ontology experiments excluded)

### Documentation (.md)
- Paper setup/verification: `paper_revision_4_experiments_IEEE_Access.md`, `VERIFICATION_4_ablations.md`
- Other experiment/analysis-related .md files at the project root

## When Re-running
- **Experiment settings**: STOP_ITERS=50, NESY_LAM=1.0, NUM_WORKERS=2, TRAIN_BATCH_SIZE=4000, ROLLOUT_FRAGMENT_LENGTH=200, MAX_EPISODE_STEPS=800 (unchanged from before).
- **Run order**: 1 Base → 2 State → 3 Reward → 4 Full NeSy
- **Analysis**: `python analyze_4_ablations.py --logdir ray_results/paper_4_ablations_10seeds_<TIMESTAMP>`
