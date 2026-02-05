# Paper NeSy Ablation Experiment Design (Complete within 7 hours)

## Experiment Goals

Maintain the paper's 4-ablation structure while applying non-NeSy techniques (Frame Stacking, Adaptive Scaling) equally to all ablations for a fair comparison.

## Ablation Setup (Option 1)

### 1. baseline
- **State Abstraction**: ❌
- **Reward Shaping**: ❌
- **Frame Stacking**: ✅ (not NeSy; applied to all modes)
- **Adaptive Scaling**: ✅ (not NeSy; applied to all modes)

### 2. state (State Abstraction only)
- **State Abstraction**: ✅ (paper core: 11,293-dim → 52-dim knowledge vector)
- **Reward Shaping**: ❌
- **Frame Stacking**: ✅
- **Adaptive Scaling**: ✅

### 3. reward (Reward Shaping only)
- **State Abstraction**: ❌
- **Reward Shaping**: ✅ (Potential-based + Critical recovery, paper core)
- **Frame Stacking**: ✅
- **Adaptive Scaling**: ✅

### 4. full (NeSy)
- **State Abstraction**: ✅
- **Reward Shaping**: ✅
- **Frame Stacking**: ✅
- **Adaptive Scaling**: ✅

## Experiment Parameters

### Time Budget
- **Total target time**: 7 hours (420 minutes)
- **Target per experiment**: ~1h 30m–1h 45m (90–105 minutes)
- **Previous run**: 2h 30m / 4 runs = 37.5 min/run (stop-iters=60)
- **This run**: stop-iters=100 for more training

### RLlib Parameters
```bash
ROLLOUT_FRAGMENT_LENGTH=128
TRAIN_BATCH_SIZE=2048
SGD_MINIBATCH_SIZE=256
STOP_ITERS=100
NUM_WORKERS=1
NUM_GPUS=0
```

### NeSy Parameters
```bash
NESY_LAM=1.0  # Reward shaping strength
NESY_FRAME_STACK=1  # Frame stacking (1 = disabled by default)
```

### Seeds
- **Number of seeds**: 1 (123)
- **Reason**: Quick validation; paper can use multiple seeds

## Expected Runtime

### Scenario 1: Conservative (1 min per iteration)
- stop-iters=100
- Per experiment: 100 min = 1h 40m
- 4 experiments: 400 min = 6h 40m ✅ (within 7h)

### Scenario 2: Based on previous run (0.625 min per iteration)
- stop-iters=100
- Per experiment: 62.5 min = 1h 2.5m
- 4 experiments: 250 min = 4h 10m ✅ (within 7h)

### Scenario 3: Worst case (1.5 min per iteration)
- stop-iters=100
- Per experiment: 150 min = 2h 30m
- 4 experiments: 600 min = 10h ❌ (exceeds 7h)

**Recommendation**: Expect between scenarios 1 and 2; **completion within 7 hours is feasible**.

## How to Run

```bash
# 1. Ensure virtual environment is active
source .venv/bin/activate  # or conda activate

# 2. Make executable (first time only)
chmod +x run_paper_ablation_wsl.sh

# 3. Run experiments
./run_paper_ablation_wsl.sh
```

## Output Files

Each experiment produces:

```
ray_results/paper_ablation/{exp_name}/
├── progress.csv          # Ray Tune auto-generated CSV
├── progress.log          # Text progress log
└── final_summary.json   # Final summary JSON
```

## Performance Metrics

Compare the following metrics across ablations:

1. **episode_reward_mean**: Mean episode reward
2. **episode_len_mean**: Mean episode length
3. **uptime_rate_mean**: System uptime (key operational metric)
4. **timesteps_total**: Total training timesteps

## Expected Results

According to the paper:
- **State Abstraction**: Main gains on operational metric (uptime)
- **Reward Shaping**: No consistent gain on its own
- **Full NeSy**: Most stable overall improvement

## If Time Limit Is Exceeded

If runs exceed 7 hours:

1. **Reduce stop-iters**: 100 → 80 or 90
2. **Resume later**: Ensure checkpoints are saved per experiment
3. **Reduce seeds**: Already using a single seed
