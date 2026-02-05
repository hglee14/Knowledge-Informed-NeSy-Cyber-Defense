# Current Experiment Configuration and Implementation Notes

## Overall Experiment Setup

There are **4** supported ablation modes:

### 1. Core Ablations (paper setup)
- `base`: Baseline (no NeSy)
- `state`: State Abstraction only
- `reward`: Reward Shaping only
- `full`: State Abstraction + Reward Shaping (paper NeSy)

---

## Implementation Notes per Experiment

### 1. `base` (Baseline)

**Implementation:**
- **Observation Space**: FixedFlatWrapper’s original observation space (11,293 dims)
- **Reward Shaping**: None (CybORG default reward only)
- **State Abstraction**: None
- **Note**: Pure baseline with no NeSy

**Code location:**
```python
# cyborg_env_maker_paper_ablation.py
self.use_state_abstraction = False
self.use_reward_shaping = False
nesy_bonus = 0.0
```

**Purpose**: Baseline for comparison with other experiments.

---

### 2. `state` (State Abstraction Only)

**Implementation:**
- **Observation Space**: 52-dim compressed (uptime + 17 hosts × 3 features)
  - `uptime`: System availability (1 dim)
  - Per-host 3 features:
    - `is_compromised`: Whether a Red agent session exists
    - `service_up`: Whether the service is running
    - `activity`: Whether compromised or service_up
- **Reward Shaping**: None
- **State Abstraction**: Uses `_nesy_state_features()`

**Code location:**
```python
# cyborg_env_maker_paper_ablation.py:342-384
def _nesy_state_features(self) -> np.ndarray:
    """Host-specific knowledge vector (52 dims)"""
    features = [uptime]
    for hostname in self.known_hosts:
        # is_compromised, service_up, activity
        ...
```

**Purpose**: Measure effect of state abstraction only.

---

### 3. `reward` (Reward Shaping Only)

**Implementation:**
- **Observation Space**: FixedFlatWrapper’s original observation space (11,293 dims)
- **Reward Shaping**: Multi-Objective Reward Shaping
  - **1. Critical Targets recovery bonus**: 20.0 × λ × recovery_count
  - **2. Uptime preservation bonus**: 5.0 × λ × (uptime_delta × 100)
  - **3. Service Availability bonus**: 2.0 × λ × (service_bonus / len(critical_targets))
  - **4. Attack Prevention bonus**: 3.0 × λ × (nearby_compromised / len(nearby))
- **State Abstraction**: None

**Code location:**
```python
# cyborg_env_maker_paper_ablation.py:1323-1421
if self.use_reward_shaping and self.nesy_lam != 0.0:
    # 1. Critical Recovery
    if recovery_count > 0:
        nesy_bonus += 20.0 * self.nesy_lam * recovery_count
    # 2. Uptime Preservation
    if uptime_delta > 0:
        nesy_bonus += 5.0 * self.nesy_lam * (uptime_delta * 100.0)
    # 3. Service Availability
    nesy_bonus += 2.0 * self.nesy_lam * (service_bonus / len(critical_targets))
    # 4. Attack Prevention
    nesy_bonus += prevention_bonus
```

**Purpose**: Measure effect of reward shaping only.

---

### 4. `full` (Full NeSy)

**Implementation:**
- **Observation Space**: 52-dim compressed (state abstraction)
- **Reward Shaping**: Same multi-objective reward shaping as reward mode
- **State Abstraction**: Uses `_nesy_state_features()`

**Code location:**
```python
# cyborg_env_maker_paper_ablation.py:146-149
self.use_state_abstraction = self.nesy_mode in {"state", "full", ...}
self.use_reward_shaping = self.nesy_mode in {"reward", "full", ...}
```

**Purpose**: Paper’s proposed method (State Abstraction + Reward Shaping).

---

## Design for Fair Comparison

### 1. Observation Space Consistency
- **Full NeSy and Ontology**: Same 52-dim observation space
- **Base and Reward**: Same 11,293-dim observation space

### 2. Reward Shaping Independence
- **Base reward**: Same for all experiments (CybORG default)
- **Reward shaping**: Computed per mode
- **Final reward**: `base_reward + nesy_bonus + logic_bonus - pruning_penalty`

---

## Recent Experiment Result Directories

```
ray_results/
├── paper_nesy_ablation_full_ontology_20260125_133424/  # Full + Ontology
│   ├── paper_nesy_full_ontology_seed0_20260125_133427/
│   ├── paper_nesy_full_ontology_seed1_20260125_134512/
│   ...
└── paper_nesy_ablation_ontology_improved_20260125_163312/  # Ontology improvement
    ├── paper_nesy_ontology_seed0_20260125_163315/
    ...
```

---

## Evaluation Criteria

### Primary metric: **Uptime** (CAGE Challenge official)
- Fraction of time the system is operational (0.0–1.0)
- Most important from a security perspective

### Secondary metrics
- **Reward**: Learning efficiency
- **Episode Length**: Exploration scope

---

## Notes

1. **Same CybORG base reward in all experiments**: Enables fair comparison.
2. **Reward shaping is added to base reward**: Standard ablation structure.
3. **Reward differences reflect reward shaping**: Compare performance using Uptime.
