# 현재 실험 구성 및 구현 특징

## 전체 실험 구성

현재 지원되는 Ablation 모드는 총 **4개**입니다:

### 1. 기본 Ablation (논문 핵심 구성)
- `base`: Baseline (NeSy 기능 없음)
- `state`: State Abstraction만
- `reward`: Reward Shaping만
- `full`: State Abstraction + Reward Shaping (논문의 NeSy)

---

## 각 실험의 구현 특징

### 1. `base` (Baseline)

**구현 특징:**
- **Observation Space**: FixedFlatWrapper의 원본 observation space 사용 (11,293차원)
- **Reward Shaping**: 없음 (CybORG 기본 reward만 사용)
- **State Abstraction**: 없음
- **특징**: NeSy 기능이 전혀 없는 순수 baseline

**코드 위치:**
```python
# cyborg_env_maker_paper_ablation.py
self.use_state_abstraction = False
self.use_reward_shaping = False
nesy_bonus = 0.0
```

**사용 목적**: 다른 실험들과 비교하기 위한 기준선

---

### 2. `state` (State Abstraction Only)

**구현 특징:**
- **Observation Space**: 52차원 압축 (uptime + 17 hosts × 3 features)
  - `uptime`: 시스템 가용성 (1차원)
  - 각 host별 3 features:
    - `is_compromised`: Red agent 세션 존재 여부
    - `service_up`: 서비스 실행 여부
    - `activity`: 활동 여부 (compromised 또는 service_up)
- **Reward Shaping**: 없음
- **State Abstraction**: `_nesy_state_features()` 메서드 사용

**코드 위치:**
```python
# cyborg_env_maker_paper_ablation.py:342-384
def _nesy_state_features(self) -> np.ndarray:
    """Host-specific knowledge vector (52차원)"""
    features = [uptime]
    for hostname in self.known_hosts:
        # is_compromised, service_up, activity 추출
        ...
```

**사용 목적**: State abstraction만으로 성능 개선 효과 측정

---

### 3. `reward` (Reward Shaping Only)

**구현 특징:**
- **Observation Space**: FixedFlatWrapper의 원본 observation space (11,293차원)
- **Reward Shaping**: Multi-Objective Reward Shaping
  - **1. Critical Targets 복구 보너스**: 20.0 × λ × recovery_count
  - **2. Uptime 보존 보너스**: 5.0 × λ × (uptime_delta × 100)
  - **3. Service Availability 보너스**: 2.0 × λ × (service_bonus / len(critical_targets))
  - **4. Attack Prevention 보너스**: 3.0 × λ × (nearby_compromised / len(nearby))
- **State Abstraction**: 없음

**코드 위치:**
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

**사용 목적**: Reward shaping만으로 성능 개선 효과 측정

---

### 4. `full` (Full NeSy)

**구현 특징:**
- **Observation Space**: 52차원 압축 (state abstraction 사용)
- **Reward Shaping**: Multi-Objective Reward Shaping (reward 모드와 동일)
- **State Abstraction**: `_nesy_state_features()` 메서드 사용

**코드 위치:**
```python
# cyborg_env_maker_paper_ablation.py:146-149
self.use_state_abstraction = self.nesy_mode in {"state", "full", ...}
self.use_reward_shaping = self.nesy_mode in {"reward", "full", ...}
```

**사용 목적**: 논문의 제안 방법 (State Abstraction + Reward Shaping)

---

## 공정한 비교를 위한 설계

### 1. Observation Space 통일
- **Full NeSy와 Ontology**: 동일한 52차원 observation space 사용
- **Base와 Reward**: 동일한 11,293차원 observation space 사용

### 2. Reward Shaping 독립성
- **Base Reward**: 모든 실험에서 동일 (CybORG 기본 reward)
- **Reward Shaping**: 각 실험 모드별로 독립적으로 계산
- **최종 Reward**: `base_reward + nesy_bonus + logic_bonus - pruning_penalty`

---

## 최근 실험 결과 디렉토리

```
ray_results/
├── paper_nesy_ablation_full_ontology_20260125_133424/  # Full + Ontology 실험
│   ├── paper_nesy_full_ontology_seed0_20260125_133427/
│   ├── paper_nesy_full_ontology_seed1_20260125_134512/
│   ├── paper_nesy_full_ontology_seed2_20260125_135544/
│   ├── paper_nesy_full_ontology_seed3_20260125_140612/
│   └── paper_nesy_full_ontology_seed4_20260125_141647/
└── paper_nesy_ablation_ontology_improved_20260125_163312/  # Ontology 개선 실험
    ├── paper_nesy_ontology_seed0_20260125_163315/
    ├── paper_nesy_ontology_seed1_20260125_164249/
    ├── paper_nesy_ontology_seed2_20260125_165221/
    ├── paper_nesy_ontology_seed3_20260125_170145/
    └── paper_nesy_ontology_seed4_20260125_171113/
```

---

## 평가 기준

### 주요 메트릭: **Uptime** (CAGE Challenge 공식 기준)
- 시스템이 정상 작동하는 시간 비율 (0.0 ~ 1.0)
- 보안 관점에서 가장 중요한 지표

### 보조 메트릭
- **Reward**: 학습 효율성 평가
- **Episode Length**: 탐색 범위 평가

---

## 참고사항

1. **모든 실험에서 동일한 CybORG 기본 reward 사용**: 공정한 비교 가능
2. **Reward Shaping은 base reward에 추가**: Ablation study의 정상적인 구조
3. **각 실험의 reward 차이는 reward shaping 효과**: 실제 성능 비교는 Uptime 기준
