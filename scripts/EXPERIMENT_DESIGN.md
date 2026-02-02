# 논문 NeSy Ablation 실험 설계 (7시간 이내 완료)

## 실험 목표

논문의 4개 Ablation 구조를 유지하면서, NeSy가 아닌 기법들(Frame Stacking, Adaptive Scaling)은 모든 ablation에 동일하게 적용하여 공정한 비교를 수행합니다.

## Ablation 구성 (옵션 1)

### 1. baseline
- **State Abstraction**: ❌
- **Reward Shaping**: ❌
- **Frame Stacking**: ✅ (NeSy 아님, 모든 모드에 적용)
- **Adaptive Scaling**: ✅ (NeSy 아님, 모든 모드에 적용)

### 2. state (State Abstraction only)
- **State Abstraction**: ✅ (논문의 핵심: 11,293차원 → 52차원 knowledge vector)
- **Reward Shaping**: ❌
- **Frame Stacking**: ✅
- **Adaptive Scaling**: ✅

### 3. reward (Reward Shaping only)
- **State Abstraction**: ❌
- **Reward Shaping**: ✅ (Potential-based + Critical recovery, 논문의 핵심)
- **Frame Stacking**: ✅
- **Adaptive Scaling**: ✅

### 4. full (NeSy)
- **State Abstraction**: ✅
- **Reward Shaping**: ✅
- **Frame Stacking**: ✅
- **Adaptive Scaling**: ✅

## 실험 파라미터

### 시간 예산
- **총 목표 시간**: 7시간 (420분)
- **각 실험당 목표**: 약 1시간 30분~1시간 45분 (90~105분)
- **어제 실험 기준**: 2시간 30분/4개 = 37.5분/실험 (stop-iters=60)
- **이번 실험**: stop-iters=100으로 증가하여 더 많은 학습

### RLlib 파라미터
```bash
ROLLOUT_FRAGMENT_LENGTH=128
TRAIN_BATCH_SIZE=2048
SGD_MINIBATCH_SIZE=256
STOP_ITERS=100
NUM_WORKERS=1
NUM_GPUS=0
```

### NeSy 파라미터
```bash
NESY_LAM=1.0  # Reward shaping 강도
NESY_FRAME_STACK=1  # Frame stacking (기본값 1 = 비활성화)
```

### 시드
- **시드 수**: 1개 (123)
- **이유**: 빠른 검증을 위해, 논문에서는 여러 시드 사용 가능

## 예상 실행 시간

### 시나리오 1: 보수적 추정 (각 iteration당 1분)
- stop-iters=100
- 각 실험당: 100분 = 1시간 40분
- 총 4개 실험: 400분 = 6시간 40분 ✅ (7시간 이내)

### 시나리오 2: 어제 실험 기준 (각 iteration당 0.625분)
- stop-iters=100
- 각 실험당: 62.5분 = 1시간 2.5분
- 총 4개 실험: 250분 = 4시간 10분 ✅ (7시간 이내)

### 시나리오 3: 최악의 경우 (각 iteration당 1.5분)
- stop-iters=100
- 각 실험당: 150분 = 2시간 30분
- 총 4개 실험: 600분 = 10시간 ❌ (7시간 초과)

**권장**: 시나리오 1~2 사이로 예상되므로, **7시간 이내 완료 가능**합니다.


## 실행 방법

```bash
# 1. 가상환경 활성화 확인
source .venv/bin/activate  # 또는 conda activate

# 2. 실행 권한 부여 (처음만)
chmod +x run_paper_ablation_wsl.sh

# 3. 실험 실행
./run_paper_ablation_wsl.sh
```

## 결과 파일

각 실험마다 다음 파일이 생성됩니다:

```
ray_results/paper_ablation/{exp_name}/
├── progress.csv          # Ray Tune 형식의 자동 생성 CSV
├── progress.log          # 텍스트 형식의 진행 로그
└── final_summary.json    # 최종 요약 JSON
```

## 성능 비교 지표

각 ablation의 다음 메트릭을 비교:

1. **episode_reward_mean**: 평균 에피소드 보상
2. **episode_len_mean**: 평균 에피소드 길이
3. **uptime_rate_mean**: 시스템 가동률 (핵심 운영 지표)
4. **timesteps_total**: 총 학습 타임스텝

## 예상 결과

논문에 따르면:
- **State Abstraction**: 운영 지표(uptime)에서 주요 개선 제공
- **Reward Shaping**: 단독으로는 일관된 개선 없음
- **Full NeSy**: 가장 안정적인 전체 개선

## 시간 초과 시 대응

만약 7시간을 초과할 경우:

1. **stop-iters 감소**: 100 → 80 또는 90
2. **실험 중단 후 재개**: 각 실험의 체크포인트 저장 확인
3. **시드 수 감소**: 이미 1개만 사용 중

