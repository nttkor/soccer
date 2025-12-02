# DACON 축구 추적 Transformer 모델

## 코드 개요

이 스크립트는 DACON 축구 추적 경진대회용 Transformer 기반 모델의 학습 및 추론 파이프라인입니다.
Jupyter notebook 로직을 Python 스크립트로 리팩토링하여 직접 실행할 수 있도록 구성되었습니다.

### 주요 구성 요소

- **CONFIG**: 축구장 스케일, 증강, 학습/모델 설정을 한 곳에 모아둔 설정 딕셔너리
- **전처리 함수**: CSV 데이터를 시퀀스(feature, target) 텐서로 변환
- **모델 정의**: Transformer 기반 인코더로 마지막 타임스텝의 임베딩에서 (end_x, end_y) 회귀 예측
- **학습/평가 유틸**: 한 epoch 학습과 검증 스코어(RMSE 기반) 계산
- **main 함수**: 전체 파이프라인 실행 (데이터 로드 → 시퀀스 생성 → 학습 → 추론 → 제출 파일 생성)

## 전체 데이터 흐름도

```
CSV 데이터 → 전처리 → 시퀀스 생성 → 모델 학습 → 추론 → 제출 파일
    ↓          ↓          ↓          ↓          ↓          ↓
train.csv → 정규화 → 8D 피처 → Transformer → 예측 → submission.csv
test.csv  → 패딩    → 증강    → 회귀      → 좌표   → 평가
```

### 상세 흐름도

```
1. 데이터 로드
   ├── train.csv 읽기
   └── 전처리 (정규화, 피처 변환)

2. 시퀀스 생성
   ├── game_episode별 그룹화
   ├── 8차원 피처 행렬 생성 [x,y,거리,각도,액션,x속도,y속도,결과]
   ├── 증강 적용 (수직/수평/양방향 플립)
   └── 시퀀스 길이 맞추기 (패딩/자르기)

3. 모델 학습
   ├── SoccerTransformer 초기화 (PositionalEncoding + Encoder)
   ├── train/val 분리 (8:2)
   ├── 100 epoch 학습
   └── 검증 평가 (RMSE 메트릭)

4. 추론 및 제출
   ├── test.csv 메타데이터 순회
   ├── 각 경기별 시퀀스 로드
   ├── 모델 예측 (정규화된 좌표)
   └── 실제 좌표로 역정규화하여 제출
```

## CONFIG 설정 설명

```python
CONFIG = {
    "field_dims": (105.0, 68.0),        # 축구장 실제 가로/세로 길이 (미터 단위)
    "action_scale": 60.0,               # action_id를 0~1 근처로 맞추기 위한 스케일
    "goal_xy": (1.0, 0.5),              # 정규화 좌표계에서의 골대 위치 (우측 중앙)
    "seq_len": 10,                      # 하나의 에피소드에서 사용할 시퀀스 길이
    "feature_size": 8,                  # 한 타임스텝에서의 피처 차원 수
    "augmentations": {                  # 데이터 증강 설정
        "vertical": True,               # y 축 기준 상하 반전
        "horizontal": True,             # x 축 기준 좌우 반전
        "both": True,                   # x, y 모두 반전
    },
    "model": {
        "d_model": 256,                 # Transformer 모델 차원
        "nhead": 8,                     # 어텐션 헤드 수
        "num_layers": 3,                # Transformer 레이어 수
        "dropout": 0.2,                 # 드롭아웃 비율
        "lr": 5e-4,                     # 학습률
    },
    "training": {
        "epochs": 100,                  # 총 학습 epoch 수
        "batch_size": 64,               # 배치 크기
        "log_interval": 5,              # 로그 출력 간격
    },
    "fallback_xy": (52.5, 34.0),        # 파일 로드 실패 시 사용할 기본 좌표
}
```

## 주요 함수 개요

### 데이터 전처리 함수들

#### `preprocess_dataframe(df, config, is_train=True)`
위치/액션/결과를 정규화하고 숫자 피처로 가공
- 좌표를 0~1 범위로 정규화 (축구장 크기로 나누기)
- action_id를 스케일로 나누어 정규화
- 텍스트 결과를 숫자 레이블로 변환

#### `build_feature_matrix(coords, actions, results, goal_xy)`
8차원 피처 행렬 생성: [x,y,거리,각도,액션,x속도,y속도,결과]
- 골대까지의 유클리드 거리 계산
- 골대를 향한 각도 계산 (라디안)
- 이전 타임스텝과의 속도 차이 계산

#### `pad_or_truncate(features, seq_len, feature_size)`
시퀀스 길이를 고정 길이에 맞게 앞쪽 패딩 또는 뒷부분 자르기
- 길이가 부족하면 앞쪽을 0으로 패딩
- 길이가 초과하면 최근 데이터만 유지

#### `generate_variants(coords, target, config, enable_augment)`
데이터 증강을 위한 변형 생성
- 수직 반전: y 좌표를 1.0에서 빼서 상하 대칭
- 수평 반전: x 좌표를 1.0에서 빼서 좌우 대칭
- 양방향 반전: x,y 모두 반전
- 최대 4배 데이터 증강

#### `create_sequences(df, config, augment=False)`
game_episode 단위로 그룹을 나누고 각 에피소드에서 시퀀스/타깃 쌓기
- 에피소드별 시간 순 정렬
- 증강 적용 시 여러 변형 생성
- 넘파이 배열로 반환

#### `load_match_sequence(file_path, config)`
단일 경기 CSV를 읽어 시퀀스로 변환 (추론용)
- 파일 존재 확인
- 전처리 및 피처 변환
- 고정 길이 시퀀스로 패딩

### 모델 관련 클래스

#### `PositionalEncoding`
Transformer용 사인/코사인 위치 인코딩
- 최대 길이까지 미리 계산하여 저장
- 짝수/홀수 인덱스에 사인/코사인 적용

#### `SoccerTransformer`
축구 시퀀스 회귀용 Transformer 모델
- 선형 임베딩 레이어
- 위치 인코딩 추가
- 다층 Transformer 인코더
- 최종 2차원 좌표 출력

### 학습/평가 함수들

#### `train_one_epoch(model, loader, criterion, optimizer, device)`
DataLoader를 한 바퀴 돌며 평균 학습 손실 계산
- 배치 단위 순전파/역전파/최적화
- 누적 손실의 평균 반환

#### `evaluate(model, val_inputs, val_targets, criterion, device, field_dims)`
검증 세트에 대한 손실과 실제 거리 기준 스코어 계산
- 예측과 실제 좌표의 유클리드 거리 평균 (RMSE)

#### `run_inference(model, meta_df, base_path, config, device)`
메타 정보를 순회하면서 각 경기 시퀀스를 읽고 모델로 좌표 예측
- 파일별 시퀀스 로드
- 배치 추론 수행
- 실제 좌표로 역정규화

### 메인 파이프라인

#### `main()`
전체 학습 및 추론 파이프라인 실행

1. **데이터 로드 단계**
   - train.csv 읽기 및 전처리
   - game_episode별 train/val 분리 (8:2 비율)

2. **시퀀스 생성 단계**
   - 학습 데이터: 4배 증강 적용
   - 검증 데이터: 증강 미적용
   - DataLoader 생성 (배치 크기 64)

3. **모델 초기화 단계**
   - SoccerTransformer 모델 생성
   - 손실 함수 (MSE), 옵티마이저 (Adam), 스케줄러 설정

4. **학습 루프 (100 epoch)**
   - 배치 단위 학습 및 검증 평가
   - 5 epoch마다 로그 출력 (손실, RMSE, 학습률)
   - 검증 손실에 따라 학습률 조정

5. **추론 단계**
   - test.csv 메타데이터 로드
   - 각 경기별 예측 수행
   - sample_submission.csv에 결과 채우기
   - submission_transformer_v5_result_feat.csv로 저장

## 실행 방법

```bash
# 학습 및 추론 실행
python transformer.py

# 또는 tmux에서 백그라운드 실행
tmux new -s soccer_training
python transformer.py
```

## 출력 파일

- `submission_transformer_v5_result_feat.csv`: DACON 제출용 예측 결과
- 학습 로그: 콘솔 출력 (epoch, 손실, 검증 스코어, 학습률)

## 주요 특징

- **Transformer 기반**: 시퀀스 의존성을 효과적으로 학습
- **데이터 증강**: 축구장 대칭성을 활용한 4배 데이터 증강
- **피처 엔지니어링**: 골대까지의 거리/각도, 속도 정보 활용
- **효율적 학습**: 배치 처리 및 GPU 가속 지원
