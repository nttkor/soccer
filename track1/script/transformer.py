#!/usr/bin/env python3
"""
Refactored training / inference pipeline for DACON soccer tracking task.
This script mirrors the Jupyter notebook logic but can be executed directly.

코드 개요
- CONFIG: 축구장 스케일, 증강, 학습/모델 설정을 한 곳에 모아둔 설정 딕셔너리
- 전처리 함수(preprocess_dataframe, build_feature_matrix, pad_or_truncate, create_sequences):
  CSV → 시퀀스(feature, target) 텐서로 변환
- 모델 정의(PositionalEncoding, SoccerTransformer):
  Transformer 기반 인코더로 마지막 타임스텝의 임베딩에서 (end_x, end_y) 회귀
- 학습/평가 유틸(train_one_epoch, evaluate):
  한 epoch 학습과 검증 스코어(RMSE 기반)를 계산
- main:
  train.csv 로드 → 시퀀스 생성 → 학습 → test.csv 에 대해 추론 → 제출 파일 생성

주요 함수 개요
- preprocess_dataframe: 위치/액션/결과를 정규화하고 숫자 피처로 가공
- build_feature_matrix: 위치 + 골대까지 거리/각도 + 액션 + 속도 + 결과를 8차원 피처로 구성
- pad_or_truncate: 에피소드 길이를 고정 시퀀스 길이에 맞게 앞쪽 패딩/뒷부분 자르기
- generate_variants: 수직/수평/양방향 플립 증강으로 여러 버전의 시퀀스 생성
- create_sequences: game_episode 단위로 그룹을 나누고, 각 에피소드에서 시퀀스/타깃을 쌓아서 넘파이 배열 반환
- load_match_sequence: 단일 경기 CSV를 읽어 하나의 시퀀스로 변환(추론용)
- run_inference: 메타 정보(DataFrame)를 순회하면서 각 경기 시퀀스를 읽고 모델로 좌표 예측
- train_one_epoch: DataLoader를 한 바퀴 돌며 평균 학습 손실 계산
- evaluate: 검증 세트에 대한 손실과 실제 거리 기준 스코어 계산
- main: 전체 파이프라인(데이터 로드 → 시퀀스 생성 → 학습 루프 → 추론/제출 저장)을 실행
"""

import os  # 경로 처리용
import math  # positional encoding 계산에 필요한 수학 함수
from time import time  # epoch 시간 측정

import numpy as np  # 수치 계산
import pandas as pd  # CSV 로드 및 테이블 전처리
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset  # 배치 생성
from sklearn.model_selection import train_test_split  # train/val 분리
from tqdm import tqdm  # 진행률 표시

# ---------------------------------------------------------------------------
# Configuration dictionary (실험 설정을 한 곳에 모아둔 딕셔너리)
# ---------------------------------------------------------------------------
CONFIG = {
    "field_dims": (105.0, 68.0),  # 축구장 실제 가로/세로 길이 (미터 단위)
    "action_scale": 60.0,  # action_id를 0~1 근처로 맞추기 위한 스케일
    "goal_xy": (1.0, 0.5),  # 정규화 좌표계에서의 골대 위치 (우측 중앙)
    "seq_len": 10,  # 하나의 에피소드에서 사용할 시퀀스 길이
    "feature_size": 8,  # 한 타임스텝에서의 피처 차원 수
    "augmentations": {  # 데이터 증강 설정
        "vertical": True,   # y 축 기준 상하 반전
        "horizontal": True, # x 축 기준 좌우 반전
        "both": True,       # x, y 모두 반전
    },
    "model": {
        "d_model": 256,
        "nhead": 8,
        "num_layers": 3,
        "dropout": 0.2,
        "lr": 5e-4,
    },
    "training": {
        "epochs": 100,
        "batch_size": 64,
        "log_interval": 5,
    },
    "fallback_xy": (52.5, 34.0),
}


# ---------------------------------------------------------------------------
# Utility functions (유틸리티 함수들)
# ---------------------------------------------------------------------------
def map_result(value: str) -> float:  # 텍스트 결과를 숫자로 변환하는 함수
    """Convert textual result labels into numeric hints."""
    if value == "Successful":  # 성공인 경우
        return 1.0  # 1.0 반환
    if value == "Unsuccessful":  # 실패인 경우
        return -1.0  # -1.0 반환
    return 0.0  # 그 외는 0.0 반환


def preprocess_dataframe(df: pd.DataFrame, config: dict, is_train: bool = True) -> pd.DataFrame:  # 데이터프레임 전처리 함수
    """Normalize columns and append result mapping."""
    field_x, field_y = config["field_dims"]  # 축구장 가로/세로 길이 가져오기
    df = df.copy()  # 원본 데이터프레임 복사 (원본 보호)
    for col in ("start_x", "end_x"):  # 시작/끝 x 좌표 정규화
        if col in df.columns:  # 해당 컬럼이 존재하면
            df[col] = df[col].astype(float) / field_x  # 실제 길이로 나누어 0~1 범위로 정규화
    for col in ("start_y", "end_y"):  # 시작/끝 y 좌표 정규화
        if col in df.columns:  # 해당 컬럼이 존재하면
            df[col] = df[col].astype(float) / field_y  # 실제 길이로 나누어 0~1 범위로 정규화
    if "action_id" in df.columns:  # 액션 ID 컬럼이 있으면
        df["action_id"] = df["action_id"].fillna(0) / config["action_scale"]  # NaN을 0으로 채우고 스케일로 나누어 정규화
    else:  # 액션 ID 컬럼이 없으면
        df["action_id"] = 0.0  # 기본값 0.0 설정
    if "result_name" in df.columns:  # 결과 텍스트 컬럼이 있으면
        df["result_mapped"] = df["result_name"].apply(map_result)  # 텍스트를 숫자로 변환하여 새 컬럼 생성
    elif "result_mapped" not in df.columns:  # 결과 매핑 컬럼이 없으면
        df["result_mapped"] = 0.0  # 기본값 0.0 설정
    else:  # 이미 결과 매핑 컬럼이 있으면
        df["result_mapped"] = df["result_mapped"].fillna(0)  # NaN만 0으로 채우기
    return df  # 전처리된 데이터프레임 반환


def build_feature_matrix(coords, actions, results, goal_xy):  # 8차원 피처 행렬 생성 함수
    """Create 8D feature matrix for a single episode."""
    goal_x, goal_y = goal_xy  # 골대 위치 가져오기 (정규화된 좌표)
    dist = np.sqrt((coords[:, 0] - goal_x) ** 2 + (coords[:, 1] - goal_y) ** 2).reshape(-1, 1)  # 각 타임스텝에서 골대까지의 유클리드 거리 계산
    angle = np.arctan2(coords[:, 1] - goal_y, coords[:, 0] - goal_x).reshape(-1, 1)  # 골대를 향한 각도 계산 (라디안)
    velocities = np.zeros_like(coords)  # 속도 배열 초기화 (첫 타임스텝은 0)
    if len(coords) > 1:  # 좌표가 2개 이상이면 속도 계산 가능
        velocities[1:, 0] = coords[1:, 0] - coords[:-1, 0]  # x 방향 속도 = 다음 x - 현재 x
        velocities[1:, 1] = coords[1:, 1] - coords[:-1, 1]  # y 방향 속도 = 다음 y - 현재 y
    return np.hstack([coords, dist, angle, actions, velocities, results])  # [x,y,거리,각도,액션,x속도,y속도,결과] 8차원 피처로 연결


def pad_or_truncate(features: np.ndarray, seq_len: int, feature_size: int) -> np.ndarray:  # 시퀀스 길이 맞추기 함수
    """Trim old steps or front-pad zeros to match the desired sequence length."""
    if len(features) >= seq_len:  # 입력 시퀀스가 목표 길이보다 길면
        return features[-seq_len:]  # 뒤에서부터 목표 길이만큼 자르기 (최근 타임스텝 우선)
    padded = np.zeros((seq_len, feature_size))  # (시퀀스길이, 피처차원) 크기의 0으로 채워진 배열 생성
    padded[-len(features):] = features  # 뒤쪽부터 실제 데이터를 채우기 (앞쪽은 0 패딩)
    return padded  # 패딩 또는 잘린 시퀀스 반환


def generate_variants(coords, target, config, enable_augment):  # 데이터 증강 변형 생성 함수
    """Augment coordinates/targets according to configuration."""
    variants = [(coords, target)]  # 원본 좌표와 타깃 쌍을 기본으로 포함
    if not enable_augment:  # 증강 옵션이 꺼져 있으면
        return variants  # 원본만 반환
    aug_cfg = config["augmentations"]  # 증강 설정 가져오기
    if aug_cfg.get("vertical"):  # 수직(y축) 반전 증강이 활성화되어 있으면
        coords_v = coords.copy()  # 좌표 배열 복사
        coords_v[:, 1] = 1.0 - coords_v[:, 1]  # y 좌표를 1.0에서 빼서 상하 반전 (축구장 대칭)
        target_v = target.copy()  # 타깃 좌표 복사
        target_v[1] = 1.0 - target_v[1]  # 타깃 y 좌표도 반전
        variants.append((coords_v, target_v))  # 수직 반전 버전 추가
    if aug_cfg.get("horizontal"):  # 수평(x축) 반전 증강이 활성화되어 있으면
        coords_h = coords.copy()  # 좌표 배열 복사
        coords_h[:, 0] = 1.0 - coords_h[:, 0]  # x 좌표를 1.0에서 빼서 좌우 반전
        target_h = target.copy()  # 타깃 좌표 복사
        target_h[0] = 1.0 - target_h[0]  # 타깃 x 좌표도 반전
        variants.append((coords_h, target_h))  # 수평 반전 버전 추가
    if aug_cfg.get("both"):  # 양방향(x,y 모두) 반전 증강이 활성화되어 있으면
        coords_hv = coords.copy()  # 좌표 배열 복사
        coords_hv[:, 0] = 1.0 - coords_hv[:, 0]  # x 좌표 반전
        coords_hv[:, 1] = 1.0 - coords_hv[:, 1]  # y 좌표 반전
        target_hv = target.copy()  # 타깃 좌표 복사
        target_hv[0] = 1.0 - target_hv[0]  # 타깃 x 좌표 반전
        target_hv[1] = 1.0 - target_hv[1]  # 타깃 y 좌표 반전
        variants.append((coords_hv, target_hv))  # 양방향 반전 버전 추가
    return variants  # 원본 + 증강된 모든 변형 리스트 반환 (최대 4배)


def create_sequences(df: pd.DataFrame, config: dict, augment: bool = False):  # 시퀀스 데이터셋 생성 함수
    """Convert a dataframe into stacked sequences and targets."""
    sequences, targets = [], []  # 시퀀스와 타깃을 저장할 리스트 초기화
    seq_len = config["seq_len"]  # 목표 시퀀스 길이 가져오기
    feature_size = config["feature_size"]  # 피처 차원 수 가져오기
    grouped = df.groupby("game_episode")  # 게임 에피소드별로 그룹화
    for _, group in tqdm(grouped, desc=f"시퀀스 생성(Augment={augment})"):  # 각 에피소드별로 반복 (진행률 표시)
        group = group.sort_values("time_seconds")  # 시간 순으로 정렬
        coords = group[["start_x", "start_y"]].values  # 시작 좌표 배열 추출
        actions = group["action_id"].values.reshape(-1, 1)  # 액션 ID를 열 벡터로 변환
        results = group["result_mapped"].values.reshape(-1, 1)  # 결과 값을 열 벡터로 변환
        target = group[["end_x", "end_y"]].values[-1].copy()  # 마지막 타임스텝의 끝 좌표를 타깃으로 설정
        for coords_variant, target_variant in generate_variants(coords, target, config, augment):  # 증강 변형별로 반복
            feats = build_feature_matrix(coords_variant, actions, results, config["goal_xy"])  # 8차원 피처 행렬 생성
            seq = pad_or_truncate(feats, seq_len, feature_size)  # 시퀀스 길이 맞추기 (패딩 또는 자르기)
            sequences.append(seq)  # 시퀀스 리스트에 추가
            targets.append(target_variant)  # 타깃 리스트에 추가
    return np.array(sequences), np.array(targets)  # 넘파이 배열로 변환하여 반환


def load_match_sequence(file_path: str, config: dict):  # 단일 경기 시퀀스 로드 함수 (추론용)
    """Load a single match CSV into a normalized, padded sequence."""
    if not os.path.exists(file_path):  # 파일이 존재하지 않으면
        return None  # None 반환
    temp_df = pd.read_csv(file_path)  # CSV 파일 읽기
    temp_df = preprocess_dataframe(temp_df, config, is_train=False)  # 전처리 (훈련 모드 아님)
    coords = temp_df[["start_x", "start_y"]].values  # 시작 좌표 추출
    actions = temp_df["action_id"].values.reshape(-1, 1)  # 액션 ID 열 벡터로 변환
    results = temp_df["result_mapped"].values.reshape(-1, 1)  # 결과 값 열 벡터로 변환
    feats = build_feature_matrix(coords, actions, results, config["goal_xy"])  # 피처 행렬 생성
    return pad_or_truncate(feats, config["seq_len"], config["feature_size"])  # 시퀀스 길이 맞추어 반환


def run_inference(model, meta_df, base_path, config, device):  # 모델 추론 실행 함수
    """Iterate over metadata rows and generate predictions."""
    model.eval()  # 모델을 평가 모드로 설정 (드롭아웃 비활성화)
    preds_x, preds_y = [], []  # 예측 결과를 저장할 리스트
    fallback_x, fallback_y = config["fallback_xy"]  # 파일 로드 실패 시 사용할 기본 좌표
    with torch.no_grad():  # 그래디언트 계산 비활성화 (추론 시 메모리 절약)
        for _, row in tqdm(meta_df.iterrows(), total=len(meta_df)):  # 메타데이터의 각 행에 대해 반복
            file_path = os.path.join(base_path, row["path"][2:])  # CSV 파일 경로 생성 (상대 경로)
            seq = load_match_sequence(file_path, config)  # CSV 파일을 시퀀스로 로드
            if seq is None:  # 파일이 없거나 로드 실패 시
                preds_x.append(fallback_x)  # 기본 x 좌표 추가
                preds_y.append(fallback_y)  # 기본 y 좌표 추가
                continue  # 다음 파일로 건너뜀
            input_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)  # 시퀀스를 텐서로 변환하고 배치 차원 추가
            pred = model(input_tensor).cpu().numpy()[0]  # 모델 예측 수행 (정규화된 좌표)
            preds_x.append(pred[0] * config["field_dims"][0])  # 실제 축구장 크기로 역정규화하여 x 좌표 저장
            preds_y.append(pred[1] * config["field_dims"][1])  # 실제 축구장 크기로 역정규화하여 y 좌표 저장
    return preds_x, preds_y  # 모든 예측 좌표 리스트 반환


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):  # 위치 인코딩 클래스 (Transformer용)
    """Sine/cosine positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500):  # 초기화 메서드
        super().__init__()  # 부모 클래스 초기화
        pe = torch.zeros(max_len, d_model)  # (최대길이, 모델차원) 크기의 0 텐서 생성
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 위치 인덱스 생성 (열 벡터)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 주기 계산을 위한 분모
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스: 사인 함수 적용
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스: 코사인 함수 적용
        self.register_buffer("pe", pe.unsqueeze(0))  # 배치 차원을 추가하여 버퍼로 등록 (학습되지 않음)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 순전파 메서드
        length = x.size(1)  # 입력 시퀀스의 길이 가져오기
        return x + self.pe[:, :length, :]  # 입력에 해당 길이만큼의 위치 인코딩을 더해서 반환


class SoccerTransformer(nn.Module):  # 축구 시퀀스 회귀용 Transformer 모델
    """Encoder-only Transformer tailored for soccer sequence regression."""

    def __init__(self, config: dict):  # 모델 초기화
        super().__init__()  # 부모 클래스 초기화
        model_cfg = config["model"]  # 모델 설정 가져오기
        feature_size = config["feature_size"]  # 입력 피처 차원
        self.embedding = nn.Linear(feature_size, model_cfg["d_model"])  # 피처를 모델 차원으로 선형 변환
        self.pos_encoder = PositionalEncoding(model_cfg["d_model"])  # 위치 인코딩 레이어
        encoder_layer = nn.TransformerEncoderLayer(  # Transformer 인코더 레이어 생성
            d_model=model_cfg["d_model"],  # 모델 차원
            nhead=model_cfg["nhead"],  # 어텐션 헤드 수
            dim_feedforward=512,  # 피드포워드 네트워크 차원
            dropout=model_cfg["dropout"],  # 드롭아웃 비율
            batch_first=True,  # 배치 차원을 첫 번째로 설정
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=model_cfg["num_layers"])  # 다층 인코더
        self.fc_out = nn.Linear(model_cfg["d_model"], 2)  # 최종 출력 레이어 (x, y 좌표 예측)

    def forward(self, src: torch.Tensor) -> torch.Tensor:  # 순전파 메서드
        x = self.embedding(src)  # 입력 피처를 임베딩 차원으로 변환
        x = self.pos_encoder(x)  # 위치 인코딩 추가
        encoded = self.transformer_encoder(x)  # Transformer 인코더 통과
        return self.fc_out(encoded[:, -1, :])  # 마지막 타임스텝의 인코딩을 사용하여 (x, y) 좌표 예측


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):  # 1 epoch 학습 함수
    model.train()  # 모델을 학습 모드로 설정 (드롭아웃 활성화)
    running_loss = 0.0  # 누적 손실 초기화
    for inputs, targets in loader:  # 배치 단위로 데이터 반복
        inputs, targets = inputs.to(device), targets.to(device)  # 데이터를 디바이스로 이동
        optimizer.zero_grad()  # 이전 그래디언트 초기화
        outputs = model(inputs)  # 모델 예측 수행
        loss = criterion(outputs, targets)  # 손실 계산 (MSE)
        loss.backward()  # 역전파로 그래디언트 계산
        optimizer.step()  # 파라미터 업데이트
        running_loss += loss.item()  # 배치 손실 누적
    return running_loss / len(loader)  # 평균 손실 반환


def evaluate(model, val_inputs, val_targets, criterion, device, field_dims):  # 검증 평가 함수
    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():  # 그래디언트 계산 비활성화
        outputs = model(val_inputs)  # 검증 데이터로 예측 수행
        loss = criterion(outputs, val_targets).item()  # 검증 손실 계산
        diff_x = (outputs[:, 0] - val_targets[:, 0]) * field_dims[0]  # x 좌표 실제 차이 (미터 단위)
        diff_y = (outputs[:, 1] - val_targets[:, 1]) * field_dims[1]  # y 좌표 실제 차이 (미터 단위)
        val_score = torch.mean(torch.sqrt(diff_x ** 2 + diff_y ** 2)).item()  # 유클리드 거리 평균 (RMSE)
    return loss, val_score  # 손실과 RMSE 스코어 반환


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------
def main():
    print("데이터 로드 중...")
    train_df = pd.read_csv("../open_track1/train.csv")  # 학습용 CSV 로드
    train_df = preprocess_dataframe(train_df, CONFIG, is_train=True)  # 위치 정규화 및 피처 가공

    unique_episodes = train_df["game_episode"].unique()  # 전체 에피소드 목록 추출
    train_eps, val_eps = train_test_split(unique_episodes, test_size=0.2, random_state=42)  # 8:2로 train/val 분리
    train_df_split = train_df[train_df["game_episode"].isin(train_eps)].copy()  # 학습용 데이터프레임
    val_df_split = train_df[train_df["game_episode"].isin(val_eps)].copy()  # 검증용 데이터프레임
    print(f"학습 에피소드: {len(train_eps)}개, 검증 에피소드: {len(val_eps)}개")

    print("학습 데이터셋 생성 중 (4배 증강 적용)...")  # 플립 증강으로 데이터 4배 증가
    X_train, y_train = create_sequences(train_df_split, CONFIG, augment=True)  # 시퀀스 생성 (증강 적용)
    print(f"학습 데이터 Shape: {X_train.shape}")  # (샘플수, 시퀀스길이, 피처차원)
    print("검증 데이터셋 생성 중 (증강 미적용)...")  # 검증은 원본만 사용
    X_val, y_val = create_sequences(val_df_split, CONFIG, augment=False)  # 시퀀스 생성 (증강 미적용)
    print(f"검증 데이터 Shape: {X_val.shape}")

    # DataLoader 생성: 배치 단위로 데이터 공급
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),  # 텐서로 변환
        batch_size=CONFIG["training"]["batch_size"],  # 한 번에 64개 샘플씩
        shuffle=True,  # 학습 시 랜덤 섞기
    )
    X_val_tensor = torch.FloatTensor(X_val)  # 검증 입력 텐서
    y_val_tensor = torch.FloatTensor(y_val)  # 검증 타깃 텐서

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 우선 사용
    print(f"Using Device: {device}")
    model = SoccerTransformer(CONFIG).to(device)  # 모델 초기화 및 디바이스 이동
    criterion = nn.MSELoss()  # 손실 함수: 평균 제곱 오차
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["model"]["lr"])  # 옵티마이저
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)  # 학습률 스케줄러
    print(model)  # 모델 구조 출력

    print("\n[Transformer 학습 시작]")
    training_cfg = CONFIG["training"]
    X_val_tensor = X_val_tensor.to(device)  # 검증 데이터를 GPU로
    y_val_tensor = y_val_tensor.to(device)
    for epoch in range(training_cfg["epochs"]):  # 설정된 epoch 수만큼 반복
        start = time()  # 시간 측정 시작
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)  # 1 epoch 학습
        val_loss, val_score = evaluate(model, X_val_tensor, y_val_tensor, criterion, device, CONFIG["field_dims"])  # 검증 평가
        scheduler.step(val_loss)  # 검증 손실에 따라 학습률 조정
        if (epoch + 1) % training_cfg["log_interval"] == 0:  # 5 epoch마다 로그 출력
            current_lr = optimizer.param_groups[0]["lr"]  # 현재 학습률
            elapsed = time() - start  # 걸린 시간
            print(
                f"Epoch {epoch+1:02d} | Loss: {avg_loss:.5f} | Val Score: {val_score:.4f} "  # MSE 손실, RMSE 스코어
                f"| LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
            )

    print("\n[추론 시작]")  # 학습 완료 후 테스트 데이터로 예측
    test_meta = pd.read_csv("../open_track1/test.csv")  # 테스트 메타 정보 로드
    preds_x, preds_y = run_inference(model, test_meta, "../open_track1", CONFIG, device)  # 각 경기별 예측

    submission = pd.read_csv("../open_track1/sample_submission.csv")  # 제출 양식 로드
    submission["end_x"] = preds_x  # 예측 x 좌표 채우기
    submission["end_y"] = preds_y  # 예측 y 좌표 채우기
    output_name = "submission_transformer_v5_result_feat.csv"  # 출력 파일명
    submission.to_csv(output_name, index=False)  # CSV로 저장 (인덱스 제외)
    print(f"저장 완료: {output_name}")  # 완료 메시지


if __name__ == "__main__":  # 스크립트 직접 실행 시
    main()  # 메인 함수 호출하여 전체 파이프라인 실행

