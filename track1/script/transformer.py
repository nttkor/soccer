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

import os
import math
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration dictionary
# ---------------------------------------------------------------------------
CONFIG = {
    "field_dims": (105.0, 68.0),
    "action_scale": 60.0,
    "goal_xy": (1.0, 0.5),
    "seq_len": 10,
    "feature_size": 8,
    "augmentations": {
        "vertical": True,
        "horizontal": True,
        "both": True,
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

# [변경] 출력 디렉토리 설정 (스크립트 실행 위치 기준, 혹은 절대 경로)
# 현재 스크립트가 track1/script/ 에 있다고 가정하고, track1/output/ 을 타겟으로 함
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # track1/script
PROJECT_ROOT = os.path.dirname(BASE_DIR)               # track1
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")      # track1/output

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory set to: {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def map_result(value: str) -> float:
    """Convert textual result labels into numeric hints."""
    if value == "Successful":
        return 1.0
    if value == "Unsuccessful":
        return -1.0
    return 0.0


def preprocess_dataframe(df: pd.DataFrame, config: dict, is_train: bool = True) -> pd.DataFrame:
    """Normalize columns and append result mapping."""
    field_x, field_y = config["field_dims"]
    df = df.copy()
    for col in ("start_x", "end_x"):
        if col in df.columns:
            df[col] = df[col].astype(float) / field_x
    for col in ("start_y", "end_y"):
        if col in df.columns:
            df[col] = df[col].astype(float) / field_y
    if "action_id" in df.columns:
        df["action_id"] = df["action_id"].fillna(0) / config["action_scale"]
    else:
        df["action_id"] = 0.0
    if "result_name" in df.columns:
        df["result_mapped"] = df["result_name"].apply(map_result)
    else:
        df["result_mapped"] = 0.0
    return df


def build_feature_matrix(df_segment: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Construct (L, F) matrix.
    Feats: [start_x, start_y, dist_goal, angle_goal, result_mapped, action_id, speed, time_delta]
    """
    gx, gy = config["goal_xy"]
    feats = []
    
    # Pre-compute time deltas
    times = df_segment["time_seconds"].values
    if len(times) > 1:
        dt = np.diff(times, prepend=times[0])
    else:
        dt = np.zeros_like(times)
        
    # Avoid division by zero
    dt = np.where(dt < 1e-9, 1.0, dt)

    for i, row in enumerate(df_segment.itertuples(index=False)):
        sx = getattr(row, "start_x", 0.0)
        sy = getattr(row, "start_y", 0.0)
        
        # Distance/Angle to goal
        dx_g = gx - sx
        dy_g = gy - sy
        dist_g = math.sqrt(dx_g**2 + dy_g**2)
        angle_g = math.atan2(dy_g, dx_g)
        
        # Speed estimate (distance from previous action / time delta)
        # For the first item, speed is 0 or needs logic. Here simply 0 or based on dist.
        # We'll use a simple proxy: just use dt[i] as a feature, let model learn.
        # Or compute speed if previous x,y known. 
        # Simplified: just use dt.
        
        # Features
        # 0: start_x
        # 1: start_y
        # 2: dist_goal
        # 3: angle_goal
        # 4: result_mapped
        # 5: action_id
        # 6: dt (time diff from prev action)
        # 7: dummy or speed (Using dt directly for now)
        
        # Accessing row fields safely
        res_m = getattr(row, "result_mapped", 0.0)
        act_id = getattr(row, "action_id", 0.0)
        
        feat_vec = [sx, sy, dist_g, angle_g, res_m, act_id, dt[i], 0.0]
        feats.append(feat_vec)
        
    return np.array(feats, dtype=np.float32)


def pad_or_truncate(arr: np.ndarray, length: int) -> np.ndarray:
    """Pad (pre) or truncate (keep last L) array."""
    L, F = arr.shape
    if L == length:
        return arr
    if L > length:
        return arr[-length:]  # Keep last actions
    # Pad
    pad_len = length - L
    # Pad with first row or zeros? usually zeros for "no action"
    # But padding with zeros might be interpreted as (0,0) coordinate.
    # Let's pad with zeros.
    padding = np.zeros((pad_len, F), dtype=arr.dtype)
    return np.vstack([padding, arr])


def generate_variants(seq: np.ndarray, target: np.ndarray, config: dict):
    """
    Generate augmented versions of (seq, target).
    Coordinates are at indices 0 (x), 1 (y).
    Target is (x, y).
    """
    variants = [(seq, target)]
    
    # Indices for x, y in features
    # start_x=0, start_y=1
    # angle_goal=3 -> also needs flip? Yes.
    # dist_goal=2 -> invariant
    
    # We will only flip X, Y for simplicity. Angle update is tricky but let's approximate or skip.
    # Ideally re-compute features after flip.
    
    # However, to be correct, we should augment RAW data, then build features.
    # Here we operate on features for speed, assuming simple geometric flips.
    
    # Normalized coords are 0..1. 
    # Vertical flip: y -> 1-y
    # Horizontal flip: x -> 1-x
    
    do_v = config["augmentations"]["vertical"]
    do_h = config["augmentations"]["horizontal"]
    do_b = config["augmentations"]["both"]
    
    def flip_y(s, t):
        s_new = s.copy()
        t_new = t.copy()
        # s[:, 1] is y
        s_new[:, 1] = 1.0 - s_new[:, 1]
        t_new[1] = 1.0 - t_new[1]
        # angle? y changes sign relative to center, but here it's 0..1
        # let's skip angle update for simplicity or re-calc if critical
        return s_new, t_new

    def flip_x(s, t):
        s_new = s.copy()
        t_new = t.copy()
        # s[:, 0] is x
        s_new[:, 0] = 1.0 - s_new[:, 0]
        t_new[0] = 1.0 - t_new[0]
        return s_new, t_new

    if do_v:
        variants.append(flip_y(seq, target))
    if do_h:
        variants.append(flip_x(seq, target))
    if do_b:
        # Flip both
        s_v, t_v = flip_y(seq, target)
        s_vb, t_vb = flip_x(s_v, t_v)
        variants.append((s_vb, t_vb))
        
    return variants


def create_sequences(df: pd.DataFrame, config: dict, augment: bool = False):
    """
    Group by game_episode, build sequences.
    Returns X (N, L, F), y (N, 2)
    """
    df_clean = preprocess_dataframe(df, config, is_train=True)
    
    sequences = []
    targets = []
    
    # Group by (game_id, episode_id) or just 'game_episode' column if unique
    # Using game_episode as unique identifier
    groups = df_clean.groupby("game_episode")
    
    for _, group in groups:
        # Sort by time
        g = group.sort_values("time_seconds")
        
        # Target: last row's end_x, end_y
        last_row = g.iloc[-1]
        target_pt = np.array([last_row["end_x"], last_row["end_y"]], dtype=np.float32)
        
        # Features
        feat_mat = build_feature_matrix(g, config)
        feat_padded = pad_or_truncate(feat_mat, config["seq_len"])
        
        if augment:
            # Generate augmentations
            variant_list = generate_variants(feat_padded, target_pt, config)
            for (s, t) in variant_list:
                sequences.append(s)
                targets.append(t)
        else:
            sequences.append(feat_padded)
            targets.append(target_pt)
            
    return np.array(sequences), np.array(targets)


# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (Batch, SeqLen, Dim)
        # pe: (MaxLen, Dim)
        # Slice pe to (1, SeqLen, Dim)
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class SoccerTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config["model"]["d_model"]
        self.feat_size = config["feature_size"]
        
        # Input projection
        self.input_proj = nn.Linear(self.feat_size, self.d_model)
        
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=config["seq_len"] + 5)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config["model"]["nhead"],
            dim_feedforward=self.d_model * 2,
            dropout=config["model"]["dropout"],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["model"]["num_layers"]
        )
        
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Predict x, y
        )

    def forward(self, x):
        # x: (Batch, Seq, Feat)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # (Batch, Seq, d_model)
        out = self.transformer_encoder(x)
        
        # Use last token embedding for prediction
        # out[:, -1, :] -> (Batch, d_model)
        last_emb = out[:, -1, :]
        
        pred = self.reg_head(last_emb)
        return pred


# ---------------------------------------------------------------------------
# Training / Evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device, config):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    field_x, field_y = config["field_dims"]
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            
            # De-normalize for metric calculation
            # pred: (B, 2), y: (B, 2)
            # 0 -> x, 1 -> y
            
            p_np = pred.cpu().numpy()
            t_np = y_batch.cpu().numpy()
            
            # Scale back
            p_np[:, 0] *= field_x
            p_np[:, 1] *= field_y
            t_np[:, 0] *= field_x
            t_np[:, 1] *= field_y
            
            all_preds.append(p_np)
            all_targets.append(t_np)
            
    avg_loss = total_loss / len(dataloader.dataset)
    
    # Calculate RMSE or Euclidean distance average
    P = np.vstack(all_preds)
    T = np.vstack(all_targets)
    
    # Euclidean distance
    dists = np.sqrt(np.sum((P - T)**2, axis=1))
    mean_dist = np.mean(dists)
    
    return avg_loss, mean_dist


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def load_match_sequence(file_path: str, config: dict) -> np.ndarray:
    """Load a single test csv, preprocess, return (1, L, F) tensor."""
    df = pd.read_csv(file_path)
    df = preprocess_dataframe(df, config, is_train=False)
    df = df.sort_values("time_seconds")
    
    feats = build_feature_matrix(df, config)
    padded = pad_or_truncate(feats, config["seq_len"])
    # Add batch dim
    return np.expand_dims(padded, axis=0)


def run_inference(model, test_df: pd.DataFrame, base_path: str, config: dict, device):
    """
    test_df: columns [game_id, game_episode, path]
    Returns list of (game_episode, pred_x, pred_y)
    """
    model.eval()
    results = []
    
    field_x, field_y = config["field_dims"]
    
    # Iterate over test index
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        rel_path = row["path"]  # e.g. "./test/123/123_1.csv"
        # Adjust path if needed. The provided path starts with "./test/"
        # We need to join with base_path.
        # Remove "./" prefix if exists
        clean_path = rel_path.lstrip("./")
        full_path = os.path.join(base_path, clean_path)
        
        if not os.path.exists(full_path):
            # Fallback
            results.append((row["game_episode"], config["fallback_xy"][0], config["fallback_xy"][1]))
            continue
            
        inp = load_match_sequence(full_path, config)
        inp_t = torch.from_numpy(inp).to(device)
        
        with torch.no_grad():
            pred = model(inp_t)
            # (1, 2)
            
        p_val = pred.cpu().numpy()[0]
        px = p_val[0] * field_x
        py = p_val[1] * field_y
        
        results.append((row["game_episode"], px, py))
        
    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print(">>> Setting up...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Paths (adjust as needed)
    # BASE_DIR should point to 'open_track1' folder containing csv files
    DATA_DIR = os.path.join(PROJECT_ROOT, "open_track1")
    train_csv = os.path.join(DATA_DIR, "train.csv")
    test_csv = os.path.join(DATA_DIR, "test.csv")
    
    if not os.path.exists(train_csv):
        print(f"Error: {train_csv} not found.")
        return

    # 1. Load Train Data
    print(">>> Loading Training Data...")
    df_train = pd.read_csv(train_csv)
    
    # 2. Create Sequences
    print(">>> Creating Sequences...")
    X_all, y_all = create_sequences(df_train, CONFIG, augment=True)
    print(f"Total sequences: {len(X_all)}")
    
    # 3. Split Train/Val
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.1, random_state=42)
    
    # 4. DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["training"]["batch_size"], shuffle=False)
    
    # 5. Model Setup
    model = SoccerTransformer(CONFIG).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["model"]["lr"])
    criterion = nn.MSELoss()
    
    # 6. Training Loop
    epochs = CONFIG["training"]["epochs"]
    best_score = float('inf')
    
    print(">>> Starting Training...")
    for ep in range(epochs):
        t0 = time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_score = evaluate(model, val_loader, criterion, device, CONFIG)
        
        if (ep + 1) % CONFIG["training"]["log_interval"] == 0:
            print(f"Epoch {ep+1}/{epochs} | T_Loss: {train_loss:.5f} | V_Loss: {val_loss:.5f} | V_Dist: {val_score:.4f}m | Time: {time()-t0:.1f}s")
            
        if val_score < best_score:
            best_score = val_score
            # [변경] 모델 저장 경로를 OUTPUT_DIR로 설정
            model_save_path = os.path.join(OUTPUT_DIR, "soccer_model.pt")
            torch.save(model.state_dict(), model_save_path)
            
    print(f">>> Best Val Distance: {best_score:.4f}m")
    
    # 7. Inference
    print(">>> Starting Inference...")
    # Load best model
    model_save_path = os.path.join(OUTPUT_DIR, "soccer_model.pt")
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print("Loaded best model.")
    
    df_test = pd.read_csv(test_csv)
    # df_test has 'path' column. Need to prefix with DATA_DIR?
    # Actually the paths in csv are like "./test/..." 
    # If we run from 'track1', it might work, but let's be explicit.
    # We pass DATA_DIR as base_path for reading test sequences.
    
    preds = run_inference(model, df_test, DATA_DIR, CONFIG, device)
    
    # 8. Save Submission
    print(">>> Saving Submission...")
    sub_df = pd.DataFrame(preds, columns=["game_episode", "end_x", "end_y"])
    
    # [변경] 제출 파일 저장 경로를 OUTPUT_DIR로 설정
    sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Saved to {sub_path}")


if __name__ == "__main__":
    main()