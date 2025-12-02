# python:track1/script/visualizer.py
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import pandas as pd
import matplotlib.cm as cm

# 터미널 실행을 위해 ipywidgets 관련 코드는 제거하고 matplotlib 이벤트로 대체합니다.

# ---------------------------------------------------------------------------
# 1. 데이터 로드
# ---------------------------------------------------------------------------
# 스크립트 위치 기준 상대 경로로 데이터 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # track1/script
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "open_track1") # track1/open_track1
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")

print(f"Loading data from {TRAIN_CSV}...")
if not os.path.exists(TRAIN_CSV):
    print("Error: train.csv not found!")
    exit(1)

df_raw = pd.read_csv(TRAIN_CSV)
print("Data loaded successfully.")

# ---------------------------------------------------------------------------
# 2. 경기장 그리기 함수
# ---------------------------------------------------------------------------
def draw_pitch(ax):
    # 경기장 크기 (105 x 68)
    # 흰색 바탕, 검정 테두리
    
    # 메인 사각형 (경기장)
    rect = patches.Rectangle((0, 0), 105, 68, linewidth=2, edgecolor='black', facecolor='white')
    ax.add_patch(rect)
    
    # 중앙선
    ax.plot([52.5, 52.5], [0, 68], color='black', linewidth=1)
    # 센터 서클
    circle = patches.Circle((52.5, 34), 9.15, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(circle)
    
    # 페널티 박스 (좌측)
    ax.add_patch(patches.Rectangle((0, 34-20.16), 16.5, 40.32, linewidth=1, edgecolor='black', facecolor='none'))
    # 페널티 박스 (우측)
    ax.add_patch(patches.Rectangle((105-16.5, 34-20.16), 16.5, 40.32, linewidth=1, edgecolor='black', facecolor='none'))
    
    # 골대 (좌측, 우측) - 빨간색으로 강조
    ax.add_patch(patches.Rectangle((-2, 34-3.66), 2, 7.32, linewidth=2, edgecolor='red', facecolor='red', alpha=0.3))
    ax.add_patch(patches.Rectangle((105, 34-3.66), 2, 7.32, linewidth=2, edgecolor='red', facecolor='red', alpha=0.3))

# ---------------------------------------------------------------------------
# 3. 시각화 로직 (Interactive)
# ---------------------------------------------------------------------------
# 첫 번째 게임 선택
target_game_id = df_raw['game_id'].unique()[0]
print(f"Visualizing Game ID: {target_game_id}")

match_df = df_raw[df_raw['game_id'] == target_game_id].sort_values('time_seconds').reset_index(drop=True)

current_idx = 0
fig, ax = plt.subplots(figsize=(14, 9))

def update_plot():
    global current_idx
    
    ax.clear() # 기존 그림 지우기
    draw_pitch(ax) # 경기장 다시 그리기
    
    row = match_df.iloc[current_idx]
    
    # 다음 수 데이터
    next_row = None
    if current_idx < len(match_df) - 1:
        next_row = match_df.iloc[current_idx + 1]
    
    past_data = match_df.iloc[:current_idx+1]
    current_positions = past_data.drop_duplicates(subset='player_id', keep='last')
    
    # A. 선수들 배치
    for _, p_row in current_positions.iterrows():
        color = 'red' if p_row['is_home'] else 'blue'
        is_active = (p_row['player_id'] == row['player_id'])
        
        action_initial = str(p_row['type_name'])[0] if pd.notna(p_row['type_name']) else "?"
        short_id = str(p_row['player_id'])[-2:]
        
        ax.plot(p_row['start_x'], p_row['start_y'], 'o', 
                markersize=16 if is_active else 12, 
                color=color, 
                alpha=1.0 if is_active else 0.4,
                markeredgecolor='black')
        
        ax.text(p_row['start_x'], p_row['start_y'], action_initial, 
                fontsize=8 if is_active else 7, 
                color='white', fontweight='bold', ha='center', va='center')
        
        id_color = 'black' if is_active else 'gray'
        id_weight = 'bold' if is_active else 'normal'
        ax.text(p_row['start_x'], p_row['start_y'] + 1.5, short_id, 
                fontsize=9, color=id_color, fontweight=id_weight, ha='center')

    # B. 현재 수의 궤적
    if not pd.isna(row['end_x']):
        color = 'red' if row['is_home'] else 'blue'
        
        if row['result_name'] == 'Unsuccessful':
            linestyle = '--'
            marker = 'X'
            marker_size = 15
            alpha = 0.6
            linewidth = 1
            edgecolor = 'red'
        else:
            linestyle = '-'
            marker = 'x'
            marker_size = 10
            alpha = 0.8
            linewidth = 1.5
            edgecolor = color
            
        dx = row['end_x'] - row['start_x']
        dy = row['end_y'] - row['start_y']
        ax.arrow(row['start_x'], row['start_y'], dx, dy, 
                 head_width=1.5, head_length=1.5, 
                 fc=color, ec=edgecolor, alpha=alpha, 
                 linestyle=linestyle, linewidth=linewidth)
        
        ax.plot(row['end_x'], row['end_y'], marker, color=edgecolor, markersize=marker_size, markeredgewidth=2)
        
    # C. 다음 수 미리보기
    if next_row is not None:
        ax.plot(next_row['start_x'], next_row['start_y'], '*', markersize=15, 
                markerfacecolor='gold', markeredgecolor='black', label='Next Move')

    # --- 정보 표시 ---
    result_str = str(row['result_name']) if pd.notna(row['result_name']) else "-"
    if result_str == 'Unsuccessful':
        result_str = "Unsuccessful (FAIL)"
        
    line1 = (f"[{current_idx}/{len(match_df)-1}] Time: {row['time_seconds']:.1f}s (P{row['period_id']})  |  "
             f"Team: {'Home' if row['is_home'] else 'Away'} ({row['team_id']})  |  "
             f"Player: {row['player_id']}  |  "
             f"Action: {row['type_name']} ({result_str})")
             
    pos_str = f"Pos: ({row['start_x']:.1f}, {row['start_y']:.1f})"
    if not pd.isna(row['end_x']):
        pos_str += f" -> ({row['end_x']:.1f}, {row['end_y']:.1f})"
        
    next_str = ""
    if next_row is not None:
        next_team = 'Home' if next_row['is_home'] else 'Away'
        next_str = f"  ||  NEXT: {next_team}(..{str(next_row['player_id'])[-2:]}) {next_row['type_name']}"
    
    line2 = f"{pos_str}  |  ActID: {row['action_id']}{next_str}"
    full_info = f"{line1}\n{line2}"
    
    title_color = 'darkred' if row['result_name'] == 'Unsuccessful' else 'black'
    ax.set_title(full_info, loc='left', fontsize=12, fontweight='bold', pad=12, color=title_color)
    
    # 범례 및 안내 문구
    ax.legend(loc='upper right')
    ax.text(0, -5, "KEYBOARD: [Left] Prev, [Right] Next", fontsize=12, color='blue', fontweight='bold')
    
    fig.canvas.draw()

# 키보드 이벤트 핸들러
def on_key(event):
    global current_idx
    if event.key == 'right':
        if current_idx < len(match_df) - 1:
            current_idx += 1
            update_plot()
    elif event.key == 'left':
        if current_idx > 0:
            current_idx -= 1
            update_plot()

# 초기 실행
fig.canvas.mpl_connect('key_press_event', on_key)
update_plot()
plt.show()
