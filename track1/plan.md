먼저 대회 구조랑 데이터 구성을 같이 확인해 보고, 그 다음에 “최소한 이 정도는 하면 된다” 수준의 단계별 플랜(환경 세팅 → 데이터 이해 → 간단한 베이스라인 코드)을 잡아 드릴게요. 일단 주어진 `readme`랑 `train.csv` 구조를 잠깐 읽어볼게요.

[5 tools called]

### 대회가 뭔지부터 정리

- **문제**: K리그 실제 경기에서, 한 플레이 시퀀스(episode)가 주어졌을 때 **마지막 패스가 어디로 갈지 (end_x, end_y)** 를 예측하는 문제입니다.  
- **평가**: 제출한 좌표와 정답 좌표 사이의 **유클리드 거리(Euclidean Distance)** 가 작을수록 점수가 높습니다.  
- **제출 형식**: `sample_submission.csv` 처럼 `game_episode,end_x,end_y` 형식으로 제출합니다.

---

### 지금 폴더 구조에서 중요한 파일들

- **`track1/open_track1/train.csv`**
  - 각 행: 한 개의 액션(패스, 드리블 등)
  - 주요 컬럼:  
    - `game_id, period_id, episode_id, time_seconds, team_id, player_id, type_name, result_name, start_x, start_y, end_x, end_y, is_home, game_episode`
  - **학습용 데이터**입니다. 여기의 `end_x`, `end_y` 를 이용해 모델을 학습하게 됩니다.
- **`track1/open_track1/test.csv`**
  - 컬럼: `game_id, game_episode, path`
  - 각 `path` 에 해당하는 `./test/게임ID/에피소드.csv` 파일에 **해당 에피소드의 액션 시퀀스**가 들어 있습니다.
- **`track1/open_track1/test/…/*.csv`**
  - 실제로 모델이 보게 될 **테스트 시퀀스 데이터**입니다.
- **`track1/open_track1/sample_submission.csv`**
  - 제출 형식 예시. 현재는 전부 `50.0, 50.0` 으로 채워둔 **예시 베이스라인**입니다.
- **`track1/open_track1/match_info.csv`**
  - 경기 단위 메타 정보(홈팀/원정팀, 스코어 등). 피처 엔지니어링할 때 참고용.

---

### 처음 참가자를 위한 “최소 플랜”

1. **환경 세팅**
   - 파이썬 3.9 이상 설치 (Anaconda 추천)
   - 터미널에서(혹은 Anaconda Prompt 에서):
     ```bash
     pip install pandas numpy scikit-learn
     ```
   - 작업 디렉토리: `c:\Users\user\git\soccer` 로 두고 진행하면 편합니다.

2. **데이터 한 번 직접 열어보기 (Jupyter / VSCode / PyCharm 아무거나)**
   - `train.csv` 몇 줄, `test.csv` 몇 줄 출력해서 **컬럼 이름**과 대략 값 범위만 확인해보세요.
   - Jupyter 예:
     ```python
     import pandas as pd

     train = pd.read_csv(r"track1/open_track1/train.csv")
     test = pd.read_csv(r"track1/open_track1/test.csv")

     print(train.head())
     print(test.head())
     ```

3. **가장 쉬운 베이스라인: “전체 평균 좌표” 예측**
   - 복잡한 시퀀스 처리는 일단 무시하고,
   - `train.csv` 전체에서 `end_x`, `end_y` 의 평균을 구해서  
     **모든 테스트 에피소드에 똑같은 좌표를 예측**하는 베이스라인을 먼저 만들어보는 게 좋습니다.
   - 이게 성능은 낮더라도,
     - 데이터 읽기
     - 제출 파일 만들기
     - 데이콘 제출 성공
     를 한 번 경험하는 게 크리티컬합니다.

---

### 평균 좌표 베이스라인 예시 코드

아래 코드를 `baseline_mean.py` 같은 이름으로 저장해서,  
현재 프로젝트 루트(`c:\Users\user\git\soccer`)에서 실행한다고 가정하겠습니다.

```python
import os
import pandas as pd

# 현재 작업 경로 기준
BASE_DIR = os.path.join("track1", "open_track1")

train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")
sample_path = os.path.join(BASE_DIR, "sample_submission.csv")
save_path = os.path.join(BASE_DIR, "submission_baseline_mean.csv")

# 1. 데이터 불러오기
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

# 2. train 전체에서 end_x, end_y 평균 구하기
mean_end_x = train["end_x"].mean()
mean_end_y = train["end_y"].mean()

print("mean_end_x:", mean_end_x)
print("mean_end_y:", mean_end_y)

# 3. sample_submission 형식을 그대로 쓰되, 좌표만 평균값으로 채우기
submission = sample_sub.copy()
submission["end_x"] = mean_end_x
submission["end_y"] = mean_end_y

# 4. 제출 파일 저장
submission.to_csv(save_path, index=False, encoding="utf-8-sig")
print("저장 완료:", save_path)
```

- 실행:
  ```bash
  cd C:\Users\user\git\soccer
  python baseline_mean.py
  ```
- 실행 후 만들어지는 `track1/open_track1/submission_baseline_mean.csv` 를 데이콘에 업로드하면 됩니다.

---

### 그 다음 단계(조금 더 잘하고 싶을 때)

지금은 **“무조건 평균”** 이라서 점수가 낮을 가능성이 큽니다.  
다음 단계로는 이런 식으로 천천히 확장해 나가면 됩니다:

- **아이디어 1**: `train` 에서 **episode별로 마지막 패스의 end_x, end_y** 만 추출해서 라벨로 사용하기  
  (각 `game_episode` 당 정답 1개로 요약)
- **아이디어 2**: 각 episode의 시퀀스(여러 액션)를  
  - 마지막 액션의 좌표,  
  - 패스/드리블 비율,  
  - 평균 위치,  
  같은 피처로 요약하고 **회귀 모델(RandomForestRegressor 등)**을 학습
- **아이디어 3**: 나중에는 RNN/Transformer 같은 시퀀스 모델도 도전해볼 수 있음 (처음엔 추천 X)

---

지금 단계에서 **가장 먼저 해볼 것**은:

1. 위 평균 베이스라인 코드 돌려서 제출까지 한 번 성공해보고  
2. 점수와 리더보드에 내 이름이 올라가는 경험을 해보는 것

입니다.  

원하시면 **“episode별 정답 추출하는 코드”** 부터 차근차근 같이 짜 보면서 한 단계씩 올려볼까요?