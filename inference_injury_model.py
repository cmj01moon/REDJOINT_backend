#!/usr/bin/env python
# coding: utf-8

"""
inference_injury_model.py

- 학습된 LSTM 모델(injury_lstm_base.h5)과 메타 정보(injury_model_meta.json) 로드
- 웹에서 들어오는 history(JSON)를 기반으로 LSTM 입력 시퀀스를 만들고
  target_cols (예: from_knee, from_head ...) 별 확률을 예측
- 이를 다시 9개 부위 카테고리(head, arm, body, hip, knee, calf, foot, hamstring, other)
  기준으로 합쳐서 반환하는 함수 제공
"""

from pathlib import Path
import json
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ==========================
# 0. 모델 / 메타 로드
# ==========================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "injury_lstm_base.h5"
META_PATH = BASE_DIR / "injury_model_meta.json"

print(f"[INFO] Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)

print(f"[INFO] Loading meta from: {META_PATH}")
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

feature_cols = meta["feature_cols"]          # ['games_missed', 'age', 'fouled', 'time', 'position_...']
target_cols = meta["target_cols"]           # ['from_arm', 'from_body', ...]
max_len = int(meta["max_len"])
PLAYER_COL = meta.get("player_col", "player")
SEASON_COL = meta.get("season_col", "injury")

num_features = len(feature_cols)
num_targets = len(target_cols)

print("[INFO] feature_cols:", feature_cols)
print("[INFO] target_cols:", target_cols)
print("[INFO] max_len:", max_len)


# ==========================
# 1. history(JSON) → DataFrame 변환
# ==========================

def history_payload_to_df(
    history: List[Dict[str, Any]],
    age: float,
    position: str
) -> pd.DataFrame:
    """
    웹에서 들어온 history(JSON 리스트)를 pandas DataFrame으로 변환.
    history 원소 예시:
      {
        "season": "24/25",
        "games_missed": 0,
        "fouled": 17,
        "time": 4282
      }

    - season → SEASON_COL ('injury')에 매핑
    - games_missed, fouled, time을 그대로 사용
    - age는 요청의 최상위 값으로 모든 row에 채움
    - position은 'position_...' 원핫으로 채움
    """
    if not history:
        raise ValueError("history is empty")

    # 기본 DataFrame 생성
    df = pd.DataFrame(history)

    # season → injury 컬럼으로
    if "season" in df.columns and SEASON_COL not in df.columns:
        df[SEASON_COL] = df["season"]

    # 숫자 피처 기본값 및 채우기
    # (있으면 그대로 사용, 없으면 0 또는 age로)
    if "games_missed" not in df.columns:
        df["games_missed"] = 0.0
    if "fouled" not in df.columns:
        df["fouled"] = 0.0
    if "time" not in df.columns:
        df["time"] = 0.0

    # age는 요청의 age를 모든 row에 동일하게 채움
    df["age"] = float(age)

    # 포지션 원핫 처리
    position_cols = [c for c in feature_cols if c.startswith("position_")]
    for c in position_cols:
        df[c] = 0.0

    if position:
        pos_col = f"position_{position}"
        if pos_col in position_cols:
            df[pos_col] = 1.0
        else:
            # 만약 학습 때 없던 포지션이면 그대로 0으로 두고 넘어감
            print(f"[WARN] Position '{position}' not found in feature_cols. All position_* remain 0.")

    # feature_cols + SEASON_COL만 남기고 나머지는 드랍
    keep_cols = set(feature_cols + [SEASON_COL])
    drop_cols = [c for c in df.columns if c not in keep_cols]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


# ==========================
# 2. 시퀀스 생성
# ==========================

def build_single_sequence_from_df(pdf: pd.DataFrame) -> np.ndarray:
    """
    한 선수의 히스토리 DataFrame(여러 시즌 row)을 받아
    길이 max_len의 시퀀스 1개를 만든다.

    - pdf: feature_cols + SEASON_COL 포함
    return:
      X: shape (1, max_len, num_features)
    """
    # 시즌 순으로 정렬
    pdf = pdf.sort_values(by=SEASON_COL).reset_index(drop=True)

    feats = pdf[feature_cols].values  # (T, num_features)
    T = len(feats)

    if T == 0:
        raise ValueError("history_df has no rows after processing.")

    # T가 max_len보다 길면, 최근 max_len만 사용
    if T > max_len:
        feats = feats[T - max_len :]
        T = max_len

    # 상단 0-padding
    pad_len = max_len - T
    if pad_len > 0:
        pad = np.zeros((pad_len, num_features), dtype=np.float32)
        padded = np.vstack([pad, feats])
    else:
        padded = feats

    # 배치 차원 추가
    X = np.expand_dims(padded, axis=0)  # (1, max_len, num_features)
    return X.astype(np.float32)


# ==========================
# 3. raw 예측 함수 (from_* 단위)
# ==========================

def predict_raw_from_history(
    history: List[Dict[str, Any]],
    age: float,
    position: str
) -> Dict[str, float]:
    """
    history(JSON 리스트) + age, position을 받아
    target_cols 각각에 대한 확률을 반환.

    return:
      {
        "from_arm": 0.10,
        "from_body": 0.05,
        ...
      }
    """
    df = history_payload_to_df(history, age=age, position=position)
    X = build_single_sequence_from_df(df)
    preds = model.predict(X, verbose=0)[0]  # shape: (num_targets,)

    probs = {col: float(p) for col, p in zip(target_cols, preds.tolist())}
    return probs


# ==========================
# 4. raw → 9개 부위(body parts) 매핑
# ==========================

# target_cols는 정확히 이 10개라고 가정 (CSV 기준):
# ['from_arm','from_body','from_calf','from_foot','from_hamstring',
#  'from_head','from_hip','from_knee','from_other','from_x']

COL_TO_PART = {
    "from_head": "head",
    "from_arm": "arm",
    "from_body": "body",
    "from_hip": "hip",
    "from_knee": "knee",
    "from_calf": "calf",
    "from_foot": "foot",
    "from_hamstring": "hamstring",
    "from_other": "other",
    "from_x": "other",   # x도 other로 묶기
}

BODY_PARTS = [
    "head",
    "arm",
    "body",
    "hip",
    "knee",
    "calf",
    "foot",
    "hamstring",
    "other",
]


def map_raw_to_body_parts(raw_probs: Dict[str, float]) -> Dict[str, float]:
    """
    raw_probs: {target_col_name: prob} 를 9개 body part로 묶어서 평균.

    여기서는 같은 부위로 매핑되는 타겟들이 여러 개면 평균 사용.
    (지금은 from_other, from_x 둘 다 other로 감)
    """
    buckets: Dict[str, List[float]] = {}

    for col, p in raw_probs.items():
        part = COL_TO_PART.get(col, "other")
        if part not in buckets:
            buckets[part] = []
        buckets[part].append(p)

    body_probs: Dict[str, float] = {}
    for part in BODY_PARTS:
        if part in buckets and len(buckets[part]) > 0:
            body_probs[part] = float(np.mean(buckets[part]))
        else:
            body_probs[part] = 0.0

    return body_probs


# ==========================
# 5. 최종 예측 래퍼
# ==========================

def predict_from_history_payload(
    history: List[Dict[str, Any]],
    age: int,
    position: str,
    season_to_predict: str = "25/26",
) -> Dict[str, Any]:
    """
    웹에서 들어온 payload 중 history/age/position을 받아,
    - raw target_cols(from_*) 확률
    - 9개 body part(head, arm, ...) 확률
    을 계산해서 묶어 반환.
    """
    if not history:
        raise ValueError("History is empty")

    raw_probs = predict_raw_from_history(history, age=age, position=position)
    body_probs = map_raw_to_body_parts(raw_probs)

    result = {
        "season": season_to_predict,
        "probabilities": body_probs,
        "raw": raw_probs,
        "meta": {
            "age": age,
            "position": position,
        }
    }
    return result
