#!/usr/bin/env python
# coding: utf-8

"""
main.py

FastAPI 백엔드

- /health          : 상태 체크용 엔드포인트
- /api/predict     : REDJOINT 웹에서 history를 보내면, 부위별 부상 확률을 반환
"""

from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference_injury_model import predict_from_history_payload

# ==========================
# 0. FastAPI 앱 생성 + CORS
# ==========================

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="REDJOINT Injury Risk API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cmj01moon.github.io",          # GitHub Pages
        "https://cmj01moon.github.io/REDJOINT"  # 프로젝트 서브 경로
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ==========================
# 1. Pydantic 모델 (요청/응답)
# ==========================

class HistoryItem(BaseModel):
    season: str          # 예: "24/25"
    games_missed: float  # 결장 경기 수
    fouled: float        # 파울 당한 횟수
    time: float          # 뛴 시간 (분)


class PredictRequest(BaseModel):
    age: int             # 현재 나이
    position: str        # 예: "Centre-Back"
    history: List[HistoryItem]


class PredictResponse(BaseModel):
    season: str
    probabilities: Dict[str, float]
    raw: Dict[str, float]
    meta: Dict[str, Any]


# ==========================
# 2. 엔드포인트
# ==========================

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    REDJOINT 웹에서 보내는 요청 형식:

    {
      "age": 23,
      "position": "Centre-Back",
      "history": [
        {
          "season": "24/25",
          "games_missed": 0,
          "fouled": 17,
          "time": 4282
        },
        {
          "season": "23/24",
          "games_missed": 62,
          "fouled": 12,
          "time": 2159
        }
      ]
    }
    """
    try:
        history_payload = [h.dict() for h in req.history]

        result = predict_from_history_payload(
            history=history_payload,
            age=req.age,
            position=req.position,
            season_to_predict="25/26",
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

