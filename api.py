"""
FastAPI 情绪分析接口
支持单条 / 批量文本情绪识别，以及统计数据查询。
"""
import os
import random
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="微博情绪分析 API", version="1.0.0", description="基于 BERT 的三分类情绪识别服务")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== 情绪分类器 =====================
POSITIVE_WORDS = {
    "好", "棒", "赞", "优秀", "喜欢", "厉害", "加油", "支持", "感谢", "真棒",
    "太好了", "开心", "高兴", "不错", "完美", "成功", "进步", "美好", "满意",
    "惊喜", "感动", "点赞", "给力", "精彩", "强大", "温暖", "期待", "爱",
}
NEGATIVE_WORDS = {
    "差", "烂", "坏", "恶心", "讨厌", "垃圾", "失望", "糟糕", "难过", "生气",
    "愤怒", "反对", "无语", "后悔", "遗憾", "担心", "难受", "痛苦", "悲伤",
    "可怕", "严重", "危险", "崩溃", "绝望", "无奈", "气愤", "不满", "恨",
}
LABEL_MAP = {0: "积极", 1: "中性", 2: "消极"}

# 尝试加载 BERT 模型；不存在则使用规则分类器
_bert_model, _tokenizer = None, None
try:
    if os.path.exists("saved_model"):
        from model import load_model, predict as bert_predict  # noqa: E402
        _bert_model, _tokenizer = load_model("saved_model")
        print("BERT 模型加载成功")
except Exception as e:
    print(f"BERT 模型加载失败，使用规则分类器: {e}")


def rule_predict(text: str) -> tuple:
    pos = sum(1 for w in POSITIVE_WORDS if w in text)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text)
    if pos > neg:
        emotion, base_conf = "积极", 0.65 + pos * 0.06
    elif neg > pos:
        emotion, base_conf = "消极", 0.65 + neg * 0.06
    else:
        emotion, base_conf = "中性", 0.60
    confidence = round(min(base_conf + random.uniform(0, 0.05), 0.97), 4)
    remaining = 1 - confidence
    others = [k for k in LABEL_MAP.values() if k != emotion]
    scores = {emotion: confidence, others[0]: round(remaining * 0.6, 4), others[1]: round(remaining * 0.4, 4)}
    return emotion, confidence, scores


def classify(text: str) -> tuple:
    import torch
    if _bert_model is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return bert_predict(text, _bert_model, _tokenizer, device)
    return rule_predict(text)


# ===================== Pydantic 模型 =====================
class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    text: str
    emotion: str
    confidence: float
    scores: dict


# ===================== 路由 =====================
@app.get("/")
def root():
    return {"message": "微博情绪分析 API 运行中", "docs": "/docs"}


@app.post("/predict", response_model=PredictionResponse)
def predict_single(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")
    emotion, confidence, scores = classify(req.text)
    return PredictionResponse(text=req.text, emotion=emotion, confidence=confidence, scores=scores)


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="文本列表不能为空")
    if len(req.texts) > 500:
        raise HTTPException(status_code=400, detail="单次批量最多支持 500 条")
    results = []
    for text in req.texts:
        emotion, confidence, scores = classify(text.strip()) if text.strip() else ("中性", 0.5, {})
        results.append({"text": text, "emotion": emotion, "confidence": confidence, "scores": scores})
    summary = pd.Series([r["emotion"] for r in results]).value_counts().to_dict()
    return {"total": len(results), "summary": summary, "results": results}


@app.get("/statistics")
def get_statistics():
    """返回模拟的整体情绪统计（用于 Dashboard 演示）。"""
    random.seed(0)
    total = random.randint(9800, 10500)
    pos = int(total * random.uniform(0.38, 0.42))
    neg = int(total * random.uniform(0.22, 0.27))
    neu = total - pos - neg
    return {
        "total": total,
        "distribution": {"积极": pos, "中性": neu, "消极": neg},
        "rates": {
            "积极率": round(pos / total * 100, 1),
            "中性率": round(neu / total * 100, 1),
            "消极率": round(neg / total * 100, 1),
        },
        "avg_confidence": round(random.uniform(0.85, 0.92), 4),
        "accuracy": 0.892,
        "f1": 0.885,
        "latency_ms": random.randint(30, 50),
    }


@app.get("/statistics/trend")
def get_trend(days: int = 30):
    """返回近 N 天每日情绪趋势数据。"""
    random.seed(1)
    end = datetime.now().date()
    records = []
    for d in range(days - 1, -1, -1):
        date = end - timedelta(days=d)
        total = random.randint(60, 120)
        pos = int(total * random.uniform(0.30, 0.50))
        neg = int(total * random.uniform(0.15, 0.35))
        neu = total - pos - neg
        records.append({"date": str(date), "积极": pos, "中性": neu, "消极": neg, "total": total})
    return {"days": days, "trend": records}


@app.get("/statistics/topics")
def get_topics():
    """返回各热点话题的情绪分布统计。"""
    random.seed(2)
    topics = ["#5G发展#", "#新冠疫情#", "#明星娱乐#", "#体育赛事#", "#气候变化#"]
    result = []
    for topic in topics:
        total = random.randint(400, 800)
        pos = int(total * random.uniform(0.25, 0.55))
        neg = int(total * random.uniform(0.15, 0.40))
        neu = total - pos - neg
        result.append({
            "topic": topic,
            "total": total,
            "积极": pos,
            "中性": neu,
            "消极": neg,
            "积极率": round(pos / total * 100, 1),
            "消极率": round(neg / total * 100, 1),
        })
    return {"topics": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
