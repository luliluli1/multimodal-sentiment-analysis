#!/usr/bin/env python3
"""
多模态情感分析 — FastAPI 接口

启动:
    python api.py
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

访问:
    POST /analyze  — 多模态分析 (multipart/form-data)
    GET  /health   — 健康检查
    GET  /docs     — Swagger 文档
"""

import io
import tempfile
import os

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile

from models.multimodal_model import MultimodalSentimentModel

app = FastAPI(
    title="多模态情感分析 API",
    description="BERT + Wav2Vec2 + ViT → Cross-Attention → 情感分数 [-3,+3]",
    version="0.2.0",
)

model = None


def get_model():
    global model
    if model is None:
        model = MultimodalSentimentModel()
        model.eval()
    return model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    text: str = Form(None),
    image: UploadFile = File(None),
    audio: UploadFile = File(None),
):
    if not any([text, image, audio]):
        return {"error": "至少需要一个模态 (text / image / audio)"}

    m = get_model()
    tmp_path = None
    audio_arr = None

    if image:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

    if audio:
        import librosa
        contents = await audio.read()
        audio_arr, _ = librosa.load(
            io.BytesIO(contents), sr=16000, mono=True, duration=10
        )

    result = m.predict(
        text=text or "",
        image_path=tmp_path,
        audio=audio_arr,
    )

    if tmp_path and os.path.exists(tmp_path):
        os.unlink(tmp_path)

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
