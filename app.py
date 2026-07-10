#!/usr/bin/env python3
"""
多模态情感分析 — Streamlit Demo

启动:
    streamlit run app.py

架构:
    BERT + Wav2Vec2 + ViT/OpenFace → Cross-Attention → 情感分数 [-3,+3]
    自动加载 checkpoints/ 下最新的模型权重。
"""

import os
import glob
import tempfile

import streamlit as st
import numpy as np
import torch
from PIL import Image

from models.multimodal_model import MultimodalSentimentModel

# ================================================================
# 页面配置
# ================================================================
st.set_page_config(
    page_title="多模态情感分析",
    page_icon="🎭",
    layout="wide",
)

st.title("🎭 多模态情感分析")
st.caption("BERT + Wav2Vec2 + ViT  →  Cross-Attention  →  情感强度 [-3, +3]")

# ================================================================
# 模型加载 (缓存)
# ================================================================
@st.cache_resource
def load_model() -> MultimodalSentimentModel:
    """加载模型，优先使用 checkpoints/ 下最新的 .pt 文件。"""
    model = MultimodalSentimentModel()

    # 查找最新 checkpoint
    ckpt_files = sorted(glob.glob("checkpoints/best_model_*.pt"))
    if ckpt_files:
        latest = ckpt_files[-1]
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        st.sidebar.success(f"✅ 已加载: {os.path.basename(latest)}")
    else:
        st.sidebar.warning("⚠️ 未找到 checkpoint，使用随机初始化权重。")

    model.eval()
    return model

# ================================================================
# 输入区域
# ================================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📝 文本")
    text_input = st.text_area("输入英文文本", placeholder="e.g. I'm so happy today!")

with col2:
    st.subheader("🖼️ 图像")
    image_file = st.file_uploader("上传图片 (jpg/png)", type=["jpg", "jpeg", "png"])

with col3:
    st.subheader("🎵 音频")
    audio_file = st.file_uploader("上传音频 (wav/mp3)", type=["wav", "mp3", "flac"])

# ================================================================
# 分析
# ================================================================
if st.button("🚀 分析", type="primary", use_container_width=True):
    has_input = any([text_input, image_file, audio_file])
    if not has_input:
        st.warning("请至少提供一个模态的输入")
    else:
        with st.spinner("推理中..."):
            model = load_model()

            # ----- 准备输入 -----
            text = text_input.strip() if text_input else ""
            audio_arr = None
            tmp_image_path = None

            if audio_file is not None:
                import librosa
                import io as _io
                audio_arr, _ = librosa.load(
                    _io.BytesIO(audio_file.read()), sr=16000, mono=True, duration=10
                )

            if image_file is not None:
                # ViT 需要文件路径: 写入临时文件
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp.write(image_file.read())
                    tmp_image_path = tmp.name

            # ----- 推理 -----
            score = model.forward(
                texts=[text or ""],
                visual_inputs=[tmp_image_path if tmp_image_path else None],
                audio_arrays=[audio_arr if audio_arr is not None
                               else np.zeros(16000, dtype=np.float32)],
            ).item()

            # 清理临时文件
            if tmp_image_path and os.path.exists(tmp_image_path):
                os.unlink(tmp_image_path)

        # ================================================================
        # 结果展示
        # ================================================================
        st.divider()
        st.subheader("📊 分析结果")

        # 连续分数 → 标签
        if score > 0.5:
            label, emoji = "positive", "😊"
        elif score < -0.5:
            label, emoji = "negative", "😞"
        else:
            label, emoji = "neutral", "😐"

        # 置信度近似
        confidence = float(torch.sigmoid(torch.tensor(abs(score) / 3.0)).item())

        # 近似分数分布
        scores = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        scores[label] = confidence

        # 综合指标
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("情感分数", f"{score:+.2f}")
        with metric_cols[1]:
            st.metric("综合情感", f"{emoji} {label.upper()}")
        with metric_cols[2]:
            st.metric("置信度", f"{confidence:.1%}")

        # 分数柱状图
        st.write("#### 情感极性分布")
        st.bar_chart(
            {
                "Negative 😞": [scores["negative"]],
                "Neutral 😐": [scores["neutral"]],
                "Positive 😊": [scores["positive"]],
            },
            horizontal=True,
        )

        # 输入摘要
        st.write("#### 输入摘要")
        st.caption(
            f"文本: {text[:80] + '...' if len(text) > 80 else text or '(未提供)'} | "
            f"图像: {'✅' if image_file else '(未提供)'} | "
            f"音频: {'✅' if audio_file else '(未提供)'}"
        )

        # 原始输出
        with st.expander("🔍 原始输出 (JSON)"):
            st.json({
                "sentiment_score": round(score, 4),
                "label": label,
                "confidence": round(confidence, 4),
                "scores": scores,
            })

# ================================================================
# 页脚
# ================================================================
st.divider()
st.caption("多模态情感分析 | BERT + Wav2Vec2 + ViT  |  Cross-Attention Fusion")
