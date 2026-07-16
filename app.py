#!/usr/bin/env python3
"""
多模态情感分析 — Streamlit Demo (V1)

使用 MOSEI 测试集样本进行推理，展示模型预测能力。

启动:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.mosei_sdk_dataset import MOSEISDKDataset
from inference import MultimodalPredictor

# ================================================================
# 页面配置
# ================================================================
st.set_page_config(
    page_title="多模态情感分析 Demo",
    page_icon="🎭",
    layout="wide",
)

st.title("🎭 多模态情感分析")
st.caption("BERT + VisualEncoder(FACET) + CovarepAdapter(COVAREP) → Cross-Attention Fusion")

# ================================================================
# 缓存：数据集 + 推理器（只加载一次）
# ================================================================
@st.cache_resource
def load_test_dataset() -> MOSEISDKDataset:
    """加载 MOSEI 测试集。"""
    return MOSEISDKDataset(split="test", modalities=["text", "audio", "visual"])


@st.cache_resource
def load_predictor() -> MultimodalPredictor:
    """加载推理器。"""
    return MultimodalPredictor()


# ================================================================
# 初始化
# ================================================================
test_ds = load_test_dataset()
predictor = load_predictor()

# ---- 侧栏：模型信息 ----
with st.sidebar:
    st.header("📋 模型信息")
    st.write(f"**Checkpoint:**")
    st.code(os.path.basename(predictor.checkpoint_path))
    st.write("**验证集指标:**")
    metrics = predictor.checkpoint_metrics
    if metrics:
        df_m = pd.DataFrame(
            {k: [f"{v:.4f}"] for k, v in metrics.items()}
        ).T.rename(columns={0: "value"})
        st.dataframe(df_m, use_container_width=True)
    st.divider()
    st.caption(f"测试集样本数: **{len(test_ds)}**")

# ================================================================
# 主区域：样本选择
# ================================================================
st.header("📝 选择测试样本")

sample_idx = st.selectbox(
    label="Sample Index",
    options=range(len(test_ds)),
    format_func=lambda i: f"[{i}] {test_ds[i]['id']} — {test_ds[i]['text'][:60]}...",
)

sample = test_ds[sample_idx]

# ---- 样本信息卡片 ----
col_info_1, col_info_2, col_info_3, col_info_4 = st.columns(4)
with col_info_1:
    st.metric("Sample ID", sample["id"])
with col_info_2:
    st.metric("Ground Truth", f"{sample['label']:+.4f}")
with col_info_3:
    has_vis = sample["visual"] is not None
    st.metric("Visual Feature", "✅ FACET" if has_vis else "❌ None")
with col_info_4:
    has_aud = sample["audio"] is not None
    st.metric("Audio Feature", "✅ COVAREP" if has_aud else "❌ None")

# ---- 文本展示 ----
with st.expander("📄 原始文本", expanded=True):
    st.write(sample["text"])

# ================================================================
# 推理按钮
# ================================================================
st.divider()

if st.button("🔍 Analyze", type="primary", use_container_width=True):
    with st.spinner("推理中…"):
        result = predictor.predict(
            text=sample["text"],
            visual=sample["visual"] if has_vis else None,
            audio=sample["audio"] if has_aud else None,
        )

    gt = float(sample["label"])
    mae = abs(result["score"] - gt)

    # ---- 结果展示 ----
    st.header("📊 分析结果")

    res_col_1, res_col_2, res_col_3 = st.columns(3)
    with res_col_1:
        st.metric(
            "Prediction Score",
            f"{result['score']:+.4f}",
            delta=f"{result['score'] - gt:+.4f} vs GT",
        )
    with res_col_2:
        sentiment_emoji = {
            "positive": "😊", "neutral": "😐", "negative": "😞",
        }
        emoji = sentiment_emoji.get(result["sentiment"], "")
        st.metric(
            "Predicted Sentiment",
            f"{emoji} {result['sentiment'].upper()}",
        )
    with res_col_3:
        st.metric("Confidence", f"{result['confidence']:.2%}")

    # ---- 对比表格 ----
    st.write("#### Prediction vs Ground Truth")

    gt_sentiment = "positive" if gt > 0.5 else ("negative" if gt < -0.5 else "neutral")
    gt_emoji = sentiment_emoji.get(gt_sentiment, "")

    df_compare = pd.DataFrame(
        {
            "": ["Score", "Sentiment", "MAE"],
            "Prediction": [
                f"{result['score']:+.4f}",
                f"{result['sentiment'].upper()} {emoji}",
                "—",
            ],
            "Ground Truth": [
                f"{gt:+.4f}",
                f"{gt_sentiment.upper()} {gt_emoji}",
                "—",
            ],
            "Δ": [
                f"{result['score'] - gt:+.4f}",
                "✅" if result["sentiment"] == gt_sentiment else "❌",
                f"{mae:.4f}",
            ],
        }
    ).set_index("")
    st.dataframe(df_compare, use_container_width=True)

    # ---- 进度条式可视化 ----
    st.write("#### 情感分数可视化")
    # 将 [-3, +3] 映射到 [0, 1]
    pred_norm = (result["score"] + 3) / 6
    gt_norm = (gt + 3) / 6

    st.caption(f"Prediction: {result['score']:+.2f}  │  Ground Truth: {gt:+.2f}")
    st.progress(
        float(np.clip(pred_norm, 0.0, 1.0)),
        text=f"😞 negative {'─' * 20} neutral {'─' * 20} positive 😊",
    )

    col_bar_1, col_bar_2 = st.columns([1, 5])
    with col_bar_1:
        st.caption(f"GT marker →")
    with col_bar_2:
        st.markdown(
            f"<span style='color:red; font-size:24px; "
            f"position:relative; left:{gt_norm * 100:.0f}%;'>▼</span>",
            unsafe_allow_html=True,
        )

# ================================================================
# 页脚
# ================================================================
st.divider()
st.caption(
    "多模态情感分析 V1 | BERT + FACET Visual + COVAREP Audio | Cross-Attention Fusion"
)
