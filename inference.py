#!/usr/bin/env python3
"""
多模态情感分析 — 推理模块

可作为命令行工具独立运行，也可被其他程序 import 调用。

命令行用法:
    python inference.py
    python inference.py --sample 5
    python inference.py --modalities text
    python inference.py --modalities visual
    python inference.py --modalities audio
    python inference.py --modalities text visual
    python inference.py --checkpoint checkpoints/best_model_epoch005_mae0.4025.pt

编程调用:
    from inference import MultimodalPredictor
    pred = MultimodalPredictor()
    result = pred.predict(text="I love this movie!")
    print(result)  # {"score": 1.23, "sentiment": "positive", "confidence": 0.62}
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEVICE, CHECKPOINT_DIR
from models.multimodal_model import MultimodalSentimentModel


class MultimodalPredictor:
    """多模态情感分析推理器。

    加载已训练的 checkpoint，提供单样本预测接口。
    """

    SENTIMENT_THRESHOLD = 0.5  # >0.5 positive, <-0.5 negative, else neutral

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """初始化推理器。

        Args:
            checkpoint_path: checkpoint 文件路径。默认使用最佳模型。
            device: 推理设备。默认使用 config.DEVICE。
        """
        self.device = device or DEVICE
        checkpoint_path = checkpoint_path or os.path.join(
            CHECKPOINT_DIR, "best_model_epoch005_mae0.4025.pt"
        )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.model = MultimodalSentimentModel()
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self._checkpoint_path = checkpoint_path
        self._checkpoint_metrics = ckpt.get("metrics", {})

    # ----------------------------------------------------------
    # 公开接口
    # ----------------------------------------------------------

    def predict(
        self,
        text: Optional[str] = None,
        visual: Optional[np.ndarray] = None,
        audio: Optional[np.ndarray] = None,
    ) -> dict:
        """对单个样本进行情感预测。

        Args:
            text:       文本输入（英文转录文本），None 表示不使用文本。
            visual:     视觉特征 (n_frames, feature_dim) float32。
                        None 表示不使用视觉模态。
            audio:      音频特征。
                        2D array (n_frames, 74) → COVAREP 特征；
                        1D array (n_samples,)   → 原始波形。
                        None 表示不使用音频模态。

        Returns:
            {
                "score":     float,  # 情感强度 [-3, +3]
                "sentiment": str,    # "positive" | "neutral" | "negative"
                "confidence": float, # 置信度 [0, 1]
            }
        """
        with torch.no_grad():
            score = self.model(
                texts=[text or ""],
                visual_inputs=[visual],
                audio_inputs=[audio],
            ).item()

        sentiment, confidence = self._score_to_result(score)

        return {
            "score": float(score),
            "sentiment": sentiment,
            "confidence": float(confidence),
        }

    @property
    def checkpoint_path(self) -> str:
        """当前加载的 checkpoint 路径。"""
        return self._checkpoint_path

    @property
    def checkpoint_metrics(self) -> dict:
        """Checkpoint 中保存的验证集指标。"""
        return dict(self._checkpoint_metrics)

    # ----------------------------------------------------------
    # 内部
    # ----------------------------------------------------------

    @classmethod
    def _score_to_result(cls, score: float) -> tuple[str, float]:
        """连续分数 → (sentiment_label, confidence)。"""
        if score > cls.SENTIMENT_THRESHOLD:
            sentiment = "positive"
        elif score < -cls.SENTIMENT_THRESHOLD:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        confidence = float(torch.sigmoid(torch.tensor(abs(score) / 3.0)).item())
        return sentiment, confidence


# ================================================================
# 命令行入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="多模态情感分析 — 推理测试")
    parser.add_argument(
        "--checkpoint", type=str,
        default=os.path.join(CHECKPOINT_DIR, "best_model_epoch005_mae0.4025.pt"),
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="测试样本索引 (0-based, 默认 0)",
    )
    parser.add_argument(
        "--device", type=str, default=DEVICE,
    )
    parser.add_argument(
        "--modalities", nargs="+", default=["text", "audio", "visual"],
        choices=["text", "audio", "visual"],
        help="text | visual | audio | text visual | text audio | text audio visual",
    )
    args = parser.parse_args()

    # 加载数据集获取测试样本
    from data.mosei_sdk_dataset import MOSEISDKDataset

    print(f"Loading test dataset ({'+'.join(args.modalities)}) ...")
    test_ds = MOSEISDKDataset(split="test", modalities=args.modalities)
    print(f"  Test samples: {len(test_ds)}")

    sample = test_ds[args.sample]
    tag = "-".join(args.modalities)

    # 不同模态组合可加载不同 checkpoint
    ckpt_map = {
        "text": "best_model_epoch026_mae0.4042.pt",
        "visual": os.path.basename(args.checkpoint),  # fallback
        "audio": os.path.basename(args.checkpoint),
        "text-visual": "best_model_epoch044_mae0.3916.pt",
        "text-audio": "best_model_epoch042_mae0.3978.pt",
        "text-audio-visual": os.path.basename(args.checkpoint),
    }
    ckpt_name = ckpt_map.get(tag, os.path.basename(args.checkpoint))
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)

    # 加载推理器
    print(f"Loading checkpoint: {ckpt_path}")
    predictor = MultimodalPredictor(
        checkpoint_path=ckpt_path,
        device=args.device,
    )
    print(f"  Checkpoint metrics: MAE={predictor.checkpoint_metrics.get('mae', 'N/A'):.4f}")

    # 推理
    has_text = "text" in args.modalities
    has_vis = "visual" in args.modalities
    has_aud = "audio" in args.modalities

    result = predictor.predict(
        text=sample["text"] if has_text else None,
        visual=sample["visual"] if has_vis else None,
        audio=sample["audio"] if has_aud else None,
    )

    # 输出
    gt = float(sample["label"])

    print()
    print("=" * 55)
    print(f"  Sample ID:           {sample['id']}")
    print(f"  Modalities:          {tag}")
    print(f"  Prediction score:    {result['score']:+.4f}")
    print(f"  Predicted sentiment: {result['sentiment'].upper()}")
    print(f"  Confidence:          {result['confidence']:.4f}")
    print(f"  Ground truth:        {gt:+.4f}")
    print(f"  Ground truth sentiment: {_gt_sentiment(gt).upper()}")
    print(f"  MAE:                 {abs(result['score'] - gt):.4f}")
    print("=" * 55)


def _gt_sentiment(score: float) -> str:
    if score > 0.5:
        return "positive"
    elif score < -0.5:
        return "negative"
    else:
        return "neutral"


if __name__ == "__main__":
    main()
