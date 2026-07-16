"""
音频编码器 — Wav2Vec2-base

输入: 音频 numpy array (可变长度, 16kHz mono)
输出: 768d 音频特征向量

Wav2Vec2 特征提取 → 768d
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoFeatureExtractor, AutoModel
from config import (
    AUDIO_ENCODER_NAME,
    AUDIO_HIDDEN,
    AUDIO_SAMPLE_RATE,
    FINETUNE_STRATEGY,
    DEVICE,
)


class AudioEncoder(nn.Module):
    def __init__(self, model_name: str = AUDIO_ENCODER_NAME):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.wav2vec2 = AutoModel.from_pretrained(model_name)

        self.hidden_dim = AUDIO_HIDDEN
        self.sample_rate = AUDIO_SAMPLE_RATE

        self._apply_finetune_strategy()

    def _apply_finetune_strategy(self):
        if FINETUNE_STRATEGY == "none":
            for p in self.wav2vec2.parameters():
                p.requires_grad = False
        elif FINETUNE_STRATEGY == "top2":
            # 冻结 feature extractor (CNN) + 前10层 transformer
            for p in self.wav2vec2.feature_extractor.parameters():
                p.requires_grad = False
            if hasattr(self.wav2vec2, "feature_projection"):
                for p in self.wav2vec2.feature_projection.parameters():
                    p.requires_grad = False
            num_layers = self.wav2vec2.config.num_hidden_layers
            for i, layer in enumerate(self.wav2vec2.encoder.layers):
                if i < num_layers - 2:
                    for p in layer.parameters():
                        p.requires_grad = False

    def forward(self, audio_list: list[np.ndarray]) -> torch.Tensor:
        """
        Args:
            audio_list: 音频数组列表，每个元素 (n_samples,) float32
        Returns:
            (batch_size, 768) 特征向量
        """
        if not audio_list:
            raise ValueError("audio_list 不能为空")

        normalized = []
        for audio in audio_list:
            if audio is None:
                normalized.append(None)
                continue
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            normalized.append(
                np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            )

        valid_lengths = [len(audio) for audio in normalized if audio is not None]
        if not valid_lengths:
            raise ValueError("audio_list 中没有有效波形")

        # 填充到统一长度；缺失样本使用同 batch 等长静音。
        max_len = max(valid_lengths)
        padded = np.zeros((len(audio_list), max_len), dtype=np.float32)
        for i, audio in enumerate(normalized):
            if audio is not None:
                padded[i, : len(audio)] = audio

        inputs = self.feature_extractor(
            padded,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.wav2vec2.device) for k, v in inputs.items()}

        outputs = self.wav2vec2(**inputs)

        # mean pooling over time → 句子级表示
        features = outputs.last_hidden_state.mean(dim=1)  # (B, 768)

        return features
