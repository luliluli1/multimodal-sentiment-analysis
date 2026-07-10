"""
COVAREP 音频特征适配器 — 74d → 768d

将 mmsdk 预提取的 COVAREP 声学特征映射到统一维度，
使其可以直接输入 Fusion 模块，替代 Wav2Vec2 路径。

参数量: ~20K (远小于 Wav2Vec2 的 95M)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class CovarepAdapter(nn.Module):
    def __init__(self, input_dim: int = 74, hidden_dim: int = 768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, hidden_dim),  # 768
        )

    def forward(self, audio_features: list[np.ndarray]) -> torch.Tensor:
        """
        Args:
            audio_features: COVAREP 特征列表，每个 (N_frames, 74)
        Returns:
            (B, 768)
        """
        pooled = []
        for feat in audio_features:
            if feat is None or feat.size == 0:
                pooled.append(np.zeros(74, dtype=np.float32))
            elif feat.ndim == 1:
                pooled.append(feat.astype(np.float32))
            else:
                pooled.append(feat.mean(axis=0).astype(np.float32))

        x = torch.tensor(np.stack(pooled, axis=0))  # (B, 74)
        return self.mlp(x)  # (B, 768)
