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
        self.input_dim = input_dim
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
        if not audio_features:
            raise ValueError("audio_features 不能为空")

        pooled = []
        for feat in audio_features:
            if feat is None:
                pooled.append(np.zeros(self.input_dim, dtype=np.float32))
                continue
            feat = np.asarray(feat, dtype=np.float32)
            if feat.size == 0:
                pooled.append(np.zeros(self.input_dim, dtype=np.float32))
                continue
            if feat.ndim == 1:
                pooled_feat = np.nan_to_num(
                    feat,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            elif feat.ndim == 2:
                finite_mask = np.isfinite(feat)
                finite_sum = np.where(
                    finite_mask,
                    feat,
                    0.0,
                ).sum(axis=0, dtype=np.float64)
                finite_count = finite_mask.sum(axis=0)
                pooled_feat = np.divide(
                    finite_sum,
                    finite_count,
                    out=np.zeros_like(finite_sum),
                    where=finite_count > 0,
                ).astype(np.float32)
            else:
                raise ValueError(
                    f"COVAREP 特征维度错误，期望 1D 或 2D，实际 shape={feat.shape}"
                )
            pooled_feat = np.nan_to_num(
                pooled_feat,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)

            if pooled_feat.shape != (self.input_dim,):
                raise ValueError(
                    f"COVAREP 特征应为 {self.input_dim} 维，"
                    f"实际 shape={pooled_feat.shape}"
                )

            pooled.append(pooled_feat)
        device = next(
            self.parameters(), torch.empty(0, device=torch.device("cpu"))
        ).device
        x = torch.as_tensor(
            np.stack(pooled, axis=0), dtype=torch.float32, device=device
        )
        return self.mlp(x)  # (B, 768)
