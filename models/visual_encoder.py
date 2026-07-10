"""
视觉编码器 — OpenFace/FACET 特征编码

输入: 预提取的面部特征 (N_frames, 41d FACET 或 68d OpenFace2)
输出: 768d 视觉特征向量

处理流程:
  1. 时序池化 (mean over frames)
  2. MLP 投影到统一维度 768
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from config import VISUAL_ENCODER_INPUT_DIM, VISUAL_HIDDEN


class VisualEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = VISUAL_ENCODER_INPUT_DIM,
        hidden_dim: int = VISUAL_HIDDEN,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, hidden_dim),  # 768
        )

    def forward(self, visual_features: list[np.ndarray]) -> torch.Tensor:
        """
        Args:
            visual_features: OpenFace/FACET 特征列表
                每个元素 (N_frames, feature_dim) float32
        Returns:
            (batch_size, 768) 特征向量
        """
        pooled = []
        for feat in visual_features:
            if feat is None or (isinstance(feat, np.ndarray) and feat.size == 0):
                pooled.append(np.zeros(self.input_dim, dtype=np.float32))
            else:
                # Mean pool over frames → (input_dim,)
                if feat.ndim == 1:
                    pooled.append(feat.astype(np.float32))
                else:
                    pooled.append(feat.mean(axis=0).astype(np.float32))

        x = torch.tensor(np.stack(pooled, axis=0))  # (B, input_dim)

        # 自适应特征维度（OpenFace2 可能是 68d）
        if x.shape[-1] != self.input_dim:
            # 即时创建适配层
            adapter = nn.Linear(x.shape[-1], self.input_dim).to(x.device)
            x = adapter(x)

        return self.mlp(x)  # (B, 768)
