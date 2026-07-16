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
        for feat in audio_features: #特征要么是二维，要么是一维
            if feat is None:
                pooled.append(np.zeros(74, dtype=np.float32))  #处理特征值为空值的情况
                continue
            feat = np.asarray(feat, dtype=np.float32)   # 统一转换为 float32 numpy 数组
            if feat.size == 0:
                pooled.append(np.zeros(74, dtype=np.float32))   #处理特征值为0的情况
                continue
            if feat.ndim == 1:#如果是一维的，先把异常值改为0，然后再进行格式转换
                pooled_feat = np.nan_to_num(
                    feat,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            elif feat.ndim ==2 :
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

            # COVAREP 应当是 74 维
            if pooled_feat.shape != (74,):
                raise ValueError(
                    f"COVAREP 特征应为 74 维，实际 shape={pooled_feat.shape}"
                )

            pooled.append(pooled_feat)
        x = torch.tensor(np.stack(pooled, axis=0))  # (B, 74)
        return self.mlp(x)  # (B, 768)
