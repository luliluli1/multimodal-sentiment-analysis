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
        # 首次遇到非配置维度时创建。Trainer 会在 forward 后把新增参数
        # 同步到 optimizer，避免参数虽注册到 Module 却没有被优化。
        self.input_adapter: nn.Linear | None = None

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
        if not visual_features:
            raise ValueError("visual_features 不能为空")

        pooled: list[np.ndarray | None] = []
        feature_dim = None
        for feat in visual_features:
            if feat is None:
                pooled.append(None)
                continue

            feat = np.asarray(feat, dtype=np.float32)
            if feat.size == 0:
                pooled.append(None)
                continue

            if feat.ndim == 1:
                pooled_feat = feat
            elif feat.ndim == 2:
                finite_mask = np.isfinite(feat)
                finite_sum = np.where(finite_mask, feat, 0.0).sum(
                    axis=0, dtype=np.float64
                )
                finite_count = finite_mask.sum(axis=0)
                pooled_feat = np.divide(
                    finite_sum,
                    finite_count,
                    out=np.zeros_like(finite_sum),
                    where=finite_count > 0,
                ).astype(np.float32)
            else:
                raise ValueError(
                    f"视觉特征维度错误，期望 1D 或 2D，实际 shape={feat.shape}"
                )

            pooled_feat = np.nan_to_num(
                pooled_feat, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            if pooled_feat.ndim != 1:
                raise ValueError(
                    f"视觉特征池化后应为 1D，实际 shape={pooled_feat.shape}"
                )
            if feature_dim is None:
                feature_dim = pooled_feat.shape[0]
            elif pooled_feat.shape[0] != feature_dim:
                raise ValueError(
                    "同一 batch 的视觉特征维度必须一致，"
                    f"期望 {feature_dim}，实际 {pooled_feat.shape[0]}"
                )
            pooled.append(pooled_feat)

        feature_dim = feature_dim or self.input_dim
        pooled = [
            np.zeros(feature_dim, dtype=np.float32) if feat is None else feat
            for feat in pooled
        ]

        device = self.mlp[0].weight.device
        x = torch.as_tensor(
            np.stack(pooled, axis=0), dtype=torch.float32, device=device
        )

        # 自适应特征维度（OpenFace2 可能是 68d, FACET 可能是 35d/42d）
        if x.shape[-1] != self.input_dim:
            if self.input_adapter is None:
                self.input_adapter = nn.Linear(x.shape[-1], self.input_dim).to(x.device)
            elif self.input_adapter.in_features != x.shape[-1]:
                raise ValueError(
                    "VisualEncoder 已适配 "
                    f"{self.input_adapter.in_features} 维输入，无法再接收 "
                    f"{x.shape[-1]} 维输入"
                )
            x = self.input_adapter(x)

        return self.mlp(x)  # (B, 768)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """按 checkpoint 中的权重形状恢复懒创建的 input_adapter。"""
        weight_key = f"{prefix}input_adapter.weight"
        bias_key = f"{prefix}input_adapter.bias"
        if self.input_adapter is None and weight_key in state_dict:
            weight = state_dict[weight_key]
            self.input_adapter = nn.Linear(
                in_features=weight.shape[1],
                out_features=weight.shape[0],
                bias=bias_key in state_dict,
            ).to(
                device=self.mlp[0].weight.device,
                dtype=self.mlp[0].weight.dtype,
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
