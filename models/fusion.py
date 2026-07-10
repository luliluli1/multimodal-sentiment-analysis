"""
多模态融合模块 — Cross-Attention + MLP

支持:
  - 3 模态 (text + image + audio) → mlp_3
  - 2 模态 (text + image) 或 (text + audio) → mlp_2
  - text-only 不经过此模块，走 classifier_head

架构:
  1. 投影到统一维度 (768 → 256)
  2. Cross-Attention: text query, 辅助模态 key/value
  3. 拼接 → MLP 回归
"""

import torch
import torch.nn as nn
from config import (
    TEXT_HIDDEN, IMAGE_HIDDEN, AUDIO_HIDDEN,
    PROJECTION_DIM, FUSION_HIDDEN, CROSS_ATTENTION_HEADS,
)


class CrossModalAttention(nn.Module):
    def __init__(self, dim: int = PROJECTION_DIM, num_heads: int = CROSS_ATTENTION_HEADS):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q = query.unsqueeze(1)
        out, _ = self.attention(q, key_value, key_value)
        return self.norm(query + out.squeeze(1))


class MultimodalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_proj  = nn.Linear(TEXT_HIDDEN, PROJECTION_DIM)
        self.image_proj = nn.Linear(IMAGE_HIDDEN, PROJECTION_DIM)
        self.audio_proj = nn.Linear(AUDIO_HIDDEN, PROJECTION_DIM)
        self.cross_attn = CrossModalAttention(PROJECTION_DIM, CROSS_ATTENTION_HEADS)

        self.mlp_3 = self._build_mlp(PROJECTION_DIM * 3)  # 768
        self.mlp_2 = self._build_mlp(PROJECTION_DIM * 2)  # 512

    def _build_mlp(self, input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, FUSION_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(FUSION_HIDDEN, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        audio_features: torch.Tensor,
        use_image: bool = True,
        use_audio: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            text_features:  (B, 768)
            image_features: (B, 768) — ignored if use_image=False
            audio_features: (B, 768) — ignored if use_audio=False
            use_image:      visual modality switch
            use_audio:      audio modality switch

        Returns:
            (B, 1) 情感分数
        """
        t = self.text_proj(text_features)

        # ── Full: text + image + audio ──
        if use_image and use_audio:
            i = self.image_proj(image_features)
            a = self.audio_proj(audio_features)
            kv = torch.stack([i, a], dim=1)        # (B, 2, 256)
            t_a = self.cross_attn(t, kv)
            fused = torch.cat([t_a, i, a], dim=-1) # (B, 768)
            return self.mlp_3(fused)

        # ── Text + Visual (audio 完全移除) ──
        elif use_image and not use_audio:
            i = self.image_proj(image_features)
            kv = i.unsqueeze(1)                     # (B, 1, 256)
            t_a = self.cross_attn(t, kv)
            fused = torch.cat([t_a, i], dim=-1)    # (B, 512)
            return self.mlp_2(fused)

        # ── Text + Audio (保持现有逻辑) ──
        elif not use_image and use_audio:
            a = self.audio_proj(audio_features)
            kv = a.unsqueeze(1)
            t_a = self.cross_attn(t, kv)
            fused = torch.cat([t_a, a], dim=-1)    # (B, 512)
            return self.mlp_2(fused)

        # 不会到达: text-only 走 classifier_head, 不经过这里
        raise RuntimeError("Fusion requires at least one auxiliary modality")
