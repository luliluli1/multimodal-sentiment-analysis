"""
多模态情感分析模型 (顶层)

组装: TextEncoder + ImageEncoder + VisualEncoder + AudioEncoder
      + CovarepAdapter + MultimodalFusion + ClassifierHead

路径:
  Exp1  Text-only:          BERT → classifier_head → score
  Exp2  Text + COVAREP:     BERT + CovarepAdapter → fusion(use_audio=True)
  Exp3  Text + FACET:       BERT + VisualEncoder  → fusion(use_image=True)
  Exp4  Full (text+visual+audio):  全部 → fusion(use_image=True, use_audio=True)
  Demo  image path:         BERT + ImageEncoder(ViT)  → fusion
  Demo  raw waveform:       BERT + AudioEncoder(Wav2Vec2) → fusion
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from models.visual_encoder import VisualEncoder
from models.audio_encoder import AudioEncoder
from models.covarep_adapter import CovarepAdapter
from models.fusion import MultimodalFusion


class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        self.visual_encoder = VisualEncoder()
        self.audio_encoder = AudioEncoder()
        self.covarep_adapter = CovarepAdapter()
        self.fusion = MultimodalFusion()

        # Text-only 分类器 (不经过 fusion)
        self.classifier_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        texts: list[str],
        visual_inputs: list,
        audio_inputs: list,
    ) -> torch.Tensor:
        text_feat = self.text_encoder(texts)  # (B, 768)

        vis_type = self._detect_visual_type(visual_inputs)
        aud_type = self._detect_audio_type(audio_inputs)
        has_vis = vis_type != "none"
        has_aud = aud_type != "none"

        # ── Exp1: Text-only → classifier_head (不经过 fusion) ──
        if not has_vis and not has_aud:
            return self.classifier_head(text_feat)

        # ── 提取辅助模态特征 ──
        if has_vis:
            if vis_type == "openface":
                vis_feat = self.visual_encoder(visual_inputs)
            else:
                vis_feat = self.image_encoder(visual_inputs)
        else:
            vis_feat = torch.zeros_like(text_feat)

        if has_aud:
            if aud_type == "covarep":
                aud_feat = self.covarep_adapter(audio_inputs)
            else:
                aud_feat = self.audio_encoder(audio_inputs)
        else:
            aud_feat = torch.zeros_like(text_feat)

        return self.fusion(text_feat, vis_feat, aud_feat,
                          use_image=has_vis, use_audio=has_aud)

    # ----------------------------------------------------------
    @staticmethod
    def _detect_visual_type(inputs: list) -> str:
        for v in inputs:
            if isinstance(v, np.ndarray):
                return "openface"
            if isinstance(v, str) and len(v) > 0:
                return "image"
        return "none"

    @staticmethod
    def _detect_audio_type(inputs: list) -> str:
        for a in inputs:
            if isinstance(a, np.ndarray) and a.size > 0:
                return "covarep" if a.ndim == 2 else "waveform"
        return "none"

    # ----------------------------------------------------------
    @torch.no_grad()
    def predict(self, text=None, image_path=None, audio=None) -> dict:
        self.eval()
        texts = [text or ""]
        visuals = [image_path or ""]
        audios = [audio if audio is not None else np.zeros(16000, dtype=np.float32)]
        score = self.forward(texts, visuals, audios).item()

        label = "positive" if score > 0.5 else ("negative" if score < -0.5 else "neutral")
        conf = float(torch.sigmoid(torch.tensor(abs(score) / 3.0)).item())
        scores = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        scores[label] = conf
        return {"label": label, "confidence": conf, "scores": scores}
