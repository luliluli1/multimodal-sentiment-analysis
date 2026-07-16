"""
多模态情感分析模型 (顶层)

组装: TextEncoder + ImageEncoder + VisualEncoder + AudioEncoder
      + CovarepAdapter + MultimodalFusion + ClassifierHead

统一路由框架 (encode → collect → route by count):
  N=1  (text | visual | audio):
       Encoder → classifier_head → score
  N=2  (text+visual | text+audio):
       Encoders → Fusion(Cross-Attention) → score
  N=3  (text+visual+audio):
       Encoders → Fusion(Cross-Attention) → score
  Demo image path:      BERT + ImageEncoder(ViT) → fusion
  Demo raw waveform:    BERT + AudioEncoder(Wav2Vec2) → fusion
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

        # 单模态分类器 — 所有 Encoder 输出均为 768d，共享此 head
        self.classifier_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    # ----------------------------------------------------------
    # 统一 forward
    # ----------------------------------------------------------
    def forward(
        self,
        texts: list[str] | None,
        visual_inputs: list | None,
        audio_inputs: list | None,
    ) -> torch.Tensor:
        # ── 1. 检测可用模态 ──
        vis_type = self._detect_visual_type(visual_inputs)
        aud_type = self._detect_audio_type(audio_inputs)
        has_text = self._has_text(texts)
        has_vis = vis_type != "none"
        has_aud = aud_type != "none"

        # ── 2. 编码可用模态 ──
        # h5py AsTypeView → np.ndarray（encoders 内部依赖 numpy 方法）
        if has_vis:
            visual_inputs = [np.asarray(v, dtype=np.float32)
                           if hasattr(v, "ndim") and not isinstance(v, np.ndarray)
                           else v for v in visual_inputs]
        if has_aud:
            audio_inputs = [np.asarray(a, dtype=np.float32)
                          if hasattr(a, "ndim") and not isinstance(a, np.ndarray)
                          else a for a in audio_inputs]

        text_feat = self.text_encoder(texts) if has_text else None

        if has_vis:
            vis_feat = (self.visual_encoder(visual_inputs)
                        if vis_type == "openface"
                        else self.image_encoder(visual_inputs))
        else:
            vis_feat = None

        if has_aud:
            aud_feat = (self.covarep_adapter(audio_inputs)
                        if aud_type == "covarep"
                        else self.audio_encoder(audio_inputs))
        else:
            aud_feat = None

        # ── 3. 收集可用特征 ──
        available: list[tuple[str, torch.Tensor]] = []
        if has_text: available.append(("text",   text_feat))
        if has_vis:  available.append(("visual", vis_feat))
        if has_aud:  available.append(("audio",  aud_feat))

        # ── 4. 按模态数量路由 ──
        if not available:
            raise ValueError("至少需要一个有效模态 (text / visual / audio)")

        if len(available) == 1:
            # 单模态 — 共享 classifier_head (768 → 1)
            return self.classifier_head(available[0][1])

        # 多模态 — Fusion (Cross-Attention)
        # 当前 Cross-Attention 以 text 为 query，不支持 visual+audio 双模态。
        if not has_text:
            raise ValueError("visual+audio 双模态暂不支持，融合时必须提供文本")
        if not has_vis:
            vis_feat = torch.zeros_like(text_feat)
        if not has_aud:
            aud_feat = torch.zeros_like(text_feat)

        return self.fusion(text_feat, vis_feat, aud_feat,
                           use_image=has_vis, use_audio=has_aud)

    # ----------------------------------------------------------
    # 模态检测
    # ----------------------------------------------------------
    @staticmethod
    def _has_text(texts: list[str] | None) -> bool:
        """检查 batch 中是否存在有效文本。"""
        if texts is None:
            return False
        return any(t for t in texts if t and str(t).strip())

    @staticmethod
    def _detect_visual_type(inputs: list) -> str:
        if not inputs:
            return "none"
        for v in inputs:
            # duck-typing: h5py AsTypeView / np.ndarray 均有 ndim+size
            if hasattr(v, "ndim") and MultimodalSentimentModel._input_size(v) > 0:
                return "openface"
            if isinstance(v, str) and v.strip():
                return "image"
        return "none"

    @staticmethod
    def _detect_audio_type(inputs: list) -> str:
        if not inputs:
            return "none"
        for a in inputs:
            if hasattr(a, "ndim") and MultimodalSentimentModel._input_size(a) > 0:
                return "covarep" if a.ndim == 2 else "waveform"
        return "none"

    @staticmethod
    def _input_size(value) -> int:
        """兼容 numpy/h5py/torch 的 size 表示。"""
        size = getattr(value, "size", 0)
        if callable(size):
            if hasattr(value, "numel"):
                return int(value.numel())
            shape = getattr(value, "shape", ())
            return int(np.prod(shape))
        return int(size)

    # ----------------------------------------------------------
    # 推理接口 (Demo / API 用)
    # ----------------------------------------------------------
    @torch.no_grad()
    def predict(self, text=None, image_path=None, audio=None) -> dict:
        self.eval()
        texts = [text or ""]
        visuals = [image_path or ""]
        audios = [audio]
        score = self.forward(texts, visuals, audios).item()

        label = "positive" if score > 0.5 else ("negative" if score < -0.5 else "neutral")
        conf = float(torch.sigmoid(torch.tensor(abs(score) / 3.0)).item())
        scores = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        scores[label] = conf
        return {"label": label, "confidence": conf, "scores": scores}
