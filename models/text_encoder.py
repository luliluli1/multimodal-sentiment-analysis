"""
文本编码器 — BERT-base

输入: 文本字符串
输出: 768d 句向量 (pooler_output / mean pooling)

fine-tune 策略:
  "all"  — 全部可训练
  "top2" — 只解冻最后2层 Transformer
  "none" — 全部冻结
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from config import (
    TEXT_ENCODER_NAME,
    TEXT_HIDDEN,
    TEXT_MAX_LENGTH,
    FINETUNE_STRATEGY,
    DEVICE,
)


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = TEXT_ENCODER_NAME):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        self.hidden_dim = TEXT_HIDDEN
        self.max_length = TEXT_MAX_LENGTH

        self._apply_finetune_strategy()

    def _apply_finetune_strategy(self):
        """根据 FINETUNE_STRATEGY 冻结/解冻 BERT 层。"""
        if FINETUNE_STRATEGY == "none":
            for p in self.bert.parameters():
                p.requires_grad = False
        elif FINETUNE_STRATEGY == "top2":
            # 冻结 embedding + 前10层 encoder，解冻最后2层 + pooler
            for p in self.bert.embeddings.parameters():
                p.requires_grad = False
            num_layers = self.bert.config.num_hidden_layers  # 12
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < num_layers - 2:
                    for p in layer.parameters():
                        p.requires_grad = False
        # "all" → 什么都不冻结

    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Args:
            texts: 文本列表
        Returns:
            (batch_size, 768) 特征向量
        """
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        encoded = {k: v.to(self.bert.device) for k, v in encoded.items()}

        outputs = self.bert(**encoded)

        # 用 [CLS] token + pooler 作为句子表示
        # 若 pooler_output 不可用则用 mean pooling
        if outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state.mean(dim=1)

        return features  # (B, 768)
