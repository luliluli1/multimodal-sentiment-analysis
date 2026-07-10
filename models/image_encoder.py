"""
图像编码器 — ViT-base

输入: 图片路径 (str) 列表
输出: 768d 图像特征向量

fine-tune 策略: 同文本编码器
"""

from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from config import (
    IMAGE_ENCODER_NAME,
    IMAGE_HIDDEN,
    IMAGE_SIZE,
    FINETUNE_STRATEGY,
    DEVICE,
)


class ImageEncoder(nn.Module):
    def __init__(self, model_name: str = IMAGE_ENCODER_NAME):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.vit = AutoModel.from_pretrained(model_name)

        self.hidden_dim = IMAGE_HIDDEN
        self.image_size = IMAGE_SIZE

        self._apply_finetune_strategy()

    def _apply_finetune_strategy(self):
        if FINETUNE_STRATEGY == "none":
            for p in self.vit.parameters():
                p.requires_grad = False
        elif FINETUNE_STRATEGY == "top2":
            # 冻结 embedding + 前10层
            for p in self.vit.embeddings.parameters():
                p.requires_grad = False
            num_layers = self.vit.config.num_hidden_layers
            for i, layer in enumerate(self.vit.encoder.layer):
                if i < num_layers - 2:
                    for p in layer.parameters():
                        p.requires_grad = False

    def forward(self, image_paths: list[str]) -> torch.Tensor:
        """
        Args:
            image_paths: 图片路径列表
        Returns:
            (batch_size, 768) 特征向量
        """
        images = []
        for path in image_paths:
            if path is None or not isinstance(path, str):
                # 缺失图片 → 全零向量
                images.append(Image.new("RGB", (self.image_size, self.image_size)))
            else:
                try:
                    images.append(Image.open(path).convert("RGB"))
                except Exception:
                    images.append(Image.new("RGB", (self.image_size, self.image_size)))

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.vit.device) for k, v in inputs.items()}

        outputs = self.vit(**inputs)

        # [CLS] token 作为图像表示
        features = outputs.last_hidden_state[:, 0, :]  # (B, 768)

        return features
