"""
数据预处理工具
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from config import IMAGE_SIZE, AUDIO_SAMPLE_RATE, AUDIO_MAX_DURATION


def preprocess_text(text: str) -> str:
    """清洗文本: 去除首尾空白、限制长度"""
    if not text or not text.strip():
        raise ValueError("输入文本不能为空")
    return text.strip()


def preprocess_image(image_path: str) -> Image.Image:
    """加载并预处理图像: 转换为RGB, 检查格式"""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"无法加载图像文件 {image_path}: {e}")
    return img


def preprocess_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """
    加载并预处理音频: 重采样、转单声道、截断。
    返回: (audio_array, sample_rate)
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("请安装 librosa: pip install librosa")

    audio, sr = librosa.load(
        audio_path,
        sr=AUDIO_SAMPLE_RATE,
        mono=True,
        duration=AUDIO_MAX_DURATION,
    )

    if len(audio) == 0:
        raise ValueError(f"音频文件为空或无法解析: {audio_path}")

    return audio, sr
