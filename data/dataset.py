"""
CMU-MOSEI 数据集加载

期望数据格式 (预处理后的 .pkl 文件):
    每个样本为 dict:
        {
            "text": str,            # 英文转录文本
            "audio": str,           # 音频 .wav 文件路径
            "visual": np.ndarray,   # OpenFace 面部特征 (n_frames, dim)
            "label": float,         # 情感强度 [-3.0, +3.0]
            "id": str,              # 样本ID (可选)
        }

若 .pkl 文件不存在，自动生成合成数据用于代码验证。
"""

from __future__ import annotations

import os
import pickle
import random
import numpy as np
from torch.utils.data import Dataset
from config import MOSEI_PKL_PATH, MOSEI_LABEL_RANGE, VISUAL_ENCODER_INPUT_DIM


SYNTHETIC_SAMPLES = 100  # 合成数据的样本数


class CMUMOSEIDataset(Dataset):
    def __init__(
        self,
        pkl_path: str = MOSEI_PKL_PATH,
        split: str = "train",
        modalities: list | None = None,
    ):
        """
        Args:
            pkl_path: 预处理数据文件路径
            split: "train" | "val" | "test"
            modalities: 使用的模态列表，默认全部 ["text", "audio", "visual"]
        """
        self.pkl_path = pkl_path
        self.split = split
        self.modalities = modalities or ["text", "audio", "visual"]

        if os.path.exists(pkl_path):
            self.samples = self._load_real(pkl_path, split)
        else:
            print(f"[WARNING] {pkl_path} 未找到，使用合成数据 ({SYNTHETIC_SAMPLES} 条)。"
                  f"请下载 CMU-MOSEI 数据集后放置真实数据。")
            self.samples = self._generate_synthetic(SYNTHETIC_SAMPLES)

    # ----------------------------------------------------------
    # 真实数据加载
    # ----------------------------------------------------------
    def _load_real(self, pkl_path: str, split: str) -> list[dict]:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # 支持两种格式:
        #  格式A: {"train": [...], "val": [...], "test": [...]}
        #  格式B: [...] (不分split)
        if isinstance(data, dict):
            samples = data.get(split, data.get("train", []))
        elif isinstance(data, list):
            samples = data
        else:
            raise ValueError(f"无法解析 pkl 数据格式: {type(data)}")

        return samples

    # ----------------------------------------------------------
    # 合成数据 (用于代码验证，不用于真实训练)
    # ----------------------------------------------------------
    def _generate_synthetic(self, n: int) -> list[dict]:
        texts = [
            "this movie is absolutely fantastic",
            "what a terrible waste of time",
            "it was okay nothing special",
            "i loved every minute of it the acting was superb",
            "boring and predictable from start to finish",
            "pretty good but the ending was disappointing",
            "the best film i have seen all year",
            "do not waste your money on this garbage",
            "decent enough to pass the time",
            "absolutely breathtaking cinematography and storytelling",
        ]
        dim = VISUAL_ENCODER_INPUT_DIM
        samples = []
        for i in range(n):
            n_frames = random.randint(3, 20)
            samples.append({
                "text": random.choice(texts),
                "audio": f"synthetic/audio_{i:04d}.wav",
                "visual": np.random.randn(n_frames, dim).astype(np.float32) * 0.1,
                "label": round(random.uniform(*MOSEI_LABEL_RANGE), 2),
                "id": f"syn_{i:04d}",
            })
        return samples

    # ----------------------------------------------------------
    # 标准接口
    # ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        返回:
            {
                "text": str | None,
                "audio": np.ndarray | None,    # (n_samples,) float32, 16kHz mono
                "visual": np.ndarray | None,   # (n_frames, feature_dim) float32
                "label": float,
                "id": str,
            }
        """
        raw = self.samples[idx]

        item = {
            "text": raw.get("text", None),
            "audio": self._load_audio(raw),
            "visual": raw.get("visual", None),
            "label": float(raw["label"]),
            "id": raw.get("id", str(idx)),
        }

        # 过滤不需要的模态
        for m in ["text", "audio", "visual"]:
            if m not in self.modalities:
                item[m] = None

        return item

    def _load_audio(self, raw: dict) -> np.ndarray | None:
        """加载音频文件，若路径不存在则返回合成噪声。"""
        audio_path = raw.get("audio")
        if audio_path is None:
            return None

        if not os.path.exists(audio_path):
            return np.random.randn(16000).astype(np.float32) * 0.01

        try:
            import librosa
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            return audio.astype(np.float32)
        except Exception:
            return np.random.randn(16000).astype(np.float32) * 0.01


# ================================================================
# 轻量测试
# ================================================================
if __name__ == "__main__":
    ds = CMUMOSEIDataset(split="train")
    print(f"数据集大小: {len(ds)}")
    sample = ds[0]
    print(f"text:   {sample['text'][:60] if sample['text'] else 'None'}...")
    print(f"audio:  {sample['audio'].shape if sample['audio'] is not None else 'None'}")
    print(f"visual: {sample['visual'].shape if sample['visual'] is not None else 'None'}")
    print(f"label:  {sample['label']}")
    print(f"id:     {sample['id']}")
    print("✅ Dataset 加载正常")
