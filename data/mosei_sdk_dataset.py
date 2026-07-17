"""
CMU-MOSEI SDK 数据集。

加载顺序:
  1. 尝试本地 .csd 文件 (手动下载后放到 data/mosei_raw/)
  2. 尝试 mmsdk 远程下载
  3. fallback 合成数据

需要的 .csd 文件 (4 个):
  CMU_MOSEI_TimestampedWords.csd   — 文本
  CMU_MOSEI_COVAREP.csd            — 音频特征 74d
  CMU_MOSEI_VisualFacet42.csd      — 视觉特征 42d
  CMU_MOSEI_Labels.csd             — 标签 [-3,+3]

下载后放到:
  ./data/mosei_raw/  (或通过 --cache-dir 指定)
"""

from __future__ import annotations

import os
import random
import numpy as np
from torch.utils.data import Dataset

from config import MOSEI_LABEL_RANGE, VISUAL_ENCODER_INPUT_DIM, DATA_DIR


SYNTHETIC_SIZE = 100
COVAREP_DIM = 74

# 必需文件清单
REQUIRED_CSD = {
    "words":  "CMU_MOSEI_TimestampedWords.csd",
    "labels": "CMU_MOSEI_Labels.csd",
    "facet":  "CMU_MOSEI_VisualFacet42.csd",
    "covarep": "CMU_MOSEI_COVAREP.csd",
}


class MOSEISDKDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        modalities: list | None = None,
        cache_dir: str | None = None,
    ):
        self.split = split
        self.modalities = modalities or ["text", "audio", "visual"]
        self.cache_dir = cache_dir or DATA_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # 加载顺序: 本地 → 远程 → 合成，避免已有数据时重复下载。
        samples = None
        for loader in [self._load_local, self._load_remote]:
            try:
                samples = loader()
                if samples:
                    break
            except Exception as e:
                print(f"  [{loader.__name__}] {e}")

        if samples:
            self.samples = samples
        else:
            print(f"[WARNING] 使用合成数据 ({SYNTHETIC_SIZE} 条)")
            self.samples = self._generate_synthetic(SYNTHETIC_SIZE)

    # ----------------------------------------------------------
    # 远程加载 (mmsdk 直接下载)
    # ----------------------------------------------------------
    def _load_remote(self) -> list[dict] | None:
        from mmsdk import mmdatasdk

        # 只下载本项目使用的 4 个序列，key 与 CSD 内部 root name 一致。
        recipe = {
            "words":      mmdatasdk.cmu_mosei.raw["words"],
            "COVAREP":    mmdatasdk.cmu_mosei.highlevel["COVAREP"],
            "FACET 4.2":  mmdatasdk.cmu_mosei.highlevel["FACET 4.2"],
            "All Labels": mmdatasdk.cmu_mosei.labels["All Labels"],
        }
        dataset = mmdatasdk.mmdataset(recipe, self.cache_dir)
        dataset.align("All Labels")

        words_seq = dataset.computational_sequences["words"]
        labels_seq = dataset.computational_sequences["All Labels"]
        facet_seq = dataset.computational_sequences.get("FACET 4.2")
        covarep_seq = dataset.computational_sequences.get("COVAREP")

        return self._build_samples(words_seq, labels_seq, facet_seq, covarep_seq)

    # ----------------------------------------------------------
    # 本地加载 (.csd 文件放 ./data/mosei_raw/)
    # ----------------------------------------------------------
    def _load_local(self) -> list[dict] | None:
        from mmsdk import mmdatasdk

        # 检查文件是否存在
        missing = []
        for key, fname in REQUIRED_CSD.items():
            path = os.path.join(self.cache_dir, fname)
            if not os.path.exists(path):
                missing.append(fname)

        if missing:
            raise FileNotFoundError(
                f"缺少 {len(missing)} 个 .csd 文件: {missing}\n"
                f"  请下载后放到 {self.cache_dir}/"
            )

        print("  加载并按标签对齐本地 .csd 文件...")
        dataset = mmdatasdk.mmdataset(self.cache_dir)
        dataset.align("All Labels")

        words_seq = dataset.computational_sequences["words"]
        labels_seq = dataset.computational_sequences["All Labels"]
        facet_seq = dataset.computational_sequences["FACET 4.2"]
        covarep_seq = dataset.computational_sequences["COVAREP"]

        # align 后再取所有序列中共有的 segment ID。
        common_ids = (
            set(words_seq.data.keys())
            & set(labels_seq.data.keys())
            & set(facet_seq.data.keys())
            & set(covarep_seq.data.keys())
        )

        if not common_ids:
            raise RuntimeError("4 个 .csd 文件之间没有共有的 segment ID，对齐失败")

        print(f"  对齐后 segment 数: {len(common_ids)}")
        return self._build_samples(words_seq, labels_seq, facet_seq, covarep_seq, common_ids)

    # ----------------------------------------------------------
    # 通用: 从计算序列构建样本列表
    # ----------------------------------------------------------
    def _build_samples(
        self,
        words_seq,
        labels_seq,
        facet_seq=None,
        covarep_seq=None,
        id_subset: set | None = None,
    ) -> list[dict]:
        split_videos = self._get_split_videos()

        if id_subset is not None:
            candidate_ids = id_subset
        else:
            candidate_ids = set(words_seq.data.keys()) & set(labels_seq.data.keys())

        samples = []
        for seg_id in sorted(candidate_ids):
            # 按 video ID 分 train/val/test
            video_id = seg_id.split("[")[0] if "[" in seg_id else seg_id[:11]
            if split_videos and video_id not in split_videos:
                continue

            # 文本 — h5py.Dataset, shape (N,1) dtype S32
            word_feat = words_seq.data[seg_id]["features"]
            try:
                words = []
                for w in word_feat:
                    # w 是 numpy array([b'hello']) → w.flat[0] → bytes
                    raw = w.flat[0] if hasattr(w, "flat") else w
                    if isinstance(raw, bytes):
                        s = raw.decode("utf-8", errors="replace").strip()
                    else:
                        s = str(raw).strip()
                    if s:
                        words.append(s)
                text = " ".join(words)
            except Exception:
                text = ""
            if not text:
                text = f"[{seg_id}]"

            # 标签 (第 0 列 = sentiment [-3, +3])
            label_arr = labels_seq.data[seg_id]["features"]
            label = float(label_arr[:, 0].mean()) if label_arr.ndim > 1 else float(label_arr.mean())

            # 视觉
            visual = None
            if facet_seq and seg_id in facet_seq.data:
                visual = facet_seq.data[seg_id]["features"].astype(np.float16)

            # 音频
            audio = None
            if covarep_seq and seg_id in covarep_seq.data:
                audio = covarep_seq.data[seg_id]["features"].astype(np.float32)

            samples.append({
                "text": text,
                "audio": audio,
                "visual": visual,
                "label": round(label, 4),
                "id": seg_id,
            })

        print(f"[SDK] {self.split} = {len(samples)} 条")
        return samples

    # ----------------------------------------------------------
    # 标准 split (从 mmsdk 读取, 失败则硬编码)
    # ----------------------------------------------------------
    def _get_split_videos(self) -> set:
        try:
            from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSEI import (
                cmu_mosei_std_folds as folds,
            )
            if self.split == "val":
                return set(folds.standard_valid_fold)
            elif self.split == "test":
                return set(folds.standard_test_fold)
            return set(folds.standard_train_fold)
        except Exception:
            # mmsdk 不可用时的 fallback: 按索引粗分
            # 这不是标准划分，仅用于代码测试
            print("  ⚠️ 无法读取标准 folds，使用索引粗分")
            # 返回空集让 _build_samples 用全部数据
            return set()  # 会被下面的逻辑处理

    # ----------------------------------------------------------
    # 合成数据
    # ----------------------------------------------------------
    def _generate_synthetic(self, n: int) -> list[dict]:
        texts = [
            "this movie is absolutely fantastic",
            "what a terrible waste of time",
            "it was okay nothing special",
            "i loved every minute of it",
            "boring and predictable",
        ]
        samples = []
        for i in range(n):
            n_frames = random.randint(3, 20)
            samples.append({
                "text": random.choice(texts),
                "audio": np.random.randn(n_frames, COVAREP_DIM).astype(np.float32) * 0.1,
                "visual": np.random.randn(n_frames, VISUAL_ENCODER_INPUT_DIM).astype(np.float32) * 0.1,
                "label": round(random.uniform(*MOSEI_LABEL_RANGE), 2),
                "id": f"syn_{i:04d}",
            })
        return samples

    # ----------------------------------------------------------
    # PyTorch Dataset 接口
    # ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        raw = self.samples[idx]

        item = {
            "text": raw["text"],
            "audio": raw.get("audio"),
            "visual": raw.get("visual"),
            "label": float(raw["label"]),
            "id": raw.get("id", str(idx)),
        }

        for m in ["text", "audio", "visual"]:
            if m not in self.modalities:
                item[m] = None

        return item


# ================================================================
# 测试
# ================================================================
if __name__ == "__main__":
    ds = MOSEISDKDataset(split="train")
    print(f"Size: {len(ds)}")
    s = ds[0]
    print(f"text:   {s['text'][:50]}")
    print(f"audio:  {s['audio'].shape if s['audio'] is not None else 'None'}")
    print(f"visual: {s['visual'].shape if s['visual'] is not None else 'None'}")
    print(f"label:  {s['label']}")
