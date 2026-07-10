#!/usr/bin/env python3
"""
从 mmsdk + CMU raw datasets 构建 CMU-MOSEI 训练数据 (.pkl)。

数据来源:
  文本 + 标签 → mmsdk (CMU_MOSEI_TimestampedWords + Labels)
  视觉特征    → mmsdk (CMU_MOSEI_VisualFacet42 / CMU_MOSEI_OpenFace2)
  音频 .wav   → CMU raw datasets (immortal.multicomp.cs.cmu.edu/raw_datasets/)

输出格式:
  {
    "train": [
      {"text": str, "audio": str, "visual": np.array, "label": float, "id": str},
      ...
    ],
    "val":   [...],
    "test":  [...],
  }

用法:
  # mmsdk 自动下载 .csd 文件
  python scripts/build_mosei_pkl.py --output ./data/mosei/mosei_data.pkl

  # 带 raw audio 匹配
  python scripts/build_mosei_pkl.py --audio-dir /path/to/MOSEI/Raw/Audio/
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MOSEI_PKL_PATH, DATA_DIR, MOSEI_SPLIT_TRAIN, MOSEI_SPLIT_VAL, VISUAL_ENCODER_INPUT_DIM,
)


def main():
    parser = argparse.ArgumentParser(description="构建 CMU-MOSEI 训练数据")
    parser.add_argument("--output", type=str, default=MOSEI_PKL_PATH,
                        help="输出 .pkl 路径")
    parser.add_argument("--audio-dir", type=str, default=None,
                        help="原始 .wav 文件所在文件夹 (可选)")
    parser.add_argument("--visual-csd", type=str, default="CMU_MOSEI_VisualFacet42",
                        help="视觉特征 .csd: CMU_MOSEI_VisualFacet42 或 CMU_MOSEI_OpenFace2")
    args = parser.parse_args()

    print("=" * 50)
    print("  构建 CMU-MOSEI 训练数据")
    print("=" * 50)

    # ----------------------------------------------------------
    # Step 1: 通过 mmsdk 下载/加载 CMU-MOSEI
    # ----------------------------------------------------------
    print("\n[1/4] 通过 mmsdk 加载 CMU-MOSEI...")
    from mmsdk import mmdatasdk

    mosei_dir = os.path.join(DATA_DIR, "mosei_raw")

    try:
        # highlevel: FACET, COVAREP, GloVe 等预提取特征
        dataset = mmdatasdk.mmdataset(
            mmdatasdk.cmu_mosei.highlevel, mosei_dir
        )
        # raw: TimestampedWords (文本) ← 关键：文字在 raw 不在 highlevel
        dataset.add_computational_sequences(
            mmdatasdk.cmu_mosei.raw, mosei_dir
        )
        # labels: All Labels
        dataset.add_computational_sequences(
            mmdatasdk.cmu_mosei.labels, mosei_dir
        )
        dataset.align("All Labels")
        print("  ✅ mmsdk 数据加载 & 对齐完成")
    except Exception as e:
        print(f"  ❌ mmsdk 加载失败: {e}")
        print("  请确保已获得 CMU-MOSEI 访问权限。")
        print("  申请地址: https://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/")
        sys.exit(1)

    # ----------------------------------------------------------
    # Step 2: 提取文本和标签
    # ----------------------------------------------------------
    print("\n[2/4] 提取文本和标签...")

    words_seq = dataset.computational_sequences["CMU_MOSEI_TimestampedWords"]
    labels_seq = dataset.computational_sequences["All Labels"]

    all_ids = sorted(set(words_seq.data.keys()) & set(labels_seq.data.keys()))

    samples = []
    for seg_id in all_ids:
        # --- 文本: 拼接单词 ---
        word_features = words_seq.data[seg_id]["features"]
        if word_features.dtype.kind in ("U", "S", "O"):
            words = [str(w) for w in word_features.flat]
        elif hasattr(word_features, "tolist"):
            tokens = word_features.tolist()
            # tokens 可能是 2D [(word, start, end), ...]
            if tokens and isinstance(tokens[0], (list, tuple)):
                words = [str(t[0]) for t in tokens]
            else:
                words = [str(t) for t in tokens]
        else:
            words = []

        text = " ".join(w for w in words if len(str(w)) > 0).strip()
        if not text:
            text = f"[segment_{seg_id}]"

        # --- 标签 ---
        label_arr = labels_seq.data[seg_id]["features"]
        if label_arr.ndim > 1:
            # MOSEI 标签可能是多列 (sentiment + emotions)
            label = float(label_arr[:, 0].mean())  # 第 0 列 = sentiment
        else:
            label = float(label_arr.mean())

        samples.append({
            "text": text,
            "audio": None,
            "visual": None,
            "label": round(label, 4),
            "id": seg_id,
        })

    print(f"  ✅ 提取到 {len(samples)} 条样本 (text + label)")

    # ----------------------------------------------------------
    # Step 2b: 提取 Visual 特征
    # ----------------------------------------------------------
    visual_csd_name = args.visual_csd
    visual_matched = 0

    if visual_csd_name in dataset.computational_sequences:
        print(f"\n[2b] 提取视觉特征: {visual_csd_name}...")
        visual_seq = dataset.computational_sequences[visual_csd_name]

        for s in samples:
            seg_id = s["id"]
            if seg_id in visual_seq.data:
                feat = visual_seq.data[seg_id]["features"]
                if isinstance(feat, np.ndarray):
                    s["visual"] = feat.astype(np.float16)
                    visual_matched += 1

        print(f"  ✅ 匹配到 {visual_matched}/{len(samples)} 条")
    else:
        avail = [k for k in dataset.computational_sequences.keys()
                 if "Visual" in k or "OpenFace" in k]
        print(f"\n[2b] 视觉特征 '{visual_csd_name}' 未找到")
        if avail:
            print(f"  可用: {avail}")
            print(f"  💡 用 --visual-csd 指定")

    # ----------------------------------------------------------
    # Step 3: 匹配音频文件 (可选)
    # ----------------------------------------------------------
    audio_matched = 0
    if args.audio_dir and os.path.isdir(args.audio_dir):
        print(f"\n[3/4] 匹配音频文件...")
        wav_files = {f for f in os.listdir(args.audio_dir) if f.endswith(".wav")}

        for s in samples:
            seg_id = s["id"]
            for wav_name in [f"{seg_id}.wav", f"{seg_id}_1.wav"]:
                if wav_name in wav_files:
                    s["audio"] = os.path.join(args.audio_dir, wav_name)
                    audio_matched += 1
                    break

        print(f"  ✅ 匹配到 {audio_matched}/{len(samples)} 条音频")
    else:
        print(f"\n[3/4] 跳过音频匹配 (未提供 --audio-dir)")
        print(f"  💡 raw audio: http://immortal.multicomp.cs.cmu.edu/raw_datasets/")

    # ----------------------------------------------------------
    # Step 4: 划分 split 并保存
    # ----------------------------------------------------------
    print("\n[4/4] 划分数据集并保存...")

    n = len(samples)
    train_end = int(n * MOSEI_SPLIT_TRAIN)          # 69%
    val_end = train_end + int(n * MOSEI_SPLIT_VAL)   # +8%

    random.seed(42)
    shuffled = samples[:]
    random.shuffle(shuffled)

    data = {
        "train": shuffled[:train_end],
        "val":   shuffled[train_end:val_end],
        "test":  shuffled[val_end:],
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data, f)

    print(f"  ✅ 已保存到 {args.output}")
    print(f"     train: {len(data['train'])} 条")
    print(f"     val:   {len(data['val'])} 条")
    print(f"     test:  {len(data['test'])} 条")

    labels = [s["label"] for s in samples]
    print(f"\n  标签范围: [{min(labels):.2f}, {max(labels):.2f}]")
    print(f"  音频: {audio_matched}/{len(samples)}")
    print(f"  视觉: {visual_matched}/{len(samples)} ({visual_csd_name})")
    print(f"\n{'=' * 50}")
    print(f"  完成! 运行训练:")
    print(f"  python scripts/train.py --data {args.output} --epochs 50")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
