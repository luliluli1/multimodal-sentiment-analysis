#!/usr/bin/env python3
"""
训练入口 — CMU-MOSEI

用法:
  # 单模态 Baseline
  python scripts/train.py --sdk --epochs 50 --modalities text
  python scripts/train.py --sdk --epochs 50 --modalities visual
  python scripts/train.py --sdk --epochs 50 --modalities audio

  # 双模态
  python scripts/train.py --sdk --epochs 50 --modalities text audio
  python scripts/train.py --sdk --epochs 50 --modalities text visual

  # Full
  python scripts/train.py --sdk --epochs 50 --modalities text audio visual

  # 继续训练
  python scripts/train.py --sdk --epochs 50 --checkpoint checkpoints/best.pt
"""

import argparse, json, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, DEVICE,
    MOSEI_PKL_PATH, MOSEI_SPLIT_TRAIN, MOSEI_SPLIT_VAL, CHECKPOINT_DIR,
)
from data.dataset import CMUMOSEIDataset
from data.mosei_sdk_dataset import MOSEISDKDataset
from models.multimodal_model import MultimodalSentimentModel
from trainers.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="CMU-MOSEI 多模态情感分析 — 训练")
    parser.add_argument("--sdk", action="store_true")
    parser.add_argument("--data", type=str, default=MOSEI_PKL_PATH)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--modalities", nargs="+", default=["text", "audio", "visual"],
                        choices=["text", "audio", "visual"],
                        help="text | visual | audio | text visual | text audio | text audio visual")
    parser.add_argument("--tag", type=str, default=None,
                        help="实验标签，用于 checkpoint 和 results.json 命名")
    args = parser.parse_args()

    tag = args.tag or "-".join(args.modalities)
    print(f"Experiment: {tag}")
    print(f"Modalities: {args.modalities}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch}")

    # 1. 数据
    if args.sdk:
        print("Data:   mmsdk (CMU-MOSEI)")
        train_ds = MOSEISDKDataset(split="train", modalities=args.modalities)
        val_ds   = MOSEISDKDataset(split="val",   modalities=args.modalities)
        test_ds  = MOSEISDKDataset(split="test",  modalities=args.modalities)
    else:
        print(f"Data:   {args.data}")
        train_ds = CMUMOSEIDataset(
            pkl_path=args.data, split="train", modalities=args.modalities
        )
        val_ds = CMUMOSEIDataset(
            pkl_path=args.data, split="val", modalities=args.modalities
        )
        test_ds = CMUMOSEIDataset(
            pkl_path=args.data, split="test", modalities=args.modalities
        )
        if isinstance(train_ds.samples, list) and len(val_ds) == len(train_ds):
            print(f"Hint: 按 {MOSEI_SPLIT_TRAIN:.0%}/{MOSEI_SPLIT_VAL:.0%}/{1-MOSEI_SPLIT_TRAIN-MOSEI_SPLIT_VAL:.0%} 切分")
            from torch.utils.data import Subset
            n = len(train_ds); full = train_ds
            te = int(n * MOSEI_SPLIT_TRAIN); ve = te + int(n * MOSEI_SPLIT_VAL)
            train_ds = Subset(full, range(0, te))
            val_ds   = Subset(full, range(te, ve))
            test_ds  = Subset(full, range(ve, n))

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # 2. 模型
    model = MultimodalSentimentModel()
    trainer = Trainer(model, train_ds, val_ds, test_ds,
                      lr=args.lr, batch_size=args.batch,
                      num_epochs=args.epochs, device=args.device,
                      tag=tag)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # 3. 训练
    trainer.train()

    # 4. 测试 + 保存结果
    test_metrics = trainer.test()
    trainer.save_results(test_metrics)

    print(f"\nCheckpoints saved to: {CHECKPOINT_DIR}/")
    print(f"Results saved to: results_{tag}.json")


if __name__ == "__main__":
    main()
