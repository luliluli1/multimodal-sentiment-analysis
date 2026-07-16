"""
训练器 — 训练循环 + 验证 + 测试

支持:
  - 逐 epoch 训练/验证
  - 基于 MAE 的 checkpoint 保存
  - 训练日志输出
  - 学习率调度
"""

from __future__ import annotations

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    GRADIENT_CLIP,
    CHECKPOINT_DIR,
    WARMUP_STEPS,
    DEVICE,
)
from trainers.metrics import compute_metrics, format_metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset=None,
        test_dataset=None,
        lr: float = LEARNING_RATE,
        batch_size: int = BATCH_SIZE,
        num_epochs: int = NUM_EPOCHS,
        device: str = DEVICE,
        tag: str = "full",
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.base_lr = lr

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=self._collate_fn,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=self._collate_fn,
            )
        self.test_loader = None
        if test_dataset is not None:
            self.test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=self._collate_fn,
            )

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=WEIGHT_DECAY,
        )
        self.criterion = nn.L1Loss()

        # 简单的线性 warmup (无外部 scheduler 依赖)
        self.warmup_steps = WARMUP_STEPS
        self.current_step = 0

        self.tag = tag
        self.best_mae = float("inf")
        self.best_metrics = None
        self.best_checkpoint_path = None
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ----------------------------------------------------------
    # Data
    # ----------------------------------------------------------
    @staticmethod
    def _collate_fn(batch: list[dict]) -> dict:
        """将 list of samples 组装为 batch。"""
        return {
            "text":   [s["text"] for s in batch],
            "visual": [s.get("visual") for s in batch],
            "audio":  [s["audio"] for s in batch],
            "label":  torch.tensor([[s["label"]] for s in batch]).float(),
        }

    # ----------------------------------------------------------
    # Training Loop
    # ----------------------------------------------------------
    def train(self):
        print(f"\n{'='*55}")
        print(f"  开始训练 — {self.num_epochs} epochs, device={self.device}")
        print(f"  可训练参数: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"{'='*55}\n")

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()

            train_loss, _ = self._run_epoch(self.train_loader, is_train=True)
            val_metrics = None
            if self.val_loader is not None:
                val_loss, val_metrics = self._run_epoch(self.val_loader, is_train=False)

            elapsed = time.time() - t0

            # 日志
            log = f"Epoch {epoch:3d}/{self.num_epochs} | train_loss={train_loss:.4f}"
            if val_metrics:
                log += f" | val_loss={val_loss:.4f} | {format_metrics(val_metrics)}"
            log += f" | {elapsed:.1f}s"
            print(log)

            # 保存最佳
            if val_metrics and val_metrics["mae"] < self.best_mae:
                self.best_mae = val_metrics["mae"]
                self.best_metrics = val_metrics
                self._save_checkpoint(epoch, val_metrics)

        if self.best_checkpoint_path is not None:
            self._restore_best_model()
        print(f"\n训练完成 — best MAE={self.best_mae:.4f}")

    # ----------------------------------------------------------
    # One epoch
    # ----------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, is_train: bool) -> tuple[float, dict | None]:
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in loader:
            texts = batch["text"]
            visuals = batch["visual"]
            audios = batch["audio"]
            labels = batch["label"].to(self.device)

            if is_train:
                self.optimizer.zero_grad()
                preds = self.model(texts, visuals, audios)
                self._sync_optimizer_parameters()
                loss = self.criterion(preds, labels)
                loss.backward()

                # 学习率 warmup
                if self.current_step < self.warmup_steps:
                    lr_scale = (self.current_step + 1) / max(self.warmup_steps, 1)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.base_lr * lr_scale
                self.current_step += 1

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
                self.optimizer.step()
            else:
                with torch.no_grad():
                    preds = self.model(texts, visuals, audios)
                    loss = self.criterion(preds, labels)

            total_loss += loss.item() * len(texts)
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        metrics = None
        if not is_train:
            preds_arr = np.concatenate(all_preds)
            labels_arr = np.concatenate(all_labels)
            metrics = compute_metrics(preds_arr, labels_arr)

        return avg_loss, metrics

    def _sync_optimizer_parameters(self):
        """把 forward 中懒创建的可训练参数加入 optimizer。"""
        optimizer_param_ids = {
            id(param)
            for group in self.optimizer.param_groups
            for param in group["params"]
        }
        new_params = [
            param
            for param in self.model.parameters()
            if param.requires_grad and id(param) not in optimizer_param_ids
        ]
        if new_params:
            self.optimizer.add_param_group({"params": new_params})

    # ----------------------------------------------------------
    # Test
    # ----------------------------------------------------------
    def test(self) -> dict:
        if self.test_loader is None:
            print("No test set available.")
            return {}

        self.model.eval()
        all_preds, all_labels = [], []

        for batch in self.test_loader:
            texts = batch["text"]
            visuals = batch["visual"]
            audios = batch["audio"]
            labels = batch["label"]
            with torch.no_grad():
                preds = self.model(texts, visuals, audios)
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        preds_arr = np.concatenate(all_preds)
        labels_arr = np.concatenate(all_labels)
        metrics = compute_metrics(preds_arr, labels_arr)
        print(f"\n[Test] {format_metrics(metrics)}")
        return metrics

    # ----------------------------------------------------------
    # Checkpoint
    # ----------------------------------------------------------
    def _save_checkpoint(self, epoch: int, metrics: dict):
        safe_tag = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in self.tag
        )
        path = os.path.join(
            CHECKPOINT_DIR,
            f"best_{safe_tag}_epoch{epoch:03d}_mae{metrics['mae']:.4f}.pt",
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "current_step": self.current_step,
                "metrics": metrics,
            },
            path,
        )
        self.best_checkpoint_path = path
        print(f"  → checkpoint saved: {os.path.basename(path)}")

    def _restore_best_model(self):
        """训练结束后恢复验证集 MAE 最优的模型权重，供测试使用。"""
        ckpt = torch.load(self.best_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(
            "  → restored best checkpoint: "
            f"{os.path.basename(self.best_checkpoint_path)}"
        )

    def save_results(self, test_metrics: dict | None = None):
        """保存实验结果到 results_{tag}.json"""
        import json
        result = {
            "tag": self.tag,
            "best_val_metrics": self.best_metrics,
            "test_metrics": test_metrics,
            "best_checkpoint": self.best_checkpoint_path,
        }
        path = f"results_{self.tag}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self._sync_optimizer_parameters()
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.current_step = ckpt.get("current_step", self.current_step)
        print(f"Loaded checkpoint: {path} (epoch {ckpt['epoch']})")
        return ckpt.get("metrics", {})
