#!/usr/bin/env python3
"""
训练曲线绘制工具

用法:
    python scripts/plot_history.py history_full_2026-07-18.json
    python scripts/plot_history.py history_full_2026-07-18.json --out-dir results/

输出:
    {out_dir}/{tag}_loss.png      — train_loss + val_loss
    {out_dir}/{tag}_metrics.png   — MAE, Corr, Acc2, F1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="训练曲线绘制")
    parser.add_argument("history", type=str, help="history JSON 文件路径")
    parser.add_argument("--out-dir", type=str, default="results",
                        help="输出目录 (默认: results/)")
    args = parser.parse_args()

    if not os.path.exists(args.history):
        print(f"ERROR: {args.history} not found")
        sys.exit(1)

    with open(args.history) as f:
        history = json.load(f)

    if not history:
        print("ERROR: empty history")
        sys.exit(1)

    # tag from filename: history_{tag}.json → {tag}
    basename = os.path.splitext(os.path.basename(args.history))[0]
    tag = basename.replace("history_", "")

    os.makedirs(args.out_dir, exist_ok=True)

    epochs = [e["epoch"] for e in history]
    train_losses = [e["train_loss"] for e in history]
    val_losses = [e["val_loss"] for e in history]
    maes  = [e["mae"]  for e in history]
    corrs = [e["corr"] for e in history]
    acc2s = [e["acc2"] for e in history]
    f1s   = [e["f1"]   for e in history]

    # ── Loss 曲线 ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_losses, "b-", linewidth=1.5, label="Train Loss")
    ax.plot(epochs, val_losses, "r-", linewidth=1.5, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MAE)")
    ax.set_title(f"Training Curves — {tag}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = os.path.join(args.out_dir, f"{tag}_loss.png")
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"  → {loss_path}")

    # ── Metrics 曲线 ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metrics = [
        (axes[0, 0], maes,  "MAE",  "blue"),
        (axes[0, 1], corrs, "Corr", "green"),
        (axes[1, 0], acc2s, "Acc-2","orange"),
        (axes[1, 1], f1s,   "F1",   "purple"),
    ]
    for ax, values, label, color in metrics:
        ax.plot(epochs, values, "-", color=color, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Validation Metrics — {tag}", fontsize=13)
    fig.tight_layout()
    metrics_path = os.path.join(args.out_dir, f"{tag}_metrics.png")
    fig.savefig(metrics_path, dpi=150)
    plt.close(fig)
    print(f"  → {metrics_path}")

    # ── 摘要 ──
    best_epoch = min(history, key=lambda e: e["mae"] or float("inf"))
    print(f"\n  Best epoch:    {best_epoch['epoch']}  (MAE={best_epoch['mae']})")
    last = history[-1]
    print(f"  Final epoch:   {last['epoch']}  train_loss={last['train_loss']}  "
          f"val_loss={last['val_loss']}  MAE={last['mae']}")


if __name__ == "__main__":
    main()
