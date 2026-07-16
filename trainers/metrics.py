"""
评估指标 — 分类 + 回归

CMU-MOSI 常用指标:
  - MAE: Mean Absolute Error
  - Corr: Pearson Correlation
  - Acc-7: 7-class accuracy (将 [-3,3] 四舍五入为整数)
  - Acc-2: Binary accuracy (正 vs 负, 排除 0)
  - F1: Binary F1
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Args:
        predictions: (N,) 模型预测的情感分数
        targets:     (N,) 真实标签

    Returns:
        dict with MAE, Corr, Acc7, Acc2, F1
    """
    preds = np.squeeze(predictions)
    y = np.squeeze(targets)

    # ---- 回归指标 ----
    mae = float(np.mean(np.abs(preds - y)))

    # Pearson correlation
    try:
        if np.ptp(preds) == 0 or np.ptp(y) == 0:
            raise ValueError("constant input")
        corr, _ = pearsonr(preds, y)
        corr = float(corr)
        if not np.isfinite(corr):
            corr = 0.0
    except Exception:
        corr = 0.0

    # ---- 分类指标 ----
    # 7-class: 四舍五入到整数，截断到 [-3, 3]
    preds_7 = np.clip(np.round(preds).astype(int), -3, 3)
    y_7 = np.clip(np.round(y).astype(int), -3, 3)

    # 标签空间可能不覆盖全部 7 类，只算存在的
    acc7 = float(accuracy_score(y_7, preds_7))

    # 2-class: 排	除正好为 0 的样本
    mask = y != 0
    if mask.sum() > 1:
        preds_2 = (preds[mask] >= 0).astype(int)
        y_2 = (y[mask] >= 0).astype(int)
        acc2 = float(accuracy_score(y_2, preds_2))
        f1 = float(f1_score(y_2, preds_2, average="weighted", zero_division=0))
    else:
        acc2 = 0.0
        f1 = 0.0

    return {
        "mae": round(mae, 4),
        "corr": round(corr, 4),
        "acc7": round(acc7, 4),
        "acc2": round(acc2, 4),
        "f1": round(f1, 4),
    }


def format_metrics(metrics: dict, phase: str = "") -> str:
    """格式化指标为一行字符串，用于日志输出。"""
    prefix = f"[{phase}] " if phase else ""
    return (
        f"{prefix}MAE={metrics['mae']:.4f}  Corr={metrics['corr']:.4f}  "
        f"Acc7={metrics['acc7']:.4f}  Acc2={metrics['acc2']:.4f}  F1={metrics['f1']:.4f}"
    )
