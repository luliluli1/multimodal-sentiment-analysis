#!/usr/bin/env python3
"""
最终消融实验结果汇总。

用法:
    python scripts/collect_results.py

输入:
    final_results/{full,text_only,text_audio,text_visual}.json

输出:
    final_results/ablation_results.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "final_results")

MODEL_MAP = {
    "text_only.json":    "Text Only",
    "text_audio.json":   "Text + Audio",
    "text_visual.json":  "Text + Visual",
    "full.json":         "Text + Audio + Visual",
}

METRIC_KEYS = ["mae", "corr", "acc7", "acc2", "f1"]
CSV_HEADER  = ["Model", "MAE", "Correlation", "Acc7", "Acc2", "F1"]


def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f"ERROR: {RESULTS_DIR} not found")
        sys.exit(1)

    rows = []
    for filename, model_name in MODEL_MAP.items():
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            print(f"  ⚠️  {filename} not found, skipping")
            continue

        with open(path) as f:
            data = json.load(f)

        test_metrics = data.get("test_metrics", {})
        row = [model_name] + [test_metrics.get(k, "-") for k in METRIC_KEYS]
        rows.append(row)

    if not rows:
        print("ERROR: no results found")
        sys.exit(1)

    # ── CSV ──
    csv_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)

    # ── 终端打印 ──
    print(f"\n{'='*70}")
    print(f"  消融实验结果汇总")
    print(f"{'='*70}")
    print(f"  {'Model':<24s} {'MAE':>6s} {'Corr':>7s} {'Acc7':>7s} {'Acc2':>7s} {'F1':>7s}")
    print(f"  {'-'*24} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for row in rows:
        name = row[0]
        vals = row[1:]
        print(f"  {name:<24s} " + " ".join(f"{v:>7.4f}" if isinstance(v, float) else f"{v:>7s}" for v in vals))
    print(f"{'='*70}")
    print(f"\n  ✅ CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
