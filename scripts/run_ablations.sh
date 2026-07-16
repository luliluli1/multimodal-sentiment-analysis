#!/bin/bash
# ================================================================
#  Baseline Comparison — 6 组实验 + 自动汇总
#
#  单模态: text | visual | audio
#  双模态: text+visual | text+audio
#  全模态: full
#
# 用法:
#   bash scripts/run_ablations.sh [EPOCHS] [DEVICE] [BATCH]
#   bash scripts/run_ablations.sh 50 cuda 2
# ================================================================

set -e
EPOCHS=${1:-50}
DEVICE=${2:-cpu}
BATCH=${3:-2}
SCRIPT="python scripts/train.py --sdk --epochs $EPOCHS --device $DEVICE --batch $BATCH"

# ── 实验矩阵 ──
# 格式: "tag|modalities_args"
EXPERIMENTS=(
    "text|text"
    "visual|visual"
    "audio|audio"
    "text_visual|text visual"
    "text_audio|text audio"
    "full|text audio visual"
)
TOTAL=${#EXPERIMENTS[@]}

echo "============================================"
echo "  Baseline Comparison — CMU-MOSEI"
echo "  ${TOTAL} experiments × ${EPOCHS} epochs  |  device=${DEVICE} batch=${BATCH}"
echo "============================================"

for i in "${!EXPERIMENTS[@]}"; do
    IFS="|" read -r TAG MODS <<< "${EXPERIMENTS[$i]}"
    N=$((i + 1))

    echo ""
    echo ">>> [$N/$TOTAL] $TAG  ($MODS)"
    echo "------------------------------------------------------------"
    $SCRIPT --modalities $MODS --tag "$TAG"
    echo "  ✅ $TAG done"
done

# ================================================================
# 汇总 — 生成 comparison.csv + comparison.md
# ================================================================
echo ""
echo "============================================"
echo "  Generating comparison table..."
echo "============================================"

python - "$TOTAL" "${EXPERIMENTS[@]}" << 'PYEOF'
import json, os, sys

# 实验列表从 shell 参数传入
total = int(sys.argv[1])
experiments_raw = sys.argv[2:]

tags = []
for raw in experiments_raw:
    tag = raw.split("|")[0]
    tags.append(tag)

# 标签显示名映射
NAME_MAP = {
    "text":        "Text Only",
    "visual":      "Visual Only",
    "audio":       "Audio Only",
    "text_visual": "Text + Visual",
    "text_audio":  "Text + Audio",
    "full":        "Full (Text+Visual+Audio)",
}

METRICS = ["mae", "corr", "acc2", "acc7", "f1"]
HEADER_CSV  = ["Experiment", "MAE", "Corr", "Acc-2", "Acc-7", "F1"]

rows = []
for tag in tags:
    path = f"results_{tag}.json"
    if not os.path.exists(path):
        print(f"  ⚠️  {path} not found, skipping")
        continue

    with open(path) as f:
        data = json.load(f)

    # 优先用 test_metrics，fallback 到 best_val_metrics
    m = data.get("test_metrics") or data.get("best_val_metrics") or {}
    name = NAME_MAP.get(tag, tag.replace("_", " ").title())

    rows.append([
        name,
        m.get("mae",  "-"),
        m.get("corr", "-"),
        m.get("acc2", "-"),
        m.get("acc7", "-"),
        m.get("f1",  "-"),
    ])

if not rows:
    print("  No results found!")
    sys.exit(1)

# ── CSV ──
csv_path = "comparison.csv"
with open(csv_path, "w") as f:
    f.write(",".join(HEADER_CSV) + "\n")
    for r in rows:
        f.write(",".join(str(x) for x in r) + "\n")
print(f"  ✅ {csv_path}")

# ── Markdown ──
md_path = "comparison.md"
with open(md_path, "w") as f:
    f.write("# Baseline Comparison — CMU-MOSEI\n\n")
    f.write(f"| {' | '.join(HEADER_CSV)} |\n")
    f.write(f"|{'|'.join(['---'] * len(HEADER_CSV))}|\n")
    for r in rows:
        f.write(f"| {' | '.join(str(x) for x in r)} |\n")
    f.write("\n*All metrics reported on test set.*\n")
print(f"  ✅ {md_path}")

# ── 终端打印 ──
print()
for r in rows:
    print(f"  {r[0]:<28s}  MAE={r[1]}  Corr={r[2]}  Acc-2={r[3]}  Acc-7={r[4]}  F1={r[5]}")
print()
print("  Done — comparison.csv / comparison.md ready for paper slides.")
PYEOF

echo ""
echo "============================================"
echo "  Baseline Comparison 完成!"
echo "============================================"
echo "Summary:"
for f in results_text.json results_visual.json results_audio.json \
         results_text_visual.json results_text_audio.json results_full.json; do
    [ -f "$f" ] && echo "  ✅ $f"
done
echo ""
echo "  📊 comparison.csv  — 论文表格 (可直接导入 Excel/Google Sheets)"
echo "  📝 comparison.md   — 论文/PPT Markdown 表格"
