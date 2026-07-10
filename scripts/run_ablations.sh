#!/bin/bash
# 消融实验 — 依次运行 4 组实验
# 用法: bash scripts/run_ablations.sh

set -e
EPOCHS=${1:-50}
DEVICE=${2:-cpu}

echo "============================================"
echo "  消融实验 — CMU-MOSEI (${EPOCHS} epochs)"
echo "============================================"

# Exp1: Text-only
echo ""
echo ">>> [1/4] Text-only"
python scripts/train.py --sdk --epochs $EPOCHS --device $DEVICE --modalities text --tag text_only

# Exp2: Text + Audio
echo ""
echo ">>> [2/4] Text + Audio"
python scripts/train.py --sdk --epochs $EPOCHS --device $DEVICE --modalities text audio --tag text_audio

# Exp3: Text + Visual
echo ""
echo ">>> [3/4] Text + Visual"
python scripts/train.py --sdk --epochs $EPOCHS --device $DEVICE --modalities text visual --tag text_visual

# Exp4: Full Model
echo ""
echo ">>> [4/4] Full Model"
python scripts/train.py --sdk --epochs $EPOCHS --device $DEVICE --modalities text audio visual --tag full

echo ""
echo "============================================"
echo "  消融实验完成!"
echo "============================================"
echo "Results:"
for f in results_text_only.json results_text_audio.json results_text_visual.json results_full.json; do
    if [ -f "$f" ]; then
        echo "  $f: $(python -c "import json; d=json.load(open('$f')); print(f\"MAE={d['best_val_metrics']['mae']}, Acc2={d['best_val_metrics']['acc2']}, F1={d['best_val_metrics']['f1']}\")")"
    fi
done
