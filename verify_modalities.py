#!/usr/bin/env python3
"""
验证修复后 6 种模态配置的网络路径。

对每种配置:
  1. 打印数据类型与 shape
  2. 追踪各模块是否被调用
  3. 输出 tensor shape
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

# ── Monkey-patch hooks ──
TRACE = {}
_orig = {}

def _hook(module, name):
    cls = type(module).__name__
    _orig[(cls, name)] = module.forward
    def patched(*args, **kwargs):
        TRACE[name] = True
        return _orig[(cls, name)](*args, **kwargs)
    module.forward = patched

TRACK = [
    ("text_encoder",    "text_encoder"),
    ("visual_encoder",  "visual_encoder"),
    ("audio_encoder",   "audio_encoder"),
    ("covarep_adapter", "covarep_adapter"),
    ("fusion",          "fusion"),
    ("classifier_head", "classifier_head"),
]

# ── Load model ──
from models.multimodal_model import MultimodalSentimentModel
from data.mosei_sdk_dataset import MOSEISDKDataset

model = MultimodalSentimentModel()
model.eval()

for attr, label in TRACK:
    _hook(getattr(model, attr), label)

# ── Configs ──
CONFIGS = [
    ("text-only",        ["text"]),
    ("visual-only",      ["visual"]),
    ("audio-only",       ["audio"]),
    ("text+visual",      ["text", "visual"]),
    ("text+audio",       ["text", "audio"]),
    ("full",             ["text", "audio", "visual"]),
]

print("=" * 72)
print("  修复后 forward() 路径验证 — 6 种模态配置")
print("=" * 72)

B = 4
ds_full = MOSEISDKDataset(split="test", modalities=["text", "audio", "visual"])
indices = list(range(min(B, len(ds_full))))

for mode_name, modalities in CONFIGS:
    TRACE.clear()

    # 取同一批样本，按 modalities 过滤
    samples = [ds_full[i] for i in indices]
    texts   = [s["text"]   if "text"   in modalities else None for s in samples]
    visuals = [s["visual"] if "visual" in modalities else None for s in samples]
    audios  = [s["audio"]  if "audio"  in modalities else None for s in samples]

    # 数据类型 + shape
    v0 = visuals[0]
    a0 = audios[0]
    v_info = f"{type(v0).__module__}.{type(v0).__qualname__}  shape={v0.shape}" if v0 is not None else "None"
    a_info = f"{type(a0).__module__}.{type(a0).__qualname__}  shape={a0.shape}" if a0 is not None else "None"
    t_info = f"str  len={len(texts[0])}" if texts[0] else "None"

    # forward
    with torch.no_grad():
        output = model(texts, visuals, audios)

    # 打印
    n_mods = sum(1 for x in [texts[0] is not None and texts[0].strip(),
                              v0 is not None,
                              a0 is not None] if x)
    lines = [
        f"\n{'─'*72}",
        f"  {mode_name:<16s}  modalities={modalities}  n_available={n_mods}",
        f"  Text={t_info}",
        f"  Visual={v_info}",
        f"  Audio={a_info}",
        f"  Output shape: {tuple(output.shape)}",
        f"  Path:",
    ]
    for _, label in TRACK:
        called = "✓" if TRACE.get(label) else "✗"
        lines.append(f"    {called} {label}")
    print("\n".join(lines))

print(f"\n{'='*72}")
print("  验证完成")
print("=" * 72)
