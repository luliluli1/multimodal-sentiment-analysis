"""
全局配置文件 — CMU-MOSEI 多模态情感分析
"""

import os
import random
import numpy as np
import torch

# ============================================================
# 随机种子（论文实验可复现性）
# ============================================================
SEED = 42


def set_seed(seed: int = SEED):
    """固定所有随机种子，确保实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================
# 设备
# ============================================================
def _detect_device() -> str:
    """自动检测可用设备: cuda > mps > cpu。"""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _detect_device()

# ============================================================
# 预训练模型 (encoder)
# ============================================================
TEXT_ENCODER_NAME = "bert-base-uncased"
IMAGE_ENCODER_NAME = "google/vit-base-patch16-224-in21k"

# OpenFace 视觉特征编码器 (MOSEI FACET 42d)
VISUAL_ENCODER_INPUT_DIM = 42
VISUAL_HIDDEN = 768

AUDIO_ENCODER_NAME = "facebook/wav2vec2-base"

# ============================================================
# 模型架构超参
# ============================================================
TEXT_HIDDEN = 768          # BERT-base hidden dim
IMAGE_HIDDEN = 768         # ViT-base hidden dim
AUDIO_HIDDEN = 768         # Wav2Vec2-base hidden dim
PROJECTION_DIM = 256       # 投影到统一维度
FUSION_HIDDEN = 256        # Fusion MLP 隐层
CROSS_ATTENTION_HEADS = 4  # Cross-attention 头数

# ============================================================
# 训练超参
# ============================================================
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
WARMUP_STEPS = 500
GRADIENT_CLIP = 1.0

# ============================================================
# Early Stopping
# ============================================================
EARLY_STOPPING_PATIENCE = 5     # 连续无改善 epoch 数后停止；设为 0 关闭
EARLY_STOPPING_MIN_DELTA = 1e-4 # MAE 改善阈值

# ============================================================
# Fine-tuning 策略
# ============================================================
# "all"   — 全部可训练
# "top2"  — 只训最后2层 (推荐)
# "none"  — 冻结全部 encoder
FINETUNE_STRATEGY = "top2"

# ============================================================
# 数据路径
# ============================================================
DATA_DIR = "./data/mosei_raw"
MOSEI_PKL_PATH = os.path.join(DATA_DIR, "mosei_data.pkl")

# MOSEI 标准 split 比例: train 69% / val 8% / test 23%
MOSEI_SPLIT_TRAIN = 0.69
MOSEI_SPLIT_VAL = 0.08

# ============================================================
# 输入尺寸
# ============================================================
TEXT_MAX_LENGTH = 128
IMAGE_SIZE = 224
AUDIO_SAMPLE_RATE = 16000
AUDIO_MAX_DURATION = 6  # 秒
AUDIO_MAX_LENGTH = AUDIO_SAMPLE_RATE * AUDIO_MAX_DURATION  # 96000 samples

# ============================================================
# 路径（AutoDL 数据盘自动检测）
# ============================================================
_AUTODL_TMP = "/root/autodl-tmp"
if os.path.isdir(_AUTODL_TMP):
    _ROOT = _AUTODL_TMP
else:
    _ROOT = "."

CHECKPOINT_DIR = os.path.join(_ROOT, "checkpoints")
EXPERIMENTS_DIR = os.path.join(_ROOT, "experiments")
LOG_DIR = os.path.join(_ROOT, "logs")
RESULTS_DIR = _ROOT  # results_xxx.json / history_xxx.json 保存位置

# ============================================================
# 标签
# ============================================================
# MOSEI 情感标签: 连续值 [-3, +3]
MOSEI_LABEL_RANGE = (-3.0, 3.0)

# 二分类阈值: score >= 0 → positive
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
