# Multimodal Sentiment Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Cross-Attention Fusion of Text, Audio, and Visual Modalities for Sentiment Regression on CMU-MOSEI.**

---

## Overview

Sentiment analysis is inherently a multimodal problem — human emotion is conveyed through **words**, **tone of voice**, and **facial expressions**. This project addresses the task of **multimodal sentiment regression** on the [CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/) dataset, predicting a continuous sentiment score in the range `[-3, +3]` from aligned text, audio, and visual sequences.

The model uses a **BERT** text encoder, a **COVAREP** adapter for acoustic features, and a **FACET** encoder for visual features. These are fused through a **cross-attention mechanism** and passed to a regression head. The project includes a full ablation study to measure the contribution of each modality.

---

## Model Architecture

```
   Text                      Audio                    Visual
    │                          │                         │
    ▼                          ▼                         ▼
┌─────────┐            ┌───────────────┐        ┌───────────────┐
│  BERT   │            │CovarepAdapter │        │ VisualEncoder │
│  base   │            │  (74d → 768d) │        │(FACET 35d/42d │
│ (768d)  │            │               │        │    → 768d)    │
└────┬────┘            └───────┬───────┘        └───────┬───────┘
     │                         │                         │
     └─────────────────────────┼─────────────────────────┘
                               │
                  ┌────────────▼────────────┐
                  │  Cross-Attention Fusion │
                  │  (text as query)        │
                  └────────────┬────────────┘
                               │
                  ┌────────────▼────────────┐
                  │   MLP Regression Head   │
                  │   (768 → 256 → 1)       │
                  └────────────┬────────────┘
                               │
                               ▼
                    Sentiment Score [-3, +3]
```

- **Single modality** (text, visual, or audio alone) bypasses fusion and uses a shared `classifier_head`.
- **Two modalities** (text+visual or text+audio) use cross-attention with the text tensor as query.
- **Three modalities** (full) use cross-attention over all three modalities.

---

## Dataset

**CMU-MOSEI** (Multimodal Opinion Sentiment and Emotion Intensity) is the largest publicly available multimodal sentiment analysis dataset:

| Statistic | Value |
|-----------|-------|
| Videos | 3,228 |
| Segments | 23,248 |
| Modalities | Text, Audio (COVAREP), Visual (FACET) |
| Labels | Sentiment `[-3, +3]` (regression) |
| Split | Official standard_folds (train: 16,322 / val: 1,871 / test: 4,659) |

The dataset is loaded via the [CMU Multimodal SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK) from local `.csd` files placed in `data/mosei_raw/`.

---

## Features

- **BERT-base** text encoder with configurable fine-tuning strategy (`all` / `top2` / `none`)
- **COVAREP features** (74-dimensional acoustic descriptors) with finite-value masked pooling to handle missing/infinite values present in the raw data
- **FACET features** (facial action units) with adaptive dimension projection
- **Cross-Attention Fusion** with text as the query modality
- **Unified modality routing** — automatically detects available modalities per batch and routes through the correct path (single → classifier, multi → fusion)
- **Full ablation study** — text-only, visual-only, audio-only, text+visual, text+audio, and full model
- **Early stopping**, learning rate warmup, gradient clipping
- **Auto device detection** (CUDA / MPS / CPU)
- **Deterministic training** with fixed random seed (`SEED = 42`) for reproducibility

---

## Experiments

Six configurations were trained under identical hyperparameters:

| Model | Modalities | Description |
|-------|-----------|-------------|
| Full | text + audio + visual | All three modalities |
| Text Only | text | BERT only, no fusion |
| Visual Only | visual | FACET features only |
| Audio Only | audio | COVAREP features only |
| Text + Audio | text + audio | Dual-modal fusion |
| Text + Visual | text + visual | Dual-modal fusion |

The first four (required for the ablation comparison) were trained on GPU, while visual-only and audio-only baselines can be trained with the same script.

---

## Results

Final test-set metrics (four primary ablation configurations):

| Model | MAE ↓ | Corr ↑ | Acc-7 ↑ | Acc-2 ↑ | F1 ↑ |
|-------|------:|------:|------:|------:|------:|
| Text Only | 0.5474 | 0.7436 | 0.5256 | 0.8465 | 0.8459 |
| Text + Audio | 0.5607 | 0.7321 | 0.5188 | 0.8401 | 0.8368 |
| Text + Visual | 0.5507 | 0.7397 | 0.5319 | 0.8431 | 0.8425 |
| **Text + Audio + Visual** | **0.5476** | **0.7437** | **0.5327** | **0.8423** | **0.8387** |

### Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **MAE** | Mean Absolute Error between predicted and ground-truth scores | `[0, 6]` (lower is better) |
| **Corr** | Pearson correlation coefficient | `[-1, 1]` (higher is better) |
| **Acc-7** | 7-class accuracy (rounded to nearest integer in `[-3, +3]`) | `[0, 1]` |
| **Acc-2** | Binary accuracy (positive vs. negative, excluding neutral) | `[0, 1]` |
| **F1** | Weighted F1 score for binary classification | `[0, 1]` |

Full results including validation metrics, training curves, and early stopping details are in `final_results/`. Generate the summary table with:

```bash
python scripts/collect_results.py
```

---

## Project Structure

```
multimodal-sentiment-analysis/
├── models/
│   ├── multimodal_model.py    # Top-level model & modality routing
│   ├── text_encoder.py        # BERT-base encoder
│   ├── visual_encoder.py      # FACET feature encoder
│   ├── audio_encoder.py       # Wav2Vec2 encoder (for raw audio)
│   ├── covarep_adapter.py     # COVAREP 74d → 768d adapter
│   ├── image_encoder.py       # ViT encoder (for image files)
│   └── fusion.py              # Cross-modal attention + MLP fusion
├── data/
│   └── mosei_sdk_dataset.py   # CMU-MOSEI SDK data loader
├── trainers/
│   ├── trainer.py             # Training loop, early stopping, checkpointing
│   └── metrics.py             # MAE, Corr, Acc-7, Acc-2, F1
├── scripts/
│   ├── train.py               # Training entry point
│   ├── collect_results.py     # Ablation results → CSV summary
│   ├── plot_history.py        # Training curve visualisation
│   └── run_ablations.sh       # Batch ablation experiment runner
├── final_results/              # Final experiment results (paper-ready)
│   ├── full.json
│   ├── text_only.json
│   ├── text_audio.json
│   ├── text_visual.json
│   ├── ablation_results.csv
│   └── README.md
├── inference.py               # Predictor class & CLI
├── app.py                     # Streamlit demo
├── api.py                     # FastAPI server
├── config.py                  # Global configuration & hyperparameters
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/your-username/multimodal-sentiment-analysis.git
cd multimodal-sentiment-analysis

python -m venv venv
source venv/bin/activate    # macOS / Linux
# venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

**CMU-MOSEI data:** download the four `.csd` files from the [official source](https://github.com/A2Zadeh/CMU-MultimodalSDK) and place them in `data/mosei_raw/`:

- `CMU_MOSEI_TimestampedWords.csd`
- `CMU_MOSEI_COVAREP.csd`
- `CMU_MOSEI_VisualFacet42.csd`
- `CMU_MOSEI_Labels.csd`

The dataset will be loaded automatically via the CMU Multimodal SDK.

---

## Training

### Full model (text + audio + visual)

```bash
python scripts/train.py --sdk --epochs 50 --modalities text audio visual --tag full
```

### Ablation experiments

```bash
# Text only
python scripts/train.py --sdk --epochs 50 --modalities text          --tag text_only

# Text + Audio
python scripts/train.py --sdk --epochs 50 --modalities text audio    --tag text_audio

# Text + Visual
python scripts/train.py --sdk --epochs 50 --modalities text visual   --tag text_visual
```

### Run all ablations sequentially

```bash
bash scripts/run_ablations.sh 50 cuda
```

Training saves checkpoints to `checkpoints/`, results to `results_{tag}.json`, per-epoch metrics to `history_{tag}.json`, and archives everything to `experiments/autodl/{tag}/`.

---

## Reproducibility

- **Random seed** is fixed at `SEED = 42` (see `config.py`) covering `random`, `numpy`, `torch`, and CUDA backends
- **Official CMU-MOSEI standard_folds** are used for train/val/test splits
- **All experiment results** are saved in `final_results/` with full metrics
- Training uses **early stopping** with patience = 5 based on validation MAE

To reproduce the exact results, use the same training commands under the same hyperparameters (`config.py`) and hardware (NVIDIA RTX 3090 / 4090).

---

## Future Work

- Integrate pre-trained audio encoders (e.g., Wav2Vec2, HuBERT) as an alternative to COVAREP features
- Explore stronger fusion mechanisms (e.g., gated fusion, transformer-based fusion)
- Support additional datasets (CMU-MOSI, IEMOCAP, MELD)
- Add visual-only and audio-only single-modality baselines to the primary ablation table
- Publish trained model weights on Hugging Face Hub
