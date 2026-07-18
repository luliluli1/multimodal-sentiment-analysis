# Final Ablation Results — CMU-MOSEI

多模态情感分析消融实验最终结果。训练于 bug-fix 分支修复后，正确使用全部三种模态数据。

## 实验设置

- **数据集**: CMU-MOSEI (official standard_folds split)
- **训练设备**: NVIDIA GPU (AutoDL)
- **随机种子**: 42 (完全可复现)
- **Early Stopping**: patience=5, min_delta=1e-4 (based on validation MAE)
- **Epochs**: 50 (max)
- **修复内容**: h5py 类型检测、COVAREP 非有限值处理、VisualEncoder 懒 adapter

## 实验组

| 文件 | 模型 | modalities 参数 |
|------|------|------|
| `full.json` | Text + Audio + Visual (Full) | `text audio visual` |
| `text_only.json` | Text Only | `text` |
| `text_audio.json` | Text + Audio | `text audio` |
| `text_visual.json` | Text + Visual | `text visual` |

## 模型结构

```
TextEncoder (BERT-base, 768d)
VisualEncoder (FACET 35d → 768d, lazy adapter)
CovarepAdapter (COVAREP 74d → 768d, finite-value pooling)
    ↓
Cross-Attention Fusion (text as query, 2+ modalities)
or ClassifierHead (single modality, 768 → 256 → 1)
    ↓
Sentiment Score [-3, +3]
```

## 指标说明

| 指标 | 全称 | 范围 | 说明 |
|------|------|------|------|
| **MAE** | Mean Absolute Error | [0, 6] | 预测值与真实值的平均绝对偏差，越低越好 |
| **Corr** | Pearson Correlation | [-1, 1] | 预测与真实情感分数的线性相关，越高越好 |
| **Acc-7** | 7-class Accuracy | [0, 1] | 四舍五入到整数 [-3, +3] 的正确率 |
| **Acc-2** | Binary Accuracy | [0, 1] | 正/负情感二分类正确率（排除 neutral） |
| **F1** | F1 Score (weighted) | [0, 1] | 二分类 F1，综合 precision 和 recall |

## 汇总表

见 `ablation_results.csv`。生成命令：

```bash
python scripts/collect_results.py
```
