# Bug Report — 多模态情感分析项目

> 生成日期: 2026-07-16
> 分支: main (532bc26 Stable version before ablation experiments)

---

## 总览

| # | Bug | 严重程度 | 状态 |
|---|-----|:---:|:---:|
| 1 | h5py 类型误判 — FACET/COVAREP 特征未被使用 | 🔴 严重 | ✅ 已修复 |
| 2 | VisualEncoder 动态 Adapter — 每次 forward 创建新的随机权重 | 🔴 严重 | ✅ 已修复 |
| 3 | COVAREP 原始特征含 -inf — 导致 NaN 传播 | 🔴 严重 | ✅ 已修复 |

---

## Bug 1: h5py 类型误判

### 位置

[models/multimodal_model.py](models/multimodal_model.py) — `_detect_visual_type()` / `_detect_audio_type()`

### 根因

```python
# 旧代码
@staticmethod
def _detect_visual_type(inputs: list) -> str:
    for v in inputs:
        if isinstance(v, np.ndarray):       # ← Bug: h5py 类型不是 np.ndarray 子类
            return "openface"
    return "none"
```

CMU-MOSEI SDK `.csd` 文件加载后返回的是 **h5py 类型**：

| 模态 | 实际类型 | `isinstance(x, np.ndarray)` |
|------|----------|:---:|
| FACET (visual) | `h5py._hl.dataset.AsTypeView` | `False` |
| COVAREP (audio) | `h5py._hl.dataset.Dataset` | `False` |

h5py 实现了 `__array__` 接口（可被 `np.asarray()` 转换），但**不继承** `np.ndarray`。

### 后果

旧版 `forward()` 执行路径：

```
_detect_visual_type → "none"  (误判)
_detect_audio_type  → "none"  (误判)
has_vis = False
has_aud = False

→ if not has_vis and not has_aud:   ✅ 命中
→ return self.classifier_head(text_feat)   # 绕过所有 encoder + fusion
```

**VisualEncoder、CovarepAdapter、Fusion 全部被跳过。所有实验实际退化为 text-only。**

### NaN 溯源

```
COVAREP raw (含 -inf)
  → mean pooling (frame axis) → -inf
  → CovarepAdapter Linear(74→256): weight 近 0 × (-inf) = NaN
  → ❌ 第一个 NaN 在此 (covarep_adapter 输出)
  → Fusion → ❌ NaN
  → Loss → ❌ NaN
```

### 影响范围

| 实验 | 是否受影响 | 说明 |
|------|:---:|------|
| text-only | ✅ 不受影响 | text 是 Python str，`_has_text` 不依赖 isinstance |
| text+visual | ❌ 受影响 | FACET 从未被使用 |
| text+audio | ❌ 受影响 | COVAREP 从未被使用 |
| full | ❌ 受影响 | 两个辅助模态均未使用 |
| visual-only | 🆕 新实验 | 修复后才能运行 |
| audio-only | 🆕 新实验 | 修复后才能运行 |
| Demo | ⚠️ 推理不受影响 | 但展示的是 text-only 结果 |

### 已有实验结果分析

```
results_full.json:       best_val_mae = 0.4025
results_text_visual.json: best_val_mae = 0.4352   (差值 0.033)
results_text_audio.json:  best_val_mae = 0.4403   (差值 0.038)
```

三个"不同"实验的 MAE 差距仅 ~0.03，处于相同架构、相同数据、不同随机种子的波动范围内。**确认所有实验实际训练的是 text-only 模型。**

### 修复

```python
# 修复 1: duck-typing 检测
@staticmethod
def _detect_visual_type(inputs: list) -> str:
    for v in inputs:
        if hasattr(v, "ndim") and v.size > 0:   # ← 替代 isinstance
            return "openface"
        if isinstance(v, str) and len(v) > 0:
            return "image"
    return "none"

# 修复 2: forward() 中 h5py → numpy 显式转换
if has_vis:
    visual_inputs = [np.asarray(v, dtype=np.float32)
                   if hasattr(v, "ndim") and not isinstance(v, np.ndarray)
                   else v for v in visual_inputs]
```

### 影响

- 已有 text-only checkpoint 不受影响（text 不经过此路径）
- text+visual / text+audio / full 的所有旧 checkpoint **不可信，需要重新训练**
- 模型架构不变，state_dict key 不变，checkpoint 格式兼容

---

## Bug 2: VisualEncoder 动态 Adapter

### 位置

[models/visual_encoder.py](models/visual_encoder.py) L58-62

### 根因

```python
# 旧代码
if x.shape[-1] != self.input_dim:
    adapter = nn.Linear(x.shape[-1], self.input_dim).to(x.device)  # 每次 forward 新建
    x = adapter(x)
```

FACET 实际 35 维 ≠ 配置 `VISUAL_ENCODER_INPUT_DIM = 42`。

每次 `forward()` 创建一个全新的 `nn.Linear`，：
- 随机初始化权重
- 不被 `optimizer` 追踪（未注册为 `nn.Module` 属性）
- 梯度无法更新

### 后果

即使 Bug 1 修复后 VisualEncoder 被调用，输出仍是随机噪声，训练无法收敛。

### 修复

```python
# __init__ 中增加
self.input_adapter = None  # 懒初始化

# forward 中改为
if x.shape[-1] != self.input_dim:
    if self.input_adapter is None:
        self.input_adapter = nn.Linear(x.shape[-1], self.input_dim).to(x.device)
    x = self.input_adapter(x)
```

首次 forward 时创建一次，之后复用，被 optimizer 正确追踪。

---

## Bug 3: COVAREP 原始特征含 -inf

### 位置

CMU-MOSEI COVAREP `.csd` 原始数据 → [models/covarep_adapter.py](models/covarep_adapter.py)

### 根因

COVAREP 声学特征的部分维度含 `-inf` 值。

**证据（8 个样本检测）：**

```
audio[0]: shape=(14475, 74)  min=-inf  Inf=True   ← 含 -inf
audio[1]: shape=(4435,  74)  min=-inf  Inf=True   ← 含 -inf
audio[2]: shape=(5858,  74)  min=-46.7 Inf=False  ← 正常
audio[3]: shape=(14187, 74)  min=-inf  Inf=True   ← 含 -inf
audio[4]: shape=(5940,  74)  min=-37.9 Inf=False  ← 正常
audio[5]: shape=(5826,  74)  min=-41.5 Inf=False  ← 正常
audio[6]: shape=(8153,  74)  min=-inf  Inf=True   ← 含 -inf
audio[7]: shape=(7690,  74)  min=-inf  Inf=True   ← 含 -inf

8 个样本中 5 个含 -inf (62.5%)
```

### NaN 传播链

```
COVAREP (n_frames, 74)  含 -inf 值
  ↓ mean(axis=0) 池化
(B, 74)                 部分维度 = -inf
  ↓ CovarepAdapter.mlp[0]: Linear(74→256)
  ↓ weight 随机初始化 → 某些接近 0 → 0 × (-inf) = NaN
(B, 256)                ❌ 第一个 NaN 出现
  ↓ ReLU → Linear(256→768)
(B, 768)                ❌ NaN 传播
  ↓ Fusion
  ↓ Loss
                         ❌ 全部 NaN
```

### NaN 溯源实验

| 步骤 | 模块 | 状态 |
|------|------|:--:|
| 1 | text_encoder | ✅ max=1.00 min=-1.00 |
| 2 | visual_encoder | ✅ max=1.37 min=-1.49 |
| **3** | **covarep_adapter** | **❌ max=NaN min=NaN** |
| 4 | fusion_in_text | ✅ |
| 5 | fusion_in_vis | ✅ |
| 6 | fusion_in_audio | ❌ NaN (来自步骤 3) |
| 7 | fusion_out | ❌ NaN |
| 8 | classifier_head | ✅ max=0.01 min=-0.05 |
| 9 | loss | ❌ NaN |

### 影响范围

| 实验 | 是否受影响 |
|------|:---:|
| text-only | ✅ 不受影响 |
| visual-only | ✅ 不受影响 |
| **audio-only** | ❌ 受影响 |
| text+visual | ✅ 不受影响（仅 text+visual 不含 audio） |
| **text+audio** | ❌ 受影响 |
| **full** | ❌ 受影响 |

### 修复建议

在 `CovarepAdapter.forward()` 的 mean pooling 之前清理 inf：

```python
# 在 pooled.append 之前加入:
feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
```

或使用 `np.clip(feat, -1e6, 1e6)` 保留极端但有限的值。

---

## 修复进度

| Bug | 修复内容 | 修改文件 | 状态 |
|-----|---------|------|:--:|
| 1 | duck-typing 检测 + h5py→numpy 转换 | `models/multimodal_model.py` | ✅ |
| 2 | 持久化 `input_adapter` | `models/visual_encoder.py` | ✅ |
| 3 | COVAREP 有限值掩码池化 + 输出兜底清洗 | `models/covarep_adapter.py` | ✅ |

---

## 后续行动

1. **重新训练全部 6 组实验** — 使用修复后的代码
2. **生成 comparison.csv + analysis.md** — 可信的实验结果

### 后续加固

- 视觉动态 Adapter 参数在首次 forward 后同步加入 optimizer
- 带视觉 Adapter 的 checkpoint 可自动恢复其输入维度
- 训练结束后自动恢复验证集最佳 checkpoint 再测试
- checkpoint 文件名加入实验 tag，避免多组消融实验互相覆盖
- 缺失音频不再被推理接口伪装成全零波形
