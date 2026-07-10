# 🎭 多模态情感分析 MVP

文本 + 图像 + 音频 → 综合情感极性 (positive / neutral / negative)

## 项目结构

```
multimodal-sentiment-analysis/
├── config.py                 # 全局配置 (模型名、设备、权重)
├── main.py                   # CLI 命令行入口
├── api.py                    # FastAPI HTTP 接口
├── app.py                    # Streamlit 可视化 Demo
├── requirements.txt          # 依赖
│
├── models/                   # 模型模块
│   ├── text_model.py         # 文本情感 (RoBERTa)
│   ├── image_model.py        # 图像情感 (ViT)
│   ├── audio_model.py        # 语音情感 (Wav2Vec2)
│   └── fusion.py             # 多模态加权融合
│
├── utils/                    # 工具模块
│   ├── preprocessing.py      # 数据预处理
│   └── postprocessing.py     # 结果格式化
│
├── tests/                    # 测试
│   └── test_basic.py
│
└── examples/                 # 示例文件 (可自行放入)
```

## 数据流

```
输入                          单模态推理                    融合                      输出
──────                      ────────────                ──────                    ──────
text  ───→ preprocess ───→ TextSentimentModel ───┐
                                                   │
image ───→ preprocess ───→ ImageEmotionModel  ────┼──→ MultimodalFusion ──→ format_result
                                                   │     (加权求和)
audio ───→ preprocess ───→ AudioEmotionModel ────┘
```

1. **预处理**: 文本清洗/图像转RGB/音频重采样 16kHz mono
2. **单模态推理**: 各自预训练模型输出 scores
3. **融合**: 按配置权重 `{text:0.5, image:0.3, audio:0.2}` 加权求和
4. **后处理**: 格式化为统一 JSON + 终端/Web 展示

## 预训练模型

| 模态 | 模型 | 输出 |
|------|------|------|
| 文本 | `cardiffnlp/twitter-roberta-base-sentiment-latest` | neg / neu / pos |
| 图像 | `trpakov/vit-face-expression` | 7种表情 → 映射到极性 |
| 音频 | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | 6种情绪 → 映射到极性 |

首次运行会自动从 HuggingFace Hub 下载模型（约 1.5GB），后续使用缓存。

## 安装 & 运行

### 1. 创建虚拟环境

```bash
cd multimodal-sentiment-analysis
python -m venv venv
source venv/bin/activate    # macOS/Linux
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
# 注意: PyTorch 建议按官方指南安装对应平台版本
# https://pytorch.org/get-started/locally/
```

### 3. 运行方式

**命令行 (CLI)**:
```bash
# 纯文本
python main.py --text "I love this movie!"

# 文本 + 图像
python main.py --text "This is amazing" --image examples/sample_image.jpg

# 全部三个模态
python main.py --text "What a great day" --image examples/sample_image.jpg --audio examples/sample_audio.wav
```

**Web UI (Streamlit)**:
```bash
streamlit run app.py
# 浏览器打开 http://localhost:8501
```

**API 服务 (FastAPI)**:
```bash
python api.py
# 浏览器打开 http://localhost:8000/docs 查看 Swagger 文档
```

```bash
# 调用示例
curl -X POST http://localhost:8000/analyze \
  -F "text=I'm feeling great today!" \
  -F "image=@examples/sample_image.jpg"
```

### 4. 运行测试

```bash
python -m pytest tests/ -v
```

## 配置说明

编辑 `config.py` 可调整:

- `DEVICE`: 推理设备 (`"cpu"` / `"cuda"` / `"mps"`)
- `FUSION_WEIGHTS`: 各模态融合权重
- `AUDIO_SAMPLE_RATE` / `AUDIO_MAX_DURATION`: 音频参数
- `TEXT_MAX_LENGTH`: 文本截断长度

## 输出示例

```json
{
  "overall": {
    "sentiment": "positive",
    "confidence": 0.8234
  },
  "scores": {
    "negative": 0.0521,
    "neutral": 0.1245,
    "positive": 0.8234
  },
  "modalities": {
    "text": {
      "label": "positive",
      "sentiment": "positive",
      "confidence": 0.91
    },
    "image": {
      "label": "happy",
      "sentiment": "positive",
      "confidence": 0.78
    },
    "audio": {
      "label": "happy",
      "sentiment": "positive",
      "confidence": 0.72
    }
  }
}
```

## 后续扩展方向

- [ ] 视频模态 (逐帧分析 + 时序聚合)
- [ ] 更复杂的融合策略 (Attention-based / Gated Fusion)
- [ ] 中文支持 (替换中文预训练模型)
- [ ] GPU 推理加速
- [ ] Docker 部署
- [ ] 实时摄像头 + 麦克风输入
