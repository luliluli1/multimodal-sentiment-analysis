#!/usr/bin/env python3
"""
多模态情感分析 — 命令行推理入口

用法:
    python main.py --text "I love this movie!"
    python main.py --text "Amazing!" --image examples/sample.jpg
    python main.py --text "Great day" --image examples/sample.jpg --audio examples/sample.wav
"""

import argparse
import os
import numpy as np
from models.multimodal_model import MultimodalSentimentModel
from utils.preprocessing import preprocess_image, preprocess_audio


def main():
    parser = argparse.ArgumentParser(description="多模态情感分析")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--audio", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if not any([args.text, args.image, args.audio]):
        parser.print_help()
        print("\n请至少提供一个模态: --text / --image / --audio")
        return

    print("加载模型中...")
    model = MultimodalSentimentModel()
    model.eval()

    # 准备输入
    text = args.text or ""
    image_path = args.image or None
    audio_arr = None

    if args.audio and os.path.exists(args.audio):
        audio_arr, _ = preprocess_audio(args.audio)

    print("推理中...")
    result = model.predict(text=text, image_path=image_path, audio=audio_arr)

    print("\n" + "=" * 40)
    print(f"  情感: {result['label'].upper()}")
    print(f"  置信度: {result['confidence']:.3f}")
    print(f"  分数: {result['scores']}")
    print("=" * 40)


if __name__ == "__main__":
    main()
