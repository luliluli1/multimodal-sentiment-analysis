"""
结果后处理与格式化
"""

import json
from typing import Optional


def format_result(fusion_result: dict, input_info: Optional[dict] = None) -> dict:
    """
    将融合结果格式化为最终输出。
    """
    per_modality = fusion_result.get("per_modality", {})

    # 构建各模态的摘要
    modality_summary = {}
    for modal_name in ["text", "image", "audio"]:
        if modal_name in per_modality:
            m = per_modality[modal_name]
            modality_summary[modal_name] = {
                "label": m.get("label") or m.get("emotion"),
                "sentiment": m.get("sentiment"),
                "confidence": round(m["confidence"], 4),
            }

    output = {
        "overall": {
            "sentiment": fusion_result["label"],
            "confidence": round(fusion_result["confidence"], 4),
        },
        "scores": {k: round(v, 4) for k, v in fusion_result["scores"].items()},
        "modalities": modality_summary,
    }

    if input_info:
        output["input"] = input_info

    return output


def to_json(result: dict, indent: int = 2) -> str:
    """将结果转为 JSON 字符串。"""
    return json.dumps(result, ensure_ascii=False, indent=indent)


def print_result(result: dict):
    """美观地打印分析结果到终端。"""
    print("\n" + "=" * 50)
    print("       多模态情感分析结果")
    print("=" * 50)

    overall = result["overall"]
    sentiment_emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}
    emoji = sentiment_emoji.get(overall["sentiment"], "")

    print(f"\n  综合情感: {overall['sentiment'].upper()} {emoji}")
    print(f"  置信度:   {overall['confidence']:.2%}")
    print(f"\n  情感分数: NEG={result['scores']['negative']:.3f}  "
          f"NEU={result['scores']['neutral']:.3f}  "
          f"POS={result['scores']['positive']:.3f}")

    for modal, info in result.get("modalities", {}).items():
        name_map = {"text": "文本", "image": "图像", "audio": "音频"}
        print(f"\n  [{name_map.get(modal, modal)}] "
              f"→ {info['label']} ({info['confidence']:.2%})")

    if "input" in result:
        print(f"\n  输入: {result['input']}")

    print("\n" + "=" * 50 + "\n")
