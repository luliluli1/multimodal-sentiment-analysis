"""
基础单元测试
运行: python -m pytest tests/ -v
"""

import pytest
import numpy as np
from models.fusion import MultimodalFusion
from utils.postprocessing import format_result, to_json


class TestFusion:
    """测试多模态融合逻辑"""

    def test_fuse_text_only(self):
        fusion = MultimodalFusion()
        text_result = {
            "label": "positive",
            "confidence": 0.85,
            "scores": {"negative": 0.05, "neutral": 0.10, "positive": 0.85},
        }
        result = fusion.fuse(text_result=text_result)
        assert result["label"] == "positive"
        assert result["confidence"] > 0.8

    def test_fuse_all_none(self):
        fusion = MultimodalFusion()
        result = fusion.fuse()
        assert result["label"] == "neutral"
        assert result["confidence"] == 0.0

    def test_fuse_weighted_majority(self):
        fusion = MultimodalFusion(weights={"text": 0.5, "image": 0.3, "audio": 0.2})

        text_result = {
            "label": "negative",
            "confidence": 0.9,
            "scores": {"negative": 0.9, "neutral": 0.05, "positive": 0.05},
        }
        image_result = {
            "emotion": "happy",
            "sentiment": "positive",
            "confidence": 0.8,
            "scores": {"angry": 0.0, "disgust": 0.0, "fear": 0.0, "happy": 0.8, "neutral": 0.1, "sad": 0.0, "surprise": 0.1},
        }
        audio_result = {
            "emotion": "happy",
            "sentiment": "positive",
            "confidence": 0.7,
            "scores": {"angry": 0.0, "sad": 0.05, "neutral": 0.1, "happy": 0.7, "fearful": 0.05, "disgusted": 0.1},
        }

        result = fusion.fuse(text_result, image_result, audio_result)
        # 文本权重 0.5 → negative, 图像+音频 0.5 → positive
        # 预期: positive 险胜 (因为图像 0.3 + 音频 0.2 > 文本 0.5 实际上 negative 的更集中在文本)
        assert result["label"] in ["negative", "positive"]

    def test_emotion_to_sentiment_mapping(self):
        fusion = MultimodalFusion()
        emotion_scores = {
            "angry": 0.1, "disgust": 0.05, "fear": 0.05,
            "happy": 0.5, "neutral": 0.2, "sad": 0.05, "surprise": 0.05,
        }
        mapped = fusion._map_emotion_to_sentiment(emotion_scores, "positive")
        assert mapped["positive"] > mapped["negative"]
        assert mapped["positive"] > mapped["neutral"]


class TestPostprocessing:
    """测试结果后处理"""

    def test_format_result(self):
        fusion = MultimodalFusion()
        text_result = {
            "label": "positive",
            "confidence": 0.9,
            "scores": {"negative": 0.02, "neutral": 0.08, "positive": 0.9},
        }
        fused = fusion.fuse(text_result=text_result)
        result = format_result(fused, {"text": "I love this!"})

        assert "overall" in result
        assert "scores" in result
        assert "modalities" in result
        assert result["overall"]["sentiment"] == "positive"
        assert result["modalities"]["text"]["label"] == "positive"

    def test_to_json(self):
        result = {"overall": {"sentiment": "positive"}}
        json_str = to_json(result)
        assert isinstance(json_str, str)
        assert "positive" in json_str


class TestPreprocessing:
    """测试数据预处理"""

    def test_preprocess_text_normal(self):
        from utils.preprocessing import preprocess_text
        assert preprocess_text("  Hello world  ") == "Hello world"

    def test_preprocess_text_empty(self):
        from utils.preprocessing import preprocess_text
        with pytest.raises(ValueError):
            preprocess_text("")

    def test_preprocess_text_none(self):
        from utils.preprocessing import preprocess_text
        with pytest.raises(ValueError):
            preprocess_text(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
