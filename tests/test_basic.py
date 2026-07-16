"""
核心回归测试
运行: python -m pytest tests/ -v
"""

import numpy as np
import pytest
import sys
import torch
import torch.nn as nn
from types import SimpleNamespace
from torch.utils.data import DataLoader

from data.mosei_sdk_dataset import MOSEISDKDataset, REQUIRED_CSD
from models.covarep_adapter import CovarepAdapter
from models.audio_encoder import AudioEncoder
from models.fusion import MultimodalFusion
from models.multimodal_model import MultimodalSentimentModel
from models.visual_encoder import VisualEncoder
from inference import MultimodalPredictor
from trainers.metrics import compute_metrics
from trainers.trainer import Trainer
from utils.postprocessing import format_result, to_json


class TestFusion:
    def test_forward_all_modalities(self):
        fusion = MultimodalFusion()
        features = [torch.randn(3, 768) for _ in range(3)]

        result = fusion(*features)

        assert result.shape == (3, 1)
        assert torch.isfinite(result).all()

    @pytest.mark.parametrize(
        ("use_image", "use_audio"),
        [(True, False), (False, True)],
    )
    def test_forward_two_modalities(self, use_image, use_audio):
        fusion = MultimodalFusion()
        features = [torch.randn(2, 768) for _ in range(3)]

        result = fusion(
            *features, use_image=use_image, use_audio=use_audio
        )

        assert result.shape == (2, 1)
        assert torch.isfinite(result).all()

    def test_requires_auxiliary_modality(self):
        fusion = MultimodalFusion()
        features = [torch.randn(1, 768) for _ in range(3)]

        with pytest.raises(RuntimeError, match="auxiliary"):
            fusion(*features, use_image=False, use_audio=False)


class TestFeatureAdapters:
    def test_covarep_ignores_nonfinite_values_when_pooling(self):
        adapter = CovarepAdapter(input_dim=3, hidden_dim=3)
        adapter.mlp = nn.Identity()
        features = np.array(
            [
                [1.0, -np.inf, np.nan],
                [3.0, 5.0, np.inf],
            ],
            dtype=np.float32,
        )

        result = adapter([features])

        assert result.shape == (1, 3)
        assert torch.allclose(result, torch.tensor([[2.0, 5.0, 0.0]]))
        assert torch.isfinite(result).all()

    def test_covarep_validates_feature_dimension(self):
        adapter = CovarepAdapter(input_dim=3, hidden_dim=3)

        with pytest.raises(ValueError, match="应为 3 维"):
            adapter([np.zeros((2, 4), dtype=np.float32)])

    def test_visual_missing_sample_uses_detected_batch_dimension(self):
        encoder = VisualEncoder(input_dim=4, hidden_dim=6)
        features = np.array(
            [[1.0, 2.0, np.inf], [3.0, 4.0, 6.0]],
            dtype=np.float32,
        )

        result = encoder([None, features])

        assert result.shape == (2, 6)
        assert encoder.input_adapter is not None
        assert encoder.input_adapter.in_features == 3
        assert torch.isfinite(result).all()

    def test_trainer_syncs_lazily_created_adapter_parameters(self):
        encoder = VisualEncoder(input_dim=4, hidden_dim=6)
        trainer = Trainer.__new__(Trainer)
        trainer.model = encoder
        trainer.optimizer = torch.optim.AdamW(encoder.parameters())

        encoder([np.ones((2, 3), dtype=np.float32)])
        adapter_param_ids = {id(p) for p in encoder.input_adapter.parameters()}
        optimizer_param_ids_before = {
            id(p)
            for group in trainer.optimizer.param_groups
            for p in group["params"]
        }

        assert adapter_param_ids.isdisjoint(optimizer_param_ids_before)

        trainer._sync_optimizer_parameters()
        optimizer_param_ids_after = {
            id(p)
            for group in trainer.optimizer.param_groups
            for p in group["params"]
        }
        assert adapter_param_ids <= optimizer_param_ids_after

    def test_visual_adapter_restores_from_checkpoint_state(self):
        original = VisualEncoder(input_dim=4, hidden_dim=6)
        original([np.ones((2, 3), dtype=np.float32)])
        state_dict = original.state_dict()

        restored = VisualEncoder(input_dim=4, hidden_dim=6)
        restored.load_state_dict(state_dict)

        assert restored.input_adapter is not None
        assert restored.input_adapter.in_features == 3
        assert torch.allclose(
            restored.input_adapter.weight,
            original.input_adapter.weight,
        )

    def test_audio_encoder_normalizes_missing_and_nonfinite_waveforms(self):
        encoder = AudioEncoder.__new__(AudioEncoder)
        nn.Module.__init__(encoder)
        encoder.sample_rate = 16_000

        class _FeatureExtractor:
            def __call__(self, values, **kwargs):
                assert values.shape == (2, 3)
                assert np.isfinite(values).all()
                assert np.all(values[0] == 0.0)
                return {"input_values": torch.as_tensor(values)}

        class _Wav2Vec(nn.Module):
            device = torch.device("cpu")

            def forward(self, input_values):
                batch = input_values.shape[0]
                hidden = torch.ones(batch, 2, 4)
                return type("Output", (), {"last_hidden_state": hidden})()

        encoder.feature_extractor = _FeatureExtractor()
        encoder.wav2vec2 = _Wav2Vec()

        result = encoder([None, np.array([1.0, np.inf, 3.0])])

        assert result.shape == (2, 4)
        assert torch.isfinite(result).all()


class _PredictOnlyModel(MultimodalSentimentModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.received = None

    def forward(self, texts, visual_inputs, audio_inputs):
        self.received = (texts, visual_inputs, audio_inputs)
        return torch.tensor([[0.75]])


class TestRouting:
    def test_modality_detection_handles_none_and_torch_tensor(self):
        assert MultimodalSentimentModel._detect_visual_type(None) == "none"
        assert MultimodalSentimentModel._detect_audio_type(None) == "none"
        assert (
            MultimodalSentimentModel._detect_audio_type([torch.ones(10)])
            == "waveform"
        )

    def test_predict_does_not_invent_missing_audio(self):
        model = _PredictOnlyModel()

        result = model.predict(text="hello")

        assert model.received == (["hello"], [""], [None])
        assert result["label"] == "positive"

    def test_checkpoint_selection_uses_lowest_mae(self, tmp_path, monkeypatch):
        import inference

        monkeypatch.setattr(inference, "CHECKPOINT_DIR", str(tmp_path))
        for name in [
            "best_full_epoch001_mae0.5000.pt",
            "best_full_epoch002_mae0.3000.pt",
            "best_text_epoch001_mae0.1000.pt",
        ]:
            (tmp_path / name).touch()

        selected = MultimodalPredictor._find_best_checkpoint("full")

        assert selected.endswith("best_full_epoch002_mae0.3000.pt")


class TestPostprocessing:
    def test_format_result(self):
        fusion_result = {
            "label": "positive",
            "confidence": 0.9,
            "scores": {"negative": 0.02, "neutral": 0.08, "positive": 0.9},
            "per_modality": {
                "text": {
                    "label": "positive",
                    "sentiment": "positive",
                    "confidence": 0.9,
                }
            },
        }

        result = format_result(fusion_result, {"text": "I love this!"})

        assert result["overall"]["sentiment"] == "positive"
        assert result["modalities"]["text"]["label"] == "positive"

    def test_to_json(self):
        result = {"overall": {"sentiment": "positive"}}
        json_str = to_json(result)
        assert isinstance(json_str, str)
        assert "positive" in json_str


class TestMetrics:
    def test_constant_predictions_have_finite_correlation(self):
        metrics = compute_metrics(
            np.ones((3, 1), dtype=np.float32),
            np.ones((3, 1), dtype=np.float32),
        )

        assert metrics["corr"] == 0.0


class TestTrainer:
    def test_test_loop_detaches_parameter_views(self):
        class _Dataset:
            def __len__(self):
                return 2

            def __getitem__(self, index):
                return {
                    "text": "x",
                    "visual": None,
                    "audio": None,
                    "label": 1.0,
                }

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.score = nn.Parameter(torch.tensor([[0.5]]))

            def forward(self, texts, visuals, audios):
                return self.score.expand(len(texts), 1)

        trainer = Trainer.__new__(Trainer)
        trainer.model = _Model()
        trainer.test_loader = DataLoader(
            _Dataset(),
            batch_size=2,
            collate_fn=Trainer._collate_fn,
        )

        metrics = trainer.test()

        assert metrics["mae"] == 0.5


class TestMOSEIDataLoading:
    @staticmethod
    def _fake_sequences():
        seg_id = "video[0]"

        def sequence(features):
            return SimpleNamespace(
                data={seg_id: {"features": features}}
            )

        return {
            "CMU_MOSEI_TimestampedWords": sequence(
                np.array([[b"hello"]], dtype=object)
            ),
            "CMU_MOSEI_Labels": sequence(
                np.array([[1.0]], dtype=np.float32)
            ),
            "CMU_MOSEI_VisualFacet42": sequence(
                np.zeros((2, 42), dtype=np.float32)
            ),
            "CMU_MOSEI_COVAREP": sequence(
                np.zeros((2, 74), dtype=np.float32)
            ),
        }

    def test_local_loader_aligns_csd_files(self, tmp_path, monkeypatch):
        for filename in REQUIRED_CSD.values():
            (tmp_path / filename).touch()

        calls = {}

        class _Dataset:
            def __init__(self):
                self.computational_sequences = self_sequences

            def align(self, reference):
                calls["reference"] = reference

        self_sequences = self._fake_sequences()
        def make_dataset(path):
            calls["path"] = path
            return _Dataset()

        fake_sdk = SimpleNamespace(
            mmdataset=make_dataset
        )
        monkeypatch.setitem(
            sys.modules,
            "mmsdk",
            SimpleNamespace(mmdatasdk=fake_sdk),
        )

        dataset = MOSEISDKDataset.__new__(MOSEISDKDataset)
        dataset.cache_dir = str(tmp_path)
        dataset.split = "train"
        dataset._get_split_videos = lambda: set()

        samples = dataset._load_local()

        assert calls["path"] == str(tmp_path)
        assert calls["reference"] == "CMU_MOSEI_Labels"
        assert samples[0]["text"] == "hello"
        assert samples[0]["visual"].shape == (2, 42)
        assert samples[0]["audio"].shape == (2, 74)

    def test_remote_loader_downloads_only_required_sequences(self, monkeypatch):
        calls = {}
        self_sequences = self._fake_sequences()

        class _Dataset:
            computational_sequences = self_sequences

            def align(self, reference):
                calls["reference"] = reference

        def make_dataset(recipe, destination):
            calls["recipe"] = recipe
            calls["destination"] = destination
            return _Dataset()

        fake_sdk = SimpleNamespace(
            mmdataset=make_dataset,
            cmu_mosei=SimpleNamespace(
                raw={"words": "words-url", "phones": "phones-url"},
                highlevel={
                    "COVAREP": "covarep-url",
                    "FACET 4.2": "facet-url",
                    "OpenFace_2": "openface-url",
                },
                labels={"All Labels": "labels-url"},
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "mmsdk",
            SimpleNamespace(mmdatasdk=fake_sdk),
        )

        dataset = MOSEISDKDataset.__new__(MOSEISDKDataset)
        dataset.cache_dir = "cache"
        dataset.split = "train"
        dataset._get_split_videos = lambda: set()

        samples = dataset._load_remote()

        assert set(calls["recipe"]) == set(self_sequences)
        assert calls["destination"] == "cache"
        assert calls["reference"] == "CMU_MOSEI_Labels"
        assert len(samples) == 1


class TestPreprocessing:
    def test_preprocess_text_normal(self):
        from utils.preprocessing import preprocess_text

        assert preprocess_text("  Hello world  ") == "Hello world"

    @pytest.mark.parametrize("value", ["", None])
    def test_preprocess_text_rejects_empty(self, value):
        from utils.preprocessing import preprocess_text

        with pytest.raises(ValueError):
            preprocess_text(value)
