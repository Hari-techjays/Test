from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import src.serving.app as serving_app_module


class TestSchemas:
    def test_predict_request_valid(self):
        from src.serving.schemas import PredictRequest

        req = PredictRequest(text="This product is great!")
        assert req.text == "This product is great!"

    def test_predict_request_empty_text_fails(self):
        from src.serving.schemas import PredictRequest

        with pytest.raises(ValidationError):
            PredictRequest(text="")

    def test_predict_response_structure(self):
        from src.serving.schemas import PredictResponse

        resp = PredictResponse(
            label="positive",
            confidence=0.95,
            probabilities={"positive": 0.95, "neutral": 0.03, "negative": 0.02},
        )
        assert resp.label == "positive"
        assert resp.confidence == 0.95

    def test_batch_predict_request(self):
        from src.serving.schemas import BatchPredictRequest

        req = BatchPredictRequest(texts=["good", "bad"])
        assert len(req.texts) == 2

    def test_batch_predict_request_empty_fails(self):
        from src.serving.schemas import BatchPredictRequest

        with pytest.raises(ValidationError):
            BatchPredictRequest(texts=[])


class TestApp:
    def test_health_endpoint(self):
        with patch.object(serving_app_module, "model", None), \
             patch.object(serving_app_module, "tokenizer", None):
            from src.serving.app import app

            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "model_loaded" in data

    def test_predict_without_model_returns_503(self):
        with patch.object(serving_app_module, "model", None), \
             patch.object(serving_app_module, "tokenizer", None):
            from src.serving.app import app

            client = TestClient(app)
            response = client.post("/predict", json={"text": "great product"})
            assert response.status_code == 503

    def test_predict_with_mock_model(self):
        import torch

        mock_model = MagicMock()
        mock_logits = torch.tensor([[0.1, 0.2, 0.9]])
        mock_model.return_value = MagicMock(logits=mock_logits)
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.device = torch.device("cpu")

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 2307, 3580, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }

        with patch.object(serving_app_module, "model", mock_model), \
             patch.object(serving_app_module, "tokenizer", mock_tokenizer):
            from src.serving.app import app

            client = TestClient(app)
            response = client.post("/predict", json={"text": "great product"})
            assert response.status_code == 200
            data = response.json()
            assert "label" in data
            assert "confidence" in data
            assert "probabilities" in data

    def test_metrics_endpoint(self):
        with patch.object(serving_app_module, "model", None), \
             patch.object(serving_app_module, "tokenizer", None):
            from src.serving.app import app

            client = TestClient(app)
            response = client.get("/metrics")
            assert response.status_code == 200
