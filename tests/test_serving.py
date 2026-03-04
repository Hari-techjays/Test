import pytest
from pydantic import ValidationError


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
