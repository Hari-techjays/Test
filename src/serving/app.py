"""FastAPI inference service for sentiment classification."""

import logging
import os
import time

import torch
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.model.train import ID2LABEL
from src.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Analysis API",
    description="DistilBERT-based sentiment classifier for retail reviews",
    version="0.1.0",
)

# Prometheus metrics
REQUEST_COUNT = Counter("predict_requests_total", "Total prediction requests", ["endpoint"])
REQUEST_LATENCY = Histogram("predict_latency_seconds", "Prediction latency", ["endpoint"])
PREDICTION_LABELS = Counter("prediction_labels_total", "Predictions by label", ["label"])

# Global model state
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join("models", "sentiment"))


@app.on_event("startup")
def load_model():
    """Load model and tokenizer on startup."""
    global model, tokenizer
    try:
        if os.path.exists(MODEL_DIR):
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            model.eval()
            model.to(device)
            logger.info(f"Model loaded from {MODEL_DIR} on {device}")
        else:
            logger.warning(f"Model directory {MODEL_DIR} not found. Server starting without model.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


def predict_sentiment(text: str) -> PredictResponse:
    """Run inference on a single text."""
    inputs = tokenizer(
        text, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    pred_idx = probs.argmax().item()
    label = ID2LABEL[pred_idx]
    confidence = probs[pred_idx].item()
    probabilities = {ID2LABEL[i]: round(p.item(), 4) for i, p in enumerate(probs)}

    PREDICTION_LABELS.labels(label=label).inc()

    return PredictResponse(label=label, confidence=round(confidence, 4), probabilities=probabilities)


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(status="healthy", model_loaded=model is not None)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predict sentiment for a single review."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    REQUEST_COUNT.labels(endpoint="/predict").inc()
    start = time.time()

    result = predict_sentiment(request.text)

    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)
    return result


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    """Predict sentiment for multiple reviews."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    REQUEST_COUNT.labels(endpoint="/predict/batch").inc()
    start = time.time()

    predictions = [predict_sentiment(text) for text in request.texts]

    REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(time.time() - start)
    return BatchPredictResponse(predictions=predictions)


@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type="text/plain; charset=utf-8")
