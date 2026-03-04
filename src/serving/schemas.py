"""Pydantic schemas for the inference API."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Review text to classify")


class PredictResponse(BaseModel):
    label: str = Field(..., description="Predicted sentiment label")
    confidence: float = Field(..., description="Confidence score for the predicted label")
    probabilities: dict[str, float] = Field(
        ..., description="Probability for each sentiment class"
    )


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="List of review texts to classify")


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
