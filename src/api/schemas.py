"""Pydantic models for API requests and responses."""

from typing import List
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to analyze for sentiment",
        example="This movie was absolutely fantastic!"
    )


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    text: str = Field(..., description="Input text (truncated if > 100 chars)")
    sentiment: str = Field(..., description="Predicted sentiment: positive or negative")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This movie was absolutely fantastic!",
                "sentiment": "positive",
                "confidence": 0.95,
                "processing_time_ms": 45.23
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of texts to analyze (max 100)",
        example=["Great product!", "Terrible service", "It was okay"]
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total_count: int = Field(..., description="Total number of predictions")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="API status: healthy or unhealthy")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    version: str = Field(..., description="API version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }
