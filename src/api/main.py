"""FastAPI application for sentiment analysis predictions."""

import logging
import pickle
import time
from typing import Dict, List
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest

from src.data.preprocess import TextPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'predictions_total',
    'Total number of predictions',
    ['sentiment']
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds'
)
REQUEST_COUNTER = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="ML-powered sentiment analysis for text classification",
    version="1.0.0"
)

# Global variables for model and preprocessor
model = None
vectorizer = None
preprocessor = None


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    text: str
    sentiment: str
    confidence: float
    processing_time_ms: float


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction endpoint."""
    texts: List[str] = Field(..., min_items=1, max_items=100)


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction endpoint."""
    predictions: List[PredictionResponse]
    total_count: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    version: str
    
    model_config = {"protected_namespaces": ()}


def load_model():
    """Load the trained model and vectorizer."""
    global model, vectorizer, preprocessor
    
    try:
        model_path = Path("models/trained/sentiment_model.pkl")
        vectorizer_path = Path("models/trained/vectorizer.pkl")
        
        logger.info(f"Loading model from {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        preprocessor = TextPreprocessor()
        
        logger.info("Model and vectorizer loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting up Sentiment Analysis API")
    load_model()


@app.get("/")
async def root():
    """Root endpoint."""
    REQUEST_COUNTER.labels(method='GET', endpoint='/', status='200').inc()
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "health": "/health",
        "predict": "/predict", 
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    REQUEST_COUNTER.labels(method='GET', endpoint='/health', status='200').inc()
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict sentiment for a single text.
    
    Args:
        request: PredictionRequest with text field
        
    Returns:
        PredictionResponse with sentiment and confidence
    """
    start_time = time.time()
    
    try:
        if model is None or vectorizer is None:
            REQUEST_COUNTER.labels(method='POST', endpoint='/predict', status='503').inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Preprocess text
        clean_text = preprocessor.preprocess(request.text)
        
        # Vectorize
        X = vectorizer.transform([clean_text])
        
        # Predict
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Map prediction to sentiment
        sentiment = "positive" if prediction == 1 else "negative"
        confidence = float(max(probabilities))
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Update metrics
        PREDICTION_COUNTER.labels(sentiment=sentiment).inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        REQUEST_COUNTER.labels(method='POST', endpoint='/predict', status='200').inc()
        
        logger.info(f"Prediction: {sentiment} ({confidence:.3f}) in {processing_time:.2f}ms")
        
        return PredictionResponse(
            text=request.text[:100] + "..." if len(request.text) > 100 else request.text,
            sentiment=sentiment,
            confidence=confidence,
            processing_time_ms=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNTER.labels(method='POST', endpoint='/predict', status='500').inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict sentiment for multiple texts.
    
    Args:
        request: BatchPredictionRequest with list of texts
        
    Returns:
        BatchPredictionResponse with list of predictions
    """
    try:
        if model is None or vectorizer is None:
            REQUEST_COUNTER.labels(method='POST', endpoint='/batch_predict', status='503').inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        predictions = []
        
        for text in request.texts:
            pred_request = PredictionRequest(text=text)
            pred_response = await predict(pred_request)
            predictions.append(pred_response)
        
        REQUEST_COUNTER.labels(method='POST', endpoint='/batch_predict', status='200').inc()
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNTER.labels(method='POST', endpoint='/batch_predict', status='500').inc()
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
