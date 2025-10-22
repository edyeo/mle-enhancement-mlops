"""
Prediction endpoint for the MLOps Endpoint API.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class PredictRequest(BaseModel):
    """Prediction request model."""
    # Common fields
    user_id: Optional[str] = None
    
    # For PyTorch models
    features: Optional[list] = None
    
    # For LLM models
    prompt: Optional[str] = None
    text: Optional[str] = None
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None


class PredictResponse(BaseModel):
    """Prediction response model."""
    prediction: Optional[Any] = None
    response: Optional[str] = None
    model_type: str
    status: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    model_service: Any = Depends(lambda: None),  # Will be injected by main app
    feature_service: Any = Depends(lambda: None)  # Will be injected by main app
) -> PredictResponse:
    """
    Prediction endpoint for model inference.
    
    Args:
        request: Prediction request data
        model_service: Injected model service
        feature_service: Injected feature service
        
    Returns:
        Prediction results
    """
    try:
        logger.info(f"Received prediction request for user: {request.user_id}")
        
        # Convert request to dictionary
        request_data = request.dict()
        
        # Feature enrichment (if feature service is available)
        if feature_service:
            enriched_data = feature_service.enrich_features(request_data)
        else:
            enriched_data = request_data
        
        # Run model inference
        if model_service:
            result = model_service.predict(enriched_data)
        else:
            # Mock response for testing
            result = {
                "prediction": [0.5, 0.3, 0.2] if request.features else "Mock LLM response",
                "model_type": "mock",
                "status": "success"
            }
        
        logger.info("Prediction completed successfully")
        return PredictResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/invoke", response_model=PredictResponse)
async def invoke(
    request: PredictRequest,
    model_service: Any = Depends(lambda: None),
    feature_service: Any = Depends(lambda: None)
) -> PredictResponse:
    """
    Alternative prediction endpoint (alias for /predict).
    
    Args:
        request: Prediction request data
        model_service: Injected model service
        feature_service: Injected feature service
        
    Returns:
        Prediction results
    """
    return await predict(request, model_service, feature_service)
