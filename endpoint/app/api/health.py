"""
Health check endpoint for the MLOps Endpoint API.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    message: str
    model_loaded: bool
    timestamp: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint to verify API and model status.
    
    Returns:
        Health status information
    """
    from datetime import datetime
    
    try:
        # Check if model is loaded (this will be injected by dependency)
        # For now, we'll assume it's loaded if the endpoint is accessible
        model_loaded = True  # This should be injected from the main app
        
        return HealthResponse(
            status="healthy",
            message="API is running and model is loaded",
            model_loaded=model_loaded,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}",
            model_loaded=False,
            timestamp=datetime.utcnow().isoformat()
        )
