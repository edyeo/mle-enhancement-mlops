"""
Main FastAPI application for MLOps Endpoint API.
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from .config import Config
from .services.model_service import ModelService
from .services.feature_service import FeatureService
from .api import health, predict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global services
model_service: ModelService = None
feature_service: FeatureService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global model_service, feature_service
    
    # Startup
    logger.info("Starting MLOps Endpoint API...")
    
    try:
        # Load configuration
        config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
        config = Config.load_from_yaml(config_path)
        
        # Initialize services
        model_service = ModelService(config)
        feature_service = FeatureService(config)
        
        # Load model
        logger.info("Loading model...")
        model_service.load_model()
        logger.info("Model loaded successfully")
        
        logger.info("MLOps Endpoint API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down MLOps Endpoint API...")


# Create FastAPI app
app = FastAPI(
    title="MLOps Endpoint API",
    description="MLOps platform endpoint for model serving",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency functions
def get_model_service() -> ModelService:
    """Get model service dependency."""
    if model_service is None:
        raise RuntimeError("Model service not initialized")
    return model_service


def get_feature_service() -> FeatureService:
    """Get feature service dependency."""
    if feature_service is None:
        raise RuntimeError("Feature service not initialized")
    return feature_service


# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(predict.router, tags=["prediction"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "MLOps Endpoint API",
        "version": "1.0.0",
        "status": "running"
    }


# Override the predict endpoints to inject dependencies
@app.post("/predict")
async def predict_endpoint(
    request: predict.PredictRequest,
    model_svc: ModelService = Depends(get_model_service),
    feature_svc: FeatureService = Depends(get_feature_service)
):
    """Prediction endpoint with injected dependencies."""
    return await predict.predict(request, model_svc, feature_svc)


@app.post("/invoke")
async def invoke_endpoint(
    request: predict.PredictRequest,
    model_svc: ModelService = Depends(get_model_service),
    feature_svc: FeatureService = Depends(get_feature_service)
):
    """Invoke endpoint with injected dependencies."""
    return await predict.invoke(request, model_svc, feature_svc)


@app.get("/health")
async def health_endpoint():
    """Health check endpoint."""
    return await health.health_check()


if __name__ == "__main__":
    import uvicorn
    
    # Load configuration for development
    config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
    config = Config.load_from_yaml(config_path)
    
    uvicorn.run(
        "main:app",
        host=config.api_host,
        port=config.api_port,
        reload=True
    )
