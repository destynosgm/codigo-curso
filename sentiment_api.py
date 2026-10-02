"""
INSTALLATION REQUIREMENTS:
pip install fastapi uvicorn transformers torch pydantic

"""

# FastAPI Sentiment Analysis Service with Transformer Model
# This script creates a REST API for sentiment analysis using Hugging Face transformers

# Standard library imports
import logging
from typing import Dict, List, Optional
from datetime import datetime

# Third-party imports
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn

# Hugging Face transformers for the sentiment analysis model
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Configure logging to track application behavior
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE
# ================================

class SentimentRequest(BaseModel):
    """
    Request model for sentiment analysis.
    Validates input data and provides documentation for the API.
    """
    text: str = Field(
        ...,  # Required field
        min_length=1,
        max_length=5000,  # Prevent extremely long texts that could cause memory issues
        description="The text to analyze for sentiment",
        example="I love this new product! It's amazing."
    )
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        """
        Custom validator to ensure text is not just whitespace.
        
        Args:
            v: The text value to validate
            
        Returns:
            str: The validated and cleaned text
            
        Raises:
            ValueError: If text is empty or only whitespace
        """
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v.strip()
    


class BatchSentimentRequest(BaseModel):
    """
    Request model for batch sentiment analysis.
    Allows processing multiple texts in a single request.
    """
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,  # Limit batch size to prevent server overload
        description="List of texts to analyze for sentiment"
    )
    
    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        """
        Validate that all texts in the batch are non-empty.
        
        Args:
            v: List of texts to validate
            
        Returns:
            List[str]: Validated list of texts
        """
        cleaned_texts = []
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty')
            if len(text) > 280:
                raise ValueError(f'Text at index {i} exceeds maximum length of 5000 characters')
            cleaned_texts.append(text.strip())
        return cleaned_texts


class SentimentResponse(BaseModel):
    """
    Response model for sentiment analysis results.
    """
    text: str = Field(description="The analyzed text")
    sentiment: str = Field(description="Predicted sentiment (POSITIVE/NEGATIVE)")
    confidence: float = Field(description="Confidence score between 0 and 1")
    processing_time: float = Field(description="Time taken to process in seconds")
    timestamp: datetime = Field(description="When the analysis was performed")


class BatchSentimentResponse(BaseModel):
    """
    Response model for batch sentiment analysis results.
    """
    results: List[SentimentResponse] = Field(description="List of sentiment analysis results")
    total_processed: int = Field(description="Total number of texts processed")
    average_processing_time: float = Field(description="Average processing time per text")


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    """
    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether the ML model is loaded")
    timestamp: datetime = Field(description="Current server time")
    version: str = Field(description="API version")


# ================================
# SENTIMENT ANALYZER CLASS
# ================================

class SentimentAnalyzer:
    """
    Encapsulates the sentiment analysis model and related functionality.
    This class handles model loading, caching, and prediction logic.
    """
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment"):
        """
        Initialize the sentiment analyzer with a pre-trained model.
        
        Args:
            model_name (str): Name of the Hugging Face model to use.
                            Default uses a RoBERTa model trained on Twitter data.
        """
        self.model_name = model_name
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        self.device = self._get_device()
        
        logger.info(f"Initializing SentimentAnalyzer with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
    def _get_device(self) -> str:
        """
        Determine the best available device for model inference.
        
        Returns:
            str: Device string ('cuda' if GPU available, otherwise 'cpu')
        """
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():  # For Apple Silicon Macs
            return "mps"
        else:
            return "cpu"
    
    def load_model(self):
        """
        Load the transformer model and tokenizer.
        This method is called during application startup.
        
        Raises:
            Exception: If model loading fails
        """
        try:
            logger.info("Loading sentiment analysis model...")
            
            # Load tokenizer and model separately for more control
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create the pipeline with loaded components
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,  # 0 for GPU, -1 for CPU
                return_all_scores=True  # Get scores for all labels
            )
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def predict(self, text: str) -> Dict:
        """
        Perform sentiment prediction on a single text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict: Contains sentiment, confidence, and processing time
            
        Raises:
            RuntimeError: If model is not loaded
            Exception: If prediction fails
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = datetime.now()
        
        try:
            # Get prediction from the model
            # The pipeline returns a list of dictionaries with label and score
            results = self.pipeline(text)
            
            # Extract the result (results is a list with one element for single text)
            prediction = results[0]
            
            # Find the prediction with highest confidence
            best_prediction = max(prediction, key=lambda x: x['score'])
            
            # Map model labels to human-readable sentiment
            # Different models may use different label formats
            sentiment = self._map_label_to_sentiment(best_prediction['label'])
            confidence = best_prediction['score']
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "sentiment": sentiment,
                "confidence": round(confidence, 4),
                "processing_time": round(processing_time, 4)
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise Exception(f"Sentiment analysis failed: {str(e)}")
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Perform sentiment prediction on multiple texts efficiently.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List[Dict]: List of prediction results
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = datetime.now()
        
        try:
            # Process all texts in batch for efficiency
            results = self.pipeline(texts)
            
            predictions = []
            for i, (text, result) in enumerate(zip(texts, results)):
                # Find the prediction with highest confidence for each text
                best_prediction = max(result, key=lambda x: x['score'])
                
                sentiment = self._map_label_to_sentiment(best_prediction['label'])
                confidence = best_prediction['score']
                
                predictions.append({
                    "sentiment": sentiment,
                    "confidence": round(confidence, 4),
                    "text": text
                })
            
            total_time = (datetime.now() - start_time).total_seconds()
            avg_time = total_time / len(texts)
            
            # Add timing information to each prediction
            for pred in predictions:
                pred["processing_time"] = round(avg_time, 4)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise Exception(f"Batch sentiment analysis failed: {str(e)}")
    
    def _map_label_to_sentiment(self, label: str) -> str:
        """
        Map model-specific labels to standardized sentiment labels.
        
        Args:
            label (str): Raw label from the model
            
        Returns:
            str: Standardized sentiment ('POSITIVE' or 'NEGATIVE')
        """
        # Different models use different labeling schemes
        # This function normalizes them to a consistent format
        label_lower = label.lower()
        
        if any(pos_word in label_lower for pos_word in ['positive', 'pos', 'label_2']):
            return "POSITIVE"
        elif any(neg_word in label_lower for neg_word in ['negative', 'neg', 'label_0']):
            return "NEGATIVE"
        elif any(neg_word in label_lower for neg_word in ['NUTRAL', 'NEU', 'label_1']): 
            # Some models have neutral category, we'll map to the closest sentiment
            # You might want to return "NEUTRAL" if your use case requires it
            return "NEUTRAL"    
        else:   
            # Fallback: return the original label if we can't map it
            logger.warning(f"Unknown sentiment label: {label}")
            return label.upper()
    
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded and ready for predictions.
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self.pipeline is not None


# ================================
# APPLICATION LIFECYCLE MANAGEMENT
# ================================

# Initialize the sentiment analyzer (model will be loaded on startup)
analyzer = SentimentAnalyzer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan events.
    This replaces the deprecated @app.on_event decorators.
    """
    # Startup
    logger.info("Starting up the application...")
    try:
        analyzer.load_model()
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise
    
    yield  # Application is running
    
    # Shutdown
    logger.info("Shutting down the application...")
    # Add any cleanup code here if needed

# ================================
# FASTAPI APPLICATION SETUP
# ================================

# Initialize the FastAPI application with lifespan management
app = FastAPI(
    title="Sentiment Analysis API",
    description="A REST API for sentiment analysis using transformer models",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI documentation
    redoc_url="/redoc",  # ReDoc documentation
    lifespan=lifespan  # New lifespan management
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================
# API ENDPOINTS
# ================================

@app.get(
    "/health", 
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API service and ML model are healthy and ready"
)
async def health_check():
    """
    Health check endpoint to verify service status.
    
    Returns:
        HealthResponse: Service health information
    """
    return HealthResponse(
        status="healthy" if analyzer.is_loaded() else "unhealthy",
        model_loaded=analyzer.is_loaded(),
        timestamp=datetime.now(),
        version="1.0.0"
    )


@app.post(
    "/predict", 
    response_model=SentimentResponse,
    summary="Analyze Sentiment",
    description="Analyze the sentiment of a single text input"
)
async def predict_sentiment(request: SentimentRequest):
    """
    Analyze sentiment for a single text.
    
    Args:
        request (SentimentRequest): Request containing text to analyze
        
    Returns:
        SentimentResponse: Sentiment analysis result
        
    Raises:
        HTTPException: If analysis fails or model is not available
    """
    try:
        # Perform the sentiment analysis
        result = analyzer.predict(request.text)
        
        # Create and return the response
        return SentimentResponse(
            text=request.text,
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            processing_time=result["processing_time"],
            timestamp=datetime.now()
        )
        
    except RuntimeError as e:
        # Model not loaded
        logger.error(f"Model not available: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment analysis model is not available"
        )
    except Exception as e:
        # Other errors
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze sentiment: {str(e)}"
        )


@app.post(
    "/predict/batch", 
    response_model=BatchSentimentResponse,
    summary="Analyze Sentiment (Batch)",
    description="Analyze the sentiment of multiple texts in a single request"
)
async def predict_sentiment_batch(request: BatchSentimentRequest):
    """
    Analyze sentiment for multiple texts in batch.
    
    Args:
        request (BatchSentimentRequest): Request containing texts to analyze
        
    Returns:
        BatchSentimentResponse: Batch sentiment analysis results
        
    Raises:
        HTTPException: If analysis fails or model is not available
    """
    try:
        # Perform batch sentiment analysis
        predictions = analyzer.predict_batch(request.texts)
        
        # Create response objects
        results = []
        for pred in predictions:
            results.append(SentimentResponse(
                text=pred["text"],
                sentiment=pred["sentiment"],
                confidence=pred["confidence"],
                processing_time=pred["processing_time"],
                timestamp=datetime.now()
            ))
        
        # Calculate average processing time
        avg_time = sum(r.processing_time for r in results) / len(results)
        
        return BatchSentimentResponse(
            results=results,
            total_processed=len(results),
            average_processing_time=round(avg_time, 4)
        )
        
    except RuntimeError as e:
        logger.error(f"Model not available: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment analysis model is not available"
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze sentiments: {str(e)}"
        )


@app.get(
    "/", 
    summary="Root",
    description="API information and links to documentation"
)
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        dict: Basic API information and links
    """
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "single_prediction": "/predict",
            "batch_prediction": "/predict/batch"
        }
    }


# ================================
# APPLICATION RUNNER
# ================================

if __name__ == "__main__":
    """
    Run the application directly with uvicorn.
    This is useful for development and testing.
    """
    # Configuration for the development server
    uvicorn.run(
        "sentiment_api:app",  # Import string format for reload to work
        host="0.0.0.0",       # Listen on all interfaces
        port=8000,            # Default port
        reload=True,          # Auto-reload on code changes (development only)
        log_level="info"      # Logging level
    )

# ================================
# USAGE INSTRUCTIONS
# ================================
