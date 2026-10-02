"""
YOLO11 Object Detection FastAPI Server

This script creates a FastAPI web server that loads a YOLO11n model and provides
an endpoint for object detection on base64-encoded images.

Requirements:
- pip install fastapi uvicorn ultralytics pillow python-multipart
- yolo11n.pt model file in the same directory

Usage:
- Run: uvicorn main:app --reload
- Send POST requests to /detect with base64 image data
"""

import base64
import io
import logging
from typing import List, Dict, Any, Optional
import traceback

# FastAPI imports
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Image processing imports
from PIL import Image
import numpy as np

# YOLO imports
from ultralytics import YOLO

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="YOLO12 Object Detection API",
    description="API for object detection using YOLO11n model",
    version="1.0.2"
)

# Add CORS middleware to allow cross-origin requests
# This is useful if you plan to call this API from a web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the loaded YOLO model
model: Optional[YOLO] = None

# Pydantic models for request/response validation
class DetectionRequest(BaseModel):
    """
    Request model for object detection endpoint
    
    Attributes:
        image_base64: Base64 encoded image string (without data:image prefix)
        confidence_threshold: Minimum confidence score for detections (0.0 to 1.0)
        iou_threshold: IoU threshold for Non-Maximum Suppression (0.0 to 1.0)
    """
    image_base64: str = Field(..., description="Base64 encoded image")
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="Confidence threshold")
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0, description="IoU threshold for NMS")

class BoundingBox(BaseModel):
    """
    Bounding box coordinates
    
    Attributes:
        x1, y1: Top-left corner coordinates
        x2, y2: Bottom-right corner coordinates
    """
    x1: float
    y1: float
    x2: float
    y2: float

class Detection(BaseModel):
    """
    Individual detection result
    
    Attributes:
        class_id: Numeric class ID
        class_name: Human-readable class name
        confidence: Detection confidence score (0.0 to 1.0)
        bbox: Bounding box coordinates
    """
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

class DetectionResponse(BaseModel):
    """
    Complete detection response
    
    Attributes:
        success: Whether detection was successful
        detections: List of detected objects
        image_shape: Original image dimensions [height, width, channels]
        processing_time_ms: Time taken for inference in milliseconds
    """
    success: bool
    detections: List[Detection]
    image_shape: List[int]
    processing_time_ms: float
    message: Optional[str] = None

def load_yolo_model(model_path: str = "yolo12n.pt") -> YOLO:
    """
    Load YOLO11n model from file
    
    Args:
        model_path: Path to the YOLO model file
        
    Returns:
        Loaded YOLO model instance
        
    Raises:
        Exception: If model loading fails
    """
    try:
        logger.info(f"Loading YOLO model from: {model_path}")
        
        # Load the YOLO model
        # The YOLO class automatically handles model initialization and weight loading
        model = YOLO(model_path)
        
        logger.info("YOLO model loaded successfully")
        logger.info(f"Model type: {type(model)}")
        
        # Print model information for debugging
        logger.info(f"Model names: {model.names}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise Exception(f"Model loading failed: {str(e)}")

def decode_base64_image(base64_string: str) -> Image.Image:
    """
    Decode base64 string to PIL Image
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL Image object
        
    Raises:
        ValueError: If base64 decoding or image loading fails
    """
    try:
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary (YOLO expects RGB format)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        logger.info(f"Decoded image: {image.size}, mode: {image.mode}")
        
        return image
        
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {str(e)}")
        raise ValueError(f"Invalid base64 image data: {str(e)}")

def process_yolo_results(results, confidence_threshold: float = 0.25) -> List[Detection]:
    """
    Process YOLO detection results into standardized format
    
    Args:
        results: YOLO model prediction results
        confidence_threshold: Minimum confidence score for filtering
        
    Returns:
        List of Detection objects
    """
    detections = []
    
    try:
        # YOLO results is a list, get the first (and typically only) result
        result = results[0]
        
        # Extract boxes, scores, and class IDs from the result
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
            scores = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
            
            # Get class names from the model
            class_names = result.names
            
            logger.info(f"Found {len(boxes)} raw detections")
            
            # Process each detection
            for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                # Filter by confidence threshold
                if score >= confidence_threshold:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box
                    
                    # Create detection object
                    detection = Detection(
                        class_id=int(class_id),
                        class_name=class_names[class_id],
                        confidence=float(score),
                        bbox=BoundingBox(
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x2),
                            y2=float(y2)
                        )
                    )
                    
                    detections.append(detection)
                    
        logger.info(f"Filtered to {len(detections)} detections above confidence threshold")
        
    except Exception as e:
        logger.error(f"Error processing YOLO results: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise Exception(f"Failed to process detection results: {str(e)}")
    
    return detections

@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler
    Loads the YOLO model when the server starts
    """
    global model
    
    try:
        logger.info("Starting up FastAPI application...")
        
        # Load the YOLO model
        model = load_yolo_model("yolo12n.pt")
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        # In a production environment, you might want to raise the exception
        # to prevent the server from starting with a broken model
        raise e

@app.get("/")
async def root():
    """
    Root endpoint for health check
    
    Returns:
        Simple status message
    """
    return {
        "message": "YOLO11 Object Detection API", 
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Detailed health status including model status
    """
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_type": "YOLO11n" if model is not None else None
    }

#@app.post("/detect", response_model=DetectionResponse)
#async def detect_objects(request: DetectionRequest):
#    """
#    Main object detection endpoint
#    
#    This endpoint accepts a base64-encoded image and returns detected objects
#    with their bounding boxes, class names, and confidence scores.
#    
#    Args:
#        request: DetectionRequest containing base64 image and parameters
#        
#    Returns:
#        DetectionResponse with detection results
#        
#    Raises:
#        HTTPException: If detection fails or model is not loaded
#    """
#    import time
#    
#    # Check if model is loaded
#    if model is None:
#        logger.error("Model is not loaded")
#        raise HTTPException(
#            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#            detail="YOLO model is not loaded. Please check server logs."
#        )
#    
#    start_time = time.time()
#    
#    try:
#        logger.info("Starting object detection...")
#        
#        # Step 1: Decode base64 image
#        logger.info("Decoding base64 image...")
#        image = decode_base64_image(request.image_base64)
#        
#        # Get image dimensions
#        image_shape = [image.height, image.width, len(image.getbands())]
#        logger.info(f"Image shape: {image_shape}")
#        
#        # Step 2: Run YOLO inference
#        logger.info("Running YOLO inference...")
#        
#        # Perform inference
#        # conf: confidence threshold for predictions
#        # iou: IoU threshold for Non-Maximum Suppression
#        results = model(
#            image, 
#            conf=request.confidence_threshold,
#            iou=request.iou_threshold,
#            verbose=False  # Suppress YOLO's verbose output
#        )
#        
#        logger.info("YOLO inference completed")
#        
#        # Step 3: Process results
#        logger.info("Processing detection results...")
#        detections = process_yolo_results(results, request.confidence_threshold)
#        
#        # Calculate processing time
#        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
#        
#        logger.info(f"Detection completed in {processing_time:.2f}ms with {len(detections)} objects")
#        
#        # Step 4: Return results
#        return DetectionResponse(
#            success=True,
#            detections=detections,
#            image_shape=image_shape,
#            processing_time_ms=processing_time,
#            message=f"Successfully detected {len(detections)} objects"
#        )
#        
#    except ValueError as e:
#        # Handle base64 decoding errors
#        logger.error(f"Validation error: {str(e)}")
#        raise HTTPException(
#            status_code=status.HTTP_400_BAD_REQUEST,
#            detail=f"Invalid input data: {str(e)}"
#        )
#        
#    except Exception as e:
#        # Handle any other errors
#        logger.error(f"Detection error: {str(e)}")
#        logger.error(f"Traceback: {traceback.format_exc()}")
#        raise HTTPException(
#            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#            detail=f"Detection failed: {str(e)}"
#        )

class multiDetectionRequest(BaseModel):
    """
    Request model for multi-image object detection endpoint
    
    Attributes:
        images_base64: List of base64 encoded image strings (without data:image prefix)
        confidence_threshold: Minimum confidence score for detections (0.0 to 1.0)
        iou_threshold: IoU threshold for Non-Maximum Suppression (0.0 to 1.0)
    """
    images_base64: List[str] = Field(..., description="List of base64 encoded images")
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="Confidence threshold")
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0, description="IoU threshold for NMS")

import time

@app.post("/multi-detect", response_model=List[DetectionResponse]) 
async def multi_detect_objects(request: multiDetectionRequest): 
    """ 
    Multi-image object detection endpoint
    This endpoint accepts a list of base64-encoded images and returns detected objects
    with their bounding boxes, class names, and confidence scores for each image."""

    responses = []
    
    for idx, base64_image in enumerate(request.images_base64):
        
        try:
            logger.info(f"Processing image {idx + 1}/{len(request.images_base64)}")
            
            # Step 1: Decode base64 image
            image = decode_base64_image(base64_image)
            
            # Get image dimensions
            image_shape = [image.height, image.width, len(image.getbands())]
            logger.info(f"Image shape: {image_shape}")
            start_time = time.time()

            # Step 2: Run YOLO inference
            results = model(
                image, 
                conf=request.confidence_threshold,
                iou=request.iou_threshold,
                verbose=False

            )
            
            logger.info("YOLO inference completed")
            processing_time_ms = (time.time() - start_time) * 1000 
            logger.info(f"Detection completed in {processing_time_ms:.2f}ms")

            # Step 3: Process results
            detections = process_yolo_results(results, request.confidence_threshold)
            
            
            # Step 4: Create response for this image
            response = DetectionResponse(
                success=True,
                detections=detections,
                image_shape=image_shape,
                processing_time_ms=processing_time_ms,  # Obligatorio
                message=f"Successfully detected {len(detections)} objects"
            )
            
            responses.append(response)
            
        

        except ValueError as e:
            logger.error(f"Validation error for image {idx + 1}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid input data for image {idx + 1}: {str(e)}"
            )
        
        except Exception as e:
            logger.error(f"Detection error for image {idx + 1}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Detection failed for image {idx + 1}: {str(e)}"
            )   
        
    return responses    

@app.get("/model-info")
async def get_model_info():
    """
    Get information about the loaded model
    
    Returns:
        Model information including available classes
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="YOLO model is not loaded"
        )
    
    return {
        "model_type": "YOLO11n",
        "classes": model.names,
        "num_classes": len(model.names)
    }

# Entry point for running the server directly
if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI application
    # host: IP address to bind to (0.0.0.0 allows external connections)
    # port: Port number to listen on
    # reload: Automatically reload on code changes (development only)
    uvicorn.run(
        "YOLO_fastAPI:app",  # module:app_instance
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info"
    )


"""
Cambiar el modelo a un modelo mas poderoso, poder mandar mas de una imagen, aumentara la confianza utilizando un modelo mas poderoso?
"""