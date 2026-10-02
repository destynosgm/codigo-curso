#!/usr/bin/env python3
"""
Install dependencies:
pip install fastapi uvicorn diffusers torch torchvision pillow transformers accelerate
"""

import logging
import base64
import io
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging to track application behavior
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variable to store the loaded model pipeline
# This will be initialized during application startup
pipeline = None

# Configuration constants
MODEL_ID = "runwayml/stable-diffusion-v1-5"  # You can change this to other SD models
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


class ImageGenerationRequest(BaseModel):
    """
    Pydantic model for validating image generation requests.
    
    This defines the structure and validation rules for incoming API requests.
    """
    prompt: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="Text prompt to generate image from"
    )
    negative_prompt: Optional[str] = Field(
        default="", 
        max_length=1000,
        description="Negative prompt to avoid certain elements"
    )
    num_inference_steps: Optional[int] = Field(
        default=20, 
        ge=1, 
        le=100,
        description="Number of denoising steps (more steps = higher quality, slower)"
    ) 
    guidance_scale: Optional[float] = Field(
        default=7.5, 
        ge=1.0, 
        le=20.0,
        description="How closely to follow the prompt (higher = more adherent)"
    )
    width: Optional[int] = Field(
        default=512, 
        ge=64, 
        le=1024,
        description="Width of generated image (should be multiple of 8)"
    )
    height: Optional[int] = Field(
        default=512, 
        ge=64, 
        le=1024,
        description="Height of generated image (should be multiple of 8)"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible results (None for random)"
    )


class ImageGenerationResponse(BaseModel):
    """
    Pydantic model for image generation responses.
    
    Defines the structure of successful API responses.
    """
    success: bool = Field(description="Whether the generation was successful")
    image_base64: Optional[str] = Field(description="Base64 encoded image data")
    prompt_used: str = Field(description="The prompt that was used")
    generation_params: Dict[str, Any] = Field(description="Parameters used for generation")
    message: Optional[str] = Field(description="Success or error message")


async def load_model():
    """
    Load the Stable Diffusion model pipeline.
    
    This function:
    1. Downloads the model from Hugging Face (if not cached)
    2. Configures it for optimal performance
    3. Loads it into memory (GPU if available)
    
    Returns:
        StableDiffusionPipeline: The loaded pipeline ready for inference
    """
    global pipeline
    
    try:
        logger.info(f"Loading Stable Diffusion model: {MODEL_ID}")
        logger.info(f"Using device: {DEVICE}")
        logger.info(f"Using torch dtype: {TORCH_DTYPE}")
        
        # Load the pipeline from Hugging Face
        # safety_checker=None disables NSFW filter (remove if you want filtering)
        pipeline = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=TORCH_DTYPE,
            safety_checker=None,  # Remove this line to enable NSFW filtering
            requires_safety_checker=False  # Remove this line too if enabling filter
        )
        
        # Use DPMSolverMultistepScheduler for better quality/speed tradeoff
        # This scheduler typically produces good results in fewer steps
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        
        # Move pipeline to appropriate device (GPU if available)
        pipeline = pipeline.to(DEVICE)
        
        # Enable memory efficient attention if using GPU
        if DEVICE == "cuda":
            try:
                # This reduces VRAM usage significantly
                pipeline.enable_attention_slicing()
                # Uncomment the next line if you have limited VRAM (< 8GB)
                # pipeline.enable_sequential_cpu_offload()
                logger.info("Enabled attention slicing for memory efficiency")
            except Exception as e:
                logger.warning(f"Could not enable attention optimizations: {e}")
        
        logger.info("Model loaded successfully!")
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Convert PIL Image to base64 encoded string.
    
    Args:
        image (Image.Image): PIL Image object
        format (str): Image format for encoding (PNG, JPEG, etc.)
    
    Returns:
        str: Base64 encoded string of the image
    """
    # Create a bytes buffer to hold the image data
    buffer = io.BytesIO()
    
    # Save the image to the buffer in the specified format
    image.save(buffer, format=format)
    
    # Get the raw bytes and encode to base64
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return img_base64


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    This function handles startup and shutdown events:
    - Startup: Load the Stable Diffusion model
    - Shutdown: Clean up resources
    """
    # Startup: Load the model
    logger.info("Starting up FastAPI server...")
    try:
        await load_model()
        logger.info("Startup complete!")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise e
    
    # Yield control to the application
    yield
    
    # Shutdown: Clean up resources
    logger.info("Shutting down server...")
    global pipeline
    if pipeline is not None:
        # Move pipeline to CPU and clear CUDA cache if using GPU
        if DEVICE == "cuda":
            pipeline = pipeline.to("cpu")
            torch.cuda.empty_cache()
        pipeline = None
    logger.info("Shutdown complete!")


# Create FastAPI application with lifespan events
app = FastAPI(
    title="Stable Diffusion API",
    description="FastAPI server for Stable Diffusion text-to-image generation",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """
    Root endpoint - basic health check.
    
    Returns:
        dict: Basic API information
    """
    return {
        "message": "Stable Diffusion FastAPI Server",
        "status": "running",
        "model": MODEL_ID,
        "device": DEVICE
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the model is loaded and ready.
    
    Returns:
        dict: Health status information
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded yet. Please wait for initialization."
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": DEVICE,
        "torch_version": torch.__version__
    }


@app.post("/generate", response_model=ImageGenerationResponse)
async def generate_image(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate an image from a text prompt.
    
    This is the main endpoint that:
    1. Validates the input request
    2. Sets up the random seed if provided
    3. Generates the image using the Stable Diffusion pipeline
    4. Converts the result to base64
    5. Returns the response
    
    Args:
        request (ImageGenerationRequest): The generation request parameters
        background_tasks (BackgroundTasks): FastAPI background tasks
    
    Returns:
        ImageGenerationResponse: The generated image and metadata
    """
    global pipeline
    
    # Check if model is loaded
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Please wait for initialization."
        )
    
    try:
        logger.info(f"Generating image for prompt: '{request.prompt[:50]}...'")
        
        # Set random seed for reproducibility if provided
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(request.seed)
            logger.info(f"Using seed: {request.seed}")
        
        # Ensure dimensions are multiples of 8 (required by Stable Diffusion)
        width = (request.width // 8) * 8
        height = (request.height // 8) * 8
        
        if width != request.width or height != request.height:
            logger.warning(
                f"Adjusted dimensions from {request.width}x{request.height} "
                f"to {width}x{height} (must be multiples of 8)"
            )
        
        # Generate the image using the pipeline
        logger.info("Starting image generation...")
        with torch.inference_mode():  # Disable gradient computation for faster inference
            result = pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt if request.negative_prompt else None,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                width=width,
                height=height,
                generator=generator,
                return_dict=True
            )
        
        # Extract the generated image (first image if multiple generated)
        generated_image = result.images[0]
        logger.info("Image generation completed successfully")
        
        # Convert image to base64 for JSON response
        image_base64 = image_to_base64(generated_image)
        logger.info("Image converted to base64")
        
        # Prepare response data
        generation_params = {
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "width": width,
            "height": height,
            "seed": request.seed,
            "negative_prompt": request.negative_prompt
        }
        
        # Create successful response
        response = ImageGenerationResponse(
            success=True,
            image_base64=image_base64,
            prompt_used=request.prompt,
            generation_params=generation_params,
            message="Image generated successfully"
        )
        
        logger.info("Request completed successfully")
        return response
        
    except torch.cuda.OutOfMemoryError:
        # Handle GPU out of memory error specifically
        logger.error("GPU out of memory during generation")
        raise HTTPException(
            status_code=507,
            detail="GPU out of memory. Try reducing image size or inference steps."
        )
    
    except Exception as e:
        # Handle any other errors
        logger.error(f"Error during image generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during image generation: {str(e)}"
        )


@app.get("/models/info")
async def model_info():
    """
    Get information about the loaded model.
    
    Returns:
        dict: Model configuration and system information
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet"
        )
    
    return {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "torch_dtype": str(TORCH_DTYPE),
        "scheduler": pipeline.scheduler.__class__.__name__,
        "vae": pipeline.vae.__class__.__name__,
        "text_encoder": pipeline.text_encoder.__class__.__name__,
        "unet": pipeline.unet.__class__.__name__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }


if __name__ == "__main__":
    """
    
    The server will start on http://localhost:8000
    
    API Documentation will be available at:
    - http://localhost:8000/docs (Swagger UI)
    - http://localhost:8000/redoc (ReDoc)
    """
    import uvicorn
    
    # Configure uvicorn server
    uvicorn.run(
        "generate_api:app",  # Change "main" to your script filename if different
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,       # Port to listen on
        reload=False,    # Set to True for development (auto-reload on changes)
        log_level="info" # Logging level
    )



"""
ahora quiero tener palabras prohibidas y quiero poder generar una iamgen a partir de otra imagen
"""
