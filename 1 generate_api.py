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
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pipelines
pipeline = None
img2img_pipeline = None

# Config
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Palabras prohibidas
FORBIDDEN_WORDS = ["sexo", "violencia", "sangre", "nazi", "porno"]


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    negative_prompt: Optional[str] = Field(default="", max_length=1000)
    num_inference_steps: Optional[int] = 20
    guidance_scale: Optional[float] = 7.5
    width: Optional[int] = 512
    height: Optional[int] = 512
    seed: Optional[int] = None


class ImageToImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    init_image_base64: str = Field(..., description="Imagen inicial en base64")
    strength: Optional[float] = Field(default=0.8, ge=0.1, le=1.0)
    num_inference_steps: Optional[int] = 20
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = None


class ImageGenerationResponse(BaseModel):
    success: bool
    image_base64: Optional[str]
    prompt_used: str
    generation_params: Dict[str, Any]
    message: Optional[str]


async def load_model():
    """Cargar los pipelines de texto→imagen e imagen→imagen"""
    global pipeline, img2img_pipeline
    try:
        logger.info(f"Loading model {MODEL_ID} en {DEVICE}")

        # text2img
        pipeline = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=TORCH_DTYPE,
            safety_checker=None, requires_safety_checker=False
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(DEVICE)

        # img2img
        img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID, torch_dtype=TORCH_DTYPE,
            safety_checker=None, requires_safety_checker=False
        )
        img2img_pipeline = img2img_pipeline.to(DEVICE)

        if DEVICE == "cuda":
            pipeline.enable_attention_slicing()
            img2img_pipeline.enable_attention_slicing()

        logger.info("Modelos cargados con éxito")

    except Exception as e:
        logger.error(f"Error al cargar modelo: {e}")
        raise e


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def check_forbidden_words(text: str):
    """Validar palabras prohibidas"""
    t = text.lower()
    for word in FORBIDDEN_WORDS:
        if word in t:
            raise HTTPException(status_code=400, detail=f"La palabra '{word}' está prohibida en el prompt.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando servidor...")
    await load_model()
    yield
    logger.info("Apagando servidor...")
    global pipeline, img2img_pipeline
    if DEVICE == "cuda":
        if pipeline: pipeline.to("cpu")
        if img2img_pipeline: img2img_pipeline.to("cpu")
        torch.cuda.empty_cache()
    pipeline, img2img_pipeline = None, None


app = FastAPI(
    title="Stable Diffusion API",
    description="API para texto→imagen e imagen→imagen",
    version="1.1.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {"message": "Stable Diffusion API", "status": "running", "model": MODEL_ID, "device": DEVICE}


@app.post("/generate", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest, background_tasks: BackgroundTasks):
    global pipeline
    if not pipeline:
        raise HTTPException(503, "Modelo no cargado")

    check_forbidden_words(request.prompt)
    if request.negative_prompt:
        check_forbidden_words(request.negative_prompt)

    generator = None
    if request.seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(request.seed)

    width = (request.width // 8) * 8
    height = (request.height // 8) * 8

    with torch.inference_mode():
        result = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=width, height=height,
            generator=generator
        )

    img = result.images[0]
    return ImageGenerationResponse(
        success=True,
        image_base64=image_to_base64(img),
        prompt_used=request.prompt,
        generation_params=request.dict(),
        message="Imagen generada con éxito"
    )


@app.post("/img2img", response_model=ImageGenerationResponse)
async def image_to_image(request: ImageToImageRequest):
    global img2img_pipeline
    if not img2img_pipeline:
        raise HTTPException(503, "Modelo img2img no cargado")

    check_forbidden_words(request.prompt)

    try:
        init_image = Image.open(io.BytesIO(base64.b64decode(request.init_image_base64))).convert("RGB")
    except Exception:
        raise HTTPException(400, "Imagen inicial inválida")

    generator = None
    if request.seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(request.seed)

    with torch.inference_mode():
        result = img2img_pipeline(
            prompt=request.prompt,
            image=init_image,
            strength=request.strength,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator
        )

    img = result.images[0]
    return ImageGenerationResponse(
        success=True,
        image_base64=image_to_base64(img),
        prompt_used=request.prompt,
        generation_params=request.dict(),
        message="Imagen generada desde otra imagen con éxito"
    )
