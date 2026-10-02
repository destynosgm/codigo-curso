"""
INSTALLATION REQUIREMENTS:
pip install fastapi uvicorn pillow pytesseract easyocr ultralytics pydantic numpy
"""

import logging
import base64
import io
import time
from typing import List, Dict
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from PIL import Image
import uvicorn
import numpy as np

# -------------------------------
# OCR Libraries
# -------------------------------
import pytesseract
import easyocr
from ultralytics import YOLO  # Optional for object detection

# -------------------------------
# Set Tesseract Path (Windows)
# -------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Tesseract-OCR\tesseract.exe"

# -------------------------------
# Logging configuration
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("OCRService")

# -------------------------------
# Pydantic Models
# -------------------------------
class OCRRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image")

    @field_validator("image_base64")
    def validate_image(cls, v):
        if not v.strip():
            raise ValueError("Image data cannot be empty")
        return v


class OCRResponse(BaseModel):
    model: str
    text: str
    words: List[str]
    processing_time: float


class BatchOCRResponse(BaseModel):
    results: List[OCRResponse]
    total_models: int
    timestamp: datetime


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    timestamp: datetime
    version: str


# -------------------------------
# OCR Models Manager
# -------------------------------
class OCRModels:
    def __init__(self):
        self.easyocr_reader = None
        self.yolo_model = None

    def load_models(self):
        logger.info("Loading OCR models...")
        try:
            self.easyocr_reader = easyocr.Reader(["en", "es"], gpu=False)
            logger.info("EasyOCR loaded")
            # Optional YOLO model
            self.yolo_model = YOLO("yolo12n.pt")
            logger.info("YOLO model loaded")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def run_tesseract(self, image: Image.Image):
        try:
            text = pytesseract.image_to_string(image, lang="spa")
            words = text.split()
            return text, words
        except Exception as e:
            return f"Error: {e}", []

    def run_easyocr(self, image: Image.Image):
        try:
            image_np = np.array(image)
            results = self.easyocr_reader.readtext(image_np)
            words = [res[1] for res in results]
            return " ".join(words), words
        except Exception as e:
            return f"Error: {e}", []

    def run_yolo(self, image: Image.Image):
        try:
            results = self.yolo_model.predict(image)
            detections = []
            for r in results:
                if hasattr(r, "boxes"):
                    detections.append(f"Detected {len(r.boxes)} objects")
            return " ".join(detections), detections
        except Exception as e:
            return f"Error: {e}", []


ocr_models = OCRModels()

# -------------------------------
# FastAPI Application Setup
# -------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting OCR service...")
    ocr_models.load_models()
    yield
    logger.info("Shutting down OCR service...")


app = FastAPI(
    title="OCR API with Multiple Models",
    description="Extract text and words from images using Tesseract, EasyOCR, and YOLO",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Utility
# -------------------------------
def decode_base64_to_image(base64_str: str) -> Image.Image:
    try:
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


# -------------------------------
# API Endpoints
# -------------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        models_loaded={
            "tesseract": True,
            "easyocr": ocr_models.easyocr_reader is not None,
            "yolo": ocr_models.yolo_model is not None,
        },
        timestamp=datetime.now(),
        version="1.0.0"
    )


@app.post("/predict", response_model=BatchOCRResponse)
async def predict_ocr(request: OCRRequest):
    image = decode_base64_to_image(request.image_base64)
    results = []

    for model_name, func in [
        ("Tesseract", ocr_models.run_tesseract),
        ("EasyOCR", ocr_models.run_easyocr),
        ("YOLO", ocr_models.run_yolo),
    ]:
        start = time.time()
        text, words = func(image)
        end = time.time()
        results.append(OCRResponse(
            model=model_name,
            text=text,
            words=words,
            processing_time=round(end - start, 4)
        ))

    return BatchOCRResponse(
        results=results,
        total_models=len(results),
        timestamp=datetime.now()
    )


@app.get("/")
async def root():
    return {
        "message": "OCR API with Multiple Models",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    }


# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    uvicorn.run("ocr_api:app", host="0.0.0.0", port=8000, reload=True)
