from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import Optional
import os
from .audio_processor import process_audio
from .inference import analyze_voice

app = FastAPI(title="Voice Detection API")

# Security configuration
API_KEY = "sk_test_123456789"  # In production, use env vars

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401, 
            detail="Invalid API key or malformed request"
        )
    return x_api_key

@app.post("/api/voice-detection")
async def detect_voice(request: VoiceRequest, api_key: str = Depends(verify_api_key)):
    try:
        if request.language not in ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]:
            return {
                "status": "error",
                "message": f"Unsupported language: {request.language}. Supported: Tamil, English, Hindi, Malayalam, Telugu"
            }
        
        if request.audioFormat.lower() != "mp3":
            return {
                "status": "error",
                "message": "Only mp3 audio format is supported"
            }

        # Process Audio
        y = process_audio(request.audioBase64)
        
        # Analyze Voice
        result = analyze_voice(y, request.language)
        
        return {
            "status": "success",
            "language": request.language,
            "classification": result["classification"],
            "confidenceScore": result["confidenceScore"],
            "explanation": result["explanation"]
        }
    
    except HTTPException as he:
        # Re-raise HTTP exceptions (like 401 from API key validation)
        raise he
    except ValueError as ve:
        return {
            "status": "error",
            "message": str(ve)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Audio processing failed: {str(e)}"
        }

# Health check
@app.get("/")
async def root():
    return {
        "service": "Voice Detection API",
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "detection": "/api/voice-detection",
            "health": "/health"
        },
        "supported_languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "voice-detection",
        "timestamp": "2026-02-04T20:00:00Z"
    }
