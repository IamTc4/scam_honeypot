from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import List, Optional, Any
from .detector import detect_scam
from .agent_engine import AgentController

app = FastAPI(title="Scam Honeypot API")

# Security
API_KEY = "sk_test_987654321" # Demo key

class MessageModel(BaseModel):
    sender: str
    text: str
    timestamp: str

class MetadataModel(BaseModel):
    channel: Optional[str] = None
    language: Optional[str] = None
    locale: Optional[str] = None

class ScamRequest(BaseModel):
    sessionId: str
    message: MessageModel
    conversationHistory: List[MessageModel] = []
    metadata: Optional[MetadataModel] = None

# Global Controller
controller = AgentController()

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

@app.post("/api/scam-honeypot")
async def handle_scam(request: ScamRequest, api_key: str = Depends(verify_api_key)):
    try:
        # 1. Detect Scam (now returns dict)
        detection_result = detect_scam(request.message.text)
        
        # 2. Process with Agent (pass full detection result)
        reply_text = controller.process_message(
            request.sessionId,
            request.message.text,
            detection_result
        )
        
        # 3. Form Response
        return {
            "status": "success",
            "reply": reply_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "service": "Scam Honeypot API",
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "honeypot": "/api/scam-honeypot",
            "health": "/health"
        },
        "features": [
            "Multi-turn conversation handling",
            "LLM-powered responses",
            "Intelligence extraction",
            "Automatic callback to GUVI"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "scam-honeypot",
        "timestamp": "2026-02-04T20:00:00Z"
    }
