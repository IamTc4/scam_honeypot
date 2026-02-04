# Deployment Guide

## Option 1: Deploy to Render (Recommended for Hackathon)

### Voice Detection Service

1. Create `render.yaml`:
```yaml
services:
  - type: web
    name: voice-detection
    env: python
    buildCommand: "pip install -r voice_detection/requirements.txt"
    startCommand: "uvicorn voice_detection.app:app --host 0.0.0.0 --port $PORT"
    envVars:
      - key: VOICE_API_KEY
        sync: false
```

2. Push to GitHub and connect to Render
3. Set environment variable `VOICE_API_KEY` in Render dashboard

### Scam Honeypot Service

Create separate `scam_honeypot/render.yaml`:
```yaml
services:
  - type: web
    name: scam-honeypot
    env: python
    buildCommand: "pip install -r scam_honeypot/requirements.txt"
    startCommand: "uvicorn scam_honeypot.app:app --host 0.0.0.0 --port $PORT"
    envVars:
      - key: HONEYPOT_API_KEY
        sync: false
```

---

## Option 2: Deploy to Railway

1. Install Railway CLI:
```bash
npm install -g @railway/cli
```

2. Initialize projects:
```bash
cd voice_detection
railway init
railway up
```

3. Repeat for `scam_honeypot`

---

## Option 3: Deploy to AWS Lambda (Serverless)

Use **Mangum** adapter for FastAPI:

```bash
pip install mangum
```

Modify `app.py`:
```python
from mangum import Mangum

# ... existing FastAPI code ...

handler = Mangum(app)  # Lambda handler
```

Package and deploy using AWS SAM or Serverless Framework.

---

## Local Testing Before Deployment

### 1. Test with Postman/Insomnia
Import the cURL commands from README.md

### 2. Test with Python script
See `test_voice.py` and `test_honeypot.py`

### 3. Test connectivity to GUVI endpoint
```bash
curl -X POST https://hackathon.guvi.in/api/updateHoneyPotFinalResult \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId": "test",
    "scamDetected": true,
    "totalMessagesExchanged": 5,
    "extractedIntelligence": {
      "bankAccounts": [],
      "upiIds": ["test@upi"],
      "phishingLinks": [],
      "phoneNumbers": [],
      "suspiciousKeywords": ["urgent"]
    },
    "agentNotes": "Test callback"
  }'
```

---

## Environment Variable Setup

Create `.env` file:
```
VOICE_API_KEY=sk_test_123456789
HONEYPOT_API_KEY=sk_test_987654321
GUVI_CALLBACK_URL=https://hackathon.guvi.in/api/updateHoneyPotFinalResult
```

Update `app.py` files to use:
```python
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("VOICE_API_KEY")
```

---

## Performance Optimization

1. **Add caching** for repeated sessions
2. **Use async HTTP client** (httpx) for callbacks
3. **Add background tasks** for callback (FastAPI BackgroundTasks)
4. **Optimize audio processing** with multiprocessing

---

## Monitoring & Logging

Add middleware:
```python
import logging
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    return response
```

---

## Production Checklist

- [ ] Replace demo API keys
- [ ] Set up environment variables
- [ ] Add rate limiting (slowapi)
- [ ] Configure CORS properly
- [ ] Add request logging
- [ ] Set up error tracking (Sentry)
- [ ] Test with real audio samples
- [ ] Verify GUVI callback works
- [ ] Load test with multiple requests
- [ ] Set up SSL/HTTPS
- [ ] Document your public API URLs

---

## Common Issues

**Issue**: Import errors when running uvicorn  
**Solution**: Run from parent directory: `uvicorn voice_detection.app:app`

**Issue**: Audio processing fails  
**Solution**: Ensure ffmpeg is installed (required by pydub for MP3)

**Issue**: Callback times out  
**Solution**: Use BackgroundTasks or async HTTP client

---

**Ready for Deployment!** 🚀
