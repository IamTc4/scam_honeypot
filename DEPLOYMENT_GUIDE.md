# 🚀 Quick Deployment Guide

## Option 1: Deploy to Render.com (Recommended - Fastest)

### Prerequisites
- GitHub account
- Render.com account (free tier available)
- Google AI Studio API key (for Gemini - get from https://makersuite.google.com/app/apikey)

### Steps

1. **Push code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - GUVI Hackathon submission"
   git branch -M main
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. **Deploy on Render**
   - Go to https://render.com/
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml`
   - Click "Apply"

3. **Set Environment Variables**
   - In Render dashboard, go to each service
   - Add `GEMINI_API_KEY` for honeypot service (get free key from Google AI Studio)
   - Other variables are already set in render.yaml

4. **Get Your Public URLs**
   - Voice Detection: `https://voice-detection-api.onrender.com`
   - Scam Honeypot: `https://scam-honeypot-api.onrender.com`

5. **Submit to Hackathon**
   - Copy the URLs above
   - Submit at hackathon portal with API keys:
     - Voice API Key: `sk_test_123456789`
     - Honeypot API Key: `sk_test_987654321`

---

## Option 2: Deploy to Railway.app

### Steps

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   railway login
   ```

2. **Deploy Voice Detection**
   ```bash
   cd voice_detection
   railway init
   railway up
   railway variables set VOICE_API_KEY=sk_test_123456789
   ```

3. **Deploy Scam Honeypot**
   ```bash
   cd ../scam_honeypot
   railway init
   railway up
   railway variables set HONEYPOT_API_KEY=sk_test_987654321
   railway variables set GEMINI_API_KEY=your_gemini_key_here
   railway variables set GUVI_CALLBACK_URL=https://hackathon.guvi.in/api/updateHoneyPotFinalResult
   ```

4. **Get URLs**
   ```bash
   railway domain
   ```

---

## Option 3: Deploy to Vercel (Serverless)

### Prerequisites
```bash
npm install -g vercel
pip install mangum
```

### Modify app.py files

Add to both `voice_detection/app.py` and `scam_honeypot/app.py`:
```python
from mangum import Mangum

# ... existing code ...

# Add at the end
handler = Mangum(app)
```

### Deploy
```bash
vercel --prod
```

---

## Testing Your Deployment

### Test Voice Detection API
```bash
curl -X POST https://YOUR-VOICE-API-URL/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_123456789" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
  }'
```

### Test Scam Honeypot API
```bash
curl -X POST https://YOUR-HONEYPOT-API-URL/api/scam-honeypot \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_987654321" \
  -d '{
    "sessionId": "test-123",
    "message": {
      "sender": "scammer",
      "text": "Your bank account will be blocked. Verify immediately.",
      "timestamp": "2026-02-04T20:00:00Z"
    },
    "conversationHistory": []
  }'
```

---

## Environment Variables Reference

### Voice Detection Service
| Variable | Value | Required |
|----------|-------|----------|
| `VOICE_API_KEY` | `sk_test_123456789` | Yes |
| `PYTHON_VERSION` | `3.10.0` | Recommended |

### Scam Honeypot Service
| Variable | Value | Required |
|----------|-------|----------|
| `HONEYPOT_API_KEY` | `sk_test_987654321` | Yes |
| `GEMINI_API_KEY` | Your Google AI key | Yes |
| `GUVI_CALLBACK_URL` | `https://hackathon.guvi.in/api/updateHoneyPotFinalResult` | Yes |
| `PYTHON_VERSION` | `3.10.0` | Recommended |

---

## Troubleshooting

### Build Fails
- **Issue**: PyTorch installation timeout
- **Solution**: Use CPU-only version (already in render.yaml)

### SpaCy Model Missing
- **Issue**: `en_core_web_sm` not found
- **Solution**: Ensure build command includes `python -m spacy download en_core_web_sm`

### Gemini API Errors
- **Issue**: LLM responses fail
- **Solution**: System falls back to template responses automatically

### Callback Fails
- **Issue**: GUVI endpoint not reachable
- **Solution**: Check logs, retry logic is automatic (3 attempts)

---

## Pre-Deployment Checklist

- [ ] Code pushed to GitHub
- [ ] Gemini API key obtained
- [ ] Render/Railway account created
- [ ] Environment variables configured
- [ ] Both services deployed successfully
- [ ] Health check endpoints responding
- [ ] Test requests successful
- [ ] Public URLs documented
- [ ] Submitted to hackathon portal

---

## Submission Information

**Endpoint URLs** (replace with your actual URLs):
- Voice Detection: `https://voice-detection-api.onrender.com/api/voice-detection`
- Scam Honeypot: `https://scam-honeypot-api.onrender.com/api/scam-honeypot`

**API Keys for Evaluators**:
- Voice API Key: `sk_test_123456789`
- Honeypot API Key: `sk_test_987654321`

**Supported Languages** (Voice Detection):
- Tamil, English, Hindi, Malayalam, Telugu

---

## Performance Notes

- **Cold Start**: First request may take 10-15 seconds (free tier)
- **Response Time**: ~1-2 seconds for voice detection
- **Response Time**: ~0.5-2 seconds for honeypot (depends on LLM)
- **Concurrent Requests**: Handles 10+ simultaneous requests

---

## Need Help?

1. Check Render/Railway logs for errors
2. Test locally first: `uvicorn voice_detection.app:app --reload`
3. Verify all dependencies installed
4. Ensure API keys are correct

**Good luck with your submission!** 🎯
