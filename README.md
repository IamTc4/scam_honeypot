# 🚀 GUVI Hackathon - Advanced ML Solutions

**Enterprise-grade implementations** with transformers, LLMs, and production-ready architecture.

## Architecture

### Voice Detection: Ensemble ML Pipeline
```
Audio → Base64 Decode → Preprocessing → Multi-Model Ensemble → Result
                           ↓
                    40+ Features + Wav2Vec2 Embeddings
                           ↓
                    Weighted Fusion (50% DL + 30% Heuristic + 20% ML)
```

### Scam Honeypot: LLM-Powered Agentic System
```
Message → Multi-Stage Detector → Gemini LLM Agent → NER Intel → Callback
            (Keyword + ML + Heuristic)      ↓
                                     Persona Manager + Emotion FSM
```

## 📡 API Endpoints

### Voice Detection API
```bash
POST http://localhost:8000/api/voice-detection
Headers: x-api-key: sk_test_123456789
Body: {
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "..."
}
```

### Scam Honeypot API
```bash
POST http://localhost:8001/api/scam-honeypot
Headers: x-api-key: sk_test_987654321
Body: {
  "sessionId": "unique-id",
  "message": {...},
  "conversationHistory": [...]
}
```

## 🎯 Key Features

### Voice Detection ✨
- **Wav2Vec2** transformer embeddings (768-dim)
- **40+ Audio Features**: MFCCs, mel-spectrograms, prosody, formants, voice quality
- **Ensemble Model**: Deep learning + traditional ML + heuristics
- **Feature Importance** explanations

### Scam Honeypot ✨
- **Gemini Pro LLM** for natural conversations
- **4 Adaptive Personas** with emotional states
- **SpaCy NER** + advanced regex intelligence extraction
- **Multi-Stage ML Detection**: DistilBERT + keywords + heuristics
- **Conversation Analysis**: sentiment, intent, urgency, manipulation tactics
- **Strategic Termination** with callback retry logic

## 🛠️ Quick Start

```bash
# Install dependencies
pip install -r voice_detection/requirements.txt
pip install -r scam_honeypot/requirements.txt
python -m spacy download en_core_web_sm

# Optional: Configure Gemini API
export GEMINI_API_KEY="your-key-here"

# Run services
uvicorn voice_detection.app:app --port 8000 --reload &
uvicorn scam_honeypot.app:app --port 8001 --reload &
```

## 📁 Project Structure

```
c:/hcl/
├── voice_detection/
│   ├── models/
│   │   ├── wav2vec_classifier.py    # Transformer model
│   │   └── ensemble_model.py        # Multi-model fusion
│   ├── feature_extractor.py         # 40+ features
│   ├── inference.py                 # Ensemble inference
│   └── config.py                    # Configuration
│
├── scam_honeypot/
│   ├── llm_agent.py                 # Gemini integration
│   ├── persona_manager.py           # 4 personas + emotions
│   ├── ner_intelligence.py          # SpaCy NER + regex
│   ├── conversation_analyzer.py     # Sentiment/intent
│   ├── ml_detector.py               # DistilBERT classifier
│   ├── detector.py                  # Multi-stage detection
│   ├── agent_engine.py              # Orchestration + callback
│   └── config.py                    # Configuration
```

## 🎓 Advanced Technologies

| Component | Technology |
|-----------|------------|
| **Deep Learning** | Wav2Vec2, DistilBERT |
| **LLM** | Google Gemini Pro |
| **NLP** | SpaCy, TextBlob |
| **Audio** | Librosa, PyDub, SciPy |
| **ML** | PyTorch, Transformers, XGBoost |

## 📊 Performance

- **Voice Detection**: ~1.5s inference with GPU
- **Honeypot Response**: <500ms (template) or ~2s (LLM)
- **Intelligence Extraction**: 15+ entity types
- **Callback**: 3 retries with exponential backoff

## 🔐 Configuration

See [config.py](file:///c:/hcl/voice_detection/config.py) and [config.py](file:///c:/hcl/scam_honeypot/config.py) for all settings.

Key environment variables:
- `GEMINI_API_KEY`: Enable LLM-powered responses
- `REDIS_HOST/PORT`: Optional caching
- `VOICE_API_KEY` / `HONEYPOT_API_KEY`: API authentication

## 📝 Documentation

- **Walkthrough**: See [walkthrough.md](file:///C:/Users/krish/.gemini/antigravity/brain/4eeabb57-3ab6-4111-8248-87715ed52cb8/walkthrough.md)
- **Deployment**: See [DEPLOYMENT.md](file:///c:/hcl/DEPLOYMENT.md)
- **Implementation Plan**: [implementation_plan.md](file:///C:/Users/krish/.gemini/antigravity/brain/4eeabb57-3ab6-4111-8248-87715ed52cb8/implementation_plan.md)

---

**Built for GUVI Hackathon 2026** | Advanced ML & LLM Implementation 🎯
