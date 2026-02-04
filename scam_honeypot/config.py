"""
Configuration for Scam Honeypot Service
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Settings
    API_KEY = os.getenv("HONEYPOT_API_KEY", "sk_test_987654321")
    
    # LLM Settings - Multiple providers supported
    GROK_API_KEY = os.getenv("GROK_API_KEY", "")  # xAI Grok (Recommended)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Google Gemini
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # OpenAI GPT
    
    # LLM Provider Priority (first available will be used)
    LLM_PROVIDER_PRIORITY = ["grok", "gemini", "openai"]
    USE_LLM = bool(GROK_API_KEY or GEMINI_API_KEY or OPENAI_API_KEY)
    
    # Model Settings
    SCAM_MODEL_PATH = "distilbert-base-uncased"
    SPACY_MODEL = "en_core_web_sm"
    
    # Detection Thresholds
    SCAM_KEYWORD_THRESHOLD = 2
    ML_SCAM_THRESHOLD = 0.7
    ENSEMBLE_THRESHOLD = 0.6
    
    # Persona Settings
    DEFAULT_PERSONA = "elderly_confused"
    AVAILABLE_PERSONAS = [
        "elderly_confused",
        "tech_unsavvy",
        "concerned_parent",
        "busy_professional"
    ]
    
    # Engagement Strategy
    MIN_TURNS_BEFORE_CALLBACK = 3
    MAX_TURNS_BEFORE_CALLBACK = 10
    INTELLIGENCE_CONFIDENCE_THRESHOLD = 0.7
    
    # Callback Settings
    GUVI_CALLBACK_URL = "https://hackathon.guvi.in/api/updateHoneyPotFinalResult"
    CALLBACK_TIMEOUT = 5
    CALLBACK_RETRIES = 3
    
    # Redis Settings
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    SESSION_TTL = 7200  # 2 hours
    
config = Config()
