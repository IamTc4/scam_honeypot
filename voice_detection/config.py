"""
Configuration for Voice Detection Service
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Settings
    API_KEY = os.getenv("VOICE_API_KEY", "sk_test_123456789")
    
    # Model Settings
    MODEL_DIR = Path(__file__).parent / "models" / "checkpoints"
    USE_GPU = torch.cuda.is_available() if 'torch' in dir() else False
    
    # Audio Processing
    TARGET_SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 30  # seconds
    MIN_AUDIO_LENGTH = 1   # seconds
    
    # Feature Extraction
    N_MFCC = 40
    N_MEL_BINS = 80
    HOP_LENGTH = 512
    N_FFT = 2048
    
    # Ensemble Model Weights
    WAV2VEC_WEIGHT = 0.5
    XGBOOST_WEIGHT = 0.3
    HEURISTIC_WEIGHT = 0.2
    
    # Confidence Thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    LOW_CONFIDENCE_THRESHOLD = 0.55
    
    # Caching
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    CACHE_TTL = 3600  # 1 hour
    
    # Languages
    SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    
config = Config()
