from .feature_extractor import extract_features
from .models.ensemble_model import EnsembleModel
import numpy as np

# Global ensemble model instance
ensemble_model = None

def get_ensemble_model():
    """Lazy load ensemble model"""
    global ensemble_model
    if ensemble_model is None:
        ensemble_model = EnsembleModel()
    return ensemble_model

def analyze_voice(audio_array: np.ndarray, language: str):
    """
    Analyzes audio using advanced ensemble model.
    Returns classification, confidence, and explanation.
    """
    # Extract comprehensive features
    features = extract_features(audio_array)
    
    # Get ensemble model
    model = get_ensemble_model()
    
    # Classify using ensemble
    classification, confidence, explanation = model.classify(audio_array, features)
    
    return {
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }
