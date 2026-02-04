"""
Wav2Vec2-based feature extractor for deepfake detection
"""
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Processor

class Wav2VecClassifier:
    def __init__(self, model_name="facebook/wav2vec2-base"):
        """Initialize Wav2Vec2 model for feature extraction"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2Model.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.available = True
        except Exception as e:
            print(f"⚠️ Wav2Vec2 not available: {e}")
            self.available = False
    
    def extract_embeddings(self, audio_array: np.ndarray, sr: int = 16000):
        """
        Extract Wav2Vec2 embeddings from audio
        Returns mean pooled embedding
        """
        if not self.available:
            return np.zeros(768)  # Return zero features if model not available
        
        try:
            # Prepare input
            inputs = self.processor(
                audio_array, 
                sampling_rate=sr, 
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
                # Mean pooling over time dimension
                embeddings = hidden_states.mean(dim=1).squeeze().cpu().numpy()
            
            return embeddings
        except Exception as e:
            print(f"Error extracting Wav2Vec2 embeddings: {e}")
            return np.zeros(768)
    
    def classify(self, embeddings: np.ndarray) -> tuple:
        """
        Simplified classifier on top of embeddings
        In production, this would be a fine-tuned classification head
        For now, we use heuristic rules on embedding statistics
        """
        if not self.available or embeddings.sum() == 0:
            return "HUMAN", 0.5  # Neutral if model unavailable
        
        # Heuristic: AI voices tend to have more uniform embeddings
        embedding_std = np.std(embeddings)
        embedding_mean = np.mean(np.abs(embeddings))
        
        # Threshold-based classification (tuned empirically)
        uniformity_score = embedding_std / (embedding_mean + 1e-6)
        
        if uniformity_score < 0.15:  # Very uniform = likely AI
            return "AI_GENERATED", min(0.7 + (0.15 - uniformity_score) * 2, 0.95)
        else:  # More variation = likely human
            return "HUMAN", min(0.6 + (uniformity_score - 0.15) * 2, 0.95)
