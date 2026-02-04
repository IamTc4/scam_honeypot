"""
Attention-Based Ensemble Model with Learned Weights
Research-grade implementation inspired by "Learn to Combine" and Attention mechanisms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List

class AttentionEnsemble(nn.Module):
    """
    Multi-head attention mechanism for adaptive model fusion.
    Learns optimal weights based on input characteristics.
    """
    def __init__(self, num_models=3, feature_dim=128, num_heads=4):
        super().__init__()
        self.num_models = num_models
        self.num_heads = num_heads
        
        # Feature encoder for input characteristics
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        # Multi-head attention for model fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Model-specific embeddings (learnable)
        self.model_embeddings = nn.Parameter(
            torch.randn(num_models, 32)
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_models + 32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # Binary classification
        )
        
    def forward(self, model_predictions: torch.Tensor, input_features: torch.Tensor):
        """
        Args:
            model_predictions: (batch, num_models, 2) - logits from each model
            input_features: (batch, feature_dim) - audio features
        Returns:
            final_logits: (batch, 2)
            attention_weights: (batch, num_models)
        """
        batch_size = model_predictions.size(0)
        
        # Encode input features
        feature_repr = self.feature_encoder(input_features)  # (batch, 32)
        
        # Prepare for attention: use model embeddings as keys/values
        model_embeds = self.model_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_models, 32)
        query = feature_repr.unsqueeze(1)  # (batch, 1, 32)
        
        # Multi-head attention
        attn_output, attn_weights = self.attention(
            query, model_embeds, model_embeds
        )  # attn_output: (batch, 1, 32), attn_weights: (batch, 1, num_models)
        
        attn_output = attn_output.squeeze(1)  # (batch, 32)
        attn_weights = attn_weights.squeeze(1)  # (batch, num_models)
        
        # Weighted combination of model predictions
        # Apply softmax to model predictions
        model_probs = F.softmax(model_predictions, dim=-1)  # (batch, num_models, 2)
        
        # Weight by attention
        weighted_probs = torch.einsum('bn,bnc->bc', attn_weights, model_probs)  # (batch, 2)
        
        # Concatenate with attention output for final fusion
        fusion_input = torch.cat([weighted_probs, attn_output], dim=1)  # (batch, 2+32)
        final_logits = self.fusion(fusion_input)  # (batch, 2)
        
        return final_logits, attn_weights

class AdaptiveEnsembleClassifier:
    """
    High-level wrapper for attention-based ensemble
    """
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = AttentionEnsemble(num_models=3, feature_dim=128).to(self.device)
        self.model.eval()
        
        # In production, load pre-trained weights
        # self.model.load_state_dict(torch.load('attention_ensemble.pth'))
    
    def classify(self, 
                 wav2vec_logits: np.ndarray,
                 heuristic_logits: np.ndarray,
                 feature_logits: np.ndarray,
                 audio_features: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Classify using attention-based ensemble
        
        Args:
            wav2vec_logits: (2,) logits from Wav2Vec2 model
            heuristic_logits: (2,) logits from heuristic classifier
            feature_logits: (2,) logits from feature-based classifier
            audio_features: (128,) flattened audio features
        
        Returns:
            classification: "AI_GENERATED" or "HUMAN"
            confidence: float
            explanation: dict with attention weights
        """
        # Stack model predictions
        model_preds = np.stack([wav2vec_logits, heuristic_logits, feature_logits], axis=0)  # (3, 2)
        
        # Convert to tensors
        model_preds_t = torch.from_numpy(model_preds).float().unsqueeze(0).to(self.device)  # (1, 3, 2)
        features_t = torch.from_numpy(audio_features).float().unsqueeze(0).to(self.device)  # (1, 128)
        
        # Forward pass
        with torch.no_grad():
            logits, attn_weights = self.model(model_preds_t, features_t)
            probs = F.softmax(logits, dim=-1)
        
        # Extract results
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_class].item()
        attn_weights_np = attn_weights[0].cpu().numpy()
        
        classification = "AI_GENERATED" if pred_class == 1 else "HUMAN"
        
        explanation = {
            "attention_weights": {
                "wav2vec2": float(attn_weights_np[0]),
                "heuristic": float(attn_weights_np[1]),
                "feature_based": float(attn_weights_np[2])
            },
            "interpretation": self._interpret_attention(attn_weights_np)
        }
        
        return classification, confidence, explanation
    
    def _interpret_attention(self, weights: np.ndarray) -> str:
        """Generate human-readable interpretation"""
        model_names = ["deep learning", "heuristic analysis", "feature-based ML"]
        top_idx = np.argmax(weights)
        top_model = model_names[top_idx]
        top_weight = weights[top_idx]
        
        return f"Decision primarily based on {top_model} ({top_weight:.1%} weight)"

# Utility: Convert existing features to fixed-size vector
def flatten_features_to_vector(features: Dict, target_dim=128) -> np.ndarray:
    """
    Convert variable-length feature dict to fixed-size vector
    """
    # Extract scalar features
    scalars = []
    for key in ['pitch_mean', 'pitch_std', 'pitch_stability', 'spectral_centroid_mean',
                'spectral_flatness_mean', 'zcr_mean', 'energy_mean', 'jitter', 'shimmer',
                'hnr', 'tempo', 'silence_ratio', 'harmonic_ratio']:
        scalars.append(features.get(key, 0.0))
    
    # Take first N elements from list features
    for key in ['mfcc_mean', 'mel_mean', 'formants']:
        vals = features.get(key, [])
        if isinstance(vals, list):
            scalars.extend(vals[:10])  # Take first 10
    
    # Pad or truncate to target_dim
    vec = np.array(scalars, dtype=np.float32)
    if len(vec) < target_dim:
        vec = np.pad(vec, (0, target_dim - len(vec)), mode='constant')
    else:
        vec = vec[:target_dim]
    
    return vec
