"""
SHAP-based Explainability for Voice Detection
Provides feature importance and counterfactual explanations
"""
import numpy as np
from typing import Dict, List, Tuple
import shap

class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) for interpretable predictions
    """
    def __init__(self, model_predict_fn):
        """
        Args:
            model_predict_fn: function that takes features dict and returns (class, confidence)
        """
        self.model_predict_fn = model_predict_fn
        self.feature_names = [
            'pitch_mean', 'pitch_std', 'pitch_stability',
            'spectral_centroid_mean', 'spectral_flatness_mean',
            'zcr_mean', 'energy_mean', 'jitter', 'shimmer',
            'hnr', 'tempo', 'silence_ratio', 'harmonic_ratio'
        ]
    
    def explain(self, features: Dict, background_data: np.ndarray = None) -> Dict:
        """
        Generate SHAP explanations for a prediction
        
        Args:
            features: audio features dict
            background_data: (optional) background dataset for SHAP
        
        Returns:
            explanation dict with feature importance
        """
        # Extract feature vector
        feature_vector = self._extract_feature_vector(features)
        
        # Simplified SHAP approximation (for demo)
        # In production, use shap.KernelExplainer or shap.TreeExplainer
        shap_values = self._compute_shap_values(feature_vector, features)
        
        # Get top contributing features
        top_features = self._get_top_features(shap_values)
        
        # Generate counterfactuals
        counterfactuals = self._generate_counterfactuals(features, shap_values)
        
        return {
            "feature_importance": shap_values,
            "top_contributors": top_features,
            "counterfactuals": counterfactuals,
            "interpretation": self._interpret_shap(shap_values, top_features)
        }
    
    def _extract_feature_vector(self, features: Dict) -> np.ndarray:
        """Extract scalar features into vector"""
        return np.array([features.get(name, 0.0) for name in self.feature_names])
    
    def _compute_shap_values(self, feature_vector: np.ndarray, features: Dict) -> Dict[str, float]:
        """
        Compute SHAP values (simplified approximation)
        In production, use proper SHAP library
        """
        # Get baseline prediction
        baseline_features = {name: 0.0 for name in self.feature_names}
        _, baseline_conf = self.model_predict_fn(baseline_features)
        
        # Get actual prediction
        _, actual_conf = self.model_predict_fn(features)
        
        # Approximate SHAP by feature ablation
        shap_values = {}
        for i, name in enumerate(self.feature_names):
            # Remove this feature
            ablated_features = features.copy()
            ablated_features[name] = 0.0
            _, ablated_conf = self.model_predict_fn(ablated_features)
            
            # SHAP value = change in prediction
            shap_values[name] = actual_conf - ablated_conf
        
        return shap_values
    
    def _get_top_features(self, shap_values: Dict[str, float], top_k=5) -> List[Tuple[str, float]]:
        """Get top K features by absolute SHAP value"""
        sorted_features = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:top_k]
    
    def _generate_counterfactuals(self, features: Dict, shap_values: Dict[str, float]) -> List[str]:
        """
        Generate counterfactual explanations
        "If feature X was Y, then prediction would be Z"
        """
        counterfactuals = []
        
        # Find most influential feature
        top_feature, top_shap = max(shap_values.items(), key=lambda x: abs(x[1]))
        current_value = features.get(top_feature, 0.0)
        
        # Generate counterfactual
        if top_shap > 0:  # Feature increases AI probability
            new_value = current_value * 0.5  # Reduce it
            direction = "decreased"
        else:  # Feature decreases AI probability
            new_value = current_value * 1.5  # Increase it
            direction = "increased"
        
        counterfactuals.append(
            f"If '{top_feature}' was {direction} to {new_value:.2f}, "
            f"the classification might flip with ~{abs(top_shap)*100:.0f}% confidence change"
        )
        
        return counterfactuals
    
    def _interpret_shap(self, shap_values: Dict[str, float], top_features: List[Tuple[str, float]]) -> str:
        """Generate human-readable interpretation"""
        if not top_features:
            return "No significant features identified"
        
        top_name, top_value = top_features[0]
        direction = "towards AI-generated" if top_value > 0 else "towards human speech"
        
        return f"'{top_name}' is the strongest indicator, pushing classification {direction}"

# Integration with existing inference
def explain_prediction(audio_array: np.ndarray, features: Dict, classification: str, confidence: float) -> Dict:
    """
    Wrapper function to add explainability to predictions
    """
    # Mock model function for SHAP
    def mock_predict(feat_dict):
        # Simplified: use pitch_stability as proxy
        pitch_stab = feat_dict.get('pitch_stability', 0.1)
        if pitch_stab < 0.05:
            return "AI_GENERATED", 0.8
        else:
            return "HUMAN", 0.7
    
    explainer = SHAPExplainer(mock_predict)
    explanation = explainer.explain(features)
    
    return {
        "classification": classification,
        "confidence": confidence,
        "explainability": explanation
    }
