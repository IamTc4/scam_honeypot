"""
Ensemble model combining Wav2Vec2, XGBoost, and Heuristics
"""
import numpy as np
from typing import Dict, Tuple
from .wav2vec_classifier import Wav2VecClassifier

class EnsembleModel:
    def __init__(self):
        """Initialize ensemble components"""
        self.wav2vec_classifier = Wav2VecClassifier()
        self.weights = {
            'wav2vec': 0.5,
            'heuristic': 0.3,
            'feature_based': 0.2
        }
        
        # Language-specific feature importance weights
        self.language_weights = {
            "Tamil": {"pitch": 1.2, "formants": 1.5, "spectral": 1.0, "prosody": 1.1},
            "English": {"pitch": 1.0, "formants": 1.0, "spectral": 1.2, "prosody": 1.3},
            "Hindi": {"pitch": 1.4, "formants": 1.1, "spectral": 1.0, "prosody": 1.2},
            "Malayalam": {"pitch": 1.1, "formants": 1.3, "spectral": 1.4, "prosody": 1.0},
            "Telugu": {"pitch": 1.3, "formants": 1.4, "spectral": 1.1, "prosody": 1.0}
        }
    
    def classify(self, audio_array: np.ndarray, features: Dict, language: str = "English") -> Tuple[str, float, str]:
        """
        Classify audio using ensemble of models with language-specific tuning
        Returns: (classification, confidence, explanation)
        """
        scores = {'AI_GENERATED': 0.0, 'HUMAN': 0.0}
        explanations = []
        
        # Get language-specific weights
        lang_weights = self.language_weights.get(language, self.language_weights["English"])
        
        # 1. Wav2Vec2 Deep Learning Model
        embeddings = self.wav2vec_classifier.extract_embeddings(audio_array)
        wav2vec_class, wav2vec_conf = self.wav2vec_classifier.classify(embeddings)
        scores[wav2vec_class] += wav2vec_conf * self.weights['wav2vec']
        explanations.append(f"Deep learning model: {wav2vec_class} ({wav2vec_conf:.2f})")
        
        # 2. Heuristic Analysis (with language-specific weights)
        heuristic_class, heuristic_conf, heuristic_exp = self._heuristic_classifier(features, lang_weights)
        scores[heuristic_class] += heuristic_conf * self.weights['heuristic']
        explanations.append(heuristic_exp)
        
        # 3. Feature-based Classifier
        feature_class, feature_conf = self._feature_classifier(features)
        scores[feature_class] += feature_conf * self.weights['feature_based']
        
        # 4. Temporal Consistency Check (NEW)
        temporal_class, temporal_conf = self._temporal_consistency_check(features)
        if temporal_conf > 0.7:  # High confidence temporal analysis
            scores[temporal_class] += temporal_conf * 0.15  # 15% weight
            explanations.append(f"Temporal analysis: {temporal_class}")
        
        # Final decision with dynamic weighting
        final_class = max(scores, key=scores.get)
        final_confidence = min(scores[final_class], 0.99)  # Cap at 0.99
        
        # Generate comprehensive explanation
        explanation = self._generate_explanation(features, final_class, explanations)
        
        return final_class, final_confidence, explanation
    
    def _heuristic_classifier(self, features: Dict, lang_weights: Dict = None) -> Tuple[str, float, str]:
        """Enhanced heuristic-based classification with multiple AI indicators"""
        indicators = []
        ai_score = 0
        human_score = 0
        
        # Default weights if not provided
        if lang_weights is None:
            lang_weights = {"pitch": 1.0, "formants": 1.0, "spectral": 1.0, "prosody": 1.0}
        
        # 1. Pitch stability (AI voices have very consistent pitch) - Language weighted
        pitch_stability = features.get('pitch_stability', 0.1)
        if pitch_stability < 0.03:  # Extremely stable = strong AI indicator
            ai_score += 4
            indicators.append("unnaturally consistent pitch")
        elif pitch_stability < 0.08:  # Moderately stable
            ai_score += 2
            indicators.append("low pitch variation")
        elif pitch_stability > 0.15:  # Natural variation
            human_score += 3
            indicators.append("natural pitch variation")
        
        # 2. Spectral flatness (AI voices can be overly smooth)
        spectral_flatness = features.get('spectral_flatness_mean', 0.1)
        spectral_flatness_std = features.get('spectral_flatness_std', 0.05)
        if spectral_flatness < 0.008:  # Too smooth
            ai_score += 3
            indicators.append("synthetic spectral characteristics")
        elif spectral_flatness > 0.05 and spectral_flatness_std > 0.02:
            human_score += 2
            indicators.append("natural spectral texture")
        
        # 3. Jitter and Shimmer (voice quality - humans have natural imperfections)
        jitter = features.get('jitter', 0)
        shimmer = features.get('shimmer', 0)
        if jitter < 0.00005 and shimmer < 0.005:  # Too perfect
            ai_score += 3
            indicators.append("abnormally low jitter/shimmer")
        elif jitter > 0.0008 and shimmer > 0.04:  # Human-like imperfection
            human_score += 3
            indicators.append("natural voice quality variations")
        elif jitter > 0.0003:
            human_score += 1
        
        # 4. Prosody features (pauses and rhythm)
        silence_ratio = features.get('silence_ratio', 0.2)
        if silence_ratio < 0.03:  # Too continuous
            ai_score += 2
            indicators.append("lacking natural pauses")
        elif silence_ratio > 0.12 and silence_ratio < 0.25:  # Natural pauses
            human_score += 2
            indicators.append("natural pause patterns")
        
        # 5. HNR (Harmonic-to-Noise Ratio) - AI is too clean
        hnr = features.get('hnr', 1.0)
        if hnr > 15:  # Unnaturally clean
            ai_score += 2
            indicators.append("unnaturally high signal clarity")
        elif hnr < 8:  # More noise (human-like)
            human_score += 1
        
        # 6. Energy consistency
        energy_std = features.get('energy_std', 0.1)
        energy_mean = features.get('energy_mean', 0.1)
        if energy_mean > 0 and energy_std / energy_mean < 0.15:  # Too consistent
            ai_score += 1
            indicators.append("uniform energy distribution")
        elif energy_std / energy_mean > 0.3:
            human_score += 1
        
        # 7. Pitch range (AI often has narrower range)
        pitch_range = features.get('pitch_range', 100)
        if pitch_range < 30:  # Very narrow
            ai_score += 2
            indicators.append("limited pitch range")
        elif pitch_range > 100:  # Wide range
            human_score += 2
        
        # 8. Spectral centroid variation (AI tends to be more stable)
        spectral_centroid_std = features.get('spectral_centroid_std', 100)
        if spectral_centroid_std < 50:
            ai_score += 1
        elif spectral_centroid_std > 200:
            human_score += 1
        
        # 9. Zero-crossing rate variation
        zcr_std = features.get('zcr_std', 0.05)
        if zcr_std < 0.01:  # Too stable
            ai_score += 1
        elif zcr_std > 0.05:
            human_score += 1
        
        # Decision with improved confidence calibration
        if ai_score > human_score:
            score_diff = ai_score - human_score
            conf = min(0.65 + (score_diff * 0.05), 0.98)
            top_indicators = [ind for ind in indicators if any(keyword in ind.lower() 
                for keyword in ['unnatural', 'synthetic', 'abnormal', 'lacking', 'limited', 'uniform'])]
            exp = f"AI indicators detected: {', '.join(top_indicators[:2])}" if top_indicators else "Multiple AI speech patterns detected"
            return "AI_GENERATED", conf, exp
        else:
            score_diff = human_score - ai_score
            conf = min(0.65 + (score_diff * 0.05), 0.98)
            top_indicators = [ind for ind in indicators if 'natural' in ind.lower()]
            exp = f"Human speech characteristics: {', '.join(top_indicators[:2])}" if top_indicators else "Natural human speech patterns detected"
            return "HUMAN", conf, exp
    
    def _temporal_consistency_check(self, features: Dict) -> Tuple[str, float]:
        """
        Analyze temporal patterns to detect AI artifacts
        AI-generated speech often has unnatural consistency over time
        """
        score = 0.0
        
        # Check pitch variation over time (if available)
        pitch_std = features.get('pitch_stability', 0.1)
        if pitch_std < 0.02:  # Too consistent = AI
            score += 0.4
        elif pitch_std > 0.15:  # Natural variation = Human
            score -= 0.4
        
        # Check energy consistency
        energy_std = features.get('energy_std', 0)
        energy_mean = features.get('energy_mean', 1)
        if energy_mean > 0:
            energy_cv = energy_std / energy_mean
            if energy_cv < 0.1:  # Too uniform = AI
                score += 0.3
            elif energy_cv > 0.3:  # Variable = Human
                score -= 0.3
        
        # Check spectral consistency
        spectral_std = features.get('spectral_centroid_std', 100)
        if spectral_std < 30:  # Too stable = AI
            score += 0.3
        elif spectral_std > 200:  # Variable = Human
            score -= 0.3
        
        # Convert score to classification
        if score > 0.5:
            return "AI_GENERATED", min(0.7 + score * 0.1, 0.95)
        elif score < -0.5:
            return "HUMAN", min(0.7 + abs(score) * 0.1, 0.95)
        else:
            return "HUMAN", 0.5  # Neutral/uncertain
    
    def _feature_classifier(self, features: Dict) -> Tuple[str, float]:
        """
        Feature-based ML classifier (placeholder for XGBoost)
        In production, this would use a trained XGBoost model
        """
        # Mock classification based on feature combinations
        score = 0
        
        # Combine multiple features
        pitch_range = features.get('pitch_range', 100)
        energy_std = features.get('energy_std', 0.1)
        tempo = features.get('tempo', 120)
        
        # AI voices tend to have narrower ranges and more consistent energy
        if pitch_range < 50 and energy_std < 0.05:
            score = 0.7
            return "AI_GENERATED", score
        else:
            score = 0.7
            return "HUMAN", score
    
    def _generate_explanation(self, features: Dict, classification: str, explanations: list) -> str:
        """Generate human-readable explanation"""
        if classification == "AI_GENERATED":
            reasons = []
            if features.get('pitch_stability', 0.1) < 0.05:
                reasons.append("unnaturally consistent pitch patterns")
            if features.get('jitter', 0) < 0.0001:
                reasons.append("lack of natural voice quality variations")
            if features.get('spectral_flatness_mean', 0.1) < 0.01:
                reasons.append("artificial spectral characteristics")
            
            if reasons:
                return f"AI-generated speech indicators: {', '.join(reasons[:2])}"
            else:
                return "Multiple AI speech patterns detected across acoustic features"
        else:
            reasons = []
            if features.get('pitch_stability', 0.1) > 0.15:
                reasons.append("natural pitch variability")
            if features.get('silence_ratio', 0.2) > 0.15:
                reasons.append("organic pause patterns")
            
            if reasons:
                return f"Human speech characteristics: {', '.join(reasons[:2])}"
            else:
                return "Natural human speech patterns detected in prosody and voice quality"
