"""
ML-based Scam Detector using transformers
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Tuple
import re

class MLScamDetector:
    def __init__(self, model_name="distilbert-base-uncased"):
        """
        Initialize ML model for scam detection
        In production, use a fine-tuned model on scam dataset
        """
        self.available = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # For demo, we'll use a sentiment model as a proxy
            # In production, fine-tune DistilBERT on labeled scam data
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english",  # Sentiment as proxy
                num_labels=2
            )
            self.model.to(self.device)
            self.model.eval()
            self.available = True
        except Exception as e:
            print(f"⚠️ ML model not available: {e}")
    
    def predict(self, text: str) -> Tuple[float, str]:
        """
        Predict scam probability
        Returns: (scam_score, scam_type)
        """
        if not self.available:
            return self._fallback_prediction(text)
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                
                # In real model, label 1 would be "scam"
                # For sentiment proxy, negative sentiment → higher scam probability
                scam_score = float(probs[0][0])  # Negative class as scam proxy
            
            # Classify scam type
            scam_type = self._classify_scam_type(text)
            
            return scam_score, scam_type
        
        except Exception as e:
            print(f"ML prediction error: {e}")
            return self._fallback_prediction(text)
    
    def _fallback_prediction(self, text: str) -> Tuple[float, str]:
        """Keyword-based fallback"""
        text_lower = text.lower()
        score = 0.0
        
        # Scam indicators
        scam_keywords = {
            'urgent': 0.15,
            'verify': 0.1,
            'blocked': 0.2,
            'suspend': 0.2,
            'expire': 0.15,
            'otp': 0.1,
            'cvv': 0.25,
            'password': 0.2,
            'bank': 0.1,
            'upi': 0.15,
            'prize': 0.2,
            'lottery': 0.25,
            'congratulations': 0.15
        }
        
        for keyword, weight in scam_keywords.items():
            if keyword in text_lower:
                score += weight
        
        scam_type = self._classify_scam_type(text)
        return min(score, 1.0), scam_type
    
    def _classify_scam_type(self, text: str) -> str:
        """Classify type of scam"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['upi', 'payment', 'transfer', 'gpay', 'paytm']):
            return "UPI_SCAM"
        elif any(word in text_lower for word in ['bank', 'account', 'blocked', 'suspend']):
            return "BANK_SCAM"
        elif any(word in text_lower for word in ['prize', 'lottery', 'winner', 'won', 'congratulations']):
            return "PRIZE_SCAM"
        elif any(word in text_lower for word in ['kyc', 'verify', 'update', 'document']):
            return "KYC_SCAM"
        elif any(word in text_lower for word in ['otp', 'code', 'cvv', 'password']):
            return "CREDENTIAL_THEFT"
        elif 'http' in text_lower or 'click' in text_lower:
            return "PHISHING"
        else:
            return "GENERIC_SCAM"
