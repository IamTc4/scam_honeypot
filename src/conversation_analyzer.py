"""
Conversation Analyzer for sentiment, intent, and manipulation detection
"""
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    TextBlob = None
from typing import Dict, List
import re

class ConversationAnalyzer:
    def __init__(self):
        """Initialize analyzer"""
        self.urgency_keywords = ['urgent', 'immediately', 'now', 'asap', 'quick', 'fast', 'hurry']
        self.threat_keywords = ['block', 'suspend', 'close', 'terminate', 'expire', 'cancel', 'freeze']
        self.money_keywords = ['pay', 'send', 'transfer','money', 'rupees', 'amount', 'deposit']
        self.trust_keywords = ['verify', 'confirm', 'secure', 'protect', 'safe', 'official']
    
    def analyze_message(self, text: str) -> Dict:
        """
        Comprehensive message analysis
        Returns sentiment, intent, urgency, manipulation tactics
        """
        analysis = {
            "sentiment": self._analyze_sentiment(text),
            "intent": self._classify_intent(text),
            "urgency_score": self._calculate_urgency(text),
            "manipulation_tactics": self._detect_manipulation(text),
            "risk_indicators": []
        }
        
        # Risk indicators
        if analysis["urgency_score"] > 0.7:
            analysis["risk_indicators"].append("high_urgency")
        if "threat" in analysis["manipulation_tactics"]:
            analysis["risk_indicators"].append("threatening_language")
        if "money_request" in analysis["intent"]:
            analysis["risk_indicators"].append("financial_request")
        
        return analysis
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Sentiment analysis using TextBlob or fallback"""
        try:
            if not TEXTBLOB_AVAILABLE:
                raise ImportError("TextBlob not available")
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"
            
            return {
                "label": label,
                "polarity": round(polarity, 2),
                "subjectivity": round(subjectivity, 2)
            }
        except:
            return {"label": "neutral", "polarity": 0.0, "subjectivity": 0.5}
    
    def _classify_intent(self, text: str) -> List[str]:
        """Classify message intent"""
        intents = []
        text_lower = text.lower()
        
        # Information request
        if any(word in text_lower for word in ['share', 'provide', 'give', 'send', 'tell']):
            intents.append("information_request")
        
        # Money request
        if any(word in text_lower for word in self.money_keywords):
            intents.append("money_request")
        
        # Verification request
        if any(word in text_lower for word in ['verify', 'confirm', 'check', 'validate']):
            intents.append("verification_request")
        
        # Link/action request
        if 'http' in text_lower or 'click' in text_lower or 'visit' in text_lower:
            intents.append("link_request")
        
        # Question
        if '?' in text:
            intents.append("question")
        
        if not intents:
            intents.append("statement")
        
        return intents
    
    def _calculate_urgency(self, text: str) -> float:
        """Calculate urgency score (0-1)"""
        text_lower = text.lower()
        score = 0.0
        
        # Urgency keywords
        urgency_count = sum(1 for word in self.urgency_keywords if word in text_lower)
        score += min(urgency_count * 0.2, 0.4)
        
        # Threat keywords
        threat_count = sum(1 for word in self.threat_keywords if word in text_lower)
        score += min(threat_count * 0.25, 0.5)
        
        # Exclamation marks
        exclamations = text.count('!')
        score += min(exclamations * 0.1, 0.2)
        
        # All caps words
        words = text.split()
        caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
        score += min(caps_words * 0.1, 0.2)
        
        return min(score, 1.0)
    
    def _detect_manipulation(self, text: str) -> List[str]:
        """Detect manipulation tactics"""
        tactics = []
        text_lower = text.lower()
        
        # Urgency/scarcity
        if any(word in text_lower for word in self.urgency_keywords):
            tactics.append("urgency")
        
        # Threats
        if any(word in text_lower for word in self.threat_keywords):
            tactics.append("threat")
        
        # Authority (impersonation)
        if any(word in text_lower for word in ['bank', 'government', 'police', 'tax', 'official']):
            tactics.append("authority_claim")
        
        # Reward/prize
        if any(word in text_lower for word in ['win', 'won', 'prize', 'lottery', 'reward', 'congratulations']):
            tactics.append("reward_lure")
        
        # Trust building
        if any(word in text_lower for word in self.trust_keywords):
            tactics.append("trust_building")
        
        # Social proof
        if any(word in text_lower for word in ['everyone', 'others', 'many people', 'customers']):
            tactics.append("social_proof")
        
        return tactics
