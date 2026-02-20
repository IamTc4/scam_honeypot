"""
Persona Manager for adaptive scam honeypot behavior
"""
from typing import Dict
import random

class PersonaManager:
    def __init__(self):
        """Initialize persona profiles"""
        self.personas = {
            "elderly_confused": {
                "name": "Elderly Confused User",
                "traits": ["easily confused", "worried about money", "not tech-savvy", "trusting"],
                "initial_emotion": "confused",
                "response_style": "hesitant and asking many questions",
                "trust_level": 0.7  # Higher = more trusting
            },
            "tech_unsavvy": {
                "name": "Tech-Unsavvy Individual",
                "traits": ["doesn't understand technology", "asks basic questions", "cautious"],
                "initial_emotion": "uncertain",
                "response_style": "slow to understand, asks for clarification",
                "trust_level": 0.5
            },
            "concerned_parent": {
                "name": "Concerned Parent",
                "traits": ["protective", "skeptical", "wants to verify", "busy"],
                "initial_emotion": "worried",
                "response_style": "asks probing questions, more suspicious",
                "trust_level": 0.3
            },
            "busy_professional": {
                "name": "Busy Professional",
                "traits": ["distracted", "wants quick solutions", "impatient"],
                "initial_emotion": "stressed",
                "response_style": "short responses, wants to resolve quickly",
                "trust_level": 0.4
            }
        }
        
        self.emotion_transitions = {
            "confused": ["worried", "curious", "frustrated"],
            "worried": ["scared", "confused", "cautious"],
            "scared": ["panicked", "worried", "compliant"],
            "curious": ["interested", "confused", "skeptical"],
            "skeptical": ["suspicious", "curious", "dismissive"],
            "compliant": ["trusting", "worried", "confused"]
        }
    
    def get_persona(self, persona_id: str = None) -> Dict:
        """Get persona profile"""
        if persona_id and persona_id in self.personas:
            return self.personas[persona_id]
        return self.personas["elderly_confused"]  # Default
    
    def update_emotion(self, current_emotion: str, scammer_message: str) -> str:
        """
        Update emotional state based on scammer's message
        """
        msg_lower = scammer_message.lower()
        
        # Urgency/threat increases fear
        if any(word in msg_lower for word in ["urgent", "immediately", "now", "blocked", "suspended"]):
            return "worried" if current_emotion not in ["scared", "panicked"] else "scared"
        
        # Requests for info increase skepticism (unless already scared)
        elif any(word in msg_lower for word in ["share", "send", "provide", "give"]):
            if current_emotion in ["scared", "worried"]:
                return "compliant"
            else:
                return "skeptical"
        
        # Questions from scammer
        elif "?" in scammer_message:
            return "curious"
        
        # Natural transition
        else:
            possible = self.emotion_transitions.get(current_emotion, ["confused"])
            return random.choice(possible)
    
    def get_engagement_strategy(self, persona_id: str, turn_count: int, emotion: str) -> Dict:
        """
        Determine engagement strategy based on persona, turn count, and emotion
        """
        persona = self.get_persona(persona_id)
        strategy = {
            "should_ask_question": True,
            "should_show_compliance": False,
            "should_express_doubt": False,
            "detail_level": "medium"
        }
        
        # Early turns: be curious/confused
        if turn_count < 3:
            strategy["should_ask_question"] = True
            strategy["detail_level"] = "high"
        
        # Middle turns: show some compliance if scared
        elif 3 <= turn_count < 7:
            if emotion in ["scared", "worried", "compliant"]:
                strategy["should_show_compliance"] = True
                strategy["should_ask_question"] = False
            elif emotion in ["skeptical", "suspicious"]:
                strategy["should_express_doubt"] = True
        
        # Late turns: start wrapping up
        else:
            if persona["trust_level"] > 0.5:
                strategy["should_show_compliance"] = True
            else:
                strategy["should_express_doubt"] = True
        
        return strategy
