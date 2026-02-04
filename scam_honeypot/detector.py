import re
from typing import List, Dict

# Optional ML detector - only if torch is available
try:
    from ml_detector import MLScamDetector
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    MLScamDetector = None

# Initialize ML detector (lazy loading)
ml_detector = None

def get_ml_detector():
    global ml_detector
    if ML_AVAILABLE and ml_detector is None:
        ml_detector = MLScamDetector()
    return ml_detector

SCAM_KEYWORDS = [
    # Urgency indicators
    "urgent", "immediately", "now", "today", "expire", "expiring", "limited time",
    # Account/verification
    "verify", "blocked", "suspended", "freeze", "locked", "deactivate", "kyc", "update",
    # Financial
    "upi", "bank", "account", "payment", "transfer", "refund", "cashback", "reward",
    # Credentials
    "password", "otp", "cvv", "pin", "atm", "card number", "account number",
    # Lottery/Prize
    "lottery", "prize", "winner", "won", "congratulations", "claim", "reward",
    # Threats
    "legal action", "police", "arrest", "fine", "penalty", "court",
    # Phishing
    "click", "link", "verify now", "confirm", "update details",
    # Tech support
    "virus", "infected", "hacked", "security alert", "suspicious activity"
]

def detect_scam(message_text: str, conversation_history: List[Dict] = None) -> Dict:
    """
    Multi-stage scam detection with conversation context analysis
    Returns: {is_scam: bool, confidence: float, scam_type: str, scores: dict}
    """
    if conversation_history is None:
        conversation_history = []
    
    scores = {
        'keyword': 0.0,
        'ml': 0.0,
        'heuristic': 0.0,
        'context': 0.0  # NEW: conversation context score
    }
    
    # Stage 1: Keyword matching
    keyword_count = sum(1 for keyword in SCAM_KEYWORDS if keyword in message_text.lower())
    scores['keyword'] = min(keyword_count / 3.0, 1.0)  # Normalize
    
    # Stage 2: ML Classifier (placeholder - would use trained model)
    scores['ml'] = _ml_classifier(message_text)
    
    # Stage 3: Heuristic analysis
    scores['heuristic'] = _heuristic_analysis(message_text)
    
    # Stage 4: Conversation context analysis (NEW)
    if conversation_history:
        context_result = analyze_conversation_context(conversation_history, message_text)
        scores['context'] = context_result['escalation_score']
        
        # Boost overall score if manipulation detected
        if context_result['manipulation_detected']:
            scores['heuristic'] = min(scores['heuristic'] + 0.2, 1.0)
    
    # Ensemble decision with dynamic weighting
    if conversation_history:
        # More weight to context in multi-turn conversations
        final_score = (
            scores['keyword'] * 0.2 +
            scores['ml'] * 0.2 +
            scores['heuristic'] * 0.3 +
            scores['context'] * 0.3
        )
    else:
        # Standard weighting for first message
        final_score = (
            scores['keyword'] * 0.25 +
            scores['ml'] * 0.25 +
            scores['heuristic'] * 0.5
        )
    
    is_scam = final_score >= 0.6
    
    # Determine scam type
    scam_type = _determine_scam_type(message_text, scores)
    
    return {
        'is_scam': is_scam,
        'confidence': round(final_score, 2),
        'scam_type': scam_type,
        'scores': scores
    }

def _ml_classifier(text: str) -> float:
    """ML classifier prediction - returns 0.0 if ML not available."""
    if not ML_AVAILABLE:
        return 0.0
    detector = get_ml_detector()
    if detector is None:
        return 0.0
    ml_score, _ = detector.predict(text)
    return ml_score

def _determine_scam_type(text: str, scores: Dict) -> str:
    """
    Determines the most likely scam type based on text and scores.
    This is a simplified placeholder. A real implementation would be more sophisticated.
    """
    text_lower = text.lower()
    
    # Prioritize based on highest score or specific keywords
    if scores['ml'] > 0.7:
        # Use ML detector's specific type if available and confident
        detector = get_ml_detector()
        _, scam_type = detector.predict(text)
        if scam_type != "NONE":
            return scam_type

    if scores['heuristic'] > 0.6:
        if any(word in text_lower for word in ['password', 'otp', 'cvv', 'pin']):
            return "Phishing/Credential Theft"
        if any(word in text_lower for word in ['lottery', 'prize', 'won', 'claim']):
            return "Lottery/Prize Scam"
        if any(word in text_lower for word in ['urgent', 'block', 'suspend', 'legal']):
            return "Urgency/Threat Scam"
        if any(word in text_lower for word in ['bank', 'upi', 'payment', 'transfer']):
            return "Financial Scam"
    
    if scores['keyword'] > 0.5:
        if any(word in text_lower for word in ['verify', 'update details', 'kyc']):
            return "Verification Scam"

    return "NONE"

def _heuristic_analysis(text: str) -> float:
    """Enhanced heuristic checks for scam detection"""
    score = 0.0
    text_lower = text.lower()
    
    # 1. Urgency + threat combination (classic scam tactic)
    has_urgency = any(word in text_lower for word in ['urgent', 'immediately', 'now', 'today', 'asap'])
    has_threat = any(word in text_lower for word in ['block', 'suspend', 'expire', 'freeze', 'lock', 'deactivate', 'close'])
    if has_urgency and has_threat:
        score += 0.5
    elif has_urgency or has_threat:
        score += 0.2
    
    # 2. Financial + credentials (very suspicious)
    has_financial = any(word in text_lower for word in ['upi', 'bank', 'account', 'payment', 'transfer', 'wallet'])
    has_credentials = any(word in text_lower for word in ['password', 'otp', 'cvv', 'pin', 'atm'])
    if has_financial and has_credentials:
        score += 0.6
    elif has_credentials:  # Credentials alone is always suspicious
        score += 0.4
    
    # 3. Links + urgency/threat (phishing)
    has_link = any(word in text_lower for word in ['http', 'www', 'click', 'link', '.com'])
    if has_link and has_urgency:
        score += 0.4
    elif has_link and has_threat:
        score += 0.5
    
    # 4. Prize/lottery scams
    has_prize = any(word in text_lower for word in ['prize', 'winner', 'won', 'lottery', 'congratulations'])
    has_claim = any(word in text_lower for word in ['claim', 'collect', 'reward'])
    if has_prize and has_claim:
        score += 0.5
    elif has_prize:
        score += 0.3
    
    # 5. Verification requests
    has_verify = any(word in text_lower for word in ['verify', 'confirm', 'update details', 'kyc'])
    if has_verify and (has_urgency or has_threat):
        score += 0.3
    
    # 6. Legal threats
    has_legal = any(word in text_lower for word in ['legal action', 'police', 'arrest', 'court', 'fine', 'penalty'])
    if has_legal:
        score += 0.4
    
    # 7. Impersonation indicators
    has_impersonation = any(word in text_lower for word in ['bank', 'government', 'official', 'authority', 'department'])
    if has_impersonation and (has_credentials or has_verify):
        score += 0.3
    
    # 8. Urgency manipulation (excessive punctuation)
    if text.count('!') > 2 or text.count('!!!') > 0:
        score += 0.1
    if text.isupper() and len(text) > 10:  # ALL CAPS
        score += 0.1
    
    # 9. Indian-specific patterns (NEW)
    has_indian_payment = any(word in text_lower for word in ['paytm', 'phonepe', 'gpay', 'bhim', 'rupay'])
    has_indian_bank = any(word in text_lower for word in ['sbi', 'hdfc', 'icici', 'axis', 'pnb', 'ifsc'])
    if has_indian_payment or has_indian_bank:
        if has_credentials or has_urgency:
            score += 0.3
    
    return min(score, 1.0)

def analyze_conversation_context(conversation_history: List[Dict], current_message: str) -> Dict:
    """
    Analyze conversation context to detect scam patterns over time
    Returns: {"escalation_score": float, "manipulation_detected": bool, "pattern": str}
    """
    if not conversation_history:
        return {"escalation_score": 0.0, "manipulation_detected": False, "pattern": "initial"}
    
    escalation_score = 0.0
    patterns_detected = []
    
    # Extract scammer messages only
    scammer_messages = [msg['text'].lower() for msg in conversation_history if msg.get('sender') == 'scammer']
    scammer_messages.append(current_message.lower())
    
    if len(scammer_messages) < 2:
        return {"escalation_score": 0.0, "manipulation_detected": False, "pattern": "initial"}
    
    # 1. Urgency escalation detection
    urgency_words = ['urgent', 'immediately', 'now', 'today', 'asap', 'quick', 'fast']
    urgency_counts = [sum(1 for word in urgency_words if word in msg) for msg in scammer_messages]
    if len(urgency_counts) >= 2 and urgency_counts[-1] > urgency_counts[0]:
        escalation_score += 0.3
        patterns_detected.append("urgency_escalation")
    
    # 2. Threat escalation
    threat_words = ['block', 'suspend', 'close', 'legal', 'police', 'arrest', 'penalty']
    threat_counts = [sum(1 for word in threat_words if word in msg) for msg in scammer_messages]
    if len(threat_counts) >= 2 and threat_counts[-1] > threat_counts[0]:
        escalation_score += 0.4
        patterns_detected.append("threat_escalation")
    
    # 3. Credential request progression
    credential_requests = [
        any(word in msg for word in ['password', 'otp', 'cvv', 'pin', 'account number'])
        for msg in scammer_messages
    ]
    if sum(credential_requests) >= 2:  # Multiple credential requests
        escalation_score += 0.5
        patterns_detected.append("credential_harvesting")
    
    # 4. Topic shift detection (scam progression)
    topics = {
        'account_issue': ['block', 'suspend', 'freeze', 'locked'],
        'verification': ['verify', 'confirm', 'update', 'kyc'],
        'payment': ['pay', 'transfer', 'send', 'upi'],
        'credentials': ['password', 'otp', 'pin', 'cvv']
    }
    
    message_topics = []
    for msg in scammer_messages:
        msg_topics = [topic for topic, keywords in topics.items() if any(kw in msg for kw in keywords)]
        message_topics.append(msg_topics)
    
    # Check for progression: account_issue -> verification -> credentials
    if len(message_topics) >= 3:
        if ('account_issue' in message_topics[0] and 
            'verification' in message_topics[1] and 
            'credentials' in message_topics[-1]):
            escalation_score += 0.6
            patterns_detected.append("classic_scam_progression")
    
    # 5. Repetition with variation (manipulation tactic)
    if len(scammer_messages) >= 3:
        last_three = scammer_messages[-3:]
        # Check if similar keywords appear repeatedly
        common_words = set(last_three[0].split()) & set(last_three[1].split()) & set(last_three[2].split())
        if len(common_words) >= 3:  # Repeated manipulation
            escalation_score += 0.2
            patterns_detected.append("repetitive_manipulation")
    
    # 6. Reward-threat cycle detection
    has_reward = any(word in current_message.lower() for word in ['cashback', 'reward', 'prize', 'bonus'])
    has_recent_threat = any(
        any(word in msg for word in ['block', 'suspend', 'legal'])
        for msg in scammer_messages[-3:]
    )
    if has_reward and has_recent_threat:
        escalation_score += 0.3
        patterns_detected.append("reward_threat_cycle")
    
    manipulation_detected = escalation_score > 0.4
    pattern = ", ".join(patterns_detected) if patterns_detected else "normal"
    
    return {
        "escalation_score": min(escalation_score, 1.0),
        "manipulation_detected": manipulation_detected,
        "pattern": pattern,
        "conversation_length": len(scammer_messages)
    }
