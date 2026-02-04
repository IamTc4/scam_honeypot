import random
import requests
from typing import Dict, List
from datetime import datetime
from ner_intelligence import NERIntelligence
from llm_agent import LLMAgent
from persona_manager import PersonaManager
from conversation_analyzer import ConversationAnalyzer
from config import config

CALLBACK_URL = config.GUVI_CALLBACK_URL

class AgentSession:
    def __init__(self, session_id: str, persona_id: str = "elderly_confused"):
        self.session_id = session_id
        self.persona_id = persona_id
        self.messages = []
        self.intel = {
            "bankAccounts": [],
            "upiIds": [],
            "phishingLinks": [],
            "phoneNumbers": [],
            "suspiciousKeywords": [],
            "ifscCodes": [],
            "cryptoAddresses": []
        }
        self.turn_count = 0
        self.is_active = True
        self.scam_detected = False
        self.scam_type = "UNKNOWN"
        self.emotion = "confused"
        self.callback_sent = False
    
    def update_intel(self, new_intel: dict):
        for key in self.intel:
            if key in new_intel and key != "_confidence_scores":
                self.intel[key].extend(new_intel[key])
                self.intel[key] = list(set(self.intel[key]))  # Dedupe

class AgentController:
    def __init__(self):
        self.sessions: Dict[str, AgentSession] = {}
        self.ner_intelligence = NERIntelligence()
        self.llm_agent = LLMAgent()
        self.persona_manager = PersonaManager()
        self.conversation_analyzer = ConversationAnalyzer()

    def get_session(self, session_id: str) -> AgentSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = AgentSession(session_id)
        return self.sessions[session_id]

    def process_message(self, session_id: str, message: Dict, conversation_history: List[Dict]) -> str:
        """
        Process incoming message and generate response with context-aware scam detection
        """
        from detector import detect_scam
        
        # Get or create session
        if session_id not in self.sessions:
            self.sessions[session_id] = AgentSession(session_id)
        
        session = self.sessions[session_id]
        session.turn_count += 1
        
        # Scam detection with conversation context
        detection_result = detect_scam(message['text'], conversation_history)
        
        if detection_result['is_scam'] and not session.scam_detected:
            session.scam_detected = True
            session.scam_type = detection_result['scam_type']
        
        # Extract intelligence
        intel = self.ner_intelligence.extract_intelligence(message['text'])
        session.update_intel(intel)
        
        # Analyze conversation for emotional state
        analysis = self.conversation_analyzer.analyze_message(message['text']) # Changed from analyze_turn
        session.emotion = analysis.get('emotion', 'confused')
        
        # Generate response using LLM
        reply_text = self.llm_agent.generate_response( # Changed from reply
            persona=session.persona_id,
            message_text=message['text'],
            conversation_history=conversation_history,
            emotion=session.emotion
        )
        
        # Log reply
        session.messages.append({
            "sender": "user",
            "text": reply_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Check termination condition
        should_terminate = self._should_terminate(session, intel, analysis)
        
        if should_terminate and not session.callback_sent:
            self._send_callback(session)
            session.callback_sent = True
        
        return reply_text
    
    def _should_terminate(self, session: AgentSession, intel: dict, analysis: dict) -> bool:
        """Determine if conversation should end and callback should be sent"""
        if not session.scam_detected:
            return False
        
        # Minimum turns reached
        if session.turn_count < config.MIN_TURNS_BEFORE_CALLBACK:
            return False
        
        # Maximum turns reached
        if session.turn_count >= config.MAX_TURNS_BEFORE_CALLBACK:
            return True
        
        # Critical intelligence gathered
        has_critical_intel = (
            len(session.intel["upiIds"]) > 0 or
            len(session.intel["bankAccounts"]) > 0 or
            len(session.intel["phishingLinks"]) > 0 or
            len(session.intel["cryptoAddresses"]) > 0
        )
        
        if has_critical_intel and session.turn_count >= 4:
            return True
        
        # High urgency and sufficient turns
        if analysis.get("urgency_score", 0) > 0.8 and session.turn_count >= 5:
            return True
        
        return False
    
    def _send_callback(self, session: AgentSession):
        """Send intelligence to GUVI evaluation endpoint"""
        try:
            payload = {
                "sessionId": session.session_id,
                "scamDetected": session.scam_detected,
                "totalMessagesExchanged": len(session.messages),
                "extractedIntelligence": {
                    "bankAccounts": session.intel["bankAccounts"],
                    "upiIds": session.intel["upiIds"],
                    "phishingLinks": session.intel["phishingLinks"],
                    "phoneNumbers": session.intel["phoneNumbers"],
                    "suspiciousKeywords": session.intel["suspiciousKeywords"]
                },
                "agentNotes": f"Scam type: {session.scam_type}. Persona: {session.persona_id}. Turns: {session.turn_count}. Intelligence gathered successfully."
            }
            
            # Send with retry logic
            for attempt in range(config.CALLBACK_RETRIES):
                try:
                    response = requests.post(
                        CALLBACK_URL,
                        json=payload,
                        timeout=config.CALLBACK_TIMEOUT
                    )
                    if response.status_code == 200:
                        print(f"✅ Callback successful for {session.session_id}")
                        return
                except requests.RequestException as e:
                    if attempt == config.CALLBACK_RETRIES - 1:
                        print(f"❌ Callback failed after {config.CALLBACK_RETRIES} attempts: {e}")
                    else:
                        import time
                        time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            print(f"❌ Callback error: {e}")
