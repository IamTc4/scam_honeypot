"""
LLM-powered agent for natural conversations with scammers
Supports multiple LLM providers: Grok (xAI), Gemini, OpenAI
"""
import os
from config import config
from typing import List, Dict, Optional
import random

class LLMAgent:
    def __init__(self):
        """Initialize LLM with automatic provider selection"""
        self.provider = None
        self.model = None
        self.available = False
        
        # Try to initialize LLM providers in priority order
        for provider in config.LLM_PROVIDER_PRIORITY:
            if self._initialize_provider(provider):
                self.provider = provider
                self.available = True
                print(f"✅ LLM initialized: {provider.upper()}")
                break
        
        if not self.available:
            print("⚠️ No LLM provider available, using template fallback")
        
        # Fallback templates
        self.fallback_templates = {
            "upi": [
                "I'm not sure what my UPI ID is, let me check... is it on the card?",
                "UPI? Is that different from my ATM PIN?",
                "My son set up UPI for me, I don't remember the ID. Can you help?"
            ],
            "verify": [
                "I tried clicking but it says page not found? Can you send it again?",
                "How do I verify? Do I need to go to the bank?",
                "Is this the official bank website? It looks different..."
            ],
            "otp": [
                "I didn't receive any code yet. Wait, is it 1234?",
                "The message says something about OTP, what should I do?",
                "Should I share this number that just came on my phone?"
            ],
            "block": [
                "Please don't block my account! I have my pension coming. What do I need to do?",
                "Why would my account be blocked? I didn't do anything wrong!",
                "This is very scary. How can I fix this immediately?"
            ],
            "default": [
                "I am a bit confused, can you explain?",
                "Is this safe? My son usually helps me with this.",
                "Okay, I am listening. What is the next step?",
                "Wait, I need to find my glasses to read this properly."
            ]
        }
    
    def _initialize_provider(self, provider: str) -> bool:
        """Initialize specific LLM provider"""
        try:
            if provider == "grok":
                return self._init_grok()
            elif provider == "gemini":
                return self._init_gemini()
            elif provider == "openai":
                return self._init_openai()
        except Exception as e:
            print(f"Failed to initialize {provider}: {e}")
            return False
        return False
    
    def _init_grok(self) -> bool:
        """Initialize Grok (xAI)"""
        if not config.GROK_API_KEY:
            return False
        try:
            from openai import OpenAI
            self.model = OpenAI(
                api_key=config.GROK_API_KEY,
                base_url="https://api.x.ai/v1"
            )
            # Test the connection
            self.model.models.list()
            return True
        except Exception as e:
            print(f"Grok init failed: {e}")
            return False
    
    def _init_gemini(self) -> bool:
        """Initialize Gemini"""
        if not config.GEMINI_API_KEY:
            return False
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-pro')
            return True
        except Exception as e:
            print(f"Gemini init failed: {e}")
            return False
    
    def _init_openai(self) -> bool:
        """Initialize OpenAI"""
        if not config.OPENAI_API_KEY:
            return False
        try:
            from openai import OpenAI
            self.model = OpenAI(api_key=config.OPENAI_API_KEY)
            return True
        except Exception as e:
            print(f"OpenAI init failed: {e}")
            return False
    
    def generate_response(self, persona: str, message_text: str, conversation_history: List[Dict], emotion: str = "confused") -> str:
        """
        Generate contextual response using LLM or fallback templates
        """
        if self.available:
            try:
                if self.provider == "grok":
                    return self._generate_with_grok(persona, message_text, conversation_history, emotion)
                elif self.provider == "gemini":
                    return self._generate_with_gemini(persona, message_text, conversation_history, emotion)
                elif self.provider == "openai":
                    return self._generate_with_openai(persona, message_text, conversation_history, emotion)
            except Exception as e:
                print(f"LLM generation failed: {e}, using fallback")
                return self._generate_fallback(message_text)
        else:
            return self._generate_fallback(message_text)
    
    def _build_prompt(self, persona: str, message_text: str, conversation_history: List[Dict], emotion: str) -> str:
        """Build unified prompt for all LLM providers"""
        # Build conversation context
        context = "\n".join([
            f"{msg['sender']}: {msg['text']}" 
            for msg in conversation_history[-5:]  # Last 5 messages
        ])
        
        # Persona-specific instructions
        persona_prompts = {
            "elderly_confused": "elderly person who is not tech-savvy, easily confused, concerned about losing money",
            "tech_unsavvy": "person who doesn't understand technology, asks basic questions",
            "concerned_parent": "worried parent trying to protect family, cautious but curious",
            "busy_professional": "busy person who is distracted, asks for quick solutions"
        }
        
        persona_desc = persona_prompts.get(persona, persona_prompts["elderly_confused"])
        
        # Construct prompt
        prompt = f"""You are acting as a {persona_desc} in a conversation with someone who might be a scammer.

CRITICAL RULES:
1. NEVER reveal you know this is a scam
2. Stay in character as the persona
3. Show emotion: {emotion}
4. Ask natural clarifying questions
5. Express confusion or concern appropriately
6. Keep response to 1-2 sentences maximum
7. Sometimes make small mistakes (typos, unclear questions)
8. DO NOT provide real personal information

Previous conversation:
{context}

Latest message from them: "{message_text}"

Your response (stay in character, be natural, show {emotion}):"""
        
        return prompt
    
    def _generate_with_grok(self, persona: str, message_text: str, conversation_history: List[Dict], emotion: str) -> str:
        """Generate response using Grok (xAI)"""
        prompt = self._build_prompt(persona, message_text, conversation_history, emotion)
        
        response = self.model.chat.completions.create(
            model="grok-beta",
            messages=[
                {"role": "system", "content": "You are a helpful assistant acting as a persona in a conversation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.8
        )
        
        reply = response.choices[0].message.content.strip()
        
        # Ensure reply is concise
        if len(reply) > 150:
            reply = reply[:147] + "..."
        
        return reply
    
    def _generate_with_gemini(self, persona: str, message_text: str, conversation_history: List[Dict], emotion: str) -> str:
        """Generate response using Gemini"""
        prompt = self._build_prompt(persona, message_text, conversation_history, emotion)
        
        response = self.model.generate_content(prompt)
        reply = response.text.strip()
        
        # Ensure reply is concise
        if len(reply) > 150:
            reply = reply[:147] + "..."
        
        return reply
    
    def _generate_with_openai(self, persona: str, message_text: str, conversation_history: List[Dict], emotion: str) -> str:
        """Generate response using OpenAI"""
        prompt = self._build_prompt(persona, message_text, conversation_history, emotion)
        
        response = self.model.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant acting as a persona in a conversation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.8
        )
        
        reply = response.choices[0].message.content.strip()
        
        # Ensure reply is concise
        if len(reply) > 150:
            reply = reply[:147] + "..."
        
        return reply
    
    def _generate_fallback(self, message_text: str) -> str:
        """Fallback template-based response"""
        text_lower = message_text.lower()
        
        # Keyword matching
        if "upi" in text_lower:
            return random.choice(self.fallback_templates["upi"])
        elif "verify" in text_lower or "link" in text_lower:
            return random.choice(self.fallback_templates["verify"])
        elif "otp" in text_lower or "code" in text_lower:
            return random.choice(self.fallback_templates["otp"])
        elif "block" in text_lower or "suspend" in text_lower:
            return random.choice(self.fallback_templates["block"])
        else:
            return random.choice(self.fallback_templates["default"])

