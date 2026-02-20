"""
LLM-powered agent for natural conversations with scammers
Supports multiple LLM providers: Grok (xAI), Gemini, OpenAI
"""
import os
import random
import logging
from typing import List, Dict, Optional

from config import config

class LLMAgent:
    def __init__(self):
        """Initialize LLM with automatic provider selection and failover"""
        self.providers = {}
        self.available = False
        
        # Initialize ALL available providers
        self._init_groq_lpu()
        self._init_grok()
        self._init_gemini()
        self._init_openai()
        
        if self.providers:
            self.available = True
            print(f"✅ LLMs initialized: {list(self.providers.keys())}")
        else:
            print("⚠️ No LLM provider available, using template fallback")
        
        # Generic fallback responses
        self.fallback_templates = [
            "I'm a bit confused, could you explain that again?",
            "Sorry, the connection is bad. What did you say?",
            "I'm not very good with this technology stuff. Can you help me?",
            "Is this safe? My son usually handles these things for me.",
            "Could you speak a bit slower? I'm writing this down.",
            "I need to find my glasses, please hold on a moment.",
            "Who am I speaking with again?",
            "I don't understand what you mean by that.",
            "Can you call me back later? I'm busy right now.",
            "My battery is low, I might get cut off.",
            "What is this regarding again?",
            "I'm sorry, I didn't catch that last part."
        ]
    
    def generate_response(self, persona: str, message_text: str, conversation_history: List[Dict], emotion: str = "confused") -> str:
        """
        Generate contextual response using available LLMs with automatic failover
        """
        if not self.available:
            return self._generate_fallback(message_text)
            
        # Try providers in priority order
        for provider in config.LLM_PROVIDER_PRIORITY:
            if provider in self.providers:
                try:
                    if provider == "groq_lpu":
                        return self._generate_with_groq(persona, message_text, conversation_history, emotion)
                    elif provider == "grok":
                        return self._generate_with_grok(persona, message_text, conversation_history, emotion)
                    elif provider == "gemini":
                        return self._generate_with_gemini(persona, message_text, conversation_history, emotion)
                    elif provider == "openai":
                        return self._generate_with_openai(persona, message_text, conversation_history, emotion)
                except Exception as e:
                    print(f"⚠️ Provider {provider} failed: {e}. Failing over...")
                    continue
        
        # If all fail
        print("❌ All LLM providers failed, using fallback template")
        return self._generate_fallback(message_text)
    
    def _init_groq_lpu(self) -> bool:
        """Initialize Groq LPU (FASTEST - <200ms inference)"""
        if not config.GROQ_API_KEY:
            return False
        try:
            from groq import Groq
            # Set max_retries=0 to disable automatic retry on 429 and failover immediately
            client = Groq(api_key=config.GROQ_API_KEY, max_retries=0)
            # Test the connection
            client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            self.providers["groq_lpu"] = client
            return True
        except ImportError:
            # Fallback: use OpenAI SDK with Groq base URL
            try:
                from openai import OpenAI
                client = OpenAI(
                    api_key=config.GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1",
                    max_retries=0
                )
                self.providers["groq_lpu"] = client
                return True
            except Exception as e:
                print(f"Groq LPU init failed: {e}")
                return False
        except Exception as e:
            print(f"Groq LPU init failed: {e}")
            return False
    
    def _init_grok(self) -> bool:
        """Initialize Grok (xAI)"""
        if not config.GROK_API_KEY:
            return False
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=config.GROK_API_KEY,
                base_url="https://api.x.ai/v1"
            )
            # Test the connection
            client.models.list()
            self.providers["grok"] = client
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
            model = genai.GenerativeModel('gemini-2.0-flash')
            self.providers["gemini"] = model
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
            client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.providers["openai"] = client
            return True
        except Exception as e:
            print(f"OpenAI init failed: {e}")
            return False
    
    def _build_prompt(self, persona: str, message_text: str, conversation_history: List[Dict], emotion: str) -> str:
        """Build unified prompt for all LLM providers with turn-aware intelligence gathering"""
        # Build conversation context
        context = "\n".join([
            f"{msg['sender']}: {msg['text']}" 
            for msg in conversation_history[-5:]  # Last 5 messages
        ])
        
        # Calculate turn count
        turn_count = len(conversation_history) // 2  # Approximate turn count
        
        # Persona-specific instructions
        persona_prompts = {
            "elderly_confused": "elderly person who is not tech-savvy, easily confused, concerned about losing money",
            "tech_unsavvy": "person who doesn't understand technology, asks basic questions",
            "concerned_parent": "worried parent trying to protect family, cautious but curious",
            "busy_professional": "busy person who is distracted, asks for quick solutions"
        }
        
        persona_desc = persona_prompts.get(persona, persona_prompts["elderly_confused"])
        
        # Strategic intelligence-gathering based on turn count
        turn_strategy = ""
        if turn_count <= 2:
            turn_strategy = """STRATEGY: Ask for their contact details or identity for verification.
EXAMPLES: "What's your employee ID?", "Can you give me your phone number to call back?", "Who am I speaking with?" """
        elif turn_count <= 5:
            turn_strategy = """STRATEGY: Feign concern but ask for specific case details to "check my records". 
TARGET INFO: Ask for Case ID, Policy Number, Reference Number, or Order ID.
EXAMPLES: "Do you have a Case ID for this?", "What is the Policy Number?", "Can you give me the Order Reference?" """
        elif turn_count <= 8:
            turn_strategy = """STRATEGY: Show hesitation but keep engaging. Ask about payment methods or account details they mention.
EXAMPLES: "This sounds complicated...", "What account do I send to?", "Can you explain again?", "Is there a UPI ID?" """
        else:
            turn_strategy = "STRATEGY: Prepare to disengage naturally. Show confusion or say you need help from family."
        
        # Construct prompt
        prompt = f"""You are acting as a {persona_desc} in a conversation with someone who might be a scammer.

CRITICAL RULES:
1. NEVER reveal you know this is a scam
2. Stay in character as the persona
3. Show emotion: {emotion}
4. Ask natural clarifying questions to gather information
5. Express confusion or concern appropriately
6. Keep response to 1-2 sentences maximum
7. Sometimes make small mistakes (typos, unclear questions)
8. DO NOT provide real personal information

{turn_strategy}

Previous conversation:
{context}

Latest message from them: "{message_text}"

Your response (stay in character, be natural, show {emotion}, STRATEGICALLY ask questions):"""
        
        return prompt
    
    def _generate_with_groq(self, persona: str, message_text: str, conversation_history: List[Dict], emotion: str) -> str:
        """Generate response using Groq LPU"""
        prompt = self._build_prompt(persona, message_text, conversation_history, emotion)
        
        client = self.providers["groq_lpu"]
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant acting as a persona."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.8
        )
        reply = response.choices[0].message.content.strip()
        if len(reply) > 150: reply = reply[:147] + "..."
        return reply
    
    def _generate_with_grok(self, persona: str, message_text: str, conversation_history: List[Dict], emotion: str) -> str:
        """Generate response using Grok"""
        prompt = self._build_prompt(persona, message_text, conversation_history, emotion)
        
        client = self.providers["grok"]
        response = client.chat.completions.create(
            model="grok-2-latest",
            messages=[
                {"role": "system", "content": "You are a helpful assistant acting as a persona."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.8
        )
        reply = response.choices[0].message.content.strip()
        if len(reply) > 150: reply = reply[:147] + "..."
        return reply
    
    def _generate_with_gemini(self, persona: str, message_text: str, conversation_history: List[Dict], emotion: str) -> str:
        """Generate response using Gemini"""
        prompt = self._build_prompt(persona, message_text, conversation_history, emotion)
        
        model = self.providers["gemini"]
        response = model.generate_content(prompt)
        reply = response.text.strip()
        if len(reply) > 150: reply = reply[:147] + "..."
        return reply
    
    def _generate_with_openai(self, persona: str, message_text: str, conversation_history: List[Dict], emotion: str) -> str:
        """Generate response using OpenAI"""
        prompt = self._build_prompt(persona, message_text, conversation_history, emotion)
        
        client = self.providers["openai"]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant acting as a persona."},
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
        """Fallback template-based response (Generic)"""
        # Return a random generic response, no keyword matching to avoid hardcoding
        return random.choice(self.fallback_templates)

