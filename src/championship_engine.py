"""
ULTRA-COMPETITIVE ENSEMBLE SYSTEM
International Competition Level - MAXIMUM FIREPOWER
"""
import asyncio
import torch
import logging
from typing import List, Dict, Optional, Tuple
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelResponse:
    """Response from a single model"""
    text: str
    confidence: float
    latency_ms: float
    source: str  # "gpu", "claude", "gpt4", "gemini", etc.
    
class UltraCompetitiveEngine:
    """
    CHAMPIONSHIP SYSTEM - Uses ALL available resources:
    
    1. LOCAL GPU (fastest, free)
    2. CLAUDE OPUS 4 (best quality, paid)
    3. GPT-4 TURBO (excellent, paid)
    4. GEMINI PRO (good, free)
    5. GROK (backup)
    
    Strategy: Run multiple in parallel, use ensemble voting
    """
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.models = {}
        self.total_requests = 0
        self.avg_response_time = 0
        
        # Initialize all engines
        self._init_gpu_model()
        self._init_claude()
        self._init_gpt4()
        self._init_gemini()
        self._init_grok()
        
        logger.info("🏆 ULTRA-COMPETITIVE ENGINE INITIALIZED")
        logger.info(f"   GPU: {'✅' if self.gpu_available else '❌'}")
        logger.info(f"   Claude: {'✅' if self.models.get('claude') else '❌'}")
        logger.info(f"   GPT-4: {'✅' if self.models.get('gpt4') else '❌'}")
        logger.info(f"   Gemini: {'✅' if self.models.get('gemini') else '❌'}")
    
    def _init_gpu_model(self):
        """Initialize GPU model"""
        if not self.gpu_available:
            return
            
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Use FASTEST model for GPU (TinyLlama for speed)
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            self.gpu_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.gpu_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cuda"
            )
            self.gpu_model.eval()
            self.models['gpu'] = True
            logger.info("✅ GPU model loaded")
        except Exception as e:
            logger.warning(f"GPU init failed: {e}")
    
    def _init_claude(self):
        """Initialize Claude Opus 4"""
        try:
            import anthropic
            import os
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                self.claude_client = anthropic.Anthropic(api_key=api_key)
                self.models['claude'] = True
                logger.info("✅ Claude Opus 4 ready (PREMIUM)")
            else:
                logger.warning("⚠️  ANTHROPIC_API_KEY not set - Add to .env for premium quality!")
        except Exception as e:
            logger.warning(f"Claude init failed: {e}")
    
    def _init_gpt4(self):
        """Initialize GPT-4 Turbo"""
        try:
            from openai import OpenAI
            import os
            
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.gpt4_client = OpenAI(api_key=api_key)
                self.models['gpt4'] = True
                logger.info("✅ GPT-4 Turbo ready (PREMIUM)")
            else:
                logger.warning("⚠️  OPENAI_API_KEY not set - Add to .env for premium quality!")
        except Exception as e:
            logger.warning(f"GPT-4 init failed: {e}")
    
    def _init_gemini(self):
        """Initialize Gemini Pro (free tier)"""
        try:
            import google.generativeai as genai
            from config import config
            
            if config.GEMINI_API_KEY:
                genai.configure(api_key=config.GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                self.models['gemini'] = True
                logger.info("✅ Gemini Pro ready (FREE)")
        except Exception as e:
            logger.warning(f"Gemini init failed: {e}")
    
    def _init_grok(self):
        """Initialize Grok (backup)"""
        try:
            from openai import OpenAI
            from config import config
            
            if config.GROK_API_KEY:
                self.grok_client = OpenAI(
                    api_key=config.GROK_API_KEY,
                    base_url="https://api.x.ai/v1"
                )
                self.models['grok'] = True
                logger.info("✅ Grok ready (BACKUP)")
        except Exception as e:
            logger.warning(f"Grok init failed: {e}")
    
    async def generate_championship_response(
        self,
        persona: str,
        message: str,
        conversation_history: List[Dict],
        emotion: str,
        mode: str = "balanced"  # "speed", "balanced", "quality"
    ) -> Tuple[str, Dict]:
        """
        Generate response using ENSEMBLE strategy
        
        Modes:
        - speed: GPU only (20-50ms)
        - balanced: GPU + 1 premium API (100-300ms)
        - quality: All models, voting (500-1000ms)
        """
        
        start = time.time()
        
        if mode == "speed":
            # SPEED MODE: GPU only
            response = await self._gpu_response(persona, message, conversation_history, emotion)
            if response:
                return response.text, {
                    "method": "speed_gpu",
                    "latency_ms": response.latency_ms,
                    "models_used": 1
                }
        
        elif mode == "balanced":
            # BALANCED MODE: GPU + Best available premium API
            tasks = []
            
            # Always try GPU first
            if self.models.get('gpu'):
                tasks.append(self._gpu_response(persona, message, conversation_history, emotion))
            
            # Add ONE premium API (fastest available)
            if self.models.get('claude'):
                tasks.append(self._claude_response(persona, message, conversation_history, emotion))
            elif self.models.get('gpt4'):
                tasks.append(self._gpt4_response(persona, message, conversation_history, emotion))
            elif self.models.get('gemini'):
                tasks.append(self._gemini_response(persona, message, conversation_history, emotion))
            
            # Wait for FIRST to complete
            if tasks:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                
                # Cancel slower ones
                for task in pending:
                    task.cancel()
                
                # Get fastest response
                response = list(done)[0].result()
                
                elapsed = (time.time() - start) * 1000
                return response.text, {
                    "method": "balanced_fastest",
                    "latency_ms": elapsed,
                    "source": response.source,
                    "models_used": len(tasks)
                }
        
        elif mode == "quality":
            # QUALITY MODE: Multi-model ensemble with voting
            tasks = []
            
            if self.models.get('gpu'):
                tasks.append(('gpu', self._gpu_response(persona, message, conversation_history, emotion)))
            if self.models.get('claude'):
                tasks.append(('claude', self._claude_response(persona, message, conversation_history, emotion)))
            if self.models.get('gpt4'):
                tasks.append(('gpt4', self._gpt4_response(persona, message, conversation_history, emotion)))
            if self.models.get('gemini'):
                tasks.append(('gemini', self._gemini_response(persona, message, conversation_history, emotion)))
            
            # Run all in parallel
            results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
            
            # Filter valid responses
            valid_responses = []
            for (name, _), result in zip(tasks, results):
                if isinstance(result, ModelResponse):
                    valid_responses.append(result)
            
            if valid_responses:
                # ENSEMBLE VOTING: Choose best based on confidence
                best_response = max(valid_responses, key=lambda r: r.confidence)
                
                elapsed = (time.time() - start) * 1000
                return best_response.text, {
                    "method": "quality_ensemble",
                    "latency_ms": elapsed,
                    "models_used": len(valid_responses),
                    "winner": best_response.source,
                    "confidence": best_response.confidence
                }
        
        # FALLBACK: Template
        from performance_optimizer import get_fast_template_response
        response = get_fast_template_response("default", message)
        return response, {"method": "template_fallback", "latency_ms": (time.time()-start)*1000}
    
    async def _gpu_response(self, persona, message, history, emotion) -> Optional[ModelResponse]:
        """GPU model response"""
        if not self.models.get('gpu'):
            return None
        
        start = time.time()
        try:
            # Build prompt
            prompt = self._build_prompt(persona, message, history, emotion)
            
            # Generate
            loop = asyncio.get_event_loop()
            
            def generate():
                inputs = self.gpu_tokenizer(prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = self.gpu_model.generate(
                        **inputs,
                        max_new_tokens=80,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.gpu_tokenizer.eos_token_id
                    )
                return self.gpu_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            response = await asyncio.wait_for(
                loop.run_in_executor(None, generate),
                timeout=0.15  # 150ms timeout for speed
            )
            
            # Clean response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            elapsed = (time.time() - start) * 1000
            
            return ModelResponse(
                text=response,
                confidence=0.7,  # GPU is fast but less confident
                latency_ms=elapsed,
                source="gpu"
            )
            
        except asyncio.TimeoutError:
            logger.warning("GPU timeout")
            return None
        except Exception as e:
            logger.error(f"GPU error: {e}")
            return None
    
    async def _claude_response(self, persona, message, history, emotion) -> Optional[ModelResponse]:
        """Claude Opus 4 response (PREMIUM QUALITY)"""
        if not self.models.get('claude'):
            return None
        
        start = time.time()
        try:
            prompt = self._build_conversation_for_api(persona, message, history, emotion)
            
            loop = asyncio.get_event_loop()
            
            def call_claude():
                response = self.claude_client.messages.create(
                    model="claude-opus-4-20250514",  # Latest Opus
                    max_tokens=150,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            response_text = await loop.run_in_executor(None, call_claude)
            elapsed = (time.time() - start) * 1000
            
            return ModelResponse(
                text=response_text,
                confidence=0.95,  # Claude highest quality
                latency_ms=elapsed,
                source="claude_opus_4"
            )
            
        except Exception as e:
            logger.error(f"Claude error: {e}")
            return None
    
    async def _gpt4_response(self, persona, message, history, emotion) -> Optional[ModelResponse]:
        """GPT-4 Turbo response (PREMIUM)"""
        if not self.models.get('gpt4'):
            return None
        
        start = time.time()
        try:
            prompt = self._build_conversation_for_api(persona, message, history, emotion)
            
            loop = asyncio.get_event_loop()
            
            def call_gpt4():
                response = self.gpt4_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.7
                )
                return response.choices[0].message.content
            
            response_text = await loop.run_in_executor(None, call_gpt4)
            elapsed = (time.time() - start) * 1000
            
            return ModelResponse(
                text=response_text,
                confidence=0.92,  # GPT-4 excellent
                latency_ms=elapsed,
                source="gpt4_turbo"
            )
            
        except Exception as e:
            logger.error(f"GPT-4 error: {e}")
            return None
    
    async def _gemini_response(self, persona, message, history, emotion) -> Optional[ModelResponse]:
        """Gemini Pro response (FREE)"""
        if not self.models.get('gemini'):
            return None
        
        start = time.time()
        try:
            prompt = self._build_conversation_for_api(persona, message, history, emotion)
            
            loop = asyncio.get_event_loop()
            
            def call_gemini():
                response = self.gemini_model.generate_content(prompt)
                return response.text
            
            response_text = await loop.run_in_executor(None, call_gemini)
            elapsed = (time.time() - start) * 1000
            
            return ModelResponse(
                text=response_text,
                confidence=0.85,
                latency_ms=elapsed,
                source="gemini_pro"
            )
            
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return None
    
    def _build_prompt(self, persona, message, history, emotion):
        """Build prompt for GPU model"""
        persona_desc = {
            "elderly_confused": "You are a confused elderly person.",
            "tech_novice": "You are nervous about technology.",
            "trusting_victim": "You are a trusting person."
        }.get(persona, "You are confused.")
        
        return f"{persona_desc} Respond briefly (1-2 sentences) to: {message}\nYou:"
    
    def _build_conversation_for_api(self, persona, message, history, emotion):
        """Build detailed prompt for premium APIs"""
        persona_desc = {
            "elderly_confused": "You are a confused 70-year-old who doesn't understand technology well. You trust people calling you and get nervous about technical terms.",
            "tech_novice": "You are a 45-year-old with limited tech knowledge. You're cautious but can be convinced if the caller sounds official.",
            "trusting_victim": "You are a kind, trusting person who wants to help others. You believe people are generally honest."
        }.get(persona, "You are a confused person.")
        
        context = ""
        if history:
            context = "\n\nConversation so far:\n"
            for msg in history[-3:]:
                role = "Caller" if msg['sender'] == 'scammer' else "You"
                context += f"{role}: {msg['text']}\n"
        
        return f"""{persona_desc}

Current emotional state: {emotion}

{context}

Caller just said: "{message}"

Respond naturally as your character would. Keep it brief (1-2 sentences) and sound human. Show your confusion or concern if appropriate."""
    
    def get_stats(self):
        """Get system stats"""
        return {
            "models_available": list(self.models.keys()),
            "gpu_enabled": self.gpu_available,
            "premium_apis": ['claude' in self.models, 'gpt4' in self.models],
            "total_engines": len(self.models)
        }
