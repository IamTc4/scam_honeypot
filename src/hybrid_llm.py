"""
Hybrid LLM System - GPU Local Model + API Fallback
Optimized for RTX 4050 (14GB VRAM)
Uses Hugging Face models for local inference
"""
import torch
import logging
from typing import Optional, List, Dict, Tuple
import time
import asyncio
import os

logger = logging.getLogger(__name__)

class HybridLLMEngine:
    """
    Intelligent LLM with 3-tier fallback:
    1. Local GPU model (fastest, ~50-200ms)
    2. Cloud API (Grok/Gemini, ~500-1500ms)
    3. Templates (emergency, <10ms)
    """
    
    def __init__(self):
        self.local_model = None
        self.local_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_available = torch.cuda.is_available()
        
        if self.gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"🎮 GPU DETECTED: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        
        # Try to load local model
        self._load_local_model()
        
        # Import API fallback
        try:
            from honeypot_agent import LLMAgent
            self.api_agent = LLMAgent()
            logger.info("✅ API fallback loaded (Grok/Gemini)")
        except:
            self.api_agent = None
            logger.warning("⚠️  API fallback not available")
    
    def _load_local_model(self):
        """Load local GPU-accelerated Hugging Face model"""
        if not self.gpu_available:
            logger.warning("⚠️  No GPU detected, skipping local model")
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Check for Hugging Face token if needed
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            
            logger.info("🔄 Loading local Hugging Face model (this may take 30-60s first time)...")
            
            # NATIONAL COMPETITION OPTIMIZATION: Use fastest model for <3s latency
            # Options ranked by speed (for <3s total response time):
            # 1. TinyLlama (1.1B) - 20-50ms, fits in 4GB - FASTEST
            # 2. Phi-2 (2.7B) - 50-100ms, fits in 6GB - GOOD BALANCE
            # 3. Qwen2.5-1.5B - 80-150ms, good quality
            # 4. Mistral-7B (7B) - 100-200ms, fits in 14GB
            
            # Use TinyLlama for maximum speed (national competition requirement: <3s)
            # Falls back to Phi-2 if TinyLlama unavailable
            model_name = os.getenv("HF_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            
            # Load tokenizer
            self.local_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=hf_token
            )
            
            # Set pad token if not set
            if self.local_tokenizer.pad_token is None:
                self.local_tokenizer.pad_token = self.local_tokenizer.eos_token
            
            # Load model with optimizations
            self.local_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Half precision for speed
                device_map="auto",  # Auto device placement
                trust_remote_code=True,
                token=hf_token,
                low_cpu_mem_usage=True  # Optimize memory usage
            )
            
            self.local_model.eval()  # Inference mode
            
            logger.info(f"✅ Local Hugging Face model loaded: {model_name}")
            if self.gpu_available:
                logger.info(f"💾 VRAM usage: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
            
        except Exception as e:
            logger.error(f"❌ Failed to load local Hugging Face model: {e}")
            logger.info("💡 Falling back to cloud API only")
            self.local_model = None
            self.local_tokenizer = None
    
    async def generate_response(
        self,
        persona: str,
        message_text: str,
        conversation_history: List[Dict],
        emotion: str,
        timeout_seconds: float = 2.0
    ) -> Tuple[str, str]:
        """
        Generate response with intelligent hybrid fallback
        Returns: (response_text, method_used)
        Priority: Local HF Model → Cloud API → Templates
        """
        
        # TIER 1: Try local Hugging Face GPU model first (fastest, ~20-100ms)
        # NATIONAL COMPETITION: Aggressive timeout for <3s total response
        if self.local_model is not None and self.local_tokenizer is not None:
            try:
                start = time.time()
                response = await self._generate_local(
                    persona, message_text, conversation_history, emotion,
                    max_time_ms=300  # 300ms budget for local model (national competition)
                )
                elapsed_ms = (time.time() - start) * 1000
                
                if response and len(response.strip()) > 5:  # Valid response
                    logger.info(f"🚀 LOCAL HF MODEL: {elapsed_ms:.0f}ms")
                    return response.strip(), f"local_hf_{elapsed_ms:.0f}ms"
                    
            except Exception as e:
                logger.warning(f"⚠️  Local HF model failed: {e}, trying API...")
        
        # TIER 2: Fall back to cloud API (Groq/Grok/Gemini)
        if self.api_agent and hasattr(self.api_agent, 'available') and self.api_agent.available:
            try:
                start = time.time()
                response = self.api_agent.generate_response(
                    persona, message_text, conversation_history, emotion
                )
                elapsed_ms = (time.time() - start) * 1000
                
                if response and len(response.strip()) > 5:
                    logger.info(f"🌐 CLOUD API: {elapsed_ms:.0f}ms")
                    return response.strip(), f"cloud_api_{elapsed_ms:.0f}ms"
                    
            except Exception as e:
                logger.warning(f"⚠️  Cloud API failed: {e}, using template...")
        
        # TIER 3: Emergency template fallback (always works, <1ms)
        from performance_optimizer import get_fast_template_response
        response = get_fast_template_response("default", message_text)
        logger.info("📋 TEMPLATE fallback")
        return response, "template_fallback"
    
    async def _generate_local(
        self,
        persona: str,
        message_text: str,
        conversation_history: List[Dict],
        emotion: str,
        max_time_ms: int = 500
    ) -> Optional[str]:
        """Generate response using local Hugging Face GPU model"""
        
        if self.local_model is None or self.local_tokenizer is None:
            return None
        
        # Build prompt
        prompt = self._build_prompt(persona, message_text, conversation_history, emotion)
        
        # Tokenize
        try:
            inputs = self.local_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            return None
        
        # Generate with timeout in executor
        # NATIONAL COMPETITION: Optimized for speed
        def generate():
            try:
                with torch.no_grad():
                    outputs = self.local_model.generate(
                        **inputs,
                        max_new_tokens=60,  # Shorter responses for speed (<3s requirement)
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.local_tokenizer.pad_token_id or self.local_tokenizer.eos_token_id,
                        eos_token_id=self.local_tokenizer.eos_token_id,
                        repetition_penalty=1.1,  # Reduce repetition
                        num_beams=1,  # Greedy decoding for speed (no beam search)
                        early_stopping=True  # Stop early if possible
                    )
                # Decode only the new tokens (skip prompt)
                generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                response = self.local_tokenizer.decode(generated_tokens, skip_special_tokens=True)
                return response.strip()
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return None
        
        try:
            # Run with timeout
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, generate),
                timeout=max_time_ms / 1000
            )
            
            if response and len(response) > 0:
                return response
            
        except asyncio.TimeoutError:
            logger.warning(f"⚠️  Local HF model timeout after {max_time_ms}ms")
        except Exception as e:
            logger.error(f"Local generation error: {e}")
        
        return None
    
    def _build_prompt(
        self,
        persona: str,
        message_text: str,
        conversation_history: List[Dict],
        emotion: str
    ) -> str:
        """Build prompt for local Hugging Face model"""
        
        # Persona descriptions
        personas = {
            "elderly_confused": "You are a confused elderly person who doesn't understand technology well. You are worried and easily confused.",
            "tech_unsavvy": "You are someone with limited tech knowledge who is nervous about scams. You ask basic questions.",
            "concerned_parent": "You are a worried parent trying to protect your family. You are cautious but curious.",
            "busy_professional": "You are a busy person who is distracted. You ask for quick solutions."
        }
        
        persona_desc = personas.get(persona, personas["elderly_confused"])
        
        # NATIONAL COMPETITION: Shorter context for speed (<3s requirement)
        # Build conversation context (last 2 messages only for speed)
        context = ""
        if conversation_history:
            for msg in conversation_history[-2:]:  # Last 2 messages only (speed optimization)
                role = "Scammer" if msg.get('sender') == 'scammer' else "You"
                context += f"{role}: {msg.get('text', '')}\n"
        
        # Build prompt in chat format (optimized for speed)
        prompt = f"""You are a {persona_desc} talking to someone who might be a scammer.

Rules: Never reveal you know it's a scam. Stay in character. Show emotion: {emotion}. Keep response to 1 sentence. Be natural. Ask for phone/account details.

{context}Scammer: {message_text}
You:"""
        
        return prompt
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        stats = {
            "gpu_available": self.gpu_available,
            "local_model_loaded": self.local_model is not None,
            "api_available": self.api_agent.available if self.api_agent else False
        }
        
        if self.gpu_available:
            stats["gpu_name"] = torch.cuda.get_device_name(0)
            stats["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            stats["vram_used_gb"] = round(torch.cuda.memory_allocated(0) / 1e9, 2)
        
        return stats
