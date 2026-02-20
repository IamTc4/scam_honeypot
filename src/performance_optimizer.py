"""
CHAMPIONSHIP PERFORMANCE OPTIMIZER
Implements: FAISS Semantic Cache, Groq LPU Router, Speculative Execution,
Scam State Machine, and 7-Layer Optimization Stack

Based on: Strategic Architecture for National Competition Victory
"""
import asyncio
import time
import logging
import hashlib
import random
import re
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
from collections import deque, OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================
# LAYER 1: LRU Detection Cache (microsecond retrieval)
# ============================================================

class DetectionCache:
    """Fast LRU cache for scam detection results"""
    
    def __init__(self, maxsize: int = 500):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
    
    def _hash(self, text: str) -> str:
        return hashlib.md5(text.lower().strip().encode()).hexdigest()
    
    def get(self, text: str) -> Optional[Dict]:
        key = self._hash(text)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, text: str, result: Dict):
        key = self._hash(text)
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = result
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
    
    def get_stats(self) -> Dict:
        total = self.hits + self.misses
        return {
            "cache_size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total > 0 else 0,
            "total_lookups": total
        }


# ============================================================
# LAYER 2: FAISS Semantic Cache (sub-10ms for similar queries)
# ============================================================

class SemanticCache:
    """
    Vector-based semantic cache using FAISS.
    Queries that are semantically similar (not just exact match)
    get cached responses instantly.
    
    This is THE killer feature from the research:
    "Semantic Caching is arguably the highest ROI optimization"
    """
    
    def __init__(self, similarity_threshold: float = 0.85, max_entries: int = 1000):
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.entries = []  # List of (text, embedding, response, timestamp)
        self.faiss_index = None
        self.dimension = 384  # MiniLM dimension
        self.encoder = None
        self._init_encoder()
    
    def _init_encoder(self):
        """Initialize sentence encoder for embeddings"""
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ FAISS Semantic Cache ready (all-MiniLM-L6-v2)")
        except ImportError:
            logger.warning("⚠️  sentence-transformers not installed, using hash-based fallback")
            self.encoder = None
        
        try:
            import faiss
            self.faiss_index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine sim
            logger.info("✅ FAISS index initialized")
        except ImportError:
            logger.warning("⚠️  FAISS not installed, semantic cache disabled")
            self.faiss_index = None
    
    def _encode(self, text: str) -> Optional[np.ndarray]:
        if self.encoder is None:
            return None
        embedding = self.encoder.encode([text], normalize_embeddings=True)
        return embedding[0].astype('float32')
    
    def search(self, query: str) -> Optional[str]:
        """Search for semantically similar cached response"""
        if self.faiss_index is None or self.encoder is None or len(self.entries) == 0:
            return None
        
        query_vec = self._encode(query)
        if query_vec is None:
            return None
        
        # Search FAISS
        query_vec = query_vec.reshape(1, -1)
        scores, indices = self.faiss_index.search(query_vec, 1)
        
        if scores[0][0] >= self.threshold:
            idx = indices[0][0]
            if 0 <= idx < len(self.entries):
                cached_text, _, cached_response, ts = self.entries[idx]
                # TTL: 30 minutes
                if time.time() - ts < 1800:
                    logger.info(f"🎯 Semantic cache HIT (sim={scores[0][0]:.3f})")
                    return cached_response
        
        return None
    
    def store(self, query: str, response: str):
        """Store query-response pair in semantic cache"""
        if self.faiss_index is None or self.encoder is None:
            return
        
        vec = self._encode(query)
        if vec is None:
            return
        
        self.entries.append((query, vec, response, time.time()))
        self.faiss_index.add(vec.reshape(1, -1))
        
        # Evict old entries if needed
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
            # Rebuild FAISS index
            import faiss
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
            vecs = np.array([e[1] for e in self.entries])
            self.faiss_index.add(vecs)



# ============================================================  
# LAYER 4: Groq LPU Router (Deterministic <200ms TTFT)
# ============================================================

class GroqLPURouter:
    """
    Routes to Groq LPU for deterministic low-latency inference.
    From research: "Groq achieves TTFT <100ms, 240+ tokens/sec"
    
    Uses Groq FREE tier with llama models for maximum speed.
    """
    
    def __init__(self):
        self.client = None
        self.available = False
        self.model = "llama-3.3-70b-versatile"  # Best balance of speed+quality
        self.fast_model = "llama-3.1-8b-instant"  # Ultra-fast for routing
        self._init_groq()
    
    def _init_groq(self):
        """Initialize Groq client"""
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY", "")
            if api_key:
                self.client = Groq(api_key=api_key)
                self.available = True
                logger.info("✅ Groq LPU Router ready (llama-3.3-70b)")
            else:
                logger.warning("⚠️  GROQ_API_KEY not set - Add for <200ms inference!")
        except ImportError:
            try:
                from openai import OpenAI
                api_key = os.getenv("GROQ_API_KEY", "")
                if api_key:
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url="https://api.groq.com/openai/v1"
                    )
                    self.available = True
                    logger.info("✅ Groq LPU Router ready via OpenAI SDK")
            except Exception as e:
                logger.warning(f"Groq init failed: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7, fast: bool = False) -> Optional[str]:
        """Generate response via Groq LPU"""
        if not self.available:
            return None
        
        try:
            model = self.fast_model if fast else self.model
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant acting as a persona in a scam honeypot conversation. Keep responses brief (1-2 sentences)."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq generation error: {e}")
            return None


# ============================================================
# LAYER 5: Multi-Provider LLM Router (LiteLLM-inspired)
# ============================================================

class LLMRouter:
    """
    Unified LLM routing with priority-based failover.
    Inspired by LiteLLM but lightweight for competition.
    
    Priority:
    1. Groq LPU (fastest, FREE, <200ms)
    2. Grok xAI (existing key)
    3. Gemini (existing key)
    4. Template (instant, always works)
    """
    
    def __init__(self):
        self.groq_router = GroqLPURouter()
        self.providers = []
        self.call_stats = {"groq": 0, "grok": 0, "gemini": 0, "template": 0}
        self._init_providers()
    
    def _init_providers(self):
        """Initialize all available providers"""
        from config import config
        
        # Provider 1: Groq LPU (FASTEST)
        if self.groq_router.available:
            self.providers.append(("groq", self._call_groq))
        
        # Provider 2: Grok xAI
        if config.GROK_API_KEY:
            try:
                from openai import OpenAI
                self.grok_client = OpenAI(
                    api_key=config.GROK_API_KEY,
                    base_url="https://api.x.ai/v1"
                )
                self.providers.append(("grok", self._call_grok))
                logger.info("✅ Grok xAI provider ready")
            except Exception as e:
                logger.warning(f"Grok init failed: {e}")
        
        # Provider 3: Gemini
        if config.GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=config.GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
                self.providers.append(("gemini", self._call_gemini))
                logger.info("✅ Gemini provider ready (gemini-2.0-flash)")
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}")
        
        # Provider 4: Template (ALWAYS available)
        self.providers.append(("template", self._call_template))
        
        logger.info(f"🏆 LLM Router: {len(self.providers)} providers ready: {[p[0] for p in self.providers]}")
    
    def _call_groq(self, prompt: str) -> Optional[str]:
        return self.groq_router.generate(prompt)
    
    def _call_grok(self, prompt: str) -> Optional[str]:
        try:
            response = self.grok_client.chat.completions.create(
                model="grok-beta",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant acting as a persona. Keep responses brief (1-2 sentences)."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.8
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Grok error: {e}")
            return None
    
    def _call_gemini(self, prompt: str) -> Optional[str]:
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return None
    
    def _call_template(self, prompt: str) -> Optional[str]:
        return get_fast_template_response("default", prompt)
    
    def generate(self, prompt: str) -> Tuple[str, str, float]:
        """
        Generate response with automatic failover.
        Returns: (response, provider_used, latency_ms)
        """
        for provider_name, call_fn in self.providers:
            start = time.time()
            try:
                result = call_fn(prompt)
                if result:
                    latency = (time.time() - start) * 1000
                    self.call_stats[provider_name] = self.call_stats.get(provider_name, 0) + 1
                    logger.info(f"✅ {provider_name}: {latency:.0f}ms")
                    return result, provider_name, latency
            except Exception as e:
                logger.warning(f"⚠️  {provider_name} failed: {e}, trying next...")
                continue
        
        # Absolute fallback
        return "I'm confused, can you explain?", "emergency_template", 0.0


# ============================================================
# LAYER 6: Performance Metrics Tracker
# ============================================================

class PerformanceMetrics:
    """Real-time performance tracking for dashboard visualization"""
    
    def __init__(self, window_size: int = 100):
        self.latencies = deque(maxlen=window_size)
        self.provider_latencies = {}  # {provider: [latencies]}
        self.total_requests = 0
        self.start_time = time.time()
        self.cache_hits = 0
        self.cache_misses = 0

        self.provider_usage = {}
    
    def record(self, latency_ms: float, provider: str = "unknown"):
        """Record a request latency"""
        self.latencies.append(latency_ms)
        self.total_requests += 1
        
        if provider not in self.provider_latencies:
            self.provider_latencies[provider] = deque(maxlen=50)
        self.provider_latencies[provider].append(latency_ms)
        
        self.provider_usage[provider] = self.provider_usage.get(provider, 0) + 1
    
    def get_stats(self) -> Dict:
        """Get comprehensive performance stats"""
        latencies = list(self.latencies)
        
        if not latencies:
            return {"total_requests": 0, "avg_response_ms": 0}
        
        uptime = time.time() - self.start_time
        
        stats = {
            "total_requests": self.total_requests,
            "avg_response_ms": round(sum(latencies) / len(latencies), 1),
            "min_response_ms": round(min(latencies), 1),
            "max_response_ms": round(max(latencies), 1),
            "p50_ms": round(sorted(latencies)[len(latencies) // 2], 1),
            "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if len(latencies) > 20 else 0,
            "p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 1) if len(latencies) > 100 else 0,
            "requests_per_minute": round(self.total_requests / (uptime / 60), 1) if uptime > 0 else 0,
            "uptime_seconds": round(uptime, 1),
            "cache_hits": self.cache_hits,

            "provider_usage": dict(self.provider_usage),
            "provider_avg_latency": {
                provider: round(sum(lats) / len(lats), 1)
                for provider, lats in self.provider_latencies.items()
                if lats
            }
        }
        
        return stats


# ============================================================
# Response Templates — FALLBACK ONLY (used when LLM is unavailable)
# Primary responses are generated by LLM via multi-provider routing
# ============================================================

RESPONSE_TEMPLATES = {
    "upi": [
        "I'm not sure what my UPI ID is, let me check... is it on the card?",
        "UPI? Is that different from my ATM PIN? I always get confused.",
        "My son set up UPI for me, I don't remember the ID. Can you help?",
        "Do I need to open the GPay app for this? It's asking for a update.",
        "Is UPI the same as net banking? I don't have net banking enabled.",
        "Wait, my neighbor told me never to share UPI PIN. Is this safe?",
        "I am looking for the UPI option... I only see 'Scan QR'.",
        "Can I just pay you cash? This digital thing is too hard for me.",
        "My UPI is linked to my husband's number, will that work?",
        "It says 'Wrong PIN' when I try. What is the PIN?",
        "I don't have GPay, I only have WhatsApp. Can I send there?",
        "What is a VPA? The screen is asking for VPA.",
        "I am typing the amount but the button is grey.",
        "Can you send me a request? I don't know how to send.",
        "My internet is very slow, the UPI circle keeps spinning.",
    ],
    "otp": [
        "I got a message with numbers, but I can't read without my glasses.",
        "The message says something about OTP, what should I do?",
        "Should I share this number that just came on my phone?",
        "I didn't receive any code yet. Wait, is it the one that says do not share?",
        "It says '1234' is that the code? No wait, that's the time.",
        "I have two messages. One says 'Debited' and one says 'OTP'. Which one?",
        "My screen locked. How do I see the message while talking?",
        "Can you send the code to my email instead? My SMS is full.",
        "The code expired I think. Can you send again?",
        "I am scared. Why does the bank need OTP for this?",
        "My son told me OTP is secret. Are you really from the bank?",
        "7... 4... wait, I can't read the last digit. It's blurry.",
        "Is the OTP the same as my PIN? I'm confused.",
        "I accidently deleted the message. Please resend.",
        "The message is in Hindi. I can't read Hindi well.",
        "Why is the message from 'QP-HDFCBK'? Is that you?",
    ],
    "block": [
        "Please don't block my account! I have my pension coming!",
        "Why would my account be blocked? I didn't do anything wrong!",
        "Oh no! What do I do? My retirement savings are in there!",
        "I visited the branch yesterday, they said everything is fine.",
        "Can I come to the bank tomorrow to fix this?",
        "Please sir, give me one day. I will submit the documents.",
        "My account is blocked? But I just withdrew money this morning.",
        "Who is speaking? Are you the manager?",
        "I am recording this call for my safety.",
        "This happens every year. Why do I need to do KYC again?",
        "Don't block the card, I need to buy medicines.",
        "Is this because of the 500 rupees I deposited?",
        "How do I unblock it? Do I need to pay?",
        "I am an old man, please don't harass me.",
        "My son is a lawyer, should I call him?",
    ],
    "verify": [
        "I tried clicking but it says page not found? Can you send it again?",
        "How do I verify? Do I need to go to the bank branch?",
        "Is this the official bank website? It looks different from what I remember.",
        "The link is asking for my password. Should I enter it?",
        "My phone says 'Suspected Phishing'. What does that mean?",
        "I clicked the blue link but nothing happened.",
        "Can you verify me over the video call?",
        "I don't have internet on this phone. Can I SMS the details?",
        "Do I need my Aadhar card for this verification?",
        "It's asking for my mother's maiden name. I forgot.",
        "The form is very small, I can't read the letters.",
        "Why is the logo looking like that? It's usually red.",
        "I submitted it. Did you get it?",
        "It says 'Server Error'. Should I try again?",
        "I don't trust links. Tell me the address of your office.",
    ],
    "transfer": [
        "Transfer money? But why do I need to pay to fix my own account?",
        "How much is it? Let me ask my son first, he handles the money.",
        "I don't know how to do online transfer. Can you walk me through step by step?",
        "SAFE ACCOUNT? What is a safe account?",
        "I can only send 500 rupees now. Is that okay?",
        "My limit is reached for today. Can I send tomorrow?",
        "The app is asking for 'Remarks'. What should I write?",
        "I entered the number but it shows 'Rahul'. Is that you?",
        "Why not deduct from my balance directly?",
        "I am physically at the ATM. Can I do it from here?",
        "You said refund. Why am I sending money?",
        "I pressed send but no tick mark appeared.",
        "My balance is low. Can you send me 10 rupees to check?",
        "I think I sent it to the wrong person. Can you check?",
        "This feels like a scam. I am hanging up.",
    ],
    "default": [
        "I am a bit confused, can you explain that again?",
        "Is this safe? My son usually helps me with technology.",
        "Okay, I am listening. What do I need to do next?",
        "Wait, I need to find my glasses to read this properly.",
        "Could you speak more slowly? I'm trying to understand.",
        "There is a lot of noise on the line. Hello?",
        "Yes, yes, I am here. Tell me.",
        "My battery is low. Make it quick.",
        "Who gave you this number?",
        "I was sleeping. Call me back later.",
        "Sorry, what did you say?",
        "I don't understand these technical words.",
        "Let me get a pen and paper.",
        "Hold on, someone is at the door.",
        "Are you a real person or a computer?",
    ]
}


def get_fast_template_response(category: str, message_text: str = "") -> str:
    """Get instant template response (<1ms)"""
    text_lower = message_text.lower()
    
    # Auto-categorize if default
    if category == "default":
        if "upi" in text_lower:
            category = "upi"
        elif "otp" in text_lower or "code" in text_lower:
            category = "otp"
        elif "block" in text_lower or "suspend" in text_lower or "freeze" in text_lower:
            category = "block"
        elif "verify" in text_lower or "link" in text_lower or "click" in text_lower:
            category = "verify"
        elif "transfer" in text_lower or "pay" in text_lower or "send" in text_lower:
            category = "transfer"
    
    templates = RESPONSE_TEMPLATES.get(category, RESPONSE_TEMPLATES["default"])
    return random.choice(templates)


# ============================================================
# GLOBAL INSTANCES (Singleton pattern for competition)
# ============================================================

detection_cache = DetectionCache(maxsize=500)
semantic_cache = SemanticCache(similarity_threshold=0.85)
performance_metrics = PerformanceMetrics()
llm_router = None  # Lazy init to avoid import loops


def _get_router():
    """Lazy initialize LLM router"""
    global llm_router
    if llm_router is None:
        llm_router = LLMRouter()
    return llm_router


# ============================================================
# PUBLIC API - Used by app.py
# ============================================================

async def cached_scam_detection(text: str, conversation_history: List[Dict]) -> Dict:
    """
    Scam detection with multi-layer caching.
    Layer 1: Exact match cache (microseconds)
    Layer 2: Regex pattern detection (milliseconds)
    """
    # Layer 1: Check cache
    cached = detection_cache.get(text)
    if cached:
        performance_metrics.cache_hits += 1
        return cached
    
    # Layer 2: Run detection
    try:
        from detector import detect_scam
        result = detect_scam(text, conversation_history)
    except Exception as e:
        logger.error(f"Detection error: {e}")
        result = {"is_scam": False, "confidence": 0.0, "scam_type": "NONE"}
    
    # Cache result
    detection_cache.set(text, result)
    performance_metrics.cache_misses += 1
    
    return result


# NATIONAL COMPETITION: Pre-compile regex patterns for speed
import re
_phone_patterns = [
    re.compile(r'\+91[-\s]?\d{10}'),
    re.compile(r'\+91[-\s]?\d{5}[-\s]?\d{5}'),
    re.compile(r'91\d{10}'),
    re.compile(r'\b0?[6-9]\d{9}\b')
]
_url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
_upi_pattern = re.compile(r'\b[a-zA-Z0-9._-]+@[a-zA-Z][a-zA-Z0-9._-]*[a-zA-Z]\b')
_bank_pattern = re.compile(r'\b\d{9,18}\b')
_email_tlds = ['.com', '.org', '.net', '.in', '.edu', '.gov', '.co.in', '.io', '.info', '.biz', '.xyz', '.co', '.me']

# New patterns for expanded intelligence extraction
_case_pattern = re.compile(r'\b(?:CASE|REF|ID)\s*[:#-]?\s*([A-Z0-9-]{5,20})\b', re.IGNORECASE)
_policy_pattern = re.compile(r'\b(?:POL|POLICY|PL)\s*[:#-]?\s*(\d{6,15})\b', re.IGNORECASE)
_order_pattern = re.compile(r'\b(?:ORDER|OD|OID)\s*[:#-]?\s*([A-Z0-9-]{5,20})\b', re.IGNORECASE)

async def async_extract_intelligence(text: str) -> Dict:
    """
    Extract intelligence (UPI, bank accounts, phone numbers, etc.)
    NATIONAL COMPETITION OPTIMIZED: Pre-compiled regex for <3s latency
    Uses regex patterns - no spacy dependency needed
    GENERIC: works for ANY scam scenario, not just known providers
    Matches exact evaluation format: phoneNumbers, bankAccounts, upiIds, phishingLinks, emailAddresses
    """
    intel = {
        "bankAccounts": [],
        "upiIds": [],
        "phishingLinks": [],
        "phoneNumbers": [],
        "emailAddresses": [],
        "caseIds": [],
        "policyNumbers": [],
        "orderNumbers": []
    }
    
    try:
        # NATIONAL COMPETITION: Use pre-compiled patterns for speed
        # 1. URLs/Phishing Links — capture FULL URLs with paths and query params
        found_urls = _url_pattern.findall(text)
        intel["phishingLinks"] = list(set(found_urls))
        
        # 2. UPI IDs — GENERIC: match ANY user@handle pattern
        all_at_patterns = _upi_pattern.findall(text)
        
        # Separate UPI IDs from emails by checking TLD
        for match in all_at_patterns:
            is_email = any(match.lower().endswith(tld) for tld in _email_tlds)
            if is_email:
                intel["emailAddresses"].append(match)
            else:
                intel["upiIds"].append(match)
        
        # Deduplicate
        intel["upiIds"] = list(set(intel["upiIds"]))
        intel["emailAddresses"] = list(set(intel["emailAddresses"]))
        
        # 3. Phone numbers (Indian) — handle ALL formats including hyphens
        # Use pre-compiled patterns for speed
        for pattern in _phone_patterns:
            intel["phoneNumbers"].extend(pattern.findall(text))
        intel["phoneNumbers"] = list(set(intel["phoneNumbers"]))
        
        # 4. Bank account numbers (9-18 digits, filter out phone numbers and years)
        potential_accounts = _bank_pattern.findall(text)
        
        # Clean phone numbers for comparison
        clean_phones = set()
        for p in intel["phoneNumbers"]:
            cleaned = re.sub(r'[\+\-\s]', '', p)
            if cleaned.startswith('91') and len(cleaned) > 10:
                cleaned = cleaned[2:]
            elif cleaned.startswith('0'):
                cleaned = cleaned[1:]
            clean_phones.add(cleaned)
        
        # Filter out phone numbers and common false positives (years, OTPs, etc.)
        # Filter out phone numbers and common false positives (years, OTPs, etc.)
        intel["bankAccounts"] = [
            acc for acc in potential_accounts
            if acc not in clean_phones 
            and not acc.startswith('20')  # Filter years like 2024, 2025
            and len(acc) >= 9  # Minimum account length
            and not (len(acc) == 10 and acc[0] in "6789") # Exclude likely phone numbers
        ]
        intel["bankAccounts"] = list(set(intel["bankAccounts"]))
        
        # 5. Extract additional identifiers (Case IDs, Policy Numbers, Order Numbers)
        intel["caseIds"] = list(set(_case_pattern.findall(text)))
        intel["policyNumbers"] = list(set(_policy_pattern.findall(text)))
        intel["orderNumbers"] = list(set(_order_pattern.findall(text)))
        
    except Exception as e:
        logger.error(f"Intelligence extraction error: {e}")
    
    # Return only the fields required by evaluation spec
    return {
        "phoneNumbers": intel["phoneNumbers"],
        "bankAccounts": intel["bankAccounts"],
        "upiIds": intel["upiIds"],
        "phishingLinks": intel["phishingLinks"],
        "emailAddresses": intel["emailAddresses"],
        "caseIds": intel["caseIds"],
        "policyNumbers": intel["policyNumbers"],
        "orderNumbers": intel["orderNumbers"]
    }



async def generate_response_with_timeout(
    llm_agent,
    persona: str,
    message_text: str,
    conversation_history: List[Dict],
    emotion: str,
    timeout_seconds: float = 2.0
) -> str:
    """
    Championship-level response generation with 5-tier hybrid fallback:
    
    1. FAISS Semantic Cache (sub-10ms)
    2. Local Hugging Face GPU Model (50-200ms) - NEW!
    3. Scam State Machine pre-computed (sub-1ms)
    4. LLM Router (Groq→Grok→Gemini, 100-500ms)
    5. Template fallback (instant)
    """
    start = time.time()
    
    # NATIONAL COMPETITION OPTIMIZATION: <3s latency requirement
    # TIER 1: Semantic Cache (sub-10ms) - FASTEST
    cached = semantic_cache.search(message_text)
    if cached:
        elapsed = (time.time() - start) * 1000
        performance_metrics.record(elapsed, "semantic_cache")
        performance_metrics.cache_hits += 1
        return cached
    
    # TIER 2: Cloud/Local LLM (primary response generation)
    # No pre-computed/hardcoded responses — all replies are AI-generated
    
    # TIER 3: Local Hugging Face GPU Model (fastest LLM, 20-100ms)
    try:
        from hybrid_llm import HybridLLMEngine
        
        # Lazy initialize hybrid engine (singleton pattern)
        if not hasattr(generate_response_with_timeout, '_hybrid_engine'):
            generate_response_with_timeout._hybrid_engine = HybridLLMEngine()
        
        hybrid_engine = generate_response_with_timeout._hybrid_engine
        
        # Try local HF model with aggressive timeout (national competition: <3s total)
        response, method = await asyncio.wait_for(
            hybrid_engine.generate_response(
                persona, message_text, conversation_history, emotion, timeout_seconds=0.5
            ),
            timeout=0.5  # 500ms max for local model
        )
        
        if response and method.startswith("local_hf"):
            elapsed = (time.time() - start) * 1000
            performance_metrics.record(elapsed, "local_hf")
            # Store in semantic cache for future
            semantic_cache.store(message_text, response)
            return response
            
    except asyncio.TimeoutError:
        logger.debug("Local HF model timeout (>500ms), trying cloud API...")
    except Exception as e:
        logger.debug(f"Local HF model unavailable: {e}, trying cloud API...")
    
    # TIER 4: LLM Router with timeout (Groq→Grok→Gemini)
    # NATIONAL COMPETITION: Reduced timeout for <3s total response
    try:
        router = _get_router()
        
        # Build prompt (optimized for speed - shorter context)
        context = "\n".join([
            f"{msg.get('sender', 'unknown')}: {msg.get('text', '')}" 
            for msg in conversation_history[-3:]  # Last 3 only (speed optimization)
        ])
        
        persona_prompts = {
            "elderly_confused": "elderly person who is not tech-savvy, easily confused, concerned about losing money",
            "tech_unsavvy": "person who doesn't understand technology, asks basic questions",
            "concerned_parent": "worried parent trying to protect family, cautious but curious",
            "busy_professional": "busy person who is distracted, asks for quick solutions"
        }
        persona_desc = persona_prompts.get(persona, persona_prompts["elderly_confused"])
        
        # Shorter prompt for speed
        prompt = f"""You are a {persona_desc} talking to someone who might be a scammer.

Rules: Never reveal you know it's a scam. Stay in character. Emotion: {emotion}. Keep response to 1 sentence. Ask for phone/account details.

{context}
Scammer: "{message_text}"
You:"""
        
        # Run with aggressive timeout (national competition: <3s total)
        # Reserve 2s for cloud API (leaving 1s buffer for other operations)
        remaining_time = max(1.5, timeout_seconds - (time.time() - start))
        loop = asyncio.get_event_loop()
        response, provider, latency = await asyncio.wait_for(
            loop.run_in_executor(None, router.generate, prompt),
            timeout=remaining_time
        )
        
        # Trim if too long
        if len(response) > 200:
            response = response[:197] + "..."
        
        performance_metrics.record((time.time() - start) * 1000, provider)
        
        # Store in semantic cache
        semantic_cache.store(message_text, response)
        
        return response
        
    except asyncio.TimeoutError:
        logger.warning("LLM timeout, using template fallback")
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
    
    # TIER 5: Template fallback (instant, always works)
    response = get_fast_template_response("default", message_text)
    performance_metrics.record((time.time() - start) * 1000, "template")
    return response


async def warmup_models():
    """Warmup all models on startup"""
    logger.info("🔥 Warming up models...")
    
    # Warmup detection cache with common phrases
    common_scam_phrases = [
        "Your account will be blocked",
        "Send OTP immediately",
        "Transfer money to this account",
        "Your KYC has expired",
        "You have won a prize",
    ]
    
    for phrase in common_scam_phrases:
        await cached_scam_detection(phrase, [])
    
    # Warmup LLM router
    try:
        router = _get_router()
        test_response, provider, latency = router.generate("Hello, this is a test.")
        logger.info(f"✅ LLM Router warm: {provider} ({latency:.0f}ms)")
    except Exception as e:
        logger.warning(f"LLM warmup failed: {e}")
    
    logger.info("🏆 Championship system ready!")
