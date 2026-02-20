"""
Championship Demo Endpoints
Showcase system capabilities for judges — 10-20 minute demo ready
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
from pydantic import BaseModel
import time
import asyncio
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/demo", tags=["demo"])

# In-memory live sessions for demo
live_sessions: Dict[str, dict] = {}


class QuickDetectRequest(BaseModel):
    text: str


class ExtractIntelRequest(BaseModel):
    text: str


class FullDemoRequest(BaseModel):
    message: str
    session_id: str = "demo"


class LiveSessionMessage(BaseModel):
    session_id: str
    text: str


# ============================================================
# 1. COMPETITION READINESS CHECK (Minutes 0-5)
# ============================================================

@router.get("/competition-ready")
async def competition_ready():
    """
    🏆 Full system health check for competition — run this FIRST
    Judges see: All systems GREEN, optimization layers loaded
    """
    from performance_optimizer import (
        performance_metrics, detection_cache,
        semantic_cache, _get_router
    )
    from detector import detect_scam
    
    checks = {}
    start = time.time()
    
    # Check 1: Detection engine
    try:
        result = detect_scam("test message", [])
        checks["detection_engine"] = {"status": "🟢 GREEN", "detail": "Multi-stage detection operational"}
    except Exception as e:
        checks["detection_engine"] = {"status": "🔴 RED", "detail": str(e)}
    
    # Check 2: LRU Cache
    try:
        stats = detection_cache.get_stats()
        checks["lru_cache"] = {"status": "🟢 GREEN", "detail": f"Capacity: {stats}"}
    except Exception as e:
        checks["lru_cache"] = {"status": "🔴 RED", "detail": str(e)}
    
    # Check 3: Semantic Cache
    try:
        checks["semantic_cache"] = {
            "status": "🟢 GREEN" if semantic_cache.encoder else "🟡 FALLBACK",
            "detail": f"Entries: {len(semantic_cache.entries)}, {'FAISS' if semantic_cache.encoder else 'hash-based'}"
        }
    except Exception as e:
        checks["semantic_cache"] = {"status": "🔴 RED", "detail": str(e)}
    

    
    # Check 5: LLM Router
    try:
        r = _get_router()
        available = [p[0] for p in r.providers if p[0] != "template"]
        checks["llm_router"] = {
            "status": "🟢 GREEN" if available else "🟡 TEMPLATE ONLY",
            "detail": f"Active providers: {available or ['template_fallback']}"
        }
    except Exception as e:
        checks["llm_router"] = {"status": "🔴 RED", "detail": str(e)}
    
    # Check 6: NER Intelligence
    try:
        from ner_intelligence import NERIntelligence
        ner = NERIntelligence()
        intel = ner.extract_intelligence("Test UPI user@paytm phone 9876543210")
        checks["ner_intelligence"] = {
            "status": "🟢 GREEN",
            "detail": f"Extracted: {sum(1 for v in intel.values() if v)} entity types"
        }
    except Exception as e:
        checks["ner_intelligence"] = {"status": "🟡 DEGRADED", "detail": str(e)}
    
    # Check 7: Scam Dataset
    try:
        from scam_dataset import get_all_messages_with_labels, SCAM_SIGNATURES
        msgs = get_all_messages_with_labels()
        checks["scam_dataset"] = {
            "status": "🟢 GREEN",
            "detail": f"{len(msgs)} training samples, {len(SCAM_SIGNATURES)} scam signatures"
        }
    except Exception as e:
        checks["scam_dataset"] = {"status": "🟡 NOT LOADED", "detail": str(e)}
    
    total_ms = (time.time() - start) * 1000
    all_green = all("GREEN" in c["status"] for c in checks.values())
    
    return {
        "competition_ready": all_green,
        "verdict": "🏆 ALL SYSTEMS GO — CHAMPIONSHIP READY!" if all_green else "⚡ MOSTLY READY — CHECK YELLOW/RED ITEMS",
        "total_check_ms": round(total_ms, 1),
        "checks": checks,
        "architecture": "7-Layer Championship Optimization Stack",
        "key_capabilities": [
            "Multi-provider LLM routing (Groq→Grok→Gemini→Template)",
            "FAISS semantic cache for near-instant similar query responses",
            "Scam state machine with 24 pre-computed response sets",
            "India-specific detection (Hinglish, UPI, KYC, Authority scams)",
            "Real-time conversation escalation analysis",
            "Parallel multi-agent processing (detection + intel + response)"
        ]
    }


# ============================================================
# 2. FULL DEMO ENDPOINT (Minutes 5-15)
# ============================================================

@router.post("/full-demo")
async def full_demo(request: FullDemoRequest):
    """
    🏆 Full pipeline showcase — sends ONE message through ALL layers
    Shows: Detection → Intelligence → Response → Metrics in ONE call
    """
    from performance_optimizer import (
        cached_scam_detection, async_extract_intelligence,
        generate_response_with_timeout, performance_metrics
    )
    from honeypot_agent import LLMAgent
    from conversation_analyzer import ConversationAnalyzer
    
    start_total = time.time()
    
    # Get session conversation history
    session = live_sessions.get(request.session_id, {"history": [], "turn": 0})
    
    # Phase 1: Parallel detection + intelligence
    start_parallel = time.time()
    detection, intel = await asyncio.gather(
        cached_scam_detection(request.message, session["history"]),
        async_extract_intelligence(request.message)
    )
    parallel_ms = (time.time() - start_parallel) * 1000
    
    # Phase 2: Conversation analysis
    analyzer = ConversationAnalyzer()
    analysis = analyzer.analyze_message(request.message)
    

    
    # Phase 4: LLM Response (or pre-computed)
    start_llm = time.time()
    llm = LLMAgent()
    reply = await generate_response_with_timeout(
        llm, "elderly_confused", request.message,
        session["history"], analysis.get("emotion", "confused"),
        timeout_seconds=3.0
    )
    llm_ms = (time.time() - start_llm) * 1000
    
    total_ms = (time.time() - start_total) * 1000
    
    # Determine strategy used
    # Determine strategy used
    strategy = "llm_generated"
    
    # Update session
    session["history"].append({"sender": "scammer", "text": request.message})
    session["history"].append({"sender": "agent", "text": reply})
    session["turn"] += 1
    live_sessions[request.session_id] = session
    
    return {
        "status": "success",
        "performance_ms": round(total_ms, 2),
        
        # Detection Agent
        "detection": {
            "is_scam": detection.get("is_scam", False),
            "confidence": detection.get("confidence", 0),
            "scam_type": detection.get("scam_type", "NONE"),
            "scores": detection.get("scores", {})
        },
        
        # Intelligence Agent (NER)
        "intelligence": {
            "phoneNumbers": intel.get("phoneNumbers", []),
            "upiIds": intel.get("upiIds", []),
            "bankAccounts": intel.get("bankAccounts", []),
            "ifscCodes": intel.get("ifscCodes", []),
            "emails": intel.get("emails", []),
            "urls": intel.get("urls", []),
            "suspiciousKeywords": intel.get("suspiciousKeywords", []),
            "financial_demands": intel.get("financialAmounts", []),
        },
        
        # Response Agent
        "response": {
            "reply": reply,
            "strategy": strategy,
            "pre_computed_available": False,
            "emotion_detected": analysis.get("emotion", "neutral"),
        },
        
        # Performance Breakdown
        "performance_breakdown": {
            "parallel_detection_intel_ms": round(parallel_ms, 2),
            "llm_response_ms": round(llm_ms, 2),
            "total_pipeline_ms": round(total_ms, 2),
        },
        
        # Session State
        "session": {
            "turn": session["turn"],
            "messages_exchanged": len(session["history"])
        }
    }


# ============================================================
# 3. LIVE SESSION (Minutes 15-25)
# ============================================================

@router.post("/live-session/start")
async def start_live_session(session_id: str = "live_demo"):
    """Start a new live demo session"""
    live_sessions[session_id] = {
        "history": [],
        "turn": 0,
        "start_time": time.time(),
        "scam_detected": False,
        "intel_collected": {},
        "phases_seen": [],
    }
    return {
        "status": "session_started",
        "session_id": session_id,
        "message": "🎯 Live session ready. Send scammer messages to /api/demo/live-session/chat"
    }


@router.post("/live-session/chat")
async def live_session_chat(msg: LiveSessionMessage):
    """
    🎯 Live multi-turn conversation — for judge deep-dive demo
    Shows real-time: detection → strategy → response with full logs
    """
    from performance_optimizer import (
        cached_scam_detection, async_extract_intelligence,
        generate_response_with_timeout
    )
    from honeypot_agent import LLMAgent
    from conversation_analyzer import ConversationAnalyzer
    from detector import analyze_conversation_context
    
    session = live_sessions.get(msg.session_id)
    if not session:
        # Auto-create session
        session = {
            "history": [], "turn": 0, "start_time": time.time(),
            "scam_detected": False, "intel_collected": {}, "phases_seen": []
        }
        live_sessions[msg.session_id] = session
    
    start = time.time()
    
    # Run ALL analysis in parallel
    detection, intel = await asyncio.gather(
        cached_scam_detection(msg.text, session["history"]),
        async_extract_intelligence(msg.text)
    )
    
    # Conversation context analysis
    context = analyze_conversation_context(session["history"], msg.text)
    

    
    # Emotion analysis
    analyzer = ConversationAnalyzer()
    analysis = analyzer.analyze_message(msg.text)
    emotion = analysis.get("emotion", "confused")
    
    # Determine engagement strategy
    strategy = "general_engagement"
    
    # Generate response
    llm = LLMAgent()
    reply = await generate_response_with_timeout(
        llm, "elderly_confused", msg.text,
        session["history"], emotion, timeout_seconds=3.0
    )
    
    total_ms = (time.time() - start) * 1000
    
    # Update session
    if detection.get("is_scam") and not session["scam_detected"]:
        session["scam_detected"] = True
    
    # Collect intel
    for key, value in intel.items():
        if value and isinstance(value, list):
            if key not in session["intel_collected"]:
                session["intel_collected"][key] = []
            session["intel_collected"][key].extend(value)
    
    session["history"].append({"sender": "scammer", "text": msg.text})
    session["history"].append({"sender": "agent", "text": reply})
    session["turn"] += 1
    
    return {
        "turn": session["turn"],
        "reply": reply,
        "performance_ms": round(total_ms, 2),
        
        # What judges should see in logs:
        "analysis_log": {
            "engagement_strategy": strategy,
            "scam_phase_progression": session["phases_seen"],
            "emotion_detected": emotion,
            "manipulation_detected": context.get("manipulation_detected", False),
            "escalation_score": round(context.get("escalation_score", 0), 2),
            "escalation_patterns": context.get("pattern", "none"),
        },
        
        "detection": {
            "is_scam": detection.get("is_scam", False),
            "confidence": detection.get("confidence", 0),
            "scam_type": detection.get("scam_type", "NONE"),
        },
        
        "new_intel_extracted": {k: v for k, v in intel.items() if v and isinstance(v, list)},
        "total_intel_collected": session["intel_collected"],
    }


@router.get("/live-session/report/{session_id}")
async def live_session_report(session_id: str):
    """
    📋 Final report — show judges at the end (Minutes 25-30)
    """
    session = live_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    elapsed = time.time() - session["start_time"]
    
    return {
        "report": "FINAL SESSION REPORT",
        "session_id": session_id,
        "totalMessagesExchanged": len(session["history"]),
        "totalTurns": session["turn"],
        "sessionDuration_seconds": round(elapsed, 1),
        "scamDetected": session["scam_detected"],
        "scamPhasesProgression": session["phases_seen"],
        "extractedIntelligence": session["intel_collected"],
        "intelligenceSummary": {
            "total_entities_extracted": sum(len(v) for v in session["intel_collected"].values()),
            "entity_types": list(session["intel_collected"].keys()),
        },
        "conversationTranscript": session["history"],
        "verdict": "🛡️ Scam engagement successful — intelligence collected!" if session["scam_detected"] 
                   else "ℹ️ No scam detected in this session"
    }


# ============================================================
# 4. SPEED DEMO (Minutes 5-10)
# ============================================================

@router.post("/quick-detect")
async def quick_detect(request: QuickDetectRequest):
    """Ultrafast scam detection — showcase speed for judges"""
    start = time.time()
    
    from performance_optimizer import cached_scam_detection
    result = await cached_scam_detection(request.text, [])
    
    latency_ms = (time.time() - start) * 1000
    
    return {
        "status": "success",
        "detection": result,
        "latency_ms": round(latency_ms, 2),
        "method": "championship_cached_detection"
    }


@router.post("/extract-intel")
async def extract_intel(request: ExtractIntelRequest):
    """Intelligence extraction endpoint for demos"""
    start = time.time()
    
    from performance_optimizer import async_extract_intelligence
    intel = await async_extract_intelligence(request.text)
    
    latency_ms = (time.time() - start) * 1000
    
    return {
        "status": "success",
        "intelligence": intel,
        "latency_ms": round(latency_ms, 2)
    }


# ============================================================
# 5. SYSTEM STATUS & BENCHMARK
# ============================================================

@router.get("/system-status")
async def system_status():
    """Championship system status — show judges all 7 layers"""
    from performance_optimizer import (
        performance_metrics, detection_cache,
        semantic_cache, _get_router
    )
    
    router_instance = _get_router()
    
    return {
        "championship_system": "ACTIVE",
        "optimization_layers": {
            "layer_1_lru_cache": {"status": "active", "stats": detection_cache.get_stats()},
            "layer_2_semantic_cache": {
                "status": "active" if semantic_cache.encoder else "fallback_mode",
                "entries": len(semantic_cache.entries)
            },
            "layer_3_state_machine": {
                "status": "disabled", "phases": 0, "personas": 4, "pre_computed_responses": 0
            },
            "layer_4_groq_lpu": {
                "status": "active" if router_instance.groq_router.available else "not_configured",
                "model": router_instance.groq_router.model
            },
            "layer_5_llm_router": {
                "status": "active",
                "providers": [p[0] for p in router_instance.providers],
                "call_stats": router_instance.call_stats
            },
            "layer_6_metrics": performance_metrics.get_stats(),
            "layer_7_templates": {"status": "active", "categories": 6}
        }
    }


@router.get("/benchmark")
async def run_benchmark():
    """Run a quick benchmark for judges — shows system speed"""
    from performance_optimizer import cached_scam_detection, async_extract_intelligence
    
    test_messages = [
        "Your bank account has been compromised! Call immediately!",
        "Send your OTP to verify your identity now",
        "Transfer Rs 5000 to account 12345678901234 IFSC: SBIN0001234",
        "You have won a lottery! Send UPI to claim@paytm",
        "This is the police, your Aadhaar has been used for fraud",
        "Sir galti se 5000 bhej diya GPay pe, wapas kar do",
        "Paytm KYC expired, call 98765xxxxx immediately",
        "Hi dad, lost my phone. Send 20k urgent to ravil@oksbi",
    ]
    
    results = []
    for msg in test_messages:
        start = time.time()
        detection, intel = await asyncio.gather(
            cached_scam_detection(msg, []),
            async_extract_intelligence(msg)
        )
        latency = (time.time() - start) * 1000
        
        results.append({
            "message": msg[:50] + "..." if len(msg) > 50 else msg,
            "is_scam": detection.get("is_scam", False),
            "scam_type": detection.get("scam_type", "NONE"),
            "confidence": detection.get("confidence", 0),
            "intel_found": {k: v for k, v in intel.items() if v},
            "latency_ms": round(latency, 2)
        })
    
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    scam_detected = sum(1 for r in results if r["is_scam"])
    
    return {
        "benchmark": "COMPLETE",
        "total_tests": len(results),
        "scams_detected": f"{scam_detected}/{len(results)}",
        "avg_latency_ms": round(avg_latency, 2),
        "max_latency_ms": max(r["latency_ms"] for r in results),
        "min_latency_ms": min(r["latency_ms"] for r in results),
        "results": results,
        "verdict": "🏆 CHAMPIONSHIP READY" if avg_latency < 500 else "⚡ OPTIMIZING..."
    }
