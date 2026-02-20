"""
Hybrid Proxy Middleware — The Competitive Edge

HF Server acts as smart gateway → forwards to local laptop via ngrok static domain.
If tunnel is down → HF processes locally using APIs.

ngrok free static domain provides a STABLE URL that never changes!

Usage:
  1. On your laptop: run start_tunnel.ps1 (uses ngrok static domain)
  2. Set your permanent ngrok URL in HF Space secrets as LOCAL_TUNNEL_URL (one-time setup!)
  3. Judges hit HF → HF forwards to your laptop → FULL LOCAL POWER
"""
import os
import httpx
import logging
import time
import json
import platform
from typing import Optional

logger = logging.getLogger(__name__)

# Check if running on Hugging Face (HF sets SPACE_ID)
# Force False if on Windows (Local Laptop) to prevent self-proxy loop
IS_HF_SPACE = "SPACE_ID" in os.environ and platform.system() != "Windows"

# Tunnel URL from environment (set as HF Space secret)
# ONLY valid if we are running on HF Space (Gateway Mode)
LOCAL_TUNNEL_URL = os.getenv("LOCAL_TUNNEL_URL", "").rstrip("/") if IS_HF_SPACE else None

# Track tunnel health
tunnel_stats = {
    "forwarded": 0,
    "fallback": 0,
    "last_tunnel_error": None,
    "tunnel_active": False,
    "avg_tunnel_latency_ms": 0.0,
}


async def proxy_to_local(
    path: str,
    method: str = "POST",
    body: dict = None,
    headers: dict = None,
    timeout_seconds: float = 10.0
) -> Optional[dict]:
    """
    Forward a request to the local laptop via ngrok static domain.
    Returns the response dict, or None if tunnel is unavailable.
    """
    if not LOCAL_TUNNEL_URL:
        return None
    
    url = f"{LOCAL_TUNNEL_URL}{path}"
    
    try:
        start = time.time()
        
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            if method.upper() == "POST":
                resp = await client.post(
                    url,
                    json=body,
                    headers=headers or {}
                )
            elif method.upper() == "GET":
                resp = await client.get(url, headers=headers or {})
            else:
                return None
            
            latency = (time.time() - start) * 1000
            
            if resp.status_code == 200:
                tunnel_stats["forwarded"] += 1
                tunnel_stats["tunnel_active"] = True
                # Rolling average
                tunnel_stats["avg_tunnel_latency_ms"] = (
                    tunnel_stats["avg_tunnel_latency_ms"] * 0.8 + latency * 0.2
                )
                
                result = resp.json()
                result["_powered_by"] = "local_laptop"
                result["_tunnel_latency_ms"] = round(latency, 2)
                return result
            else:
                logger.warning(f"Tunnel returned {resp.status_code}: {resp.text[:200]}")
                tunnel_stats["last_tunnel_error"] = f"HTTP {resp.status_code}"
                return None
                
    except httpx.ConnectError:
        logger.warning("🔌 Tunnel connection failed — falling back to cloud")
        tunnel_stats["tunnel_active"] = False
        tunnel_stats["last_tunnel_error"] = "Connection refused"
        tunnel_stats["fallback"] += 1
        return None
    except httpx.TimeoutException:
        logger.warning("⏰ Tunnel timeout — falling back to cloud")
        tunnel_stats["tunnel_active"] = False
        tunnel_stats["last_tunnel_error"] = "Timeout"
        tunnel_stats["fallback"] += 1
        return None
    except Exception as e:
        logger.warning(f"❌ Tunnel error: {e} — falling back to cloud")
        tunnel_stats["tunnel_active"] = False
        tunnel_stats["last_tunnel_error"] = str(e)
        tunnel_stats["fallback"] += 1
        return None


async def check_tunnel_health() -> dict:
    """Check if the tunnel to local laptop is healthy."""
    if not LOCAL_TUNNEL_URL:
        return {
            "status": "not_configured",
            "message": "LOCAL_TUNNEL_URL not set. Running cloud-only mode.",
            "tunnel_url": None
        }
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{LOCAL_TUNNEL_URL}/health")
            if resp.status_code == 200:
                return {
                    "status": "connected",
                    "message": f"🚀 Local laptop connected via tunnel!",
                    "tunnel_url": LOCAL_TUNNEL_URL,
                    "stats": tunnel_stats
                }
    except Exception:
        pass
    
    return {
        "status": "disconnected",
        "message": "⚠️ Tunnel configured but not responding. Using cloud fallback.",
        "tunnel_url": LOCAL_TUNNEL_URL,
        "stats": tunnel_stats
    }


def get_proxy_status() -> dict:
    """Get current proxy/tunnel status for dashboard."""
    return {
        "mode": "hybrid" if LOCAL_TUNNEL_URL else "cloud_only",
        "tunnel_url": LOCAL_TUNNEL_URL or "not_set",
        "tunnel_active": tunnel_stats["tunnel_active"],
        "requests_forwarded": tunnel_stats["forwarded"],
        "requests_fallback": tunnel_stats["fallback"],
        "avg_tunnel_latency_ms": round(tunnel_stats["avg_tunnel_latency_ms"], 2),
        "last_error": tunnel_stats["last_tunnel_error"],
    }
