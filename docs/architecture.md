# System Architecture

## Overview
The Scam Honeypot is a sophisticated AI-driven system designed to detect, engage, and extract intelligence from scammers. It uses a **7-Layer Optimization Stack** to ensure low-latency responses while maintaining conversational depth.

## Core Components

### 1. `src/main.py`
The FastAPI entry point. Handles incoming requests, routing, and background task management.

### 2. `src/honeypot_agent.py` (formerly llm_agent)
The central intelligence engine. Manages LLM provider connections (Groq, Grok, Gemini) and falls back to templates if needed.

### 3. `src/performance_optimizer.py`
Implements the optimization layers:
- **Layer 1: LRU Cache**: Microsecond-level exact match caching.
- **Layer 2: FAISS Semantic Cache**: Vector-based similar query caching.
- **Layer 3: Scam State Machine**: Pre-computed responses for common patterns.
- **Layer 4: Groq LPU**: High-speed inference (<200ms).

### 4. `src/detector.py`
Real-time scam detection engine using keyword analysis and conversation patterns.

## Data Flow

1. **Request**: User sends message to `/api/scam-honeypot`.
2. **Fast Path**: 
   - Check LRU Cache → Return if hit.
   - Check Semantic Cache → Return if hit.
3. **Execution**:
   - Classify intent and emotion.
   - Generate response via LLM Router (Groq > Grok > Gemini).
4. **Background**:
   - Extract intelligence (UPI, Phone, Bank details).
   - Update session state.
   - Auto-submit to evaluation system if critical intel found.

## Deployment
Running on Hugging Face Spaces (Docker) with ngrok tunneling for stable endpoint access.
