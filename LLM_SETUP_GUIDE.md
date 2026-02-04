# 🤖 LLM Provider Setup Guide

## Overview

The Scam Honeypot now supports **3 LLM providers** with automatic fallback:

1. **Grok (xAI)** - ⭐ **RECOMMENDED** for hackathon
2. **Gemini (Google)** - Alternative option
3. **OpenAI (GPT)** - Alternative option

The system automatically uses the first available provider based on priority.

---

## Why Grok is Recommended

### ✅ Advantages
- **Free tier available** - Perfect for hackathons
- **Fast responses** - ~1-2 second latency
- **Strong reasoning** - Excellent for conversational AI
- **Easy setup** - Simple API similar to OpenAI
- **Reliable** - xAI infrastructure is robust
- **Good context handling** - Handles multi-turn conversations well

### 📊 Comparison

| Feature | Grok | Gemini | OpenAI |
|---------|------|--------|--------|
| **Free Tier** | ✅ Yes | ✅ Yes | ❌ No |
| **Speed** | ⚡ Fast | ⚡ Fast | ⚡ Fast |
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Setup** | 🟢 Easy | 🟢 Easy | 🟢 Easy |
| **Reliability** | 🟢 High | 🟢 High | 🟢 High |
| **Cost** | 💰 Free | 💰 Free | 💰 Paid |

---

## 🚀 Quick Setup - Grok (Recommended)

### Step 1: Get Grok API Key (2 minutes)

1. Visit: **https://console.x.ai/**
2. Sign up with your email or X (Twitter) account
3. Navigate to **API Keys** section
4. Click **"Create New API Key"**
5. Copy the API key (starts with `xai-...`)

### Step 2: Add to Environment

**Option A: Using .env file**
```bash
# Create .env file in project root
GROK_API_KEY=xai-your-api-key-here
```

**Option B: For deployment (Render/Railway)**
- Add environment variable in dashboard
- Name: `GROK_API_KEY`
- Value: Your Grok API key

### Step 3: Test It

```bash
# Run the honeypot service
uvicorn scam_honeypot.app:app --reload

# You should see:
# ✅ LLM initialized: GROK
```

---

## 🔄 Alternative: Gemini Setup

### Step 1: Get Gemini API Key

1. Visit: **https://makersuite.google.com/app/apikey**
2. Sign in with Google account
3. Click **"Create API Key"**
4. Copy the API key

### Step 2: Add to Environment

```bash
GEMINI_API_KEY=your-gemini-api-key-here
```

---

## 🔄 Alternative: OpenAI Setup

### Step 1: Get OpenAI API Key

1. Visit: **https://platform.openai.com/api-keys**
2. Sign up (requires payment method)
3. Create API key
4. Copy the key

### Step 2: Add to Environment

```bash
OPENAI_API_KEY=sk-your-openai-key-here
```

---

## 🎯 Provider Priority

The system tries providers in this order:

1. **Grok** (if `GROK_API_KEY` is set)
2. **Gemini** (if `GEMINI_API_KEY` is set)
3. **OpenAI** (if `OPENAI_API_KEY` is set)
4. **Template Fallback** (if no LLM available)

You can set multiple API keys for redundancy. If one fails, it automatically falls back to the next.

---

## 💡 Using Multiple Providers

For maximum reliability, set all three:

```bash
# .env file
GROK_API_KEY=xai-your-key
GEMINI_API_KEY=your-gemini-key
OPENAI_API_KEY=sk-your-openai-key
```

The system will:
1. Try Grok first
2. If Grok fails → try Gemini
3. If Gemini fails → try OpenAI
4. If all fail → use template responses

---

## 🧪 Testing Your Setup

### Test Script

```python
# test_llm.py
from scam_honeypot.llm_agent import LLMAgent

agent = LLMAgent()

if agent.available:
    print(f"✅ LLM Provider: {agent.provider.upper()}")
    
    # Test response
    response = agent.generate_response(
        persona="elderly_confused",
        message_text="Your bank account will be blocked. Verify now!",
        conversation_history=[],
        emotion="worried"
    )
    
    print(f"🤖 Response: {response}")
else:
    print("❌ No LLM provider available")
```

Run:
```bash
python test_llm.py
```

---

## 📊 Expected Output

### With Grok
```
✅ LLM initialized: GROK
✅ LLM Provider: GROK
🤖 Response: Oh no! Why would my account be blocked? I haven't done anything wrong. What should I do?
```

### With Gemini
```
✅ LLM initialized: GEMINI
✅ LLM Provider: GEMINI
🤖 Response: I'm very worried. Can you please explain why this is happening?
```

### Without LLM (Fallback)
```
⚠️ No LLM provider available, using template fallback
❌ No LLM provider available
🤖 Response: Please don't block my account! I have my pension coming. What do I need to do?
```

---

## 🔧 Troubleshooting

### Issue: "Failed to initialize grok"

**Cause**: Invalid API key or network issue

**Solution**:
1. Verify API key is correct
2. Check internet connection
3. Try regenerating API key

### Issue: "No LLM provider available"

**Cause**: No API keys set

**Solution**:
1. Set at least one API key in `.env`
2. Restart the service
3. Check environment variables loaded

### Issue: "LLM generation failed, using fallback"

**Cause**: API rate limit or temporary error

**Solution**:
- System automatically uses template fallback
- No action needed - responses still work
- Check API usage limits

---

## 💰 Cost Comparison

| Provider | Free Tier | Paid Pricing |
|----------|-----------|--------------|
| **Grok** | ✅ Generous free tier | ~$5/1M tokens |
| **Gemini** | ✅ 60 requests/min free | ~$0.50/1M tokens |
| **OpenAI** | ❌ No free tier | ~$0.50/1M tokens |

**For Hackathon**: Grok or Gemini free tiers are more than sufficient.

---

## 🎯 Recommendation for Hackathon

### Best Setup:

```bash
# Primary: Grok (free, fast, reliable)
GROK_API_KEY=xai-your-key

# Backup: Gemini (free, good fallback)
GEMINI_API_KEY=your-gemini-key
```

This gives you:
- ✅ Free tier for both
- ✅ Automatic fallback if one fails
- ✅ High reliability
- ✅ Great conversation quality

---

## 📝 Deployment Configuration

### Render.com

Add environment variables in dashboard:
```
GROK_API_KEY = xai-your-key
GEMINI_API_KEY = your-gemini-key  # Optional backup
```

### Railway.app

```bash
railway variables set GROK_API_KEY=xai-your-key
railway variables set GEMINI_API_KEY=your-gemini-key
```

### Vercel

```bash
vercel env add GROK_API_KEY
# Enter: xai-your-key

vercel env add GEMINI_API_KEY
# Enter: your-gemini-key
```

---

## ✅ Verification Checklist

- [ ] Grok API key obtained from console.x.ai
- [ ] API key added to `.env` or deployment platform
- [ ] Service started successfully
- [ ] Logs show "✅ LLM initialized: GROK"
- [ ] Test request returns natural response
- [ ] (Optional) Backup provider configured

---

## 🎓 Summary

**For the hackathon, use Grok:**
1. Free tier available
2. Fast and reliable
3. Easy 2-minute setup
4. Excellent conversation quality
5. Automatic fallback to templates if needed

**Your honeypot will have:**
- Natural, human-like responses
- Better scammer engagement
- Higher intelligence extraction
- Competitive advantage over template-only solutions

**Get started**: https://console.x.ai/ 🚀
