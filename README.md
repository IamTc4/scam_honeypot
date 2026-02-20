# Honeypot API

## Description
An advanced **LLM-powered Scam Honeypot** designed to detect fraud, extract actionable intelligence, and engage scammers in realistic, multi-turn conversations. The system uses a multi-stage detection engine (Keyword + ML + Heuristic) and an adaptive persona system to maintain engagement while safely extracting phone numbers, UPI IDs, bank accounts, and phishing links.

## Tech Stack
- **Language**: Python 3.10+
- **Framework**: FastAPI (High-performance web framework)
- **LLM Providers**: Groq LPU (Llama 3), xAI Grok, Google Gemini, OpenAI GPT-4
- **ML/NLP**: SpaCy (NER), Scikit-learn (Detection), TextBlob (Sentiment Analysis)
- **Infrastructure**: Docker, Redis (Caching), Uvicorn

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd scam-honeypot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Set environment variables**
   Create a `.env` file in the root directory:
   ```env
   # Recommended for speed
   GROQ_API_KEY=gsk_...
   
   # Optional Fallbacks
   GROK_API_KEY=...
   GEMINI_API_KEY=...
   OPENAI_API_KEY=...
   ```

4. **Run the application**
   ```bash
   cd scam_honeypot
   python -m uvicorn app:app --host 0.0.0.0 --port 8000
   ```

## API Endpoint
- **URL**: `https://<your-deployed-url>/api/scam-honeypot`
- **Method**: `POST`
- **Authentication**: `x-api-key` header (value: whatever you set in env, default optional)

### Request Format
```json
{
  "sessionId": "12345",
  "message": {
    "text": "Your account is blocked. Share OTP."
  },
  "conversationHistory": []
}
```

## Approach

### 1. Scam Detection
We use a **hybrid detection strategy**:
- **Keyword Matching**: Instant flagging of high-risk terms (OTP, Block, KYC).
- **Heuristic Analysis**: analyzing urgency, financial demands, and authority impersonation.
- **Contextual Analysis**: Evaluating the conversation history for manipulation patterns.

### 2. Intelligence Extraction
- **NER (Named Entity Recognition)**: Using SpaCy to identify organizations, dates, and persons.
- **Regex Patterns**: Specialized patterns for Indian phone numbers, UPI IDs, IFSC codes, and crypto addresses.
- **Link Analysis**: Extracting and categorizing suspicious URLs.

### 3. Engagement & Personas
- **Adaptive Personas**: The system selects from 4 personas (e.g., "Elderly Confused", "Busy Professional") to match the scammer's tactic.
- **Emotional State Machine**: The agent transitions between emotional states (Confused → Worried → Trusting → Suspicious) to mimic human behavior.
- **LLM-First Generation**: Responses are generated dynamically by large language models (Llama 3, Grok, etc.) to ensure context awareness and fluency. **No hardcoded responses are used.**
