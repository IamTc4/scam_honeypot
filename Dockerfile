# Competition Dockerfile — Hybrid Architecture
# Optimized: No torch, no transformers = 10x smaller image, instant startup
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /code

# Copy COMPETITION requirements (lightweight)
# Copy requirements (now in root)
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /code/src/

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Expose port for Hugging Face
EXPOSE 7860

# Run uvicorn (single worker for Hugging Face Spaces compatibility)
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]
