FROM python:3.12-slim

LABEL maintainer="LLM Gateway"
LABEL description="OpenAI-compatible API gateway for Ollama"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY server.py .
COPY model_manager.py .
COPY live_handler.py .
COPY static/ ./static/

# Create data directory for persistence
RUN mkdir -p /app/data

# Expose the gateway port
EXPOSE 4000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:4000/health || exit 1

# Run the gateway with single worker for simplicity
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "4000", "--workers", "1"]
