# TB-Registry Dockerfile
# Multi-stage build for optimized production image

FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Production stage
FROM python:3.12-slim

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 registry

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY registry ./registry

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Create data directory
RUN mkdir -p /app/data && chown -R registry:registry /app

# Switch to non-root user
USER registry

# Expose port
EXPOSE 4025

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:4025/health')" || exit 1

# Run the application
CMD ["python", "-m", "registry"]

