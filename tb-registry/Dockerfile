# --- Builder Stage ---
FROM python:3.12-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

# Kopiere Metadaten für die Abhängigkeits-Installation
COPY pyproject.toml uv.lock README.md ./

# Installiere NUR die Abhängigkeiten (ohne das Projekt selbst)
# Das macht das Venv portabel
RUN uv sync --frozen --no-dev --no-install-project

# --- Production Stage ---
FROM python:3.12-slim

WORKDIR /app

RUN useradd -m -u 1000 registry

# Kopiere das virtuelle Environment vom Builder
COPY --from=builder /app/.venv /app/.venv
# Kopiere die pyproject.toml (wichtig für Paket-Erkennung)
COPY pyproject.toml ./
# Kopiere den Quellcode
COPY registry ./registry

# Falls deine App die Migrations oder Daten braucht, entkommentiere dies:
# COPY migrations ./migrations

# Umgebungsvariablen
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /app/data && chown -R registry:registry /app

USER registry
EXPOSE 4025

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:4025/health')" || exit 1

CMD ["python", "-m", "registry"]
