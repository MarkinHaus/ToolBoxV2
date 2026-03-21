# 06_PRODUCTION_SETUP - Deployment

## Problem
Wie deploye ich einen ISAA-Agenten in Produktion?

## Loesung
1. Konfiguration - Environment-basiert
2. Dockerisierung - Container
3. Monitoring - Logs und Metriken
4. Health Checks - Verfuegbarkeit

## Projektstruktur

    /my-agent/
    |-- agent.py
    |-- config/base.yaml
    |-- config/prod.yaml
    |-- docker/Dockerfile
    |-- docker/docker-compose.yml

## Dockerfile

    FROM python:3.12-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    CMD ["python", "-m", "agent.main"]

## docker-compose.yml

    services:
      agent:
        build: .
        restart: unless-stopped
        ports:
          - "8080:8080"
      redis:
        image: redis:7-alpine

## Health Check

    @app.route("/health")
    def health():
        return jsonify({"status": "healthy", "agent": "ProductionAgent"})

## Deployment Checklist

- Config in prod.yaml angepasst
- Log-Level auf INFO/WARN
- Debug-Code entfernt
- Secrets als Environment-Vars
- Health-Check implementiert
