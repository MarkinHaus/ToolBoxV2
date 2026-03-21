# ISAA Jobs System

## Übersicht

Das Jobs-System (`isaa_mod/extras/jobs/`) ermöglicht geplante und zeitgesteuerte Agent-Ausführungen.

## Komponenten

| Komponente | Datei | Beschreibung |
|------------|-------|---------------|
| `JobManager` | `job_manager.py` | Zentrale Job-Verwaltung |
| `JobLiveState` | `job_live_state.py` | Live-Zustand der Jobs |
| `JobAPI` | `job_api_models.py` | REST-API Modelle |
| `HeadlessRunner` | `headless_runner.py` | Job-Ausführung ohne UI |
| `OSScheduler` | `os_scheduler.py` | OS-Level Scheduling |

## Trigger Types

```python
from isaa_mod.extras.jobs import TriggerConfig

# Cron Trigger
trigger = TriggerConfig(
    type=\"cron\",
    cron=\"0 9 * * *\"  # Täglich um 9:00
)

# Interval Trigger
trigger = TriggerConfig(
    type=\"interval\",
    seconds=3600  # Alle Stunde
)

# Event Trigger
trigger = TriggerConfig(
    type=\"event\",
    event=\"file_uploaded\"
)
```

## API Usage

```python
from toolboxv2 import Application

app = Application()
isaa = app.get_mod(\"isaa\")

# Job erstellen
await isaa.job_add(
    name=\"daily_report\",
    query=\"Erstelle Tagesbericht\",
    trigger=trigger,
    agent_name=\"reporter\"
)

# Jobs auflisten
jobs = await isaa.job_list()

# Job pausieren
await isaa.job_pause(\"daily_report\")

# Job fortsetzen
await isaa.job_resume(\"daily_report\")

# Job löschen
await isaa.job_remove(\"daily_report\")
```
