# ISAA Job System

> **Location**: `toolboxv2/mods/isaa/extras/jobs/`
> **Zweck**: Persistente, geplante Agent-Tasks die CLI-Neustarts überleben und sich über OS-Scheduler automatisch reaktivieren.

## Übersicht

Das ISAA Job System ermöglicht persistente, geplante Agent-Tasks die CLI-Neustarts überleben und sich über OS-Scheduler (Windows schtasks, Linux crontab, macOS LaunchAgent) automatisch reaktivieren können.

**Kernkomponenten:**

| Komponente | Aufgabe |
|---|---|
| `JobDefinition` | Datenhaltung: was, wann, welcher Agent |
| `TriggerConfig` | Wann soll der Job feuern |
| `TriggerRegistry` | Plugin-System für eigene Trigger-Typen |
| `JobScheduler` | Async Tick-Loop, evaluiert Trigger, feuert Jobs |
| `JobEventBus` | Ermöglicht Job-Chaining (A fertig → B startet) |
| `headless_runner` | Entry-Point für OS-Scheduler wenn CLI nicht läuft |
| `os_scheduler` | Registriert/entfernt OS-Level Scheduled Tasks |

**Persistenz:** Alle Jobs werden als JSON-Datei gespeichert (`jobs.json`). Jede Änderung schreibt sofort auf Disk.

---

## Setup

### Scheduler initialisieren

```python
from pathlib import Path
from toolboxv2.mods.isaa.extras.jobs import JobScheduler

async def fire_callback(job):
    agent = await isaa.get_agent(job.agent_name)
    result = await agent.a_run(job.query, session_id=job.session_id)
    return result

scheduler = JobScheduler(
    jobs_file=Path("data/jobs.json"),
    fire_callback=fire_callback,
)

await scheduler.start()  # Async Tick-Loop
# ...
await scheduler.stop()
```

Der Scheduler tickt jede Sekunde und prüft alle aktiven Jobs gegen ihre Trigger-Evaluatoren.

### OS Auto-Wake (optional)

```python
from toolboxv2.mods.isaa.extras.jobs.os_scheduler import install_autowake, remove_autowake, autowake_status

result = install_autowake(Path("data/jobs.json"))
# → "Auto-wake installed (Windows schtasks, every 15min + on boot)"

autowake_status()  # Status prüfen
remove_autowake()  # Entfernen
```

### Dependencies

| Feature | Dependency | Pflicht? |
|---|---|---|
| Cron-Trigger | `pip install croniter` | Nur für `on_cron` |
| File-Watching | `pip install watchdog` | Nur für `on_file_changed` |
| System-Idle (Linux) | `xprintidle` Binary | Nur für `on_system_idle` auf Linux |

---

## Jobs erstellen

```python
from toolboxv2.mods.isaa.extras.jobs import JobDefinition, TriggerConfig

job = JobDefinition(
    job_id=JobDefinition.generate_id(),
    name="daily-backup",
    agent_name="main_agent",
    query="Erstelle ein Backup aller Datenbanken",
    trigger=TriggerConfig(
        trigger_type="on_cron",
        cron_expression="0 2 * * *",
    ),
    session_id="default",
    timeout_seconds=300,
    max_retries=0,
)

job_id = scheduler.add_job(job)
```

### CRUD-Operationen

```python
job_id = scheduler.add_job(job)        # Erstellen
job = scheduler.get_job(job_id)        # Abfragen
all_jobs = scheduler.list_jobs()       # Alle
scheduler.pause_job(job_id)            # Pausieren
scheduler.resume_job(job_id)           # Fortsetzen
scheduler.remove_job(job_id)           # Löschen
print(scheduler.active_count)          # Aktive Anzahl
```

### Job-Status Lifecycle

```
active → (trigger feuert) → running → completed/failed/timeout
active → pause_job()      → paused  → resume_job() → active
active → (on_time fired)  → expired
active → remove_job()     → gelöscht
```

---

## Trigger-Typen

### Zeitbasiert

| Trigger | Beispiel | Beschreibung |
|---------|----------|-------------|
| `on_time` | `at_datetime="2025-06-15T14:30:00+02:00"` | Einmalig zu Zeitpunkt, danach `expired` |
| `on_interval` | `interval_seconds=300` | Alle N Sekunden, zählt ab letztem Feuern |
| `on_cron` | `cron_expression="0 3 * * *"` | Cron-Schedule (benötigt `croniter`) |

### System-Events

| Trigger | Beschreibung |
|---------|-------------|
| `on_cli_start` / `on_cli_exit` | CLI Lifecycle |
| `on_system_boot` | Nach Systemstart (nur mit `install_autowake()`) |
| `on_system_idle` | `idle_seconds=600` — System-Leerlauf |
| `on_system_shutdown` | Vor Herunterfahren (via `atexit`, `SIGTERM`) |
| `on_network_available` | Übergang offline → online |

### Datei-basiert

```python
TriggerConfig(
    trigger_type="on_file_changed",
    watch_path="/srv/data/configs",
    watch_patterns=["*.yaml", "*.json"],  # Optional
)
```

Benötigt `watchdog`. 2-Sekunden Debouncing, rekursiv.

### Job-Chaining

```python
# Job B startet wenn Job A erfolgreich war
scheduler.add_job(JobDefinition(
    ...,
    trigger=TriggerConfig(
        trigger_type="on_job_completed",  # oder on_job_failed / on_job_timeout
        watch_job_id="job_backup",
    ),
))
```

### Webhooks

Jobs können auf externe HTTP-Trigger warten (`on_webhook_received`). Scheduler startet keinen HTTP-Server — Integration über bestehende Web-Infrastruktur.

---

## Dream-Trigger (Meta-Learning)

Spezielle Trigger für die Dreamer-Integration:

| Trigger | Beschreibung |
|---------|-------------|
| `on_agent_idle` | Auto-Dream bei Leerlauf (`agent_idle_seconds`) |
| `on_dream_start` / `on_dream_end` | Dream Lifecycle Events |
| `on_dream_budget_hit` | Token-Budget erschöpft |
| `on_dream_skill_evolved` | Skill wurde verändert |

### Convenience: `add_dream_job()`

```python
# Nightly Dream um 03:00
scheduler.add_dream_job("main_agent")

# Dream nach 10 Min Leerlauf
scheduler.add_dream_job("main_agent", trigger_type="on_agent_idle", agent_idle_seconds=600)

# Dream mit Custom-Config
scheduler.add_dream_job("main_agent", dream_config={
    "max_budget": 5000,
    "do_skill_split": True,
    "hard_stop": False,
})
```

---

## Custom Trigger

```python
from toolboxv2.mods.isaa.extras.jobs import TriggerEvaluator, JobDefinition

class OnDiskSpaceLowEvaluator:
    async def setup(self, job, scheduler): pass
    async def evaluate(self, job) -> bool:
        # True = Job soll feuern
        ...
    async def teardown(self, job): pass

scheduler.trigger_registry.register("on_disk_space_low", OnDiskSpaceLowEvaluator())
```

`evaluate()` wird jede Sekunde aufgerufen — teure Operationen throttlen.

---

## EventBus

```python
scheduler.event_bus.on("job_completed", lambda e, d: print(f"Done: {d['job_id']}"))
scheduler.event_bus.on("dream_end", lambda e, d: print(f"Dream finished"))

scheduler.event_bus.off("job_completed", callback)
```

| Event | Wann |
|-------|------|
| `job_completed` / `job_failed` / `job_timeout` | Job Lifecycle |
| `dream_start` / `dream_end` | Dreamer-Zyklus |
| `dream_budget_hit` | Token-Budget erschöpft |
| `dream_skill_evolved` | Skill verändert |

---

## Headless Runner

Entry-Point für OS-Scheduler. Läuft ohne CLI.

```bash
python -m toolboxv2.mods.isaa.extras.jobs.headless_runner --jobs-file data/jobs.json
```

Unterstützte Trigger im Headless Mode: `on_time`, `on_interval`, `on_cron`, `on_system_boot`.

---

## Agent-Tool Integration

Innerhalb der ExecutionEngine registrierte Tools:

```python
createJob(name="weekly-report", trigger_type="on_cron", cron_expression="0 9 * * 1", query="...")
listJobs()
deleteJob(job_id="job_abc123")
```

---

## Debugging

```python
import logging
logging.getLogger("toolboxv2.mods.isaa.extras.jobs.job_manager").setLevel(logging.DEBUG)

for job in scheduler.list_jobs():
    print(f"{job.name:30s} | {job.status:8s} | runs={job.run_count} fails={job.fail_count}")
```
