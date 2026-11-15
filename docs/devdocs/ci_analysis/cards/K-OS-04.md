# Karte K-OS-04

```
Karte:       K-OS-04
Paket:       P6-PLATTFORM – Plattform- und versionsabhängig
Status:      UNGETESTET (braucht ein System mit `crontab`; in der Analyseumgebung nicht vorhanden)
Fix-Typ:     B (Darf ein Test die echte crontab des Nutzers ändern? Auf dem Entwicklerrechner ist das riskant (Regel 4). Alternative: `crontab`-Aufruf im Test durch ein Schein-Programm im `PATH` ersetzen.)
Fix-Auftrag: FIX-39
```

**Ursache:** `test_job_os_scheduler.py::TestRealFilesystemIntegration` schreibt echte crontab-Einträge und liest sie zurück; auf dem Ubuntu-Runner kommen sie nicht an.

**Betroffen:** 2 Tests in 4 Jobs (ubuntu-3.10, ubuntu-3.11, ubuntu-3.12, ubuntu-3.13); Liste in catalog.csv, Spalte karte=K-OS-04

**Beleg:** CI nur ubuntu: `AssertionError: 0 != 2 : Expected 2 ISAA entries, got 0:`, `'*/15' not found in '# ISAA_JobRunner - ISAA Job Runner (auto-generated)'`. Lokal (ohne `crontab`-Programm): 5 rot mit `FileNotFoundError: … 'crontab'`.

**Hypothese:** Der Test verändert die echte crontab des Nutzers. Auf dem Runner wird nur der Kommentarkopf geschrieben, die Einträge fehlen (Format, oder `crontab -` lehnt eine Zeile ab). macOS-Jobs zeigen die IDs nicht.

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_jobs/test_job_os_scheduler.py -q   # ubuntu mit cron
```

**Vorher:** CI: 2 rot

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_isaa/test_jobs/test_job_os_scheduler.py` – 2

Beispiel aus `raw/ubuntu-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_jobs/test_job_os_scheduler.py::TestRealFilesystemIntegration::test_crontab_entry_has_correct_interval | FAILED | AssertionError | '*/15' not found in '# ISAA_JobRunner - ISAA Job Runner (auto-generated)'
```
