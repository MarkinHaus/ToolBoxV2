# FIX-39 – aus Karte K-OS-04

Paket: **P6-PLATTFORM** (Plattform- und versionsabhängig) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 2 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Darf ein Test die echte crontab des Nutzers ändern? Auf dem Entwicklerrechner ist das riskant (Regel 4). Alternative: `crontab`-Aufruf im Test durch ein Schein-Programm im `PATH` ersetzen.

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-39  (aus Karte K-OS-04)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_isaa/test_jobs/test_job_os_scheduler.py; ggf. toolboxv2/mods/isaa/extras/jobs/ (Scheduler)   (nur diese ändern)
Vorher lesen:  `TestRealFilesystemIntegration`, die Funktion, die crontab-Zeilen baut
```

**Schritte:**

1. Auf Ubuntu mit cron reproduzieren, geschriebene crontab ausgeben.
2. Fehler im Zeilenformat beheben, oder nach Antwort den Test auf ein Schein-`crontab` umstellen.

**Nicht tun:** Nicht auf dem Owner-Rechner gegen die echte crontab laufen lassen. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 2 IDs grün auf ubuntu.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_jobs/test_job_os_scheduler.py::TestRealFilesystemIntegration::test_crontab_entry_has_correct_interval
toolboxv2/tests/test_isaa/test_jobs/test_job_os_scheduler.py::TestRealFilesystemIntegration::test_idempotent_install
```
