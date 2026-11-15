# Karte K-API-19

```
Karte:       K-API-19
Paket:       P4-ENTSCHEIDUNG – Braucht Entscheidung des Owners
Status:      UNGETESTET (Verhaltensfrage)
Fix-Typ:     B (Soll jede HTML-Zeichenkette in die Seitenhülle eingebettet werden, auch ein Fragment? (Für HTMX-Teilantworten wäre das falsch.))
Fix-Auftrag: FIX-24
```

**Ursache:** `test_fast_tb.py::test_format_html_string` erwartet das rohe `<h1>Hi</h1>`; das Produkt liefert die Seite in der Hülle (`<html data-style="paper"> …`, ~171 000 Zeichen).

**Betroffen:** 1 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-19

**Beleg:** CI: `AssertionError: b'<html data-style="paper">\n<head><link rel=[171200 chars]/h1>' != b'<h1>Hi</h1>'`

**Hypothese:** HTML-Antworten werden jetzt immer in das Seitengerüst eingebettet (Live-UI „one stack“).

**Repro:**

```
python -m pytest toolboxv2/tests/test_web/test_workers/test_fast/test_fast_tb.py -q
```

**Vorher:** `1 failed, 86 passed`

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_web/test_workers/test_fast/test_fast_tb.py` – 1

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_web/test_workers/test_fast/test_fast_tb.py::TestHandlerResponseFormatting::test_format_html_string | FAILED | AssertionError | b'<html data-style="paper">\n<head><link rel=[171200 chars]/h1>' != b'<h1>Hi</h1>'
```
