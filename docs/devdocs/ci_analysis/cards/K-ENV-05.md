# Karte K-ENV-05

```
Karte:       K-ENV-05
Paket:       P5-ARCH – Architektur
Status:      UNGETESTET (Grundsatzfrage statt Einzel-Fix; Ursache je Paket durch die Fehlermeldung belegt, kein Probe-Fix angewendet)
Fix-Typ:     C (-> ARCH-03)
Fix-Auftrag: – (ARCH-03)
```

**Ursache:** Tests brauchen optionale Pakete (`pyautogui`, `torch`, `fastapi`, `litellm`), die die CI nicht installiert.

**Betroffen:** 26 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-ENV-05

**Beleg:** `toolboxv2/mods/isaa/extras/destop_auto.py:18` `import pyautogui`; `toolboxv2/mods/icli_web/server.py:29` `fastapi`; `test_omni_p2_fixes.py:166` `torch`; CI: `ModuleNotFoundError: No module named 'pyautogui'` (12), `'torch'` (3), `'fastapi'` (3, plus 2 Folgefehler `module 'toolboxv2.mods.icli_web' has no attribute 'server'`), `<MagicMock …> does not have the attribute 'acompletion'/'completion'` (6, `litellm` ist nur als Stub vorhanden)

**Hypothese:** Die CI installiert `.` ohne Extras. Tests für Desktop-Automatisierung, Audio-Diarisierung, icli_web und den litellm-Fallback setzen die Pakete voraus. `pyautogui` lässt sich auf einem Runner ohne Display auch nicht einfach nachinstallieren.

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_icli_web.py toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p2_fixes.py -q
```

**Vorher:** `test_icli_web.py`: `5 failed, 8 passed, 7 skipped`; `test_omni_p2_fixes.py`: `3 failed, 7 passed, 1 skipped`

**Probe-Fix:** kein einzelner Probe-Fix: die Lösung ist eine Regel für alle optionalen Pakete (→ ARCH-03)

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py` – 12
- `toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_performance_stability.py` – 3
- `toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_view.py` – 3
- `toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p2_fixes.py` – 3
- `toolboxv2/tests/test_isaa/test_icli_web.py` – 5

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestDesktopAutoFeatureLive::test_A_loader_registers_feature | FAILED | ModuleNotFoundError | No module named 'pyautogui'
```
