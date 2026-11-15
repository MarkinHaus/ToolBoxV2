# Karte K-API-11

```
Karte:       K-API-11
Paket:       P5-ARCH – Architektur
Status:      UNGETESTET (hängt an ARCH-01: erst `conftest2.py` ohne globale Attrappen, dann die echte Datenklasse nutzen)
Fix-Typ:     C (-> ARCH-01)
Fix-Auftrag: – (ARCH-01)
```

**Ursache:** `FlowAgent.__init__` liest `amd.web_config.enable_web`; die Attrappe `MockAgentModelData` in `conftest2.py` hat `web_config` nicht – und `conftest2.py` ersetzt das Modul mit `AgentWebConfig` selbst durch `MagicMock`.

**Betroffen:** 8 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-11

**Beleg:** `toolboxv2/mods/isaa/base/Agent/flow_agent.py:515`; `toolboxv2/tests/test_isaa/test_base/test_agent/conftest2.py:8, 31-45`. CI: `AttributeError: 'MockAgentModelData' object has no attribute 'web_config'` (8)

**Hypothese:** Die Attrappe ist hinter der echten Datenklasse zurück. Ein Ein-Zeilen-Fix (`self.web_config = AgentWebConfig()`) geht nicht sauber, weil `conftest2.py` genau das Modul `…Agent.types` global durch eine Attrappe ersetzt (K-STATE-02).

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_system_integration.py -q
```

**Vorher:** `8 failed`

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

**Notiz:** Nebenbefund an derselben Stelle: `flow_agent.py:512` enthält als Vorgabe den Pfad `C:/Users/Markin/Workspace/ToolBoxV2/dist/toolboxv2-0.1.24-py2.py3-none-any.whl`.

## Test-IDs

- `toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_system_integration.py` – 8

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_system_integration.py::TestIntegration::test_01_initialization | FAILED | AttributeError | 'MockAgentModelData' object has no attribute 'web_config'
```
