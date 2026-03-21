# format_class() API

## Datei
`toolboxv2/mods/isaa/module.py:1980`

## ⚠️ WICHTIG: minitask() existiert NICHT

In ISAA V2 wurde `minitask()` **entfernt**. Die Funktionalität wurde durch `format_class()` ersetzt.

---

## `format_class()`

**Signatur:** `module.py:1980`

```python
async def format_class(
    self,
    format_schema: type[BaseModel] | None = None,
    task: str | None = None,
    agent_name: str = \"TaskCompletion\",
    auto_context: bool = False,
    request: RequestData | None = None,
    form_data: dict | None = None,
    data: dict | None = None,
    **kwargs,
) -> BaseModel
```

### Parameter

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|---------------|
| `format_schema` | `type[BaseModel]` | **Pflicht** | Pydantic-Model als Ausgabeformat |
| `task` | `str` | **Pflicht** | Prompt/Beschreibung für den Agent |
| `agent_name` | `str` | `\"TaskCompletion\"` | Welcher Agent soll formatieren |
| `auto_context` | `bool` | `False` | Automatisch Kontext hinzufügen |
| `request` | `RequestData` | `None` | HTTP Request Data |
| `form_data` | `dict` | `None` | Form-Daten |
| `data` | `dict` | `None` | Direkte Dict-Eingabe |

### Return
`BaseModel` Instance (Typ des übergebenen format_schema)

### Logik (Code-Referenz)

```python
# module.py:1982-1998
if request is not None or form_data is not None or data is not None:
    data_dict = (request.request.body if request else None) or form_data or data
    format_schema = format_schema or data_dict.get(\"format_schema\")
    task = task or data_dict.get(\"task\")
    agent_name = data_dict.get(\"agent_name\") or agent_name
    auto_context = auto_context or data_dict.get(\"auto_context\")

# Validierung
if format_schema is None or not task:
    return None

# Agent holen
agent = await self.get_agent(agent_name)

# An FlowAgent delegieren
return await agent.a_format_class(format_schema, task, auto_context=auto_context)
```

---

## Beispiel: structuriertes JSON mit Pydantic

```python
from pydantic import BaseModel
from typing import Optional

# 1. Output-Schema definieren
class UserProfile(BaseModel):
    name: str
    email: str
    age: Optional[int] = None
    tags: list[str] = []

# 2. Task formulieren
task = \"\"\"
Analysiere folgenden Text und extrahiere die Benutzerinformationen:

Input: \"Max Mustermann (max@example.com), 30 Jahre alt, interessiert an Python und AI.\"
\"\"\"

# 3. format_class aufrufen
isaa = app.get_mod(\"isaa\")
result: UserProfile = await isaa.format_class(
    format_schema=UserProfile,
    task=task,
    agent_name=\"TaskCompletion\"
)

# 4. Result nutzen
print(result.name)      # \"Max Mustermann\"
print(result.email)     # \"max@example.com\"
print(result.age)       # 30
print(result.tags)      # [\"Python\", \"AI\"]
```

---

## API-Endpoint (HTTP)

`format_class()` ist auch als REST-API verfügbar:

```
POST /api/isaa/format_class
Content-Type: application/json

{
    \"format_schema\": \"UserProfile\",  // Schema-Name oder Definition
    \"task\": \"Extrahiere Daten aus...\",
    \"agent_name\": \"TaskCompletion\",
    \"auto_context\": false
}
```

### Mit Request-Body

```python
# Automatische Extraktion aus request.body
@app.post(\"/api/extract\")
async def extract_data(request: RequestData):
    return await isaa.format_class(request=request)
```

---

## FlowAgent.a_format_class()

**Datei:** `toolboxv2/mods/isaa/base/Agent/flow_agent.py`

Die Methode wird an den FlowAgent delegiert:

```python
async def a_format_class(
    self,
    format_schema: type[BaseModel],
    task: str,
    auto_context: bool = False
) -> BaseModel:
    # Generiert Prompt mit Schema-Beschreibung
    # Ruft LLM auf
    # Parst Response als format_schema
    ...
```

---

## Vergleich: V1 minitask() → V2 format_class()

| Feature | V1 minitask() | V2 format_class() |
|---------|---------------|-------------------|
| Input | Custom Object | Pydantic BaseModel |
| Output | String/Object | Pydantic Instance |
| Validation | Manuell | Automatisch via Pydantic |
| Type-Safety | Schwach | Stark (MyPy/Pyright) |
| API-Export | Nein | Ja (@export decorator) |

---

## Nächste Schritte

👉 **[Agent Management](./AGENT_MANAGEMENT.md)** - Agents konfigurieren
👉 **[Chain System](./CHAIN_SYSTEM.md)** - Agents verketten