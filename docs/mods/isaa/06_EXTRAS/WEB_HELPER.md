# ISAA Web Helper

## Übersicht

Web-Such- und Scraping-Tools für ISAA.

## Komponenten

| Komponente | Datei | Beschreibung |
|------------|-------|---------------|
| `WebSearch` | `web_search.py` | Google Dorks Suche |
| `SearxngSetup` | `searxng_setup.py` | Searxng Integration |
| `WebAgent` | `web_agent.py` | Automatisierte Web-Tasks |

## Usage

```python
from isaa_mod.extras.web_helper import WebSearch

search = WebSearch()

# Einfache Suche
results = await search.query(\"Python async\")

# Mit Dorks
results = await search.query(
    \"site:github.com filetype:py tensorflow\"
)
```

## Agent Integration

```python
agent = await isaa.get_agent(\"web_scraper\")
result = await agent.a_run(
    \"Finde alle Python Projekte auf GitHub zum Thema ML\"
)
```
