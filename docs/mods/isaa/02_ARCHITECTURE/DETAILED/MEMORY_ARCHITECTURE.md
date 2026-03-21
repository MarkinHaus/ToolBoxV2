# ISAA Memory Architektur

## Übersicht

Das Memory-System bietet verschiedene Speicher-Typen für Agents.

## Komponenten

### AISemanticMemory
`ai_semantic_memory.py` - Semantische Suche basierend auf Concept-Matching.

```python
memory = AISemanticMemory()

# Speichern mit Konzepten
await memory.add_data(
    text=\"Python ist eine Programmiersprache\",
    concepts=[\"python\", \"programmiersprache\"]
)

# Semantische Suche
results = await memory.query(\"Programmieren\")
```

### HybridMemory
`hybrid_memory.py` - Kombination aus Vektor-Suche und Graph-Speicher.

### KnowledgeBase
`KnowledgeBase.py` - Strukturierte Wissensbasis.

## Vector Stores

| Store | Datei | Beschreibung |
|-------|-------|-------------|
| Faiss | `FaissVectorStore.py` | Facebook AI Similarity Search |
| Redis | `RedisVectorStore.py` | Redis-basierter Store |
| Taichi | `taichiNumpyNumbaVectorStores.py` | GPU-beschleunigt |

## Daten-Flow

```
Add Data
    │
    ▼
Embedding Model
    │
    ▼
Vector Store
    │
    ▼
Index Update
    │
    ▼
Query
    │
    ▼
Top-K Retrieval
    │
    ▼
Reranking
    │
    ▼
Results
```
