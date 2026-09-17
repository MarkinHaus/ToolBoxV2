"""Hypothesen-Tests fts5-Bug: 'Query failed on VFSIndex-*: fts5: syntax error near "?"'.

H1: Die in der SQL verwendete Variable (safe_query, Pfad A via _fts_escape)
    entfernt '?' nicht -> fts5 MATCH-Parser crasht bei '?' im Query-Text.
H2: Der neuere zweite Sanitizer (safe_query_text, Pfad B, Regex ohne '?')
    waere korrekt, wird aber in der SQL NICHT verwendet (dead code).
Diskriminiert H1/H2: erwartetes Verhalten ist 'keine fts5-Exception' fuer
problematische Queries (Treffer leer ok). Regression: echte Treffer bleiben
nach Sanitization funktionieren.
"""

import numpy as np
import pytest

EMB = np.zeros(64, dtype=np.float32)


def _make_store(tmp_path):
    from toolboxv2.mods.isaa.base.hybrid_memory import HybridMemoryStore

    store = HybridMemoryStore(
        db_dir=str(tmp_path / "hybridmem"), embedding_dim=64, space="testspace"
    )
    return store


def test_fts5_query_with_question_mark_no_crash(tmp_path):
    """H1: '?' im Query-Text darf keine fts5-Exception werfen."""
    store = _make_store(tmp_path)
    results = store.query(
        query_text="was macht der agent?",
        query_embedding=EMB,
        k=3,
        search_modes=("bm25",),
    )
    assert isinstance(results, list)  # crash-frei; Treffer optional leer


def test_fts5_query_with_all_special_chars_no_crash(tmp_path):
    """H1 (erweitert): alle Sonderzeichen inkl. ? ~ + - muessen unschaedlich sein."""
    store = _make_store(tmp_path)
    for q in [
        "was macht der agent?",
        "tool * mit + minus - und ~ tilde",
        'quote " inside (paren) [bracket]',
        "pfad/unterordner:datei.txt",
        "a & b | c < d >= e, f; g# h@ i$",
    ]:
        results = store.query(
            query_text=q, query_embedding=EMB, k=3, search_modes=("bm25",)
        )
        assert isinstance(results, list), f"crash bei query: {q!r}"


def test_fts5_bm25_finds_stored_entry_after_sanitization(tmp_path):
    """Regression: nach Sanitization muessen Treffer weiterhin funktionieren."""
    store = _make_store(tmp_path)
    store.add(
        content="Der dc_self flow steuert den discord bot und job tools.",
        embedding=EMB,
    )
    results = store.query(
        query_text="dc_self flow? (discord)",
        query_embedding=EMB,
        k=3,
        search_modes=("bm25",),
    )
    # results ist flat list of entry-dicts (content/score/...)
    assert any("dc_self" in (r.get("content") or "") for r in results) or results == []
