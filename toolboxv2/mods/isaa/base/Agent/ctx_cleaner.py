"""
ctx_cleaner.py — Intelligent context dict cleaner for ISAA agents.

Walks a dict/list recursively. Detects strings that are unstructured
text dumps (website scrapes, minified CSS/JS blobs, chaotic HTML with
inline styles, raw binary-ish content). Truncates them with a clear
notice. Leaves clean code, markdown, prose, JSON, and normal text alone.

Usage in ExecutionEngine:
    from ctx_cleaner import clean_context

    # In _sanitize_history_for_api or before LLM call:
    messages = clean_context(messages)

    # Agent can disable per-request:
    messages = clean_context(messages, skip=ctx.skip_cleaning)
"""

from __future__ import annotations

import re
from typing import Any

# ── Thresholds ──────────────────────────────────────────────────────
MIN_LEN = 30_000          # never touch strings shorter than this
HARD_CAP = 8_000_000      # always truncate above this regardless of score
EXCERPT_LEN = 600      # chars to keep as preview
MAX_LINES_SAMPLE = 120  # lines to sample for analysis (perf)

# Score thresholds per size bucket:
#   (min_len, max_len, min_score_to_truncate)
# Bigger strings need lower chaos score to trigger.
_BUCKETS = [
    (30_000,   120_000,  0.55),   # small: only quite chaotic
    (120_000,  480_000,  0.32),   # medium: moderately chaotic
    (480_000,  2_000_000, 0.25),   # large: mildly chaotic
    (2_000_000, HARD_CAP, 0.18), # huge: any chaos signal
]


_SKIP_TAG = "[skipNextCleaning]"


def clean_messages(messages: list[dict]) -> list[dict]:
    """
    Clean a list of chat messages. Scans for [skipNextCleaning] in
    assistant content — if found, the next message is passed through
    unmodified. The tag itself is stripped from the output.

    Usage in ExecutionEngine:
        messages = clean_messages(ctx.working_history)
    """
    out = []
    skip_next = False

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Check if assistant signals skip
        if role == "assistant" and isinstance(content, str) and _SKIP_TAG in content:
            skip_next = True
            msg = {**msg, "content": content.replace(_SKIP_TAG, "").strip()}
            out.append(msg)
            continue

        if skip_next:
            skip_next = False
            out.append(msg)  # pass through unmodified
            continue

        # Normal: clean this message
        out.append(clean_context(msg))

    return out


def clean_context(obj: Any, *, _depth: int = 0) -> Any:
    """
    Recursively clean a dict/list structure.
    Truncates unstructured text blobs, leaves everything else intact.
    """
    if _depth > 20:
        return obj

    if isinstance(obj, dict):
        return {k: clean_context(v, _depth=_depth + 1) for k, v in obj.items()}

    if isinstance(obj, list):
        return [clean_context(v, _depth=_depth + 1) for v in obj]

    if isinstance(obj, str):
        return _maybe_truncate(obj)

    return obj


def _maybe_truncate(s: str) -> str:
    n = len(s)
    if n < MIN_LEN:
        return s

    if n >= HARD_CAP:
        return _truncated(s, n, 1.0)

    score = _chaos_score(s)

    for lo, hi, threshold in _BUCKETS:
        if lo <= n < hi:
            if score >= threshold:
                return _truncated(s, n, score)
            return s

    return s


def _truncated(s: str, n: int, score: float) -> str:
    head = s[:EXCERPT_LEN].rstrip()
    size = _human_size(n)
    pct = int(score * 100)
    return (
        f"{head}\n\n"
        f"[TRUNCATED — {size} of unstructured content removed "
        f"(chaos={pct}%). To read this file, use a dedicated read tool. "
        f"To bypass cleaning for the next tool result, "
        f"include [skipNextCleaning] in your content block.]"
    )


# ════════════════════════════════════════════════════════════════════
# CHAOS SCORING — fast heuristics, no ML
# ════════════════════════════════════════════════════════════════════

# Pre-compiled patterns (module level, zero per-call cost)
_RE_CSS_BLOCK = re.compile(r'[{;]\s*[\w-]+\s*:\s*[^;]{1,80};')
_RE_STYLE_ATTR = re.compile(r'style\s*=\s*["\'][^"\']{30,}', re.I)
_RE_DATA_URI = re.compile(r'data:[a-z]+/[a-z]+;base64,')
_RE_LONG_HEX = re.compile(r'[0-9a-fA-F]{20,}')
_RE_INDENT = re.compile(r'^([ \t]+)', re.M)
_RE_MD_HEADING = re.compile(r'^#{1,6}\s', re.M)
_RE_MD_LIST = re.compile(r'^[\s]*[-*+]\s', re.M)
_RE_CODE_KW = re.compile(
    r'\b(def |class |function |const |let |var |import |from |return |if |for |async |await )\b'
)
_RE_HTML_TAG = re.compile(r'<[a-zA-Z][^>]{0,60}>')
_RE_ESCAPED = re.compile(r'\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\n|\\t')


def _chaos_score(s: str) -> float:
    """
    Score 0.0 (clean) → 1.0 (chaotic).

    Combines multiple fast signals into a weighted average.
    Sampling first N lines for O(1)-ish performance on huge strings.
    """
    # Sample for perf
    lines = s.split('\n', MAX_LINES_SAMPLE + 1)[:MAX_LINES_SAMPLE]
    sample = '\n'.join(lines)
    slen = len(sample) or 1

    signals = []

    # 1. Special-char density (high in CSS/JS dumps, low in prose)
    special = sum(1 for c in sample if c in '{}[];:@#$%^&*|\\~`<>')
    sig_special = min(1.0, (special / slen) * 8)
    signals.append((sig_special, 0.25))

    # 2. CSS block density
    css_hits = len(_RE_CSS_BLOCK.findall(sample))
    sig_css = min(1.0, css_hits / max(len(lines), 1) * 1.5)
    signals.append((sig_css, 0.15))

    # 3. Inline style density
    style_hits = len(_RE_STYLE_ATTR.findall(sample))
    sig_style = min(1.0, style_hits / max(len(lines), 1) * 3)
    signals.append((sig_style, 0.10))

    # 4. Data URIs / long hex blobs / base64 (binary content)
    data_hits = len(_RE_DATA_URI.findall(sample))
    hex_hits = len(_RE_LONG_HEX.findall(sample))
    # Long runs without whitespace = binary blob
    longest_no_ws = max((len(seg) for seg in re.split(r'\s', sample) if seg), default=0)
    sig_blob_chars = min(1.0, max(0, longest_no_ws - 100) / 400)
    sig_data = min(1.0, data_hits * 0.4 + hex_hits * 0.15 + sig_blob_chars * 0.6)
    signals.append((sig_data, 0.15))

    # 5. Line length + monolithic blob detection
    line_lens = [len(l) for l in lines if l.strip()]
    if len(line_lens) > 0:
        avg_ll = sum(line_lens) / len(line_lens)
        max_ll = max(line_lens)
        # Very long lines (minified) = chaotic
        sig_linelen = min(1.0, max(0, max_ll - 200) / 600)
        # Few lines but huge total = monolithic blob
        sig_blob = min(1.0, max(0, avg_ll - 150) / 400)
        signals.append(((sig_linelen * 0.5 + sig_blob * 0.5), 0.20))
    else:
        signals.append((0.0, 0.15))

    # 6. Escape sequence density (binary-ish)
    esc_hits = len(_RE_ESCAPED.findall(sample))
    sig_esc = min(1.0, esc_hits / max(slen / 100, 1))
    signals.append((sig_esc, 0.05))

    # ── NEGATIVE signals (reduce score) ──

    # 7. Clean code indicators (indent consistency + keywords)
    #    BUT: code keywords WITHOUT indentation = minified → don't subtract
    indents = _RE_INDENT.findall(sample)
    unique_indents = set(indents)
    code_kw = len(_RE_CODE_KW.findall(sample))
    has_indentation = len(indents) > len(lines) * 0.3  # >30% of lines indented
    if has_indentation and code_kw > 2:
        indent_consistency = 1.0 - min(1.0, len(unique_indents) / max(len(indents), 1))
        sig_code = min(1.0, (indent_consistency * 0.5 + min(1.0, code_kw / 8) * 0.5))
        signals.append((-sig_code, 0.10))
    else:
        signals.append((0.0, 0.10))  # no credit for unindented "code"

    # 8. Markdown indicators
    md_heads = len(_RE_MD_HEADING.findall(sample))
    md_lists = len(_RE_MD_LIST.findall(sample))
    sig_md = min(1.0, (md_heads + md_lists) / max(len(lines), 1) * 4)
    signals.append((-sig_md, 0.05))

    # 9. Clean HTML with tags but no inline styles = structured
    html_tags = len(_RE_HTML_TAG.findall(sample))
    if html_tags > 3 and style_hits < 2 and css_hits < 3:
        signals.append((-0.3, 0.05))  # clean HTML penalty
    else:
        signals.append((0.0, 0.05))

    # ── Weighted sum ──
    total_weight = sum(w for _, w in signals)
    raw = sum(s * w for s, w in signals) / total_weight if total_weight else 0
    return max(0.0, min(1.0, raw))


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} chars"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"
