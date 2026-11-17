"""Aussehen der icli 2.0 - ausschliesslich aus dem vorhandenen ISAA-Branding.

Farben/Logo/Begruessung: mods/isaa/extras/isaa_branding.py
Symbole:                 mods/isaa/extras/zen/zen_renderer.py (SYM)
"""
from __future__ import annotations

from textual.theme import Theme

from toolboxv2.mods.isaa.extras.isaa_branding import (
    LOGO_LINES, GradientCyan as G, StateColors as S, get_greeting,
)
from toolboxv2.mods.isaa.extras.zen.zen_renderer import C as ZEN, SYM

BG = "#070d10"                      # dunkler als L1_DEEP, damit der Verlauf traegt
DIM = ZEN["dim"]

ISAA_THEME = Theme(
    name="isaa",
    primary=G.L3_MID,
    secondary=G.L2_DARK,
    accent=G.L4_BRIGHT,
    foreground=ZEN["white"],
    background=BG,
    surface=BG,
    panel=G.L1_DEEP,
    success=ZEN["green"],
    warning=ZEN["amber"],
    error=ZEN["red"],
    dark=True,
    variables={
        "footer-key-foreground": G.L4_BRIGHT,
        "input-selection-background": f"{G.L2_DARK} 60%",
        "scrollbar": G.L1_DEEP,
        "scrollbar-hover": G.L2_DARK,
        "scrollbar-active": G.L3_MID,
        "scrollbar-background": BG,
        "block-cursor-background": G.L4_BRIGHT,
        "block-cursor-foreground": BG,
    },
)

_LOGO_SHADES = (G.L5_GLOW, G.L5_GLOW, G.L4_BRIGHT, G.L4_BRIGHT, G.L3_MID, G.L2_DARK)


def banner(version: str) -> str:
    """Logo mit Cyan-Verlauf + Begruessung als Textual-Markup."""
    lines = [f"[{shade}]{line}[/]" for shade, line in zip(_LOGO_SHADES, LOGO_LINES)]
    lines.append("")
    lines.append(f" [{G.WHITE}]{get_greeting()}[/]  [{DIM}]icli {version}[/]")
    lines.append(f" [{DIM}]Strg+P Befehle   F2 Leiste   Esc pausieren   Strg+O Tools   Strg+R Gedanken[/]")
    return "\n".join(lines)


__all__ = ["ISAA_THEME", "SYM", "G", "S", "DIM", "BG", "banner"]
