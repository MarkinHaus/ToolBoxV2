"""
ISAA Branding & Flow Matrix Animation
======================================

Animated startup header with non-blocking idle animation.
The flow lines (│ ╎ ┃ ▼) become a live Matrix Rain that runs
as an async context manager during prompt input.

Usage:
    anim = FlowMatrixAnimation()

    # Startup: rain → logo reveal
    await anim.play_startup(duration=2.5)

    # Print branded header (static, flow area reserved)
    total = print_isaa_header(...)

    # Tell animation how far up the flow area is
    anim.set_cursor_offset(total)

    # Non-blocking idle animation during prompt
    async with anim:
        result = await session.prompt_async("ISAA › ")
"""

import asyncio
import html
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Optional

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import HTML


try:
    import psutil
except ImportError:
    psutil = None

import random
from datetime import datetime

# Begrüßungen nach Tageszeit (2-Stunden-Intervalle)
greetings = {
    "de": {
        "0-2": [
            "Guten Mitternacht!",
            "Schlafe gut, Mitternächtler!",
            "Mitternächtliche Grüße!",
            "Hallo Nachtschwärmer!",
            "Träume süß!"
        ],
        "2-4": [
            "Gute Nacht noch!",
            "Schlummer gut!",
            "Mitternachtliche Grüße!",
            "Hallo Nachtwanderer!",
            "Zeit für Ruhe!"
        ],
        "4-6": [
            "Guten frühen Morgen!",
            "Schönen Sonnenaufgang!",
            "Aufwachen!",
            "Morgenstund’ hat Gold im Mund!",
            "Früh aufstehen lohnt sich!"
        ],
        "6-8": [
            "Guten Morgen!",
            "Hallo Frühaufsteher!",
            "Schönen Start in den Tag!",
            "Morgen, Zeit für Kaffee!",
            "Auf in einen neuen Tag!"
        ],
        "8-10": [
            "Guten Vormittag!",
            "Schönen Arbeitstag!",
            "Hallo!",
            "Guten Tag!",
            "Viel Energie für heute!"
        ],
        "10-12": [
            "Fast Mittag!",
            "Guten Vormittag noch!",
            "Zeit für eine kurze Pause?",
            "Schönen Tag weiterhin!",
            "Hallo da draußen!"
        ],
        "12-14": [
            "Guten Mittag!",
            "Mittagspause gefällig?",
            "Schönen Mittag!",
            "Zeit für Essen!",
            "Hallo Mittagsmensch!"
        ],
        "14-16": [
            "Guten Nachmittag!",
            "Hallo!",
            "Schönen Nachmittag!",
            "Kaffeezeit!",
            "Halbzeit des Tages!"
        ],
        "16-18": [
            "Späten Nachmittag guten!",
            "Fast Feierabend!",
            "Hallo noch!",
            "Genieße den Nachmittag!",
            "Zeit für kleine Erholung!"
        ],
        "18-20": [
            "Guten Abend!",
            "Schönen Abend!",
            "Hallo Abendmensch!",
            "Abendstimmung genießen!",
            "Zeit zu entspannen!"
        ],
        "20-22": [
            "Hallo am Abend!",
            "Schönen Abend noch!",
            "Zeit für Ruhe!",
            "Guten Abend noch!",
            "Abendliche Grüße!"
        ],
        "22-24": [
            "Gute Nacht!",
            "Schlafe gut!",
            "Träume süß!",
            "Hallo Nachtschwärmer!",
            "Mitternacht naht!"
        ]
    },
    "en": {
        "0-2": [
            "Good midnight!",
            "Sleep tight!",
            "Midnight greetings!",
            "Hello night owl!",
            "Sweet dreams!"
        ],
        "2-4": [
            "Still awake?",
            "Night greetings!",
            "Time to rest!",
            "Hello late night!",
            "Sleep well soon!"
        ],
        "4-6": [
            "Early morning greetings!",
            "Good sunrise!",
            "Wake up!",
            "Morning has gold in its mouth!",
            "Time to start!"
        ],
        "6-8": [
            "Good morning!",
            "Hello early bird!",
            "Have a great day!",
            "Morning coffee time!",
            "Rise and shine!"
        ],
        "8-10": [
            "Good forenoon!",
            "Hello!",
            "Have an energetic morning!",
            "Good day!",
            "Time to work!"
        ],
        "10-12": [
            "Almost noon!",
            "Good late morning!",
            "Take a short break?",
            "Keep up the day!",
            "Hello there!"
        ],
        "12-14": [
            "Good noon!",
            "Lunch time!",
            "Have a nice midday!",
            "Time to eat!",
            "Hello midday person!"
        ],
        "14-16": [
            "Good afternoon!",
            "Hello!",
            "Enjoy your afternoon!",
            "Coffee time!",
            "Halfway through the day!"
        ],
        "16-18": [
            "Late afternoon greetings!",
            "Almost quitting time!",
            "Hello!",
            "Relax a bit!",
            "Keep going!"
        ],
        "18-20": [
            "Good evening!",
            "Enjoy your evening!",
            "Hello evening person!",
            "Evening vibes!",
            "Time to unwind!"
        ],
        "20-22": [
            "Evening greetings!",
            "Have a nice evening!",
            "Time to relax!",
            "Good evening still!",
            "Hello night person!"
        ],
        "22-24": [
            "Good night!",
            "Sleep tight!",
            "Sweet dreams!",
            "Hello night owl!",
            "Midnight is near!"
        ]
    }
}

def get_greeting(lang="de"):
    now = datetime.now()
    hour = now.hour
    # Finde passendes 2h-Intervall
    interval = f"{(hour//2)*2}-{((hour//2)*2)+2}"
    return random.choice(greetings.get(lang, greetings["de"]).get(interval, ["Hallo!"]))

# ═══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

class GradientCyan:
    L1_DEEP    = '#0e2a35'
    L2_DARK    = '#155e75'
    L3_MID     = '#22d3ee'
    L4_BRIGHT  = '#67e8f9'
    L5_GLOW    = '#a5f3fc'
    WHITE      = '#ecfeff'


class StateColors:
    INIT_PULSE   = '#facc15'
    INIT_DIM     = '#854d0e'
    DREAM_VIOLET = '#a78bfa'
    DREAM_DIM    = '#4c1d95'
    DREAM_MOON   = '#fde68a'
    LOAD_RED     = '#fb7185'
    LOAD_ORANGE  = '#fb923c'
    LOAD_BORDER  = '#ef4444'

# ANSI RGB escape codes for raw terminal writes (animation engine)
_ANSI = {
    'L1':     '\033[38;2;14;42;53m',
    'L2':     '\033[38;2;21;94;117m',
    'L3':     '\033[38;2;34;211;238m',
    'L4':     '\033[38;2;103;232;249m',
    'L5':     '\033[38;2;165;243;252m',
    'WHITE':  '\033[38;2;236;254;255m',
    'DIM':    '\033[38;2;30;60;75m',
    'RESET':  '\033[0m',
    'GOLD':   '\033[38;2;250;204;21m',
    'VIOLET': '\033[38;2;167;139;250m',
    'ORANGE': '\033[38;2;251;146;60m',
}


def _esc(text: Any) -> str:
    return html.escape(str(text).encode().decode(encoding="utf-8", errors="replace"))


# ═══════════════════════════════════════════════════════════════════════════════
# MATRIX RAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

MATRIX_CHARS = "ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉ0123456789"
FLOW_CHARS   = "│╎┃║╏┊┋▏▎▍▌▐░"
MIXED_CHARS  = MATRIX_CHARS + FLOW_CHARS * 3  # Flow-heavy mix


class _Stream:
    __slots__ = ('pos', 'length', 'speed', 'active')

    def __init__(self, height: int):
        self.reset(height)

    def reset(self, height: int):
        self.pos = random.randint(-height * 2, -1)
        self.length = random.randint(2, max(3, height - 1))
        self.speed = random.choice([1, 1, 1, 2])
        self.active = True

    def tick(self, height: int):
        self.pos += self.speed
        if self.pos - self.length > height + 2:
            self.reset(height)


class MatrixRainEngine:
    """Efficient Matrix Rain renderer with mixed Flow + Katakana characters."""

    def __init__(self, height: int = 4, width: int = 72):
        self.h = height
        self.w = width
        self.streams: list[_Stream] = [_Stream(height) for _ in range(width)]
        self.grid: list[list[str]] = [[' '] * width for _ in range(height)]
        self.brightness: list[list[int]] = [[0] * width for _ in range(height)]
        self._color_map = ['', _ANSI['L1'], _ANSI['L2'], _ANSI['L3'], _ANSI['L5']]

    def tick(self):
        for col, stream in enumerate(self.streams):
            stream.tick(self.h)
            for row in range(self.h):
                dist = stream.pos - row
                if 0 <= dist < stream.length:
                    if dist == 0:    self.brightness[row][col] = 4
                    elif dist == 1:  self.brightness[row][col] = 3
                    elif dist < 4:   self.brightness[row][col] = 2
                    else:            self.brightness[row][col] = 1

                    if random.random() < 0.2:
                        self.grid[row][col] = random.choice(MIXED_CHARS)
                    elif self.grid[row][col] == ' ':
                        self.grid[row][col] = random.choice(MIXED_CHARS)
                else:
                    if self.brightness[row][col] > 0:
                        self.brightness[row][col] -= 1
                    if self.brightness[row][col] == 0:
                        self.grid[row][col] = ' '

    def render_ansi(self, indent: int = 4) -> str:
        """Render frame as ANSI string (no cursor movement)."""
        lines = []
        pad = ' ' * indent
        reset = _ANSI['RESET']
        for row in range(self.h):
            parts = [pad]
            prev_c = ''
            for col in range(self.w):
                b = self.brightness[row][col]
                ch = self.grid[row][col]
                if b > 0 and ch != ' ':
                    c = self._color_map[b]
                    if c != prev_c:
                        parts.append(c)
                        prev_c = c
                    parts.append(ch)
                else:
                    if prev_c:
                        parts.append(reset)
                        prev_c = ''
                    parts.append(' ')
            if prev_c:
                parts.append(reset)
            lines.append(''.join(parts))
        return '\n'.join(lines)

    def render_overwrite(self, indent: int = 4) -> str:
        """Render with cursor-up prefix for in-place overwrite."""
        return f"\033[{self.h}A{self.render_ansi(indent)}\n"


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW MATRIX ANIMATION — Async Context Manager
# ═══════════════════════════════════════════════════════════════════════════════

FLOW_HEIGHT = 4


class FlowMatrixAnimation:
    """
    Non-blocking Matrix Flow animation as async context manager.

    The animation overwrites the flow area (first 4 lines of header)
    using ANSI cursor save/restore — safe alongside prompt_toolkit.

    Usage:
        anim = FlowMatrixAnimation()
        await anim.play_startup()           # blocking 2.5s rain
        total = print_isaa_header(...)      # static header
        anim.set_cursor_offset(total)       # where flow area is

        async with anim:                    # non-blocking idle
            result = await session.prompt_async(...)
    """

    def __init__(self, height: int = FLOW_HEIGHT, width: int = 72,
                 fps: float = 8, state: str = 'online'):
        self.engine = MatrixRainEngine(height=height, width=width)
        self.fps = fps
        self.state = state
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._offset: int = 0

        # State-specific color palettes
        if state == 'dreaming':
            self.engine._color_map = [
                '', _ANSI['DIM'], '\033[38;2;76;29;149m',
                _ANSI['VIOLET'], '\033[38;2;196;181;253m',
            ]
        elif state == 'high_load':
            self.engine._color_map[4] = _ANSI['ORANGE']

    def set_cursor_offset(self, total_header_lines: int):
        """Set how many lines from cursor bottom to top of flow area."""
        self._offset = total_header_lines

    # ── Startup (blocking) ─────────────────────────────────────────────────

    async def play_startup(self, duration: float = 2.5, fps: int = 12):
        """Blocking startup rain. Call BEFORE print_isaa_header()."""
        h = self.engine.h
        for _ in range(h):
            sys.stdout.write('\n')
        sys.stdout.flush()

        start = time.time()
        while time.time() - start < duration:
            self.engine.tick()
            sys.stdout.write(self.engine.render_overwrite())
            sys.stdout.flush()
            await asyncio.sleep(1.0 / fps)

        # Clear canvas
        sys.stdout.write(f"\033[{h}A")
        for _ in range(h):
            sys.stdout.write(' ' * (self.engine.w + 8) + '\n')
        sys.stdout.write(f"\033[{h}A")
        sys.stdout.flush()

    # ── Idle (non-blocking context manager) ────────────────────────────────

    async def _loop(self):
        interval = 1.0 / self.fps
        while self._running:
            try:
                self.engine.tick()
                frame = self.engine.render_ansi()
                if self._offset > 0:
                    sys.stdout.write(
                        f"\033[s"                # save cursor position
                        f"\033[{self._offset}A"  # jump up to flow area
                        f"\r"                    # start of line
                        f"{frame}"               # draw FLOW_HEIGHT lines
                        f"\033[u"                # restore cursor position
                    )
                    sys.stdout.flush()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1.0)

    async def __aenter__(self):
        self._running = True
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        return False

    def stop(self):
        """Sync stop for signal handlers."""
        self._running = False
        if self._task:
            self._task.cancel()


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM INFO
# ═══════════════════════════════════════════════════════════════════════════════

def _get_system_metrics() -> dict:
    metrics = {
        'os': platform.system(), 'arch': platform.machine(),
        'python': platform.python_version(),
        'cpu_percent': '?', 'mem_used_gb': '?',
        'mem_total_gb': '?', 'mem_percent': '?',
        'term_width': shutil.get_terminal_size().columns,
    }
    if psutil:
        try:
            metrics['cpu_percent'] = f"{psutil.cpu_percent(interval=0.1):.0f}"
            mem = psutil.virtual_memory()
            metrics['mem_used_gb'] = f"{mem.used / (1024**3):.1f}"
            metrics['mem_total_gb'] = f"{mem.total / (1024**3):.1f}"
            metrics['mem_percent'] = f"{mem.percent:.0f}"
        except Exception:
            pass
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER RENDERING
# ═══════════════════════════════════════════════════════════════════════════════

LOGO_LINES = [
    " ██╗   ███████╗    █████╗     █████╗ ",
    " ██║   ██╔════╝   ██╔══██╗   ██╔══██╗",
    " ██║   ███████╗   ███████║   ███████║",
    " ██║   ╚════██║   ██╔══██║   ██╔══██║",
    " ██║   ███████║   ██║  ██║   ██║  ██║",
    " ╚═╝   ╚══════╝   ╚═╝  ╚═╝   ╚═╝  ╚═╝",
]


def _state_badge(state: str) -> str:
    badges = {
        'online':       ("⚡ ONLINE",       GradientCyan.L4_BRIGHT),
        'initializing': ("◐ INITIALIZING",  StateColors.INIT_PULSE),
        'dreaming':     ("☾ DREAMING",      StateColors.DREAM_VIOLET),
        'high_load':    ("▲ HIGH LOAD",     StateColors.LOAD_RED),
        'error':        ("✗ ERROR",         StateColors.LOAD_RED),
    }
    text, color = badges.get(state, badges['online'])
    return f"<style fg='{color}' font-weight='bold'>{text}</style>"


def _build_status_bar(host_id, uptime, version, state='online', agent_count=0, task_count=0):
    ts = int(uptime.total_seconds())
    if ts < 3600:    up = f"{ts // 60}m {ts % 60}s"
    elif ts < 86400: up = f"{ts // 3600}h {(ts % 3600) // 60}m"
    else:            up = f"{ts // 86400}d {(ts % 86400) // 3600}h"

    sc = GradientCyan.L3_MID
    if state == 'high_load':      sc = StateColors.LOAD_ORANGE
    elif state == 'dreaming':     sc = StateColors.DREAM_VIOLET
    elif state == 'initializing': sc = StateColors.INIT_DIM

    sep = f"<style fg='{sc}'> │ </style>"
    d, b = GradientCyan.L3_MID, GradientCyan.WHITE
    segs = [
        f"<style fg='{b}'>{_esc(host_id)}</style>",
        f"<style fg='{d}'>⏱</style> <style fg='{b}'>{up}</style>",
        f"<style fg='{d}'>v</style><style fg='{b}'>{_esc(version)}</style>",
    ]
    if agent_count > 0:
        segs.append(f"<style fg='{d}'>agents:</style><style fg='{GradientCyan.L4_BRIGHT}'>{agent_count}</style>")
    if task_count > 0:
        tc = StateColors.LOAD_ORANGE if task_count > 3 else GradientCyan.L4_BRIGHT
        segs.append(f"<style fg='{d}'>tasks:</style><style fg='{tc}'>{task_count}</style>")
    return sep.join(segs)


def _build_system_bar():
    m = _get_system_metrics()
    d, v = GradientCyan.L2_DARK, GradientCyan.L3_MID
    sep = f"<style fg='{d}'> · </style>"
    parts = [
        f"<style fg='{d}'>{_esc(m['os'])}</style>",
        f"<style fg='{d}'>{_esc(m['arch'])}</style>",
        f"<style fg='{d}'>py</style><style fg='{v}'>{_esc(m['python'])}</style>",
    ]
    if m['cpu_percent'] != '?':
        cpu = float(m['cpu_percent'])
        cc = StateColors.LOAD_RED if cpu > 80 else (StateColors.INIT_PULSE if cpu > 50 else v)
        parts.append(f"<style fg='{d}'>cpu:</style><style fg='{cc}'>{m['cpu_percent']}%</style>")
    if m['mem_percent'] != '?':
        mc = StateColors.LOAD_RED if float(m['mem_percent']) > 85 else v
        parts.append(f"<style fg='{d}'>mem:</style><style fg='{mc}'>{m['mem_used_gb']}/{m['mem_total_gb']}G</style>")
    return sep.join(parts)


def print_isaa_header(
    host_id: str = "????????",
    uptime: Optional[timedelta] = None,
    version: str = "0.0.0",
    state: str = "online",
    agent_count: int = 0,
    task_count: int = 0,
    show_system_bar: bool = True,
    subtitle: str = "Intelligent Semantic Agent Architecture",
) -> int:
    """
    Print branded ISAA header. First FLOW_HEIGHT lines = animation zone.
    Returns total lines printed (pass to anim.set_cursor_offset()).
    """
    if uptime is None:
        uptime = timedelta(seconds=0)
    if state == 'online' and task_count > 3:
        state = 'high_load'

    logo_c = GradientCyan.L5_GLOW
    border_c = GradientCyan.L3_MID
    fc = (GradientCyan.L1_DEEP, GradientCyan.L2_DARK)

    if state == 'initializing':
        logo_c, border_c = StateColors.INIT_PULSE, StateColors.INIT_DIM
    elif state == 'dreaming':
        logo_c, border_c = StateColors.DREAM_VIOLET, StateColors.DREAM_DIM
        fc = (StateColors.DREAM_DIM, StateColors.DREAM_VIOLET)
    elif state == 'high_load':
        border_c = StateColors.LOAD_ORANGE
    elif state == 'error':
        logo_c, border_c = StateColors.LOAD_RED, StateColors.LOAD_RED

    n = 0  # line counter

    # ── Flow zone (FLOW_HEIGHT lines) — overwritten by animation ──
    flow_pats = [
        "│  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │",
        "╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎",
        "┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃",
        "▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼",
    ]
    for i, pat in enumerate(flow_pats[:FLOW_HEIGHT]):
        c = fc[0] if i < 2 else fc[1]
        print_formatted_text(HTML(f"<style fg='{c}'>    {_esc(pat)}</style>"))
        n += 1

    # ── Box ──
    bw = 72
    print_formatted_text(HTML(f"   <style fg='{border_c}'>╔{'═' * bw}╗</style>"))
    n += 1

    for ll in LOGO_LINES:
        pad = bw - len(ll) - 1
        r = f"<style fg='{logo_c}' font-weight='bold'>{_esc(ll)}</style>"
        print_formatted_text(HTML(
            f"   <style fg='{border_c}'>║</style> {r}{' ' * max(0, pad)}<style fg='{border_c}'>║</style>"
        ))
        n += 1

    badge = _state_badge(state)

    if subtitle == "time-random":
        subtitle = get_greeting('en')
    if subtitle == "zeit-random":
        subtitle = get_greeting('de')
    sub = _esc(subtitle[:42])
    sp = bw - len(subtitle[:42]) - 12
    print_formatted_text(HTML(
        f"   <style fg='{border_c}'>║</style>  <style fg='{GradientCyan.L3_MID}'>{sub}</style>"
        f"{' ' * max(2, sp)}{badge}  <style fg='{border_c}'>║</style>"
    ))
    n += 1

    print_formatted_text(HTML(f"   <style fg='{border_c}'>╚{'═' * bw}╝</style>"))
    n += 1

    # ── Status ──
    print_formatted_text(HTML(f"    {_build_status_bar(host_id, uptime, version, state, agent_count, task_count)}"))
    n += 1

    if show_system_bar:
        print_formatted_text(HTML(f"    {_build_system_bar()}"))
        n += 1

    print_formatted_text(HTML(""))
    n += 1

    return n


# ═══════════════════════════════════════════════════════════════════════════════
# EXTENDED PRINTER API (additive)
# ═══════════════════════════════════════════════════════════════════════════════

def print_brand_separator(style: str = "default", width: int = 76):
    patterns = {
        'default': ('─', GradientCyan.L2_DARK),
        'double':  ('═', GradientCyan.L3_MID),
        'dots':    ('·', GradientCyan.L1_DEEP),
    }
    if style == 'gradient':
        cs = [GradientCyan.L1_DEEP, GradientCyan.L2_DARK, GradientCyan.L3_MID,
              GradientCyan.L4_BRIGHT, GradientCyan.L3_MID, GradientCyan.L2_DARK, GradientCyan.L1_DEEP]
        s = width // len(cs)
        print_formatted_text(HTML("".join(f"<style fg='{c}'>{'━' * s}</style>" for c in cs)))
    else:
        ch, co = patterns.get(style, patterns['default'])
        print_formatted_text(HTML(f"<style fg='{co}'>{ch * width}</style>"))


def print_brand_metric(label, value, unit="", color=GradientCyan.L4_BRIGHT, width=30):
    dl = max(2, width - len(label) - len(value) - len(unit) - 2)
    print_formatted_text(HTML(
        f"  <style fg='{GradientCyan.L2_DARK}'>{_esc(label)}</style> "
        f"<style fg='{GradientCyan.L1_DEEP}'>{'·' * dl}</style> "
        f"<style fg='{color}' font-weight='bold'>{_esc(value)}</style>"
        f"<style fg='{GradientCyan.L2_DARK}'>{_esc(unit)}</style>"
    ))


def print_brand_progress(label, current, maximum, width=30, color=GradientCyan.L4_BRIGHT):
    r = min(1.0, current / maximum) if maximum > 0 else 0.0
    f = int(width * r)
    print_formatted_text(HTML(
        f"  <style fg='{GradientCyan.L2_DARK}'>{_esc(label)}</style> "
        f"<style fg='{color}'>{'━' * f}</style>"
        f"<style fg='{GradientCyan.L1_DEEP}'>{'─' * (width - f)}</style> "
        f"<style fg='{color}'>{r * 100:.0f}%</style>"
    ))


def print_brand_kv_row(pairs):
    sep = f"<style fg='{GradientCyan.L1_DEEP}'> │ </style>"
    ps = [
        f"<style fg='{GradientCyan.L2_DARK}'>{_esc(k)}:</style>"
        f"<style fg='{GradientCyan.L4_BRIGHT}'>{_esc(v)}</style>"
        for k, v in pairs
    ]
    print_formatted_text(HTML(f"  {sep.join(ps)}"))


def print_brand_section(title, icon="◈"):
    print_formatted_text(HTML(
        f"\n  <style fg='{GradientCyan.L4_BRIGHT}' font-weight='bold'>{icon} {_esc(title)}</style>"
        f"\n  <style fg='{GradientCyan.L2_DARK}'>{'─' * (len(title) + 4)}</style>"
    ))


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD V2
# ═══════════════════════════════════════════════════════════════════════════════

async def print_status_dashboard_v2(host: 'ISAA_Host') -> int:
    """Drop-in replacement. Returns total lines for animation offset."""
    from toolboxv2.flows.cli_v4 import (
        print_table_header, print_table_row, c_print, AgentInfo
    )

    running = [t for t in host.background_tasks.values() if t.status == "running"]
    agents = host.isaa_tools.config.get("agents-name-list", [])

    state = 'online'
    if not host._self_agent_initialized:
        state = 'initializing'
    elif len(running) > 3:
        state = 'high_load'
    elif host.job_scheduler:
        for j in host.job_scheduler._jobs.values():
            if 'dream' in j.name.lower() and j.status == 'active':
                state = 'dreaming'
                break

    n = print_isaa_header(
        host_id=host.host_id,
        uptime=datetime.now() - host.started_at,
        version=host.version,
        state=state,
        agent_count=len(agents),
        task_count=len(running),
    )

    if agents:
        print_brand_section("Agents", "🤖"); n += 2
        mnl = max(len(x) for x in agents)
        ws = [mnl, 8, 22, 5]
        print_table_header([("Name",mnl),("Status",8),("Persona",22),("Tasks",5)], ws); n += 2
        for name in agents[:10]:
            info = host.agent_registry.get(name, AgentInfo(name=name))
            ik = f"agent-instance-{name}"
            st = "Active" if ik in host.isaa_tools.config else "Idle"
            bg = sum(1 for t in host.background_tasks.values() if t.agent_name == name and t.status == "running")
            p = info.persona[:20]+".." if len(info.persona)>22 else info.persona
            print_table_row([name, st, p, str(bg)], ws,
                ["cyan" if info.is_self_agent else "white", "green" if st=="Active" else "grey", "grey", "yellow"])
            n += 1

    if running:
        print_brand_section("Running Tasks", "⟳"); n += 2
        print_table_header([("Agent",14),("Progress",22),("Phase",10),("Focus",18)], [14,22,10,18]); n += 2
        for t in running:
            ph, bar, fo = "-", "", "-"
            try:
                ag = await host.isaa_tools.get_agent(t.agent_name)
                live = ag._get_execution_engine().live
                it, mx = live.iteration, live.max_iterations
                fl = int(16*it/mx) if mx>0 else 0
                bar = f"{'━'*fl}{'─'*(16-fl)} {it}/{mx if mx>0 else '?'}"
                ph = live.phase.value[:10]
                if live.tool.name: fo = f"◇ {live.tool.name[:16]}"
                elif live.thought: fo = f"◎ {live.thought[:16]}"
                elif live.status_msg: fo = live.status_msg[:18]
            except Exception:
                bar = f"{'─'*16} {(datetime.now()-t.started_at).total_seconds():.0f}s"
            print_table_row([t.agent_name[:14], bar, ph, fo], [14,22,10,18], ["cyan","green","white","grey"])
            n += 1

    if host.job_scheduler and host.job_scheduler.total_count > 0:
        print_brand_kv_row([("Jobs", f"{host.job_scheduler.active_count}/{host.job_scheduler.total_count}")]); n += 1
    if host.feature_manager:
        en = [f for f in host.feature_manager.features if host.feature_manager.features[f].get("is_enabled", False)]
        if en:
            print_brand_kv_row([("Features", ", ".join(en[:5]) + (f" +{len(en)-5}" if len(en)>5 else ""))]); n += 1

    print_brand_separator('dots'); c_print(); n += 2
    return n


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

async def demo():
    from prompt_toolkit import PromptSession

    print_formatted_text(HTML(
        f"\n<style fg='{GradientCyan.L4_BRIGHT}' font-weight='bold'>"
        f"  ══ ISAA BRANDING + ANIMATION DEMO ══</style>\n"
    ))

    anim = FlowMatrixAnimation(height=FLOW_HEIGHT, width=72, fps=8)

    # 1. Startup rain
    await anim.play_startup(duration=2.5, fps=12)

    # 2. Static header
    total = print_isaa_header(
        host_id="7e9d3c64",
        uptime=timedelta(hours=2, minutes=15),
        version="4.0.0",
        state="online",
        agent_count=3,
        task_count=1,
    )

    # 3. Non-blocking animation during prompt
    print_formatted_text(HTML(
        f"<style fg='{GradientCyan.L3_MID}'>"
        f"  Flow lines above are now live. Type + Enter to stop.</style>\n"
    ))

    anim.set_cursor_offset(total + 3)  # +3 for info text + blank + prompt

    session = PromptSession()
    async with anim:
        result = await session.prompt_async(
            HTML(f"<style fg='{GradientCyan.L4_BRIGHT}'>ISAA › </style>"),
        )

    print_formatted_text(HTML(
        f"\n<style fg='{GradientCyan.L4_BRIGHT}'>✓ Animation stopped. Input: {_esc(result)}</style>\n"
    ))

    # Show other states
    for st in ['dreaming', 'high_load']:
        print_formatted_text(HTML(f"\n<style fg='{GradientCyan.L2_DARK}'>  ── State: {st} ──</style>"))
        sa = FlowMatrixAnimation(height=FLOW_HEIGHT, width=72, state=st)
        await sa.play_startup(duration=1.5, fps=10)
        print_isaa_header(
            host_id="7e9d3c64", uptime=timedelta(hours=14), version="4.0.0",
            state=st, agent_count=5, task_count=5 if st == 'high_load' else 1,
            subtitle="Dream Consolidation Active" if st == 'dreaming' else "Intelligent Semantic Agent Architecture",
        )


if __name__ == "__main__":
    asyncio.run(demo())
