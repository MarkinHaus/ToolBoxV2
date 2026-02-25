"""
ISAA Branding & Flow Matrix Animation (prompt_toolkit-safe)
============================================================

All animation runs through prompt_toolkit's native rendering:
  FormattedTextControl → (style, text) tuples → Application(refresh_interval)

No raw ANSI escapes. No sys.stdout cursor hacks.
Safe alongside PromptSession, completers, key bindings.

Usage:
    from isaa_branding import FlowMatrixAnimation, print_isaa_header

    anim = FlowMatrixAnimation(state='online')

    # Startup: animated rain for 2.5s
    await anim.play_startup(duration=2.5)

    # Static branded header
    print_isaa_header(host_id=..., state='online', ...)

    # Animated prompt (replaces session.prompt_async)
    result = await anim.prompt_async(
        message="ISAA › ",
        completer=my_completer,
        key_bindings=my_kb,
    )

    # Switch to audio recording mode (changes animation)
    anim.set_mode('audio')
    result = await anim.prompt_async(message="🎙 › ")
    anim.set_mode('idle')
"""

import asyncio
import html
import platform
import random
import shutil
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from prompt_toolkit import Application, print_formatted_text
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.history import History
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.layout import Layout, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl
from prompt_toolkit.layout.processors import BeforeInput

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
    AUDIO_GREEN  = '#4ade80'
    AUDIO_DIM    = '#166534'
    AUDIO_PULSE  = '#86efac'


def _esc(text: Any) -> str:
    return html.escape(str(text).encode().decode(encoding="utf-8", errors="replace"))


# ═══════════════════════════════════════════════════════════════════════════════
# MATRIX RAIN ENGINE — renders to (style, text) tuples
# ═══════════════════════════════════════════════════════════════════════════════

# Character sets per mode
CHARS_IDLE  = "│╎┃║╏┊┋▏▎▍▌▐░" * 3 + "ｱｲｳｴｵｶｷｸｹｺ0123456789"
CHARS_AUDIO = "▁▂▃▄▅▆▇█░▒▓│║┃" + "∿∾≋~" * 2

# Color palettes per mode (index 0=off, 1=dim..4=bright)
PALETTE_IDLE = ['', f'fg:{GradientCyan.L1_DEEP}', f'fg:{GradientCyan.L2_DARK}',
                f'fg:{GradientCyan.L3_MID}', f'fg:{GradientCyan.L5_GLOW}']
PALETTE_AUDIO = ['', f'fg:{StateColors.AUDIO_DIM}', f'fg:{StateColors.AUDIO_GREEN}',
                 f'fg:{StateColors.AUDIO_PULSE}', f'fg:{StateColors.AUDIO_GREEN} bold']
PALETTE_DREAM = ['', f'fg:{StateColors.DREAM_DIM}', f'fg:#7c3aed',
                 f'fg:{StateColors.DREAM_VIOLET}', f'fg:#c4b5fd']
PALETTE_LOAD  = ['', f'fg:{GradientCyan.L1_DEEP}', f'fg:{GradientCyan.L2_DARK}',
                 f'fg:{GradientCyan.L3_MID}', f'fg:{StateColors.LOAD_ORANGE} bold']


class _Stream:
    __slots__ = ('pos', 'length', 'speed')

    def __init__(self, height: int):
        self.reset(height)

    def reset(self, height: int):
        self.pos = random.randint(-height * 2, -1)
        self.length = random.randint(2, max(3, height - 1))
        self.speed = random.choice([1, 1, 1, 2])

    def tick(self, height: int):
        self.pos += self.speed
        if self.pos - self.length > height + 2:
            self.reset(height)


class MatrixRainEngine:
    """
    Renders matrix rain as prompt_toolkit (style, text) tuples.
    Supports mode switching for idle / audio / dreaming / high_load.
    """

    def __init__(self, height: int = 4, width: int = 72):
        self.h = height
        self.w = width
        self.streams = [_Stream(height) for _ in range(width)]
        self.grid = [[' '] * width for _ in range(height)]
        self.brightness = [[0] * width for _ in range(height)]

        # Active mode config
        self._chars = CHARS_IDLE
        self._palette = PALETTE_IDLE
        self._mode = 'idle'

        # Audio-specific state
        self._audio_phase = 0.0

    def set_mode(self, mode: str):
        """Switch animation mode: 'idle', 'audio', 'dreaming', 'high_load'."""
        self._mode = mode
        if mode == 'audio':
            self._chars = CHARS_AUDIO
            self._palette = PALETTE_AUDIO
        elif mode == 'dreaming':
            self._chars = CHARS_IDLE
            self._palette = PALETTE_DREAM
        elif mode == 'high_load':
            self._chars = CHARS_IDLE
            self._palette = PALETTE_LOAD
        else:
            self._chars = CHARS_IDLE
            self._palette = PALETTE_IDLE

    def tick(self):
        if self._mode == 'audio':
            self._tick_audio()
        else:
            self._tick_rain()

    def _tick_rain(self):
        """Standard matrix rain tick."""
        for col, stream in enumerate(self.streams):
            stream.tick(self.h)
            for row in range(self.h):
                dist = stream.pos - row
                if 0 <= dist < stream.length:
                    b = 4 if dist == 0 else (3 if dist == 1 else (2 if dist < 4 else 1))
                    self.brightness[row][col] = b
                    if random.random() < 0.2 or self.grid[row][col] == ' ':
                        self.grid[row][col] = random.choice(self._chars)
                else:
                    if self.brightness[row][col] > 0:
                        self.brightness[row][col] -= 1
                    if self.brightness[row][col] == 0:
                        self.grid[row][col] = ' '

    def _tick_audio(self):
        """Audio waveform tick — pulsing columns that react to 'sound'."""
        import math
        self._audio_phase += 0.3
        for col in range(self.w):
            # Simulate audio waveform with overlapping sine waves
            wave = (
                math.sin(self._audio_phase + col * 0.15) * 0.4
                + math.sin(self._audio_phase * 1.7 + col * 0.08) * 0.3
                + random.uniform(-0.15, 0.15)  # noise
            )
            # Map wave amplitude to which rows are lit
            center = (self.h - 1) / 2
            amplitude = abs(wave) * self.h * 0.8

            for row in range(self.h):
                dist_from_center = abs(row - center)
                if dist_from_center < amplitude:
                    # Brightness based on distance from center
                    b = 4 if dist_from_center < amplitude * 0.3 else (
                        3 if dist_from_center < amplitude * 0.6 else 2)
                    self.brightness[row][col] = b
                    if random.random() < 0.25 or self.grid[row][col] == ' ':
                        self.grid[row][col] = random.choice(self._chars)
                else:
                    if self.brightness[row][col] > 0:
                        self.brightness[row][col] -= 1
                    if self.brightness[row][col] == 0:
                        self.grid[row][col] = ' '

    def render_tuples(self, indent: int = 4) -> list[tuple[str, str]]:
        """Render current frame as prompt_toolkit (style, text) tuples."""
        frags: list[tuple[str, str]] = []
        pad = ' ' * indent

        for row in range(self.h):
            frags.append(('', pad))
            for col in range(self.w):
                b = self.brightness[row][col]
                ch = self.grid[row][col]
                if b > 0 and ch != ' ':
                    frags.append((self._palette[b], ch))
                else:
                    frags.append(('', ' '))
            if row < self.h - 1:
                frags.append(('', '\n'))

        return frags


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW MATRIX ANIMATION — prompt_toolkit native
# ═══════════════════════════════════════════════════════════════════════════════

FLOW_HEIGHT = 4


class FlowMatrixAnimation:
    """
    prompt_toolkit-safe Matrix Flow animation.
    All rendering via FormattedTextControl → Application(refresh_interval).

    Modes:
      'idle'      — Matrix rain with flow chars (│╎┃ + ｱｲｳ), cyan gradient
      'audio'     — Waveform bars (▁▂▃▄▅▆▇█), green pulsing
      'dreaming'  — Slow rain, violet palette
      'high_load' — Fast rain, orange-tipped
    """

    def __init__(self, height: int = FLOW_HEIGHT, width: int = 72,
                 fps: float = 8, state: str = 'online'):
        self.engine = MatrixRainEngine(height=height, width=width)
        self.fps = fps
        self._refresh = max(0.05, 1.0 / fps)

        # Map host state → animation mode
        mode_map = {'online': 'idle', 'initializing': 'idle',
                    'dreaming': 'dreaming', 'high_load': 'high_load',
                    'error': 'high_load', 'audio': 'audio'}
        self.engine.set_mode(mode_map.get(state, 'idle'))

    def set_mode(self, mode: str):
        """Switch animation: 'idle', 'audio', 'dreaming', 'high_load'."""
        if mode == "stream":
            self.fps += 2
        if mode == "idle":
            self.fps += 2
        self.engine.set_mode(mode)

    def _get_flow_text(self) -> list[tuple[str, str]]:
        """Callback for FormattedTextControl — one tick + render."""
        self.engine.tick()
        return self.engine.render_tuples()

    # ── Startup Animation (blocking) ──────────────────────────────────────

    async def play_startup(self, duration: float = 2.5):
        """
        Show Matrix Rain for `duration` seconds, then return.
        Uses a temporary Application — fully prompt_toolkit-safe.
        """
        control = FormattedTextControl(self._get_flow_text)
        window = Window(content=control, height=self.engine.h, dont_extend_height=True)
        layout = Layout(Window(content=control, height=self.engine.h))

        app: Application = Application(
            layout=layout,
            refresh_interval=self._refresh,
        )

        async def _auto_exit():
            await asyncio.sleep(duration)
            app.exit()

        task = asyncio.create_task(_auto_exit())
        try:
            await app.run_async()
        finally:
            task.cancel()

    # ── Animated Prompt (non-blocking animation during input) ─────────────

    async def prompt_async(
        self,
        message: str = "ISAA › ",
        *,
        completer: Optional[Completer] = None,
        history: Optional[History] = None,
        key_bindings: Optional[KeyBindings] = None,
        is_password: bool = False,
        prompt_style: str = f'fg:{GradientCyan.L4_BRIGHT}',
        extra_key_bindings: Optional[KeyBindings] = None,
        on_text_changed: Optional[Callable] = None,
    ) -> str:
        """
        Animated prompt — replaces PromptSession.prompt_async().
        Flow animation runs above the input line via Application layout.

        Returns the user's input string.
        """
        # ── Flow animation window ──
        flow_control = FormattedTextControl(self._get_flow_text)
        flow_window = Window(
            content=flow_control,
            height=self.engine.h,
            dont_extend_height=True,
        )

        # ── Separator ──
        sep_control = FormattedTextControl(
            lambda: [(f'fg:{GradientCyan.L2_DARK}', '─' * self.engine.w)]
        )
        sep_window = Window(content=sep_control, height=1, dont_extend_height=True)

        # ── Input buffer ──
        buffer = Buffer(
            completer=completer,
            history=history,
            name='isaa_input',
            on_text_changed=lambda buf: on_text_changed(buf) if on_text_changed else None,
        )
        input_control = BufferControl(
            buffer=buffer,
            input_processors=[BeforeInput(lambda: [(prompt_style, message)])],
        )
        input_window = Window(content=input_control, height=1, dont_extend_height=True)

        # ── Key bindings ──
        base_kb = KeyBindings()

        @base_kb.add('enter')
        def _accept(event):
            event.app.exit(result=buffer.text)

        @base_kb.add('c-c')
        def _abort(event):
            event.app.exit(result='')

        @base_kb.add('c-d')
        def _eof(event):
            if not buffer.text:
                event.app.exit(result='exit')

        all_kb = [base_kb]
        if key_bindings:
            all_kb.append(key_bindings)
        if extra_key_bindings:
            all_kb.append(extra_key_bindings)

        merged_kb = merge_key_bindings(all_kb) if len(all_kb) > 1 else base_kb

        # ── Layout ──
        layout = Layout(
            HSplit([flow_window, sep_window, input_window]),
            focused_element=input_window,
        )

        # ── Application ──
        app: Application[str] = Application(
            layout=layout,
            key_bindings=merged_kb,
            refresh_interval=self._refresh,
            mouse_support=False,
        )

        return await app.run_async()


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM INFO
# ═══════════════════════════════════════════════════════════════════════════════

def _get_system_metrics() -> dict:
    m = {
        'os': platform.system(), 'arch': platform.machine(),
        'python': platform.python_version(),
        'cpu_percent': '?', 'mem_used_gb': '?',
        'mem_total_gb': '?', 'mem_percent': '?',
        'term_width': shutil.get_terminal_size().columns,
    }
    if psutil:
        try:
            m['cpu_percent'] = f"{psutil.cpu_percent(interval=0.1):.0f}"
            mem = psutil.virtual_memory()
            m['mem_used_gb'] = f"{mem.used / (1024**3):.1f}"
            m['mem_total_gb'] = f"{mem.total / (1024**3):.1f}"
            m['mem_percent'] = f"{mem.percent:.0f}"
        except Exception:
            pass
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# STATIC HEADER
# ═══════════════════════════════════════════════════════════════════════════════

LOGO_LINES = [
    " ██╗   ███████╗    █████╗     █████╗ ",
    " ██║   ██╔════╝   ██╔══██╗   ██╔══██╗",
    " ██║   ███████╗   ███████║   ███████║",
    " ██║   ╚════██║   ██╔══██║   ██╔══██║",
    " ██║   ███████║   ██║  ██║   ██║  ██║",
    " ╚═╝   ╚══════╝   ╚═╝  ╚═╝   ╚═╝  ╚═╝",
]


def _state_badge(state: str) -> (str, int):
    badges = {
        'online':       ("⚡ ONLINE",       GradientCyan.L4_BRIGHT, 1+len("⚡ ONLINE")),
        'initializing': ("◐ INITIALIZING",  StateColors.INIT_PULSE, len("◐ INITIALIZING")),
        'dreaming':     ("☾ DREAMING",      StateColors.DREAM_VIOLET, 1+len("☾ DREAMING")),
        'high_load':    ("▲ HIGH LOAD",     StateColors.LOAD_RED, 1+len("▲ HIGH LOAD")),
        'error':        ("✗ ERROR",         StateColors.LOAD_RED, 1+len("✗ ERROR")),
        'audio':        ("🎙 RECORDING",    StateColors.AUDIO_GREEN, len("🎙 RECORDING")),
    }
    text, color, text_len = badges.get(state, badges['online'])
    return f"<style fg='{color}' font-weight='bold'>{text}</style>", text_len


def _build_status_bar(host_id, uptime, version, state='online', agent_count=0, task_count=0):
    ts = int(uptime.total_seconds())
    if ts < 3600:    up = f"{ts // 60}m {ts % 60}s"
    elif ts < 86400: up = f"{ts // 3600}h {(ts % 3600) // 60}m"
    else:            up = f"{ts // 86400}d {(ts % 86400) // 3600}h"

    sc = GradientCyan.L3_MID
    if state == 'high_load':      sc = StateColors.LOAD_ORANGE
    elif state == 'dreaming':     sc = StateColors.DREAM_VIOLET
    elif state == 'initializing': sc = StateColors.INIT_DIM
    elif state == 'audio':        sc = StateColors.AUDIO_GREEN

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
    show_flows: bool = True,
    subtitle: str = "Intelligent Semantic Agent Architecture",
):
    """Print static branded ISAA header."""
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
    elif state == 'audio':
        logo_c, border_c = StateColors.AUDIO_GREEN, StateColors.AUDIO_DIM
        fc = (StateColors.AUDIO_DIM, StateColors.AUDIO_GREEN)

    # ── Static flow lines (placeholder — animation replaces these) ──
    if show_flows:
        pats = [
            "│  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │",
            "╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎  │  ╎",
            "┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃",
            "▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼",
        ]
        for i, p in enumerate(pats[:FLOW_HEIGHT]):
            c = fc[0] if i < 2 else fc[1]
            print_formatted_text(HTML(f"<style fg='{c}'>    {_esc(p)}</style>"))

    # ── Box ──
    bw = 72
    print_formatted_text(HTML(f"   <style fg='{border_c}'>╔{'═' * bw}╗</style>"))
    for ll in LOGO_LINES:
        pad = bw - len(ll) - 1
        r = f"<style fg='{logo_c}' font-weight='bold'>{_esc(ll)}</style>"
        print_formatted_text(HTML(
            f"   <style fg='{border_c}'>║</style> {r}{' ' * max(0, pad)}<style fg='{border_c}'>║</style>"
        ))

    badge, offset = _state_badge(state)

    if subtitle == "time-random":
        subtitle = get_greeting('en')
    if subtitle == "zeit-random":
        subtitle = get_greeting('de')
    sub = _esc(subtitle[:42])
    sp = bw - len(subtitle[:42]) - (18 - (14 - offset))
    print_formatted_text(HTML(
        f"   <style fg='{border_c}'>║</style>  <style fg='{GradientCyan.L3_MID}'>{sub}</style>"
        f"{' ' * max(2, sp)}{badge}  <style fg='{border_c}'>║</style>"
    ))
    print_formatted_text(HTML(f"   <style fg='{border_c}'>╚{'═' * bw}╝</style>"))

    # ── Status ──
    print_formatted_text(HTML(f"    {_build_status_bar(host_id, uptime, version, state, agent_count, task_count)}"))
    if show_system_bar:
        print_formatted_text(HTML(f"    {_build_system_bar()}"))
    print_formatted_text(HTML(""))


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
    if not host._self_agent_initialized:   state = 'initializing'
    elif len(running) > 3:                 state = 'high_load'
    elif host.job_scheduler:
        for j in host.job_scheduler._jobs.values():
            if 'dream' in j.name.lower() and j.status == 'active':
                state = 'dreaming'; break

    print_isaa_header(
        host_id=host.host_id,
        uptime=datetime.now() - host.started_at,
        version=host.version, state=state,
        agent_count=len(agents), task_count=len(running),
    )

    if agents:
        print_brand_section("Agents", "🤖")
        mnl = max(len(x) for x in agents)
        ws = [mnl, 8, 22, 5]
        print_table_header([("Name",mnl),("Status",8),("Persona",22),("Tasks",5)], ws)
        for name in agents[:10]:
            info = host.agent_registry.get(name, AgentInfo(name=name))
            st = "Active" if f"agent-instance-{name}" in host.isaa_tools.config else "Idle"
            bg = sum(1 for t in host.background_tasks.values() if t.agent_name == name and t.status == "running")
            p = info.persona[:20]+".." if len(info.persona)>22 else info.persona
            print_table_row([name, st, p, str(bg)], ws,
                ["cyan" if info.is_self_agent else "white", "green" if st=="Active" else "grey", "grey", "yellow"])

    if running:
        print_brand_section("Running Tasks", "⟳")
        print_table_header([("Agent",14),("Progress",22),("Phase",10),("Focus",18)], [14,22,10,18])
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

    if host.job_scheduler and host.job_scheduler.total_count > 0:
        print_brand_kv_row([("Jobs", f"{host.job_scheduler.active_count}/{host.job_scheduler.total_count}")])
    if host.feature_manager:
        en = [f for f in host.feature_manager.features if host.feature_manager.features[f].get("is_enabled", False)]
        if en:
            print_brand_kv_row([("Features", ", ".join(en[:5]) + (f" +{len(en)-5}" if len(en)>5 else ""))])

    print_brand_separator('dots')
    c_print()


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

async def demo():
    print_formatted_text(HTML(
        f"\n<style fg='{GradientCyan.L4_BRIGHT}' font-weight='bold'>"
        f"  ══ ISAA FLOW MATRIX ANIMATION ══</style>\n"
    ))

    anim = FlowMatrixAnimation(fps=8, state='online')

    # 1. Startup rain (2.5s)
    print_formatted_text(HTML(f"<style fg='{GradientCyan.L2_DARK}'>  ┌ Startup Animation...</style>"))
    await anim.play_startup(duration=2.5)

    # 2. Static header
    print_isaa_header(
        host_id="7e9d3c64", uptime=timedelta(hours=2, minutes=15),
        version="4.0.0", state="online", agent_count=3, task_count=1,
    )

    # 3. Animated prompt (idle mode)
    print_formatted_text(HTML(f"<style fg='{GradientCyan.L3_MID}'>  Flow lines animate above. Type + Enter.</style>\n"))
    result = await anim.prompt_async(message="ISAA › ")
    print_formatted_text(HTML(f"<style fg='{GradientCyan.L4_BRIGHT}'>  ✓ Input: {_esc(result)}</style>\n"))

    # 4. Switch to audio mode
    print_formatted_text(HTML(f"<style fg='{StateColors.AUDIO_GREEN}'>  ┌ Audio Recording Mode...</style>"))
    anim.set_mode('audio')
    result = await anim.prompt_async(message="🎙 › ", prompt_style=f'fg:{StateColors.AUDIO_GREEN}')
    print_formatted_text(HTML(f"<style fg='{StateColors.AUDIO_GREEN}'>  ✓ Audio Input: {_esc(result)}</style>\n"))

    # 5. Dreaming mode
    print_formatted_text(HTML(f"<style fg='{StateColors.DREAM_VIOLET}'>  ┌ Dreaming Mode...</style>"))
    anim.set_mode('dreaming')
    result = await anim.prompt_async(message="☾ › ", prompt_style=f'fg:{StateColors.DREAM_VIOLET}')
    print_formatted_text(HTML(f"<style fg='{StateColors.DREAM_VIOLET}'>  ✓ Dream: {_esc(result)}</style>"))


if __name__ == "__main__":
    asyncio.run(demo())
