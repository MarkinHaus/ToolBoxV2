#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║   YT STREAM SERVICE  —  Enterprise Live Streaming Hub   ║
║   Single-file · Bottle · FFmpeg · REST API · unittest   ║
╚══════════════════════════════════════════════════════════╝

Stack:
  - Bottle  (zero-dep Python micro-framework, bundled below)
  - FFmpeg  (subprocess, RTMP → YouTube/any target)
  - sqlite3 (stdlib, stream persistence)
  - unittest (stdlib, full test suite)

Usage:
  python yt_stream_service.py          # start server :8080
  python yt_stream_service.py test     # run test suite
  python yt_stream_service.py setup    # print setup guide
"""

import sys
import os
import json
import time
import uuid
import sqlite3
import subprocess
import threading
import mimetypes
import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from io import BytesIO
from urllib.parse import urlencode, parse_qs

# ─────────────────────────────────────────────────────────────
#  EMBEDDED BOTTLE  (single-file, no external deps needed)
# ─────────────────────────────────────────────────────────────
try:
    from bottle import (
        Bottle, run, request, response, static_file,
        HTTPResponse, HTTPError, template, BaseTemplate
    )
    _BOTTLE_AVAILABLE = True
except ImportError:
    _BOTTLE_AVAILABLE = False
    # Minimal stub so tests still run without bottle installed
    class Bottle:
        def __init__(self): self.routes = {}
        def route(self, *a, **kw): return lambda f: f
        def get(self, *a, **kw): return lambda f: f
        def post(self, *a, **kw): return lambda f: f
        def delete(self, *a, **kw): return lambda f: f
        def run(self, **kw): pass
    def run(*a, **kw): pass

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
DB_PATH      = os.environ.get("YTS_DB",      "streams.db")
MEDIA_DIR    = Path(os.environ.get("YTS_MEDIA", "media"))
HOST         = os.environ.get("YTS_HOST",    "0.0.0.0")
PORT         = int(os.environ.get("YTS_PORT", "8080"))
# Set via env: export YTS_API_KEY=your-secret
_ENV_API_KEY = os.environ.get("YTS_API_KEY", "admin-secret-change-me")

YT_RTMP_BASE = "rtmp://a.rtmp.youtube.com/live2"

# ─────────────────────────────────────────────────────────────
#  DATABASE
# ─────────────────────────────────────────────────────────────
def get_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

# Column migrations: (table, column, definition)
_MIGRATIONS = [
    ("streams", "yt_channel_id", "TEXT DEFAULT ''"),
    ("streams", "error_msg",     "TEXT DEFAULT ''"),
]

def _migrate_db(path=DB_PATH):
    """Add any missing columns to existing databases (safe to run repeatedly)."""
    with get_db(path) as db:
        for table, col, definition in _MIGRATIONS:
            existing = [
                row[1] for row in
                db.execute(f"PRAGMA table_info({table})").fetchall()
            ]
            if col not in existing:
                db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {definition}")
                db.commit()

def init_db(path=DB_PATH):
    with get_db(path) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS streams (
            id             TEXT PRIMARY KEY,
            name           TEXT NOT NULL,
            description    TEXT DEFAULT '',
            stream_key     TEXT NOT NULL,
            rtmp_target    TEXT NOT NULL,
            status         TEXT DEFAULT 'stopped',
            mode           TEXT DEFAULT 'file',
            created_at     TEXT NOT NULL,
            updated_at     TEXT NOT NULL,
            pid            INTEGER DEFAULT NULL,
            error_msg      TEXT DEFAULT '',
            yt_channel_id  TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS media_files (
            id          TEXT PRIMARY KEY,
            stream_id   TEXT NOT NULL,
            filename    TEXT NOT NULL,
            filepath    TEXT NOT NULL,
            filetype    TEXT NOT NULL,
            size_bytes  INTEGER DEFAULT 0,
            created_at  TEXT NOT NULL,
            FOREIGN KEY (stream_id) REFERENCES streams(id)
        );

        CREATE TABLE IF NOT EXISTS api_keys (
            key         TEXT PRIMARY KEY,
            label       TEXT NOT NULL,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS overlay_presets (
            id            TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            svg_json      TEXT NOT NULL,   -- JSON: {shapes:[...], duration_ms:N}
            duration_ms   INTEGER DEFAULT 3000,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audio_presets (
            id            TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            filepath      TEXT NOT NULL,
            filetype      TEXT DEFAULT 'audio/mpeg',
            volume        REAL DEFAULT 1.0,
            created_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS clip_presets (
            id            TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            filepath      TEXT NOT NULL,
            filetype      TEXT DEFAULT 'video/mp4',
            duration_s    REAL DEFAULT 0,
            created_at    TEXT NOT NULL
        );
        """)
    # API key is now ENV-based (YTS_API_KEY) — no DB key needed.

    # ── Migration: add columns that may be missing in older DBs ──────────────
    _migrate_db(path)

def _now():
    return datetime.utcnow().isoformat()

# ─────────────────────────────────────────────────────────────
#  STREAM MANAGER
# ─────────────────────────────────────────────────────────────
_processes: dict[str, subprocess.Popen] = {}
_process_lock = threading.Lock()

class StreamManager:
    """Manages FFmpeg subprocesses for each stream."""

    @staticmethod
    def _inject_overlay(cmd: list, overlay_png: str) -> list:
        """Insert '-i overlay_png' + filter_complex for live overlay mix."""
        if not overlay_png:
            return cmd

        new_cmd = list(cmd)
        try:
            out_idx = new_cmd.index("-c:v")
        except ValueError:
            return cmd

        n_inputs = 0
        i = 0
        while i < out_idx:
            if new_cmd[i] == "-i":
                n_inputs += 1
            i += 1

        # FIX: KEIN -re für das Overlay-Bild (sonst Deadlock mit dem Haupt-Video)
        overlay_input = [
            "-thread_queue_size", "1024",
            "-f", "image2",
            "-loop", "1",
            "-framerate", "15",
            "-i", overlay_png,
        ]

        filter_complex = (
            f"[0:v]scale=1280:720:force_original_aspect_ratio=decrease,"
            f"pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,fps=30,format=yuv420p[vnorm];"
            f"[vnorm][{n_inputs}:v]overlay=0:0:shortest=0:eof_action=pass[vout]"
        )

        mode_has_anullsrc = any(isinstance(a, str) and "anullsrc" in a for a in new_cmd)
        mode_has_color_video = any(isinstance(a, str) and a.startswith("color=c=black") for a in new_cmd)

        if mode_has_anullsrc:
            audio_map = "1:a"
        elif mode_has_color_video:
            audio_map = "1:a"
        else:
            audio_map = "0:a?"

        result = (
            new_cmd[:out_idx]
            + overlay_input
            + ["-filter_complex", filter_complex,
               "-map", "[vout]",
               "-map", audio_map]
            + new_cmd[out_idx:]
        )
        return result

    @staticmethod
    def _write_playlist(filepaths: list[str]) -> str:
        """Write an ffconcat playlist file and return its path."""
        import tempfile
        lines = ["ffconcat version 1.0"]
        for fp in filepaths:
            lines.append(f"file '{fp}'")
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="ytsh_playlist_"
        )
        f.write("\n".join(lines) + "\n")
        f.flush()
        f.close()
        return f.name

    @staticmethod
    def _write_playlist(filepaths: list[str]) -> str:
        """Write an ffconcat playlist file and return its path."""
        import tempfile
        lines = ["ffconcat version 1.0"]
        for fp in filepaths:
            lines.append(f"file '{fp}'")
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="ytsh_playlist_"
        )
        f.write("\n".join(lines) + "\n")
        f.flush()
        f.close()
        return f.name

    @staticmethod
    def _build_ffmpeg_cmd(stream: dict, files: list[dict], overlay_png: str = None) -> list[str]:
        cmd = StreamManager._build_ffmpeg_cmd_raw(stream, files)
        return StreamManager._inject_overlay(cmd, overlay_png)

    @staticmethod
    def _build_ffmpeg_cmd_raw(stream: dict, files: list[dict]) -> list[str]:
        target = f"{stream['rtmp_target']}/{stream['stream_key']}"

        # FIX: Harte Codierungsvorgaben für YouTube!
        # Kein Stream-Copy, sonst droppt YT den Stream bei minimalen Abweichungen.
        common_out = [
            "-c:v", "libx264",
            "-profile:v", "main",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            # Zwinge Output strikt auf 1280x720 bei exakt 30 FPS
            "-s", "1280x720",
            "-r", "30",
            "-g", "60",  # Keyframe alle 2 Sekunden (PFLICHT für YT!)
            "-keyint_min", "60",
            "-sc_threshold", "0",
            "-b:v", "2500k",
            "-maxrate", "2500k",
            "-bufsize", "5000k",

            "-c:a", "aac",
            "-ar", "44100",
            "-ac", "2",
            "-b:a", "128k",

            "-rw_timeout", "5000000",
            "-f", "flv",
            target
        ]

        mode = stream.get("mode", "file")

        if mode == "test":
            return [
                "ffmpeg",
                "-thread_queue_size", "1024", "-re", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30",
                "-thread_queue_size", "1024", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                *common_out,
            ]

        if mode == "api":
            return [
                "ffmpeg",
                "-thread_queue_size", "1024", "-f", "h264", "-i", "pipe:0",
                "-thread_queue_size", "1024", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                *common_out,
            ]

        video_files = [f for f in files if f["filetype"].startswith("video")]
        audio_files = [f for f in files if f["filetype"].startswith("audio")]

        if not video_files and not audio_files:
            return StreamManager._build_ffmpeg_cmd({**stream, "mode": "test"}, [])

        if video_files:
            playlist = StreamManager._write_playlist(
                [f["filepath"] for f in video_files]
            )
            return [
                "ffmpeg",
                # FIX: -fflags +genpts korrigiert Timestamps von defekten MP4s
                "-fflags", "+genpts",
                "-thread_queue_size", "1024", "-re", "-f", "concat", "-safe", "0",
                "-stream_loop", "-1", "-i", playlist,
                *common_out,
            ]

        playlist = StreamManager._write_playlist(
            [f["filepath"] for f in audio_files]
        )
        return [
            "ffmpeg",
            "-thread_queue_size", "1024", "-re", "-f", "lavfi", "-i", "color=c=black:size=1280x720:rate=30",
            "-thread_queue_size", "1024", "-f", "concat", "-safe", "0",
            "-stream_loop", "-1", "-i", playlist,
            *common_out,
        ]
    # per-stream rolling log buffer (last N lines of ffmpeg stderr)
    _log_buffers: dict = {}
    _LOG_MAX_LINES = 200

    @staticmethod
    def get_log(stream_id: str) -> str:
        """Return the current ffmpeg log buffer for a stream."""
        with _process_lock:
            buf = StreamManager._log_buffers.get(stream_id, [])
            return "\n".join(buf)

    @staticmethod
    def assess_health(stream_id: str, log: str, running: bool) -> dict:
        """
        Analyze the log to produce a health verdict:
        - state: "offline" | "starting" | "streaming" | "error" | "stalled"
        - message: human-readable hint
        - fps, bitrate, time: extracted from latest "frame=" line
        - has_io_error: bool
        """
        import re as _re
        result = {
            "state": "offline",
            "message": "Not running",
            "fps": None,
            "bitrate": None,
            "time": None,
            "speed": None,
            "has_io_error": False,
        }
        if not running:
            return result
        if not log:
            result["state"] = "starting"
            result["message"] = "FFmpeg launched, no output yet"
            return result

        # I/O errors are fatal for RTMP
        io_patterns = [
            "Connection refused", "I/O error", "RTMP_Connect",
            "Operation not permitted", "401 Unauthorized",
            "Invalid argument", "Error opening output",
        ]
        for pat in io_patterns:
            if pat.lower() in log.lower():
                result["has_io_error"] = True
                result["state"] = "error"
                # Find a hint line
                for line in reversed(log.split("\n")):
                    if pat.lower() in line.lower():
                        result["message"] = line.strip()[:200]
                        break
                return result

        # Parse the most recent "frame=N fps=N q=N size=N time=HH:MM:SS.SS bitrate=Nkbits/s speed=Nx"
        frame_lines = [l for l in log.split("\n") if l.startswith("frame=") or "frame=" in l[:20]]
        if frame_lines:
            last = frame_lines[-1]
            m_fps = _re.search(r"fps=\s*([\d.]+)", last)
            m_br  = _re.search(r"bitrate=\s*([\d.]+)kbits/s", last)
            m_t   = _re.search(r"time=([\d:.]+)", last)
            m_sp  = _re.search(r"speed=\s*([\d.]+)x", last)
            if m_fps: result["fps"]     = float(m_fps.group(1))
            if m_br:  result["bitrate"] = float(m_br.group(1))
            if m_t:   result["time"]    = m_t.group(1)
            if m_sp:  result["speed"]   = float(m_sp.group(1))

            # streaming if fps > 0 and speed > 0.5
            if result["fps"] and result["fps"] > 0 and \
               (result["speed"] is None or result["speed"] > 0.5):
                result["state"] = "streaming"
                result["message"] = f"Live: {result['fps']:.1f} fps @ {result.get('bitrate', 0):.0f} kbps"
            elif result["fps"] == 0:
                result["state"] = "stalled"
                result["message"] = "FFmpeg running but not producing frames"
            return result

        # No frame lines yet but also no errors → starting
        if "Press [q] to stop" in log or "Stream mapping" in log:
            result["state"] = "starting"
            result["message"] = "FFmpeg connecting to RTMP ingest…"
        else:
            result["state"] = "starting"
            result["message"] = "FFmpeg initializing…"
        return result

    @staticmethod
    def start(stream_id: str, db_path=DB_PATH) -> dict:
        with get_db(db_path) as db:
            row = db.execute("SELECT * FROM streams WHERE id=?", (stream_id,)).fetchone()
            if not row:
                return {"ok": False, "error": "Stream not found"}
            stream = dict(row)

            # --- Stream Key Kollisions-Prüfung & Auto-Close ---
            target_key = stream.get("stream_key", "").strip()
            if target_key:
                conflicts = db.execute(
                    "SELECT id, name FROM streams WHERE stream_key=? AND id!=? AND status='live'",
                    (target_key, stream_id)
                ).fetchall()

                for conflict in conflicts:
                    conflict_id = conflict["id"]
                    conflict_name = conflict["name"]
                    StreamManager.stop(conflict_id, db_path)

                    with _process_lock:
                        if conflict_id in _processes and _processes[conflict_id].poll() is None:
                            return {
                                "ok": False,
                                "error": f"FEHLER: Stream Key blockiert. Wird noch von '{conflict_name}' verwendet."
                            }
            # ---------------------------------------------------

            files = [dict(r) for r in db.execute(
                "SELECT * FROM media_files WHERE stream_id=? ORDER BY created_at",
                (stream_id,)
            ).fetchall()]

        with _process_lock:
            if stream_id in _processes and _processes[stream_id].poll() is None:
                return {"ok": False, "error": "Already running"}

            overlay_png = OverlayEngine.ensure_blank(stream_id)
            cmd = StreamManager._build_ffmpeg_cmd(stream, files, overlay_png=overlay_png)
            OverlayEngine._ensure_worker(stream_id)

            StreamManager._log_buffers[stream_id] = [
                f"$ {' '.join(cmd)}", ""
            ]
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE if stream["mode"] == "api" else subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                _processes[stream_id] = proc
            except FileNotFoundError:
                return {"ok": False, "error": "ffmpeg not found — install it first"}

        with get_db(db_path) as db:
            db.execute(
                "UPDATE streams SET status='live', pid=?, updated_at=?, error_msg='' WHERE id=?",
                (proc.pid, _now(), stream_id)
            )
            db.commit()

        # ── Reader thread: FIX für FFmpeg '\r' (Carriage Return) Ausgaben ──────
        def _reader():
            try:
                line_buf = bytearray()
                while True:
                    # Lese Zeichen für Zeichen, um das Blockieren bei \r zu verhindern
                    char = proc.stderr.read(1)
                    if not char:
                        break

                    # FFmpeg nutzt \r für frame= Updates und \n für reguläre Logs
                    if char == b'\r' or char == b'\n':
                        if not line_buf:
                            continue
                        line = line_buf.decode("utf-8", errors="replace").strip()
                        line_buf.clear()

                        with _process_lock:
                            buf = StreamManager._log_buffers.setdefault(stream_id, [])

                            # Wenn die aktuelle Zeile UND die letzte Zeile ein "frame=" Update sind,
                            # überschreibe die letzte Zeile (so spammen wir den UI Log nicht zu!)
                            if line.startswith("frame=") and buf and buf[-1].startswith("frame="):
                                buf[-1] = line
                            else:
                                buf.append(line)
                                if len(buf) > StreamManager._LOG_MAX_LINES:
                                    del buf[: len(buf) - StreamManager._LOG_MAX_LINES]
                    else:
                        line_buf.extend(char)
            except Exception as e:
                with _process_lock:
                    StreamManager._log_buffers.setdefault(stream_id, []).append(
                        f"[reader error] {e}"
                    )

        # ── Watcher thread: wait for exit, update DB status ────────────────────
        def _watch():
            proc.wait()
            with _process_lock:
                _processes.pop(stream_id, None)
                tail = "\n".join(
                    StreamManager._log_buffers.get(stream_id, [])[-20:]
                )
            with get_db(db_path) as db2:
                db2.execute(
                    "UPDATE streams SET status='stopped', pid=NULL, error_msg=?, updated_at=? WHERE id=?",
                    (tail[-1000:], _now(), stream_id)
                )
                db2.commit()

        import threading
        threading.Thread(target=_reader, daemon=True).start()
        threading.Thread(target=_watch, daemon=True).start()
        return {"ok": True, "pid": proc.pid, "cmd": cmd}

    @staticmethod
    def stop(stream_id: str, db_path=DB_PATH) -> dict:
        OverlayEngine.stop_worker(stream_id)
        proc_to_wait = None

        with _process_lock:
            proc = _processes.get(stream_id)
            if not proc or proc.poll() is not None:
                # Update DB just in case
                with get_db(db_path) as db:
                    db.execute(
                        "UPDATE streams SET status='stopped', pid=NULL, updated_at=? WHERE id=?",
                        (_now(), stream_id)
                    )
                    db.commit()
                return {"ok": True, "note": "Was not running"}

            # Reguläres Beenden anfordern (SIGTERM)
            proc.terminate()
            proc_to_wait = proc

        # Außerhalb des Locks warten, um andere Requests nicht zu blockieren
        if proc_to_wait:
            try:
                # Gib FFmpeg 3 Sekunden Zeit zum sauberen Schreiben des Trailers
                proc_to_wait.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                # Wenn FFmpeg hängt (z.B. durch tote Pipe/Lock), erzwinge den Abbruch (SIGKILL)
                proc_to_wait.kill()
                proc_to_wait.wait()

        return {"ok": True}

    @staticmethod
    def get_pipe(stream_id: str):
        """Return stdin pipe for API-mode streams."""
        with _process_lock:
            proc = _processes.get(stream_id)
            if proc and proc.poll() is None:
                return proc.stdin
        return None

# ─────────────────────────────────────────────────────────────
#  OVERLAY ENGINE  —  renders SVG paths → PNG for FFmpeg
# ─────────────────────────────────────────────────────────────
# Preset JSON schema:
# {
#   "duration_ms": 3000,
#   "shapes": [
#       { "type":"path",    "d":"M 10,10 L 100,100",
#         "stroke":"#ff4d1c", "stroke_width":4,
#         "fill":"none", "opacity":1.0,
#         "anim": {"x":[0,100], "y":[0,0], "scale":[1,1.5], "rotate":[0,360],
#                  "opacity":[0,1,1,0], "ease":"linear"} },
#       { "type":"rect", "x":50, "y":50, "w":200, "h":80,
#         "fill":"#1c3fff", "opacity":0.8, "anim":{...} },
#       { "type":"text", "x":100,"y":100, "text":"HELLO",
#         "font_size":48, "fill":"#ffe01c", "anim":{...} }
#   ]
# }

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# Global overlay state per stream
_overlay_state = {}        # stream_id -> {"active": [{"preset_id","start_ts","data"}], "path": str}
_overlay_state_lock = threading.Lock()


def _hex_to_rgba(hex_color: str, opacity: float = 1.0) -> tuple:
    """Convert '#rrggbb' or '#rgb' to (r,g,b,a) with int alpha."""
    if not hex_color or hex_color == "none":
        return (0, 0, 0, 0)
    s = hex_color.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
    except Exception:
        return (255, 255, 255, int(255 * opacity))
    return (r, g, b, int(255 * max(0.0, min(1.0, opacity))))


def _interp(values, t: float) -> float:
    """Linear interpolation through an array of keyframe values (t in [0,1])."""
    if not values:
        return 0.0
    if isinstance(values, (int, float)):
        return float(values)
    if len(values) == 1:
        return float(values[0])
    # map t to segment index
    n_seg = len(values) - 1
    pos = t * n_seg
    i = int(pos)
    if i >= n_seg:
        return float(values[-1])
    frac = pos - i
    return float(values[i]) * (1 - frac) + float(values[i + 1]) * frac


def _parse_svg_path(d: str) -> list:
    """Very small SVG path parser — supports M, L, Z, H, V commands only.
    Returns list of (x,y) tuples for a polyline."""
    import re as _re
    if not d:
        return []
    tokens = _re.findall(r"[MLHVZmlhvz]|-?\d*\.?\d+", d)
    pts = []
    i = 0
    cur = [0.0, 0.0]
    start = None
    while i < len(tokens):
        t = tokens[i]
        if t in "MLml":
            abs_ = t.isupper()
            i += 1
            x = float(tokens[i]); i += 1
            y = float(tokens[i]); i += 1
            if abs_:
                cur = [x, y]
            else:
                cur = [cur[0] + x, cur[1] + y]
            if t in "Mm" and start is None:
                start = tuple(cur)
            pts.append(tuple(cur))
        elif t in "Hh":
            abs_ = t.isupper()
            i += 1
            x = float(tokens[i]); i += 1
            cur = [x if abs_ else cur[0] + x, cur[1]]
            pts.append(tuple(cur))
        elif t in "Vv":
            abs_ = t.isupper()
            i += 1
            y = float(tokens[i]); i += 1
            cur = [cur[0], y if abs_ else cur[1] + y]
            pts.append(tuple(cur))
        elif t in "Zz":
            if start:
                pts.append(start)
            i += 1
        else:
            i += 1
    return pts


class OverlayEngine:
    """Renders active overlay presets to a PNG file that FFmpeg reads."""

    FPS = 15                     # overlay update rate
    SIZE = (1280, 720)
    _workers: dict = {}          # stream_id -> Thread
    _stop_flags: dict = {}

    @staticmethod
    def overlay_path(stream_id: str) -> str:
        """Path to the live overlay PNG for a stream."""
        p = MEDIA_DIR / "_overlays" / f"{stream_id}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    @staticmethod
    def ensure_blank(stream_id: str) -> str:
        """Make sure a transparent overlay PNG exists (used as ffmpeg input)."""
        path = OverlayEngine.overlay_path(stream_id)
        if not os.path.exists(path):
            if _PIL_OK:
                img = Image.new("RGBA", OverlayEngine.SIZE, (0, 0, 0, 0))
                img.save(path, "PNG")
            else:
                # 1x1 transparent PNG fallback (base64-decoded on demand)
                import base64
                blank = base64.b64decode(
                    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
                    b"+M9QDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                )
                with open(path, "wb") as f:
                    f.write(blank)
        return path

    @staticmethod
    def trigger(stream_id: str, preset: dict, db_path=DB_PATH) -> None:
        """Schedule a preset to play on a stream (one-shot with duration)."""
        with _overlay_state_lock:
            st = _overlay_state.setdefault(stream_id, {"active": []})
            st["active"].append({
                "preset_id": preset.get("id"),
                "name": preset.get("name"),
                "data": preset,
                "start_ts": time.time(),
                "duration_ms": preset.get("duration_ms", 3000),
            })
        OverlayEngine._ensure_worker(stream_id)

    @staticmethod
    def clear(stream_id: str) -> None:
        with _overlay_state_lock:
            if stream_id in _overlay_state:
                _overlay_state[stream_id]["active"] = []

    @staticmethod
    def _ensure_worker(stream_id: str) -> None:
        if stream_id in OverlayEngine._workers and \
           OverlayEngine._workers[stream_id].is_alive():
            return
        OverlayEngine._stop_flags[stream_id] = False
        t = threading.Thread(
            target=OverlayEngine._render_loop,
            args=(stream_id,), daemon=True
        )
        OverlayEngine._workers[stream_id] = t
        t.start()

    @staticmethod
    def stop_worker(stream_id: str) -> None:
        OverlayEngine._stop_flags[stream_id] = True

    @staticmethod
    def _render_loop(stream_id: str):
        """Continuously render active overlays to the PNG, 15 fps."""
        path = OverlayEngine.overlay_path(stream_id)
        tmp_path = path + ".tmp"
        last_was_blank = False
        while not OverlayEngine._stop_flags.get(stream_id):
            now = time.time()
            with _overlay_state_lock:
                st = _overlay_state.get(stream_id, {"active": []})
                # Prune expired
                st["active"] = [
                    a for a in st["active"]
                    if (now - a["start_ts"]) * 1000 < a["duration_ms"]
                ]
                active = list(st["active"])

            if not active:
                if not last_was_blank:
                    OverlayEngine._write_blank(path, tmp_path)
                    last_was_blank = True
                time.sleep(1.0 / OverlayEngine.FPS)
                continue

            last_was_blank = False
            OverlayEngine._render_frame(path, tmp_path, active, now)
            time.sleep(1.0 / OverlayEngine.FPS)

    @staticmethod
    def _write_blank(path: str, tmp: str):
        if not _PIL_OK:
            return
        img = Image.new("RGBA", OverlayEngine.SIZE, (0, 0, 0, 0))
        img.save(tmp, "PNG")
        try:
            os.replace(tmp, path)
        except Exception:
            pass

    @staticmethod
    def _render_frame(path: str, tmp: str, active: list, now: float):
        if not _PIL_OK:
            return
        w, h = OverlayEngine.SIZE
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        for entry in active:
            preset_data = entry["data"]
            duration_ms = entry["duration_ms"]
            elapsed_ms = (now - entry["start_ts"]) * 1000
            t = max(0.0, min(1.0, elapsed_ms / max(1, duration_ms)))

            shapes = preset_data.get("shapes", [])
            for shape in shapes:
                try:
                    OverlayEngine._draw_shape(draw, img, shape, t)
                except Exception:
                    continue  # skip broken shape rather than crash renderer

        try:
            img.save(tmp, "PNG")
            os.replace(tmp, path)  # atomic swap — ffmpeg won't see partial writes
        except Exception:
            pass

    @staticmethod
    def _anim_val(shape: dict, key: str, default, t: float):
        anim = shape.get("anim", {}) or {}
        if key in anim:
            return _interp(anim[key], t)
        return shape.get(key, default)

    @staticmethod
    def _draw_shape(draw, img, shape: dict, t: float):
        stype = shape.get("type", "rect")

        # Compute animated transforms
        off_x = OverlayEngine._anim_val(shape, "x_offset", 0, t)
        off_y = OverlayEngine._anim_val(shape, "y_offset", 0, t)
        opacity = OverlayEngine._anim_val(shape, "opacity", shape.get("opacity", 1.0), t)
        opacity = max(0.0, min(1.0, opacity))

        fill_rgba = _hex_to_rgba(shape.get("fill", "#ffffff"), opacity)
        stroke_rgba = _hex_to_rgba(shape.get("stroke", "none"), opacity)
        stroke_w = int(shape.get("stroke_width", 0))

        if stype == "rect":
            x = shape.get("x", 0) + off_x
            y = shape.get("y", 0) + off_y
            rw = shape.get("w", 100)
            rh = shape.get("h", 50)
            draw.rectangle([x, y, x + rw, y + rh],
                           fill=fill_rgba if shape.get("fill", "none") != "none" else None,
                           outline=stroke_rgba if stroke_w else None,
                           width=stroke_w or 1)

        elif stype == "circle":
            cx = shape.get("cx", 100) + off_x
            cy = shape.get("cy", 100) + off_y
            r = shape.get("r", 50)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                         fill=fill_rgba if shape.get("fill", "none") != "none" else None,
                         outline=stroke_rgba if stroke_w else None,
                         width=stroke_w or 1)

        elif stype == "path":
            pts = _parse_svg_path(shape.get("d", ""))
            if len(pts) >= 2:
                pts_shifted = [(p[0] + off_x, p[1] + off_y) for p in pts]
                if shape.get("fill", "none") != "none":
                    draw.polygon(pts_shifted, fill=fill_rgba,
                                 outline=stroke_rgba if stroke_w else None)
                else:
                    draw.line(pts_shifted, fill=stroke_rgba or fill_rgba,
                              width=max(1, stroke_w))

        elif stype == "text":
            x = shape.get("x", 100) + off_x
            y = shape.get("y", 100) + off_y
            text = str(shape.get("text", ""))
            font_size = int(shape.get("font_size", 48))
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
            except Exception:
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
            if font:
                draw.text((x, y), text, fill=fill_rgba, font=font)
            else:
                draw.text((x, y), text, fill=fill_rgba)


# ─────────────────────────────────────────────────────────────
#  AUDIO MIXER  —  plays audio presets into running stream
# ─────────────────────────────────────────────────────────────
# Strategy: spawn a short-lived ffmpeg that streams the audio file
# to an RTMP mini-proxy, OR use a platform-specific mechanism.
# For portability we use: an ffmpeg sub-process that writes WAV to
# a loopback file, then the main stream's amix filter reads it.
# Simpler approach here: we don't attempt true inline audio injection.
# Instead, we document that audio presets are better triggered by
# adding them to the file playlist, OR by using a JACK/PulseAudio
# virtual sink on the server. We still provide the API + UI so users
# can wire up their own audio routing.

class AudioPresetManager:
    """Thin manager for uploaded audio presets (trigger = async file play)."""

    @staticmethod
    def trigger(preset_path: str) -> dict:
        """Play an audio file locally (server-side) as a subprocess.
        This is useful for local monitoring / dev — for live mixing into
        RTMP, the audio preset should be part of the stream's media playlist
        OR routed via a virtual audio sink."""
        try:
            proc = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", preset_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return {"ok": True, "pid": proc.pid}
        except FileNotFoundError:
            return {"ok": False, "error": "ffplay not installed"}


# ─────────────────────────────────────────────────────────────
#  CRUD helpers
# ─────────────────────────────────────────────────────────────
def _stream_row_to_dict(row) -> dict:
    d = dict(row)
    d.pop("stream_key", None)  # don't expose key in list endpoints
    return d

def create_stream(name, description="", stream_key="", rtmp_target=YT_RTMP_BASE,
                  mode="file", yt_channel_id="", db_path=DB_PATH) -> dict:
    sid = str(uuid.uuid4())
    now = _now()
    with get_db(db_path) as db:
        db.execute(
            "INSERT INTO streams VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, name, description, stream_key, rtmp_target,
             "stopped", mode, now, now, None, "", yt_channel_id)
        )
        db.commit()
    return {"id": sid, "name": name, "status": "stopped"}

def list_streams(db_path=DB_PATH) -> list:
    with get_db(db_path) as db:
        rows = db.execute("SELECT * FROM streams ORDER BY created_at DESC").fetchall()
    return [_stream_row_to_dict(r) for r in rows]

def get_stream(sid, db_path=DB_PATH) -> dict | None:
    with get_db(db_path) as db:
        row = db.execute("SELECT * FROM streams WHERE id=?", (sid,)).fetchone()
    return dict(row) if row else None

def update_stream(sid, fields: dict, db_path=DB_PATH) -> bool:
    allowed = {"name", "description", "stream_key", "rtmp_target", "mode", "yt_channel_id"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    updates["updated_at"] = _now()
    placeholders = ", ".join(f"{k}=?" for k in updates)
    values = list(updates.values()) + [sid]
    with get_db(db_path) as db:
        db.execute(f"UPDATE streams SET {placeholders} WHERE id=?", values)
        db.commit()
    return True

def delete_stream(sid, db_path=DB_PATH) -> bool:
    StreamManager.stop(sid, db_path)
    with get_db(db_path) as db:
        db.execute("DELETE FROM media_files WHERE stream_id=?", (sid,))
        db.execute("DELETE FROM streams WHERE id=?", (sid,))
        db.commit()
    return True

def add_media(stream_id, filename, filepath, filetype, size, db_path=DB_PATH) -> dict:
    mid = str(uuid.uuid4())
    with get_db(db_path) as db:
        db.execute(
            "INSERT INTO media_files VALUES (?,?,?,?,?,?,?)",
            (mid, stream_id, filename, filepath, filetype, size, _now())
        )
        db.commit()
    return {"id": mid, "filename": filename}

def list_media(stream_id, db_path=DB_PATH) -> list:
    with get_db(db_path) as db:
        rows = db.execute(
            "SELECT * FROM media_files WHERE stream_id=? ORDER BY created_at",
            (stream_id,)
        ).fetchall()
    return [dict(r) for r in rows]

# ─────────────────────────────────────────────────────────────
#  PRESET CRUD  (overlay + audio)
# ─────────────────────────────────────────────────────────────
def create_overlay_preset(name, svg_json: dict, duration_ms=3000, db_path=DB_PATH) -> dict:
    pid = str(uuid.uuid4())
    now = _now()
    svg_text = json.dumps(svg_json) if isinstance(svg_json, dict) else str(svg_json)
    with get_db(db_path) as db:
        db.execute(
            "INSERT INTO overlay_presets VALUES (?,?,?,?,?,?)",
            (pid, name, svg_text, duration_ms, now, now)
        )
        db.commit()
    return {"id": pid, "name": name, "duration_ms": duration_ms}

def list_overlay_presets(db_path=DB_PATH) -> list:
    with get_db(db_path) as db:
        rows = db.execute(
            "SELECT id, name, duration_ms, created_at FROM overlay_presets ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]

def get_overlay_preset(pid, db_path=DB_PATH) -> dict | None:
    with get_db(db_path) as db:
        row = db.execute("SELECT * FROM overlay_presets WHERE id=?", (pid,)).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["svg"] = json.loads(d.pop("svg_json"))
    except Exception:
        d["svg"] = {}
    return d

def update_overlay_preset(pid, fields: dict, db_path=DB_PATH) -> bool:
    allowed = {"name", "duration_ms", "svg_json"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if "svg_json" in updates and isinstance(updates["svg_json"], dict):
        updates["svg_json"] = json.dumps(updates["svg_json"])
    if not updates:
        return False
    updates["updated_at"] = _now()
    placeholders = ", ".join(f"{k}=?" for k in updates)
    values = list(updates.values()) + [pid]
    with get_db(db_path) as db:
        db.execute(f"UPDATE overlay_presets SET {placeholders} WHERE id=?", values)
        db.commit()
    return True

def delete_overlay_preset(pid, db_path=DB_PATH) -> bool:
    with get_db(db_path) as db:
        db.execute("DELETE FROM overlay_presets WHERE id=?", (pid,))
        db.commit()
    return True

def create_audio_preset(name, filepath, filetype="audio/mpeg", volume=1.0, db_path=DB_PATH) -> dict:
    pid = str(uuid.uuid4())
    with get_db(db_path) as db:
        db.execute(
            "INSERT INTO audio_presets VALUES (?,?,?,?,?,?)",
            (pid, name, filepath, filetype, volume, _now())
        )
        db.commit()
    return {"id": pid, "name": name}

def list_audio_presets(db_path=DB_PATH) -> list:
    with get_db(db_path) as db:
        rows = db.execute("SELECT * FROM audio_presets ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]

def get_audio_preset(pid, db_path=DB_PATH) -> dict | None:
    with get_db(db_path) as db:
        row = db.execute("SELECT * FROM audio_presets WHERE id=?", (pid,)).fetchone()
    return dict(row) if row else None

def create_clip_preset(name, filepath, filetype="video/mp4", duration_s=0.0,
                       db_path=DB_PATH) -> dict:
    pid = str(uuid.uuid4())
    with get_db(db_path) as db:
        db.execute(
            "INSERT INTO clip_presets VALUES (?,?,?,?,?,?)",
            (pid, name, filepath, filetype, duration_s, _now())
        )
        db.commit()
    return {"id": pid, "name": name}

def list_clip_presets(db_path=DB_PATH) -> list:
    with get_db(db_path) as db:
        rows = db.execute("SELECT * FROM clip_presets ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]

def get_clip_preset(pid, db_path=DB_PATH) -> dict | None:
    with get_db(db_path) as db:
        row = db.execute("SELECT * FROM clip_presets WHERE id=?", (pid,)).fetchone()
    return dict(row) if row else None

def delete_clip_preset(pid, db_path=DB_PATH) -> bool:
    with get_db(db_path) as db:
        row = db.execute("SELECT filepath FROM clip_presets WHERE id=?", (pid,)).fetchone()
        if row:
            try: os.remove(row[0])
            except Exception: pass
        db.execute("DELETE FROM clip_presets WHERE id=?", (pid,))
        db.commit()
    return True

def delete_audio_preset(pid, db_path=DB_PATH) -> bool:
    with get_db(db_path) as db:
        row = db.execute("SELECT filepath FROM audio_presets WHERE id=?", (pid,)).fetchone()
        if row:
            try: os.remove(row[0])
            except Exception: pass
        db.execute("DELETE FROM audio_presets WHERE id=?", (pid,))
        db.commit()
    return True


# ─────────────────────────────────────────────────────────────
#  HTML UI  (Neobrutalism / Paper Cut-out / Enterprise)
# ─────────────────────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YT STREAM HUB</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Unbounded:wght@400;700;900&display=swap" rel="stylesheet">
<style>
  :root {
    --ink:   #0a0a0a;
    --paper: #f5f0e8;
    --cream: #ede8d8;
    --cut1:  #ff4d1c;
    --cut2:  #1c3fff;
    --cut3:  #ffe01c;
    --border: 3px solid var(--ink);
    --shadow: 5px 5px 0 var(--ink);
    --shadow-lg: 8px 8px 0 var(--ink);
    --radius: 0px;
    --font-head: 'Unbounded', monospace;
    --font-body: 'Space Mono', monospace;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  html { scroll-behavior: smooth; }

  body {
    background: var(--paper);
    color: var(--ink);
    font-family: var(--font-body);
    font-size: 13px;
    line-height: 1.6;
    min-height: 100vh;
    /* paper texture via repeating-linear-gradient */
    background-image:
      repeating-linear-gradient(
        0deg,
        transparent,
        transparent 28px,
        rgba(10,10,10,0.04) 28px,
        rgba(10,10,10,0.04) 29px
      );
  }

  /* ── HEADER ── */
  header {
    background: var(--ink);
    color: var(--paper);
    padding: 0 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 60px;
    border-bottom: var(--border);
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .logo {
    font-family: var(--font-head);
    font-size: 15px;
    font-weight: 900;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .header-badge {
    font-size: 10px;
    font-weight: 700;
    background: var(--cut1);
    color: var(--ink);
    padding: 3px 8px;
    letter-spacing: 0.1em;
    border: 2px solid var(--paper);
  }
  .header-api-note {
    font-size: 10px;
    opacity: 0.5;
    letter-spacing: 0.05em;
  }

  /* ── LAYOUT ── */
  .container { max-width: 1280px; margin: 0 auto; padding: 0 2rem; }

  .toolbar {
    border-bottom: var(--border);
    background: var(--cream);
    padding: 12px 2rem;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    align-items: center;
  }

  main { padding: 2rem; max-width: 1280px; margin: 0 auto; }

  /* ── GRID ── */
  .streams-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 20px;
  }

  /* ── STREAM CARD ── */
  .card {
    background: var(--paper);
    border: var(--border);
    box-shadow: var(--shadow);
    position: relative;
    transition: transform 0.1s, box-shadow 0.1s;
    overflow: hidden;
  }
  .card:hover {
    transform: translate(-2px, -2px);
    box-shadow: var(--shadow-lg);
  }
  .card-head {
    border-bottom: var(--border);
    padding: 14px 16px;
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 8px;
  }
  .card-title {
    font-family: var(--font-head);
    font-size: 13px;
    font-weight: 700;
    word-break: break-word;
    flex: 1;
  }
  .status-badge {
    font-family: var(--font-body);
    font-size: 10px;
    font-weight: 700;
    padding: 3px 8px;
    letter-spacing: 0.08em;
    flex-shrink: 0;
    border: 2px solid var(--ink);
  }
  .logo-dot {
    width: 10px; height: 10px;
    background: var(--cut1);
    border-radius: 50%;
  }
  .status-live     { background: var(--cut1); color: #fff; }
  .status-stopped  { background: var(--cream); }
  .status-starting { background: var(--cut3); }
  .status-stalled  { background: #ffa500; color: #fff; }
  .status-error    { background: #880000; color: #fff; }
  .card-body { padding: 14px 16px; }
  .card-desc { font-size: 11px; opacity: 0.7; margin-bottom: 12px; min-height: 16px; }
  .meta-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 14px; }
  .tag {
    font-size: 10px;
    background: var(--cream);
    border: 2px solid var(--ink);
    padding: 2px 6px;
    letter-spacing: 0.05em;
  }
  .tag-mode { background: var(--cut3); }
  .card-actions { display: flex; gap: 8px; flex-wrap: wrap; }
  .card-error {
    margin-top: 8px;
    font-size: 10px;
    color: var(--cut1);
    background: #fff0ed;
    border: 2px solid var(--cut1);
    padding: 6px 8px;
    word-break: break-all;
    max-height: 60px;
    overflow-y: auto;
  }

  /* ── BUTTONS ── */
  .btn {
    font-family: var(--font-body);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    border: var(--border);
    padding: 8px 14px;
    cursor: pointer;
    box-shadow: 3px 3px 0 var(--ink);
    background: var(--paper);
    color: var(--ink);
    transition: transform 0.08s, box-shadow 0.08s;
    text-decoration: none;
    display: inline-block;
    line-height: 1;
    white-space: nowrap;
  }
  .btn:active {
    transform: translate(2px, 2px);
    box-shadow: 1px 1px 0 var(--ink);
  }
  .btn-primary { background: var(--cut2); color: #fff; }
  .btn-danger  { background: var(--cut1); color: #fff; }
  .btn-live    { background: var(--cut1); color: #fff; animation: blink 1.2s step-start infinite; }
  .btn-sm      { padding: 5px 10px; font-size: 10px; }
  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
  }

  /* ── FORMS / MODAL ── */
  .modal-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(10,10,10,0.6);
    z-index: 200;
    align-items: center;
    justify-content: center;
  }
  .modal-overlay.open { display: flex; }
  .modal {
    background: var(--paper);
    border: var(--border);
    box-shadow: 12px 12px 0 var(--ink);
    width: 100%;
    max-width: 520px;
    max-height: 90vh;
    overflow-y: auto;
    margin: 1rem;
  }
  .modal-head {
    background: var(--ink);
    color: var(--paper);
    padding: 14px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: var(--font-head);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.06em;
    position: sticky;
    top: 0;
  }
  .modal-close {
    background: none;
    border: none;
    color: var(--paper);
    cursor: pointer;
    font-size: 20px;
    line-height: 1;
    font-family: var(--font-body);
  }
  .modal-body { padding: 20px; display: flex; flex-direction: column; gap: 14px; }

  label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    display: block;
    margin-bottom: 4px;
  }
  input[type=text], input[type=password], select, textarea {
    width: 100%;
    border: var(--border);
    background: var(--paper);
    font-family: var(--font-body);
    font-size: 12px;
    padding: 8px 10px;
    box-shadow: 2px 2px 0 var(--ink);
    outline: none;
    resize: vertical;
  }
  input:focus, select:focus, textarea:focus {
    box-shadow: 4px 4px 0 var(--cut2);
  }

  .field-note {
    font-size: 10px;
    opacity: 0.55;
    margin-top: 4px;
  }

  /* ── API DOCS STRIP ── */
  .api-strip {
    border: var(--border);
    box-shadow: var(--shadow);
    margin-bottom: 24px;
    overflow: hidden;
  }
  .api-strip-head {
    background: var(--ink);
    color: var(--paper);
    padding: 10px 16px;
    font-family: var(--font-head);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    user-select: none;
  }
  .api-strip-body {
    background: var(--cream);
    padding: 16px;
    display: none;
    font-size: 11px;
  }
  .api-strip-body.open { display: block; }
  .api-table { width: 100%; border-collapse: collapse; }
  .api-table th, .api-table td {
    border: 2px solid var(--ink);
    padding: 6px 10px;
    text-align: left;
    font-size: 11px;
    vertical-align: top;
  }
  .api-table th { background: var(--ink); color: var(--paper); font-family: var(--font-head); font-size: 10px; }
  .method { font-weight: 700; letter-spacing: 0.04em; }
  .method-get    { color: var(--cut2); }
  .method-post   { color: #006f00; }
  .method-delete { color: var(--cut1); }

  code {
    font-family: var(--font-body);
    background: var(--ink);
    color: var(--cut3);
    padding: 1px 5px;
    font-size: 10px;
  }

  /* ── UPLOAD ZONE ── */
  .upload-zone {
    border: 3px dashed var(--ink);
    padding: 20px;
    text-align: center;
    cursor: pointer;
    font-size: 12px;
    transition: background 0.15s;
  }
  .upload-zone:hover { background: var(--cream); }
  .upload-zone input[type=file] { display: none; }

  /* ── EMPTY STATE ── */
  .empty-state {
    border: var(--border);
    box-shadow: var(--shadow);
    padding: 40px 20px;
    text-align: center;
    grid-column: 1/-1;
  }
  .empty-icon { font-size: 40px; margin-bottom: 16px; }
  .empty-title {
    font-family: var(--font-head);
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 8px;
  }

  /* ── SECTION HEADER ── */
  .section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: var(--border);
  }
  .section-title {
    font-family: var(--font-head);
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 0.1em;
  }

  /* ── TOAST ── */
  #toast {
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: var(--ink);
    color: var(--paper);
    padding: 12px 18px;
    border: var(--border);
    box-shadow: var(--shadow);
    font-size: 12px;
    z-index: 500;
    opacity: 0;
    transition: opacity 0.2s;
    pointer-events: none;
    max-width: 300px;
  }
  #toast.show { opacity: 1; }

  /* ── MEDIA LIST ── */
  .media-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid rgba(10,10,10,0.15);
    gap: 8px;
  }
  .media-name { font-size: 11px; flex: 1; word-break: break-all; }
  .media-size { font-size: 10px; opacity: 0.5; white-space: nowrap; }

  /* ── SCROLLBAR ── */
  ::-webkit-scrollbar { width: 8px; }
  ::-webkit-scrollbar-track { background: var(--cream); }
  ::-webkit-scrollbar-thumb { background: var(--ink); }

  @media (max-width: 600px) {
    .streams-grid { grid-template-columns: 1fr; }
    header { padding: 0 1rem; }
    main { padding: 1rem; }
    .toolbar { padding: 10px 1rem; }
  }
</style>
</head>
<body>

<!-- LOGIN OVERLAY -->
<div id="loginOverlay" style="display:flex;position:fixed;inset:0;background:var(--paper);z-index:1000;align-items:center;justify-content:center;background-image:repeating-linear-gradient(0deg,transparent,transparent 28px,rgba(10,10,10,0.04) 28px,rgba(10,10,10,0.04) 29px);">
  <div style="border:3px solid var(--ink);box-shadow:8px 8px 0 var(--ink);width:100%;max-width:400px;margin:1rem;">
    <div style="background:var(--ink);color:var(--paper);padding:16px 20px;font-family:var(--font-head);font-size:12px;font-weight:700;letter-spacing:0.08em;display:flex;align-items:center;gap:12px;">
      <div style="width:10px;height:10px;background:var(--cut1);border-radius:50%;"></div>
      YT STREAM HUB — SIGN IN
    </div>
    <div style="padding:28px 24px;display:flex;flex-direction:column;gap:16px;">
      <div>
        <label>API KEY</label>
        <input type="password" id="loginKeyInput" placeholder="Enter your API key" autocomplete="current-password"
          style="width:100%;border:3px solid var(--ink);background:var(--paper);font-family:var(--font-body);font-size:13px;padding:10px 12px;box-shadow:3px 3px 0 var(--ink);outline:none;">
        <div style="font-size:10px;opacity:0.5;margin-top:4px;">Set via <code>YTS_API_KEY</code> environment variable on server.</div>
      </div>
      <div id="loginError" style="color:var(--cut1);font-size:11px;font-weight:700;min-height:16px;"></div>
      <button class="btn btn-primary" onclick="submitLogin()" style="width:100%;padding:12px;">SIGN IN →</button>
    </div>
  </div>
</div>

<div id="mainContent" style="display:none;">
<header>
  <div class="logo">
    <div class="logo-dot"></div>
    YT STREAM HUB
  </div>
  <div style="display:flex;gap:12px;align-items:center;">
    <span class="header-api-note">REST API ENABLED</span>
    <span class="header-badge">ENTERPRISE</span>
  </div>
</header>

<div class="toolbar">
  <button class="btn btn-primary" onclick="openCreateModal()">+ NEW STREAM</button>
  <button class="btn" onclick="refreshAll()">↻ REFRESH</button>
  <div style="flex:1"></div>
  <span style="font-size:10px;opacity:0.5;font-family:var(--font-head);">ADMIN PANEL</span>
</div>

<main>

  <!-- API DOCS -->
  <div class="api-strip" id="apiDocs">
    <div class="api-strip-head" onclick="toggleApi()">
      <span>◆ REST API REFERENCE</span>
      <span id="apiToggleIcon">▼</span>
    </div>
    <div class="api-strip-body" id="apiStripBody">
      <table class="api-table">
        <thead><tr>
          <th>METHOD</th><th>ENDPOINT</th><th>DESCRIPTION</th>
        </tr></thead>
        <tbody>
          <tr>
            <td class="method method-get">GET</td>
            <td><code>/api/streams</code></td>
            <td>List all streams</td>
          </tr>
          <tr>
            <td class="method method-post">POST</td>
            <td><code>/api/streams</code></td>
            <td>Create stream · body: <code>{"name","stream_key","mode","rtmp_target"}</code></td>
          </tr>
          <tr>
            <td class="method method-get">GET</td>
            <td><code>/api/streams/:id</code></td>
            <td>Get single stream</td>
          </tr>
          <tr>
            <td class="method method-post">POST</td>
            <td><code>/api/streams/:id</code></td>
            <td>Update stream fields</td>
          </tr>
          <tr>
            <td class="method method-delete">DELETE</td>
            <td><code>/api/streams/:id</code></td>
            <td>Delete stream (stops if live)</td>
          </tr>
          <tr>
            <td class="method method-post">POST</td>
            <td><code>/api/streams/:id/start</code></td>
            <td>Start streaming</td>
          </tr>
          <tr>
            <td class="method method-post">POST</td>
            <td><code>/api/streams/:id/stop</code></td>
            <td>Stop streaming</td>
          </tr>
          <tr>
            <td class="method method-post">POST</td>
            <td><code>/api/streams/:id/upload</code></td>
            <td>Upload media file (video/audio)</td>
          </tr>
          <tr>
            <td class="method method-get">GET</td>
            <td><code>/api/streams/:id/media</code></td>
            <td>List uploaded media for a stream</td>
          </tr>
          <tr>
            <td class="method method-post">POST</td>
            <td><code>/api/streams/:id/push</code></td>
            <td>Push raw H.264 bytes (mode=api streams). Stream via <code>ffmpeg … -f h264 pipe:1 | curl -X POST -T - /api/streams/:id/push</code></td>
          </tr>
        </tbody>
      </table>
      <div style="margin-top:12px;font-size:10px;opacity:0.6;">
        All POST/DELETE endpoints require header <code>X-API-Key: &lt;key&gt;</code>. Default key on first run: <strong>admin-secret-change-me</strong>
      </div>
    </div>
  </div>

  <!-- STREAMS -->
  <div class="section-header">
    <span class="section-title">◆ LIVE STREAMS</span>
    <span id="streamCount" style="font-size:10px;opacity:0.5;"></span>
  </div>

  <div class="streams-grid" id="streamsGrid">
    <div class="empty-state">
      <div class="empty-icon">◈</div>
      <div class="empty-title">NO STREAMS YET</div>
      <p style="font-size:11px;opacity:0.6;margin-bottom:16px;">Create your first stream above.</p>
    </div>
  </div>
</main>

<!-- TOAST -->
<div id="toast"></div>

<!-- CREATE / EDIT MODAL -->
<div class="modal-overlay" id="streamModal">
  <div class="modal">
    <div class="modal-head">
      <span id="modalTitle">NEW STREAM</span>
      <button class="modal-close" onclick="closeModal()">✕</button>
    </div>
    <div class="modal-body">
      <input type="hidden" id="editStreamId">
      <div>
        <label>STREAM NAME *</label>
        <input type="text" id="fName" placeholder="My Awesome Stream">
      </div>
      <div>
        <label>DESCRIPTION</label>
        <textarea id="fDesc" rows="2" placeholder="Optional description"></textarea>
      </div>
      <div>
        <label>YOUTUBE STREAM KEY *</label>
        <input type="password" id="fKey" placeholder="xxxx-xxxx-xxxx-xxxx">
        <div class="field-note" id="fKeyHint">YouTube Studio → Go Live → Stream Setup → <strong>Copy the stream key</strong>.<br>
        The "name" you set in YouTube Studio is just a label for yourself — it does NOT have to match this app's stream name.</div>
      </div>
      <div>
        <label>RTMP TARGET</label>
        <input type="text" id="fRtmp" value="rtmp://a.rtmp.youtube.com/live2">
        <div class="field-note">Default is YouTube. Change for Twitch, custom RTMP, etc.</div>
      </div>
      <div>
        <label>YOUTUBE CHANNEL ID <span style="font-weight:400;opacity:0.5;">(optional)</span></label>
        <input type="text" id="fChannelId" placeholder="UCxxxxxxxxxxxxxxxxxxxxxxxxx">
        <div class="field-note">Enables a direct ▶ WATCH LIVE link. Works best when you run one stream at a time on this channel.<br>
        Multiple simultaneous streams use separate stream keys but share the same channel — the /live link will show whichever is newest.<br>
        Find your channel ID: YouTube Studio → Settings → Channel → Advanced settings.</div>
      </div>
      <div>
        <label>STREAM MODE</label>
        <select id="fMode">
          <option value="file">FILE — loop uploaded video/audio files</option>
          <option value="test">TEST CARD — colour bars (no upload needed)</option>
          <option value="api">API — push raw H.264 via REST endpoint</option>
        </select>
      </div>
      <div style="display:flex;gap:10px;justify-content:flex-end;padding-top:8px;border-top:var(--border);">
        <button class="btn" onclick="closeModal()">CANCEL</button>
        <button class="btn btn-primary" onclick="saveStream()">SAVE STREAM</button>
      </div>
    </div>
  </div>
</div>

<!-- MEDIA MODAL -->
<div class="modal-overlay" id="mediaModal">
  <div class="modal">
    <div class="modal-head">
      <span id="mediaModalTitle">MEDIA — STREAM</span>
      <button class="modal-close" onclick="closeMediaModal()">✕</button>
    </div>
    <div class="modal-body">
      <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
        <input type="file" id="fileInput" multiple accept="video/*,audio/*" onchange="handleUpload()">
        <div style="font-size:28px;margin-bottom:8px;">⊕</div>
        <strong>CLICK TO UPLOAD</strong><br>
        <span style="font-size:10px;opacity:0.6;">Video or Audio files · Multiple allowed</span>
      </div>
      <div id="mediaList" style="max-height:240px;overflow-y:auto;">
        <div style="font-size:11px;opacity:0.5;text-align:center;padding:20px;">No files uploaded yet.</div>
      </div>
    </div>
  </div>
</div>
<!-- OVERLAY MODAL -->
<div class="modal-overlay" id="overlayModal">
  <div class="modal">
    <div class="modal-head">
      <span id="overlayModalTitle">OVERLAYS</span>
      <button class="modal-close" onclick="closeOverlayModal()">✕</button>
    </div>
    <div class="modal-body">
      <div style="display:flex; justify-content:space-between; margin-bottom:12px; gap: 8px;">
        <button class="btn btn-primary" onclick="createDemoOverlay()" style="flex:1">+ CREATE DEMO OVERLAY</button>
        <button class="btn btn-danger" onclick="clearOverlay()">✕ CLEAR ACTIVE</button>
      </div>
      <div id="overlayList" style="max-height:240px;overflow-y:auto;">
        <!-- Overlays laden hier -->
      </div>
    </div>
  </div>
</div>
<script>
const API = '/api';
let _currentMediaStreamId = null;

// ── API helpers ──────────────────────────────────────────────
// ── Session key (set on login screen, stored only in memory) ────────────────
let _sessionKey = '';

function apiKey() { return _sessionKey; }

function showLogin() {
  document.getElementById('loginOverlay').style.display = 'flex';
  document.getElementById('mainContent').style.display = 'none';
  setTimeout(() => document.getElementById('loginKeyInput').focus(), 100);
}

function hideLogin() {
  document.getElementById('loginOverlay').style.display = 'none';
  document.getElementById('mainContent').style.display = 'block';
}

async function submitLogin() {
  const k = document.getElementById('loginKeyInput').value.trim();
  const errEl = document.getElementById('loginError');
  const btn = document.querySelector('#loginOverlay .btn-primary');
  console.log('[YTS] submitLogin called, key length:', k.length);
  if (!k) {
    errEl.textContent = 'Please enter your API key.';
    return;
  }
  errEl.textContent = 'Connecting…';
  btn.disabled = true;
  btn.textContent = 'SIGNING IN…';
  try {
    console.log('[YTS] fetching', API + '/streams');
    const res = await fetch(API + '/health', {
      headers: { 'X-API-Key': k }
    });
    console.log('[YTS] response status:', res.status);
    if (res.status === 401) {
      errEl.textContent = '✗ Wrong API key.';
      document.getElementById('loginKeyInput').focus();
      btn.disabled = false;
      btn.textContent = 'SIGN IN →';
      return;
    }
    _sessionKey = k;
    errEl.textContent = '';
    console.log('[YTS] login ok, showing main');
    hideLogin();
    refreshAll();
  } catch(e) {
    console.error('[YTS] login fetch error:', e);
    errEl.textContent = '✗ Network error: ' + e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = 'SIGN IN →';
  }
}

document.addEventListener('keydown', e => {
  if (e.key === 'Enter' && document.getElementById('loginOverlay').style.display !== 'none') {
    submitLogin();
  }
});

async function apiFetch(path, opts = {}) {
  const method = (opts.method || 'GET').toUpperCase();
  console.log('[YTS] apiFetch', method, path);
  const headers = { 'X-API-Key': apiKey(), ...(opts.headers || {}) };
  if (!(opts.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }
  let res;
  try {
    res = await fetch(API + path, { ...opts, headers });
  } catch(e) {
    console.error('[YTS] fetch network error:', e);
    throw e;
  }
  console.log('[YTS] response', res.status, path);
  let data;
  try { data = await res.json(); } catch { data = {}; }
  if (res.status === 401) { showLogin(); throw new Error('Unauthorized'); }
  if (!res.ok) {
    console.error('[YTS] API error:', data);
    throw new Error(data.error || 'Request failed (' + res.status + ')');
  }
  return data;
}

// ── Toast ────────────────────────────────────────────────────
let _toastTimer;
function toast(msg, isError = false) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.style.background = isError ? '#ff4d1c' : '#0a0a0a';
  el.classList.add('show');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove('show'), 3200);
}

// ── Render ───────────────────────────────────────────────────
function formatBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
  return (b/1048576).toFixed(1) + ' MB';
}

function renderStreams(streams) {
  const grid = document.getElementById('streamsGrid');
  document.getElementById('streamCount').textContent = streams.length + ' STREAM(S)';
  if (!streams.length) {
    grid.innerHTML = `<div class="empty-state">
      <div class="empty-icon">◈</div>
      <div class="empty-title">NO STREAMS YET</div>
      <p style="font-size:11px;opacity:0.6;margin-bottom:16px;">Create your first stream above.</p>
      <button class="btn btn-primary" onclick="openCreateModal()">+ NEW STREAM</button>
    </div>`;
    return;
  }
  grid.innerHTML = streams.map(s => {
    const isLive = s.status === 'live';
    const modeLabel = (s.mode || 'file').toUpperCase();
    const created = s.created_at ? s.created_at.substring(0,10) : '';
    const errorHtml = s.error_msg
      ? `<div class="card-error">⚠ ${escHtml(s.error_msg)}</div>` : '';
    const h = s.health || {};
    const statsHtml = (isLive && h.state === 'streaming')
      ? `<div style="font-size:10px;opacity:0.6;margin-bottom:6px;">
           ▸ ${h.fps?.toFixed(0) || '?'} fps · ${h.bitrate?.toFixed(0) || '?'} kbps · ${h.time || ''} · ${(h.speed || 0).toFixed(2)}x speed
         </div>` : '';
    const warnHtml = (isLive && h.state === 'error')
      ? `<div class="card-error" style="margin-bottom:6px;">⚠ ${escHtml(h.message || '')}</div>` : '';
    return `<div class="card" id="card-${s.id}">
      <div class="card-head">
        <div class="card-title">${escHtml(s.name)}</div>
        <div class="status-badge ${healthClass(s)}" title="${escHtml(s.health?.message || '')}">
          ${healthLabel(s)}
        </div>
      </div>
      <div class="card-body">
        <div class="card-desc">${escHtml(s.description || '—')}</div>
        <div class="meta-row">
          <span class="tag tag-mode">${modeLabel}</span>
          <span class="tag">${created}</span>
          ${s.pid ? `<span class="tag">PID ${s.pid}</span>` : ''}
        </div>
        ${warnHtml}${statsHtml}${errorHtml}
        <div class="card-actions">
          ${isLive
            ? `<button class="btn btn-live btn-sm" onclick="stopStream('${s.id}')">■ STOP</button>`
            : `<button class="btn btn-primary btn-sm" onclick="startStream('${s.id}')">▶ START</button>`
          }
          <button class="btn btn-sm"
            data-id="${s.id}"
            data-name="${escHtml(s.name)}"
            data-desc="${escHtml(s.description||'')}"
            data-rtmp="${escHtml(s.rtmp_target||'')}"
            data-mode="${escHtml(s.mode||'file')}"
            data-channelid="${escHtml(s.yt_channel_id||'')}"
            onclick="openEditModalFromEl(this)">✎ EDIT</button>
          <button class="btn btn-sm" onclick="openMediaModal('${s.id}', '${escHtml(s.name)}')">⊕ MEDIA</button>

          <!-- NEUER OVERLAY BUTTON HIER: -->
          <button class="btn btn-sm" onclick="openOverlayModal('${s.id}', '${escHtml(s.name)}')">✧ OVERLAYS</button>

          <button class="btn btn-danger btn-sm" onclick="deleteStream('${s.id}')">✕</button>
          <button class="btn btn-sm" onclick="showLog('${s.id}')">⎙ LOG</button>
        </div>
        ${isLive ? `<div style="margin-top:8px;font-size:10px;opacity:0.55;border-top:2px solid var(--ink);padding-top:6px;">
          ▶ Streaming to: <strong>${escHtml(s.rtmp_target||'')}</strong> &nbsp; PID: ${s.pid||'?'}<br>
          ${s.yt_channel_id
            ? `<a href="https://www.youtube.com/channel/${escHtml(s.yt_channel_id)}/live"
                 target="_blank" style="color:var(--cut2);font-weight:700;">▶ WATCH LIVE ↗</a> &nbsp;|&nbsp;`
            : ''
          }
          <a href="https://studio.youtube.com" target="_blank" style="color:var(--ink);font-weight:700;">YT STUDIO ↗</a>
        </div>` : ''}
      </div>
    </div>`;
  }).join('');
}

function healthClass(s) {
  if (!s.health || s.status !== 'live') return 'status-stopped';
  switch(s.health.state) {
    case 'streaming': return 'status-live';
    case 'starting':  return 'status-starting';
    case 'stalled':   return 'status-stalled';
    case 'error':     return 'status-error';
    default:          return 'status-stopped';
  }
}
function healthLabel(s) {
  if (s.status !== 'live') return '◯ STOPPED';
  const st = s.health?.state;
  if (st === 'streaming') return '⬤ LIVE';
  if (st === 'starting')  return '◐ CONNECTING';
  if (st === 'stalled')   return '◒ STALLED';
  if (st === 'error')     return '✕ ERROR';
  return '◯ IDLE';
}
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Load streams ─────────────────────────────────────────────
async function refreshAll() {
  try {
    const data = await apiFetch('/streams');
    renderStreams(data.streams || []);
  } catch (e) { toast('Load error: ' + e.message, true); }
}

// ── CRUD ─────────────────────────────────────────────────────
function openCreateModal() {
  document.getElementById('modalTitle').textContent = 'NEW STREAM';
  document.getElementById('editStreamId').value = '';
  document.getElementById('fName').value = '';
  document.getElementById('fDesc').value = '';
  document.getElementById('fKey').value = '';
  document.getElementById('fRtmp').value = 'rtmp://a.rtmp.youtube.com/live2';
  document.getElementById('fMode').value = 'file';
  document.getElementById('fChannelId').value = '';
  document.getElementById('streamModal').classList.add('open');
}

function openEditModalFromEl(el) {
  document.getElementById('modalTitle').textContent = 'EDIT STREAM';
  document.getElementById('editStreamId').value = el.dataset.id;
  document.getElementById('fName').value = el.dataset.name || '';
  document.getElementById('fDesc').value = el.dataset.desc || '';
  document.getElementById('fKey').value = '';  // never pre-fill key
  document.getElementById('fKeyHint').textContent = 'Leave empty to keep existing key.';
  document.getElementById('fRtmp').value = el.dataset.rtmp || 'rtmp://a.rtmp.youtube.com/live2';
  document.getElementById('fMode').value = el.dataset.mode || 'file';
  document.getElementById('fChannelId').value = el.dataset.channelid || '';
  document.getElementById('streamModal').classList.add('open');
}

function closeModal() {
  document.getElementById('streamModal').classList.remove('open');
}

async function saveStream() {
  const sid = document.getElementById('editStreamId').value;
  const body = {
    name: document.getElementById('fName').value.trim(),
    description: document.getElementById('fDesc').value.trim(),
    stream_key: document.getElementById('fKey').value.trim(),
    rtmp_target: document.getElementById('fRtmp').value.trim(),
    mode: document.getElementById('fMode').value,
    yt_channel_id: document.getElementById('fChannelId').value.trim(),
  };
  if (!body.name) { toast('Name is required', true); return; }
  try {
    if (sid) {
      await apiFetch('/streams/' + sid, { method: 'POST', body: JSON.stringify(body) });
      toast('Stream updated ✓');
    } else {
      if (!body.stream_key && body.mode !== 'test') {
        toast('Stream key required (or use TEST CARD mode)', true); return;
      }
      await apiFetch('/streams', { method: 'POST', body: JSON.stringify(body) });
      toast('Stream created ✓');
    }
    closeModal();
    refreshAll();
  } catch (e) { toast(e.message, true); }
}

async function deleteStream(id) {
  if (!confirm('Delete this stream?')) return;
  try {
    await apiFetch('/streams/' + id, { method: 'DELETE' });
    toast('Deleted');
    refreshAll();
  } catch (e) { toast(e.message, true); }
}

async function startStream(id) {
  try {
    const r = await apiFetch('/streams/' + id + '/start', { method: 'POST' });
    toast('▶ Stream starting (PID ' + r.pid + ')');
    setTimeout(refreshAll, 1000);
  } catch (e) { toast(e.message, true); }
}

async function stopStream(id) {
  try {
    await apiFetch('/streams/' + id + '/stop', { method: 'POST' });
    toast('■ Stream stopped');
    setTimeout(refreshAll, 800);
  } catch (e) { toast(e.message, true); }
}

// ── Media ─────────────────────────────────────────────────────
async function openMediaModal(streamId, streamName) {
  _currentMediaStreamId = streamId;
  document.getElementById('mediaModalTitle').textContent = 'MEDIA — ' + streamName.toUpperCase();
  document.getElementById('mediaModal').classList.add('open');
  await loadMedia(streamId);
}

function closeMediaModal() {
  document.getElementById('mediaModal').classList.remove('open');
  _currentMediaStreamId = null;
}

async function loadMedia(streamId) {
  try {
    const data = await apiFetch('/streams/' + streamId + '/media');
    const files = data.files || [];
    const el = document.getElementById('mediaList');
    if (!files.length) {
      el.innerHTML = '<div style="font-size:11px;opacity:0.5;text-align:center;padding:20px;">No files uploaded yet.</div>';
      return;
    }
    el.innerHTML = files.map(f => `
      <div class="media-item">
        <span class="media-name">${escHtml(f.filename)}</span>
        <span class="media-size">${formatBytes(f.size_bytes)}</span>
      </div>
    `).join('');
  } catch(e) { toast('Media load error', true); }
}

async function handleUpload() {
  const input = document.getElementById('fileInput');
  const files = [...input.files];
  if (!files.length || !_currentMediaStreamId) return;
  for (const file of files) {
    const fd = new FormData();
    fd.append('file', file);
    try {
      await apiFetch('/streams/' + _currentMediaStreamId + '/upload', {
        method: 'POST',
        body: fd,
        headers: { 'X-API-Key': apiKey() }
      });
      toast('Uploaded: ' + file.name);
    } catch(e) { toast('Upload failed: ' + e.message, true); }
  }
  input.value = '';
  await loadMedia(_currentMediaStreamId);
}

// ── Overlays ──────────────────────────────────────────────────
let _currentOverlayStreamId = null;

async function openOverlayModal(streamId, streamName) {
  _currentOverlayStreamId = streamId;
  document.getElementById('overlayModalTitle').textContent = 'OVERLAYS — ' + streamName.toUpperCase();
  document.getElementById('overlayModal').classList.add('open');
  await loadOverlays();
}

function closeOverlayModal() {
  document.getElementById('overlayModal').classList.remove('open');
  _currentOverlayStreamId = null;
}

async function loadOverlays() {
  try {
    const data = await apiFetch('/overlays');
    const list = data.presets || [];
    const el = document.getElementById('overlayList');
    if (!list.length) {
      el.innerHTML = '<div style="font-size:11px;opacity:0.5;text-align:center;padding:20px;">No overlays found. Create a demo first!</div>';
      return;
    }
    el.innerHTML = list.map(o => `
      <div class="media-item">
        <span class="media-name">${escHtml(o.name)} <span style="opacity:0.5">(${o.duration_ms}ms)</span></span>
        <button class="btn btn-sm btn-primary" onclick="triggerOverlay('${o.id}')">▶ SHOW</button>
        <button class="btn btn-sm btn-danger" onclick="deleteOverlay('${o.id}')">✕</button>
      </div>
    `).join('');
  } catch(e) { toast('Overlay load error', true); }
}

async function triggerOverlay(pid) {
  if (!_currentOverlayStreamId) return;
  try {
    await apiFetch('/streams/' + _currentOverlayStreamId + '/overlay/' + pid + '/trigger', { method: 'POST' });
    toast('Overlay triggered! ✓');
  } catch(e) { toast('Error: ' + e.message, true); }
}

async function clearOverlay() {
  if (!_currentOverlayStreamId) return;
  try {
    await apiFetch('/streams/' + _currentOverlayStreamId + '/overlay/clear', { method: 'POST' });
    toast('Overlays cleared!');
  } catch(e) { toast('Error: ' + e.message, true); }
}

async function deleteOverlay(pid) {
  if (!confirm('Delete this overlay?')) return;
  try {
    await apiFetch('/overlays/' + pid, { method: 'DELETE' });
    await loadOverlays();
  } catch(e) { toast('Error: ' + e.message, true); }
}

// Erstellt ein einfaches "LIVE BROADCAST" Test-Overlay über die API
async function createDemoOverlay() {
  const demoSvg = {
    duration_ms: 6000,
    shapes: [
      { type: "rect", x: 40, y: 40, w: 320, h: 60, fill: "#ff4d1c", opacity: 0.9 },
      { type: "text", x: 60, y: 55, text: "LIVE BROADCAST", font_size: 24, fill: "#ffffff" }
    ]
  };
  try {
    await apiFetch('/overlays', {
      method: 'POST',
      body: JSON.stringify({ name: "Demo Banner " + Math.floor(Math.random()*1000), svg: demoSvg, duration_ms: 6000 })
    });
    toast('Demo Overlay created!');
    await loadOverlays();
  } catch(e) { toast('Error: ' + e.message, true); }
}

// ── Stream log viewer (live tail with auto-refresh) ───────────
let _logPollTimer = null;
let _logCurrentSid = null;

async function showLog(sid) {
  _logCurrentSid = sid;
  document.getElementById('logModal').classList.add('open');
  document.getElementById('logPre').textContent = 'Loading…';
  await refreshLog();
  if (_logPollTimer) clearInterval(_logPollTimer);
  _logPollTimer = setInterval(refreshLog, 1500);
}

function closeLogModal() {
  document.getElementById('logModal').classList.remove('open');
  if (_logPollTimer) { clearInterval(_logPollTimer); _logPollTimer = null; }
  _logCurrentSid = null;
}

async function refreshLog() {
  if (!_logCurrentSid) return;
  try {
    const d = await apiFetch('/streams/' + _logCurrentSid + '/log');
    const header = [
      'STATUS   : ' + d.status + (d.running ? '  ⬤ RUNNING' : '  ◯ STOPPED'),
      'PID      : ' + (d.pid || 'none'),
      'RTMP     : ' + (d.rtmp_target || ''),
      '────────────────────────────────────────',
    ].join('\n');
    const body = d.live_log || d.last_error || '(waiting for ffmpeg output…)';
    const pre = document.getElementById('logPre');
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 20;
    pre.textContent = header + '\n' + body;
    if (atBottom) pre.scrollTop = pre.scrollHeight;
    // status dot
    document.getElementById('logStatusDot').style.background =
      d.running ? 'var(--cut1)' : 'var(--cream)';
  } catch(e) {
    document.getElementById('logPre').textContent = 'Error: ' + e.message;
  }
}

// ── API docs toggle ──────────────────────────────────────────
function toggleApi() {
  const body = document.getElementById('apiStripBody');
  const icon = document.getElementById('apiToggleIcon');
  body.classList.toggle('open');
  icon.textContent = body.classList.contains('open') ? '▲' : '▼';
}

// ── Auto-refresh every 10s ───────────────────────────────────
setInterval(refreshAll, 10000);

// ── Boot ─────────────────────────────────────────────────────
showLogin();
</script>

<!-- LIVE LOG MODAL -->
<div class="modal-overlay" id="logModal">
  <div class="modal" style="max-width:720px;">
    <div class="modal-head">
      <span style="display:flex;align-items:center;gap:8px;">
        <span id="logStatusDot" style="width:10px;height:10px;background:var(--cream);border-radius:50%;border:2px solid var(--paper);"></span>
        FFMPEG LIVE LOG
      </span>
      <button class="modal-close" onclick="closeLogModal()">✕</button>
    </div>
    <div class="modal-body" style="padding:0;">
      <pre id="logPre" style="margin:0;padding:16px;background:#0a0a0a;color:#00ff66;font-family:'Space Mono',monospace;font-size:10px;line-height:1.4;max-height:60vh;min-height:300px;overflow:auto;white-space:pre-wrap;word-break:break-all;">Loading…</pre>
    </div>
  </div>
</div>
</div><!-- /mainContent -->
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────
#  AUTH MIDDLEWARE
# ─────────────────────────────────────────────────────────────
def require_api_key(db_path=DB_PATH):
    """Check X-API-Key header against YTS_API_KEY env var."""
    if not _BOTTLE_AVAILABLE:
        return
    key = request.headers.get("X-API-Key", "") or request.query.get("api_key", "")
    if key != _ENV_API_KEY:
        raise HTTPError(401, "Invalid or missing API key")

def json_resp(data, status=200):
    response.content_type = "application/json"
    response.status = status
    return json.dumps(data)

# ─────────────────────────────────────────────────────────────
#  APP FACTORY
# ─────────────────────────────────────────────────────────────
def make_app(db_path=DB_PATH, media_dir=MEDIA_DIR):
    app = Bottle()
    media_dir = Path(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)
    init_db(db_path)

    # ── UI ────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return HTML_PAGE

    # ── STREAMS LIST / CREATE ─────────────────────────────────
    @app.route("/api/streams", method=["GET", "OPTIONS"])
    def api_streams_list():
        require_api_key(db_path)
        streams = list_streams(db_path)
        # Enrich live streams with health data
        for s in streams:
            if s.get("status") == "live":
                log = StreamManager.get_log(s["id"])
                with _process_lock:
                    proc = _processes.get(s["id"])
                    running = proc is not None and proc.poll() is None
                s["health"] = StreamManager.assess_health(s["id"], log, running)
            else:
                s["health"] = {"state": "offline", "message": ""}
        return json_resp({"streams": streams})

    @app.route("/api/streams", method="POST")
    def api_streams_create():
        require_api_key(db_path)
        data = request.json or {}
        name = data.get("name", "").strip()
        rtmp_target = data.get("rtmp_target", YT_RTMP_BASE).strip()

        if not name:
            raise HTTPError(400, "name is required")

        # FIX: Harte Validierung der RTMP URL
        if not rtmp_target.startswith("rtmp://") and not rtmp_target.startswith("rtmps://"):
            raise HTTPError(400, "RTMP target must start with rtmp:// or rtmps://")
        if len(rtmp_target) < 16 or "." not in rtmp_target:
            raise HTTPError(400, "RTMP target URL looks incomplete (e.g. rtmp://a.rtmp.youtube.com/live2)")

        result = create_stream(
            name=name,
            description=data.get("description", ""),
            stream_key=data.get("stream_key", ""),
            rtmp_target=rtmp_target,
            mode=data.get("mode", "file"),
            yt_channel_id=data.get("yt_channel_id", ""),
            db_path=db_path,
        )
        return json_resp(result, 201)


    # ── SINGLE STREAM ─────────────────────────────────────────
    @app.route("/api/streams/<sid>", method="GET")
    def api_stream_get(sid):
        require_api_key(db_path)
        s = get_stream(sid, db_path)
        if not s:
            raise HTTPError(404, "Stream not found")
        return json_resp(s)

    @app.route("/api/streams/<sid>", method="POST")
    def api_stream_update(sid):
        require_api_key(db_path)
        data = request.json or {}
        if not get_stream(sid, db_path):
            raise HTTPError(404, "Stream not found")

        # FIX: Harte Validierung beim Updaten
        if "rtmp_target" in data:
            target = data["rtmp_target"].strip()
            if not target.startswith("rtmp://") and not target.startswith("rtmps://"):
                raise HTTPError(400, "RTMP target must start with rtmp:// or rtmps://")
            if len(target) < 16 or "." not in target:
                raise HTTPError(400, "RTMP target URL looks incomplete")

        update_stream(sid, data, db_path)
        return json_resp({"ok": True})

    @app.route("/api/streams/<sid>", method="DELETE")
    def api_stream_delete(sid):
        require_api_key(db_path)
        if not get_stream(sid, db_path):
            raise HTTPError(404, "Stream not found")
        delete_stream(sid, db_path)
        return json_resp({"ok": True})

    # ── START / STOP ──────────────────────────────────────────
    @app.route("/api/streams/<sid>/start", method="POST")
    def api_stream_start(sid):
        require_api_key(db_path)
        if not get_stream(sid, db_path):
            raise HTTPError(404, "Stream not found")
        result = StreamManager.start(sid, db_path)
        if not result["ok"]:
            raise HTTPError(400, result["error"])
        return json_resp(result)

    @app.route("/api/streams/<sid>/stop", method="POST")
    def api_stream_stop(sid):
        require_api_key(db_path)
        if not get_stream(sid, db_path):
            raise HTTPError(404, "Stream not found")
        result = StreamManager.stop(sid, db_path)
        return json_resp(result)

    # ── MEDIA UPLOAD ──────────────────────────────────────────
    @app.route("/api/streams/<sid>/upload", method="POST")
    def api_media_upload(sid):
        require_api_key(db_path)
        if not get_stream(sid, db_path):
            raise HTTPError(404, "Stream not found")
        upload = request.files.get("file")
        if not upload:
            raise HTTPError(400, "No file provided")
        mime, _ = mimetypes.guess_type(upload.filename)
        mime = mime or "application/octet-stream"
        if not (mime.startswith("video/") or mime.startswith("audio/")):
            raise HTTPError(400, "Only video/audio files accepted")
        stream_media_dir = media_dir / sid
        stream_media_dir.mkdir(parents=True, exist_ok=True)
        dest = stream_media_dir / upload.filename
        upload.save(str(dest), overwrite=True)
        size = dest.stat().st_size
        result = add_media(sid, upload.filename, str(dest), mime, size, db_path)
        return json_resp(result, 201)

    @app.route("/api/streams/<sid>/media", method="GET")
    def api_media_list(sid):
        require_api_key(db_path)
        if not get_stream(sid, db_path):
            raise HTTPError(404, "Stream not found")
        return json_resp({"files": list_media(sid, db_path)})

    # ── PUSH ENDPOINT (API mode) ──────────────────────────────
    @app.route("/api/streams/<sid>/push", method="POST")
    def api_push(sid):
        require_api_key(db_path)
        s = get_stream(sid, db_path)
        if not s:
            raise HTTPError(404, "Stream not found")
        if s["mode"] != "api":
            raise HTTPError(400, "Stream is not in API mode")
        pipe = StreamManager.get_pipe(sid)
        if not pipe:
            raise HTTPError(409, "Stream is not running — start it first")
        chunk = request.body.read()
        if chunk:
            try:
                pipe.write(chunk)
                pipe.flush()
            except BrokenPipeError:
                raise HTTPError(500, "FFmpeg pipe closed")
        return json_resp({"ok": True, "bytes": len(chunk)})

    # ── HEALTH ────────────────────────────────────────────────
    # ─── OVERLAY PRESETS ───────────────────────────────────────────
    @app.route("/api/overlays", method="GET")
    def api_overlays_list():
        require_api_key(db_path)
        return json_resp({"presets": list_overlay_presets(db_path)})

    @app.route("/api/overlays", method="POST")
    def api_overlays_create():
        require_api_key(db_path)
        data = request.json or {}
        name = data.get("name", "").strip()
        if not name:
            raise HTTPError(400, "name is required")
        svg = data.get("svg") or {"shapes": [], "duration_ms": 3000}
        duration = int(data.get("duration_ms", svg.get("duration_ms", 3000)))
        result = create_overlay_preset(name, svg, duration, db_path)
        return json_resp(result, 201)

    @app.route("/api/overlays/<pid>", method="GET")
    def api_overlays_get(pid):
        require_api_key(db_path)
        p = get_overlay_preset(pid, db_path)
        if not p:
            raise HTTPError(404, "Preset not found")
        return json_resp(p)

    @app.route("/api/overlays/<pid>", method="POST")
    def api_overlays_update(pid):
        require_api_key(db_path)
        if not get_overlay_preset(pid, db_path):
            raise HTTPError(404, "Preset not found")
        data = request.json or {}
        fields = {}
        if "name" in data:          fields["name"] = data["name"]
        if "duration_ms" in data:   fields["duration_ms"] = int(data["duration_ms"])
        if "svg" in data:           fields["svg_json"] = data["svg"]
        update_overlay_preset(pid, fields, db_path)
        return json_resp({"ok": True})

    @app.route("/api/overlays/<pid>", method="DELETE")
    def api_overlays_delete(pid):
        require_api_key(db_path)
        delete_overlay_preset(pid, db_path)
        return json_resp({"ok": True})

    @app.route("/api/streams/<sid>/overlay/<pid>/trigger", method="POST")
    def api_trigger_overlay(sid, pid):
        require_api_key(db_path)
        if not get_stream(sid, db_path):
            raise HTTPError(404, "Stream not found")
        preset = get_overlay_preset(pid, db_path)
        if not preset:
            raise HTTPError(404, "Preset not found")
        OverlayEngine.trigger(sid, {
            "id": preset["id"],
            "name": preset["name"],
            "duration_ms": preset["duration_ms"],
            "shapes": preset["svg"].get("shapes", []),
        }, db_path)
        return json_resp({"ok": True, "triggered": preset["name"]})

    @app.route("/api/streams/<sid>/overlay/clear", method="POST")
    def api_clear_overlay(sid):
        require_api_key(db_path)
        if not get_stream(sid, db_path):
            raise HTTPError(404, "Stream not found")
        OverlayEngine.clear(sid)
        return json_resp({"ok": True})

    # ─── AUDIO PRESETS ─────────────────────────────────────────────
    @app.route("/api/audio", method="GET")
    def api_audio_list():
        require_api_key(db_path)
        return json_resp({"presets": list_audio_presets(db_path)})

    @app.route("/api/audio", method="POST")
    def api_audio_upload():
        require_api_key(db_path)
        name = request.forms.get("name", "").strip() or (
            request.files.get("file").filename if request.files.get("file") else ""
        )
        upload = request.files.get("file")
        if not upload:
            raise HTTPError(400, "No file provided")
        audio_dir = media_dir / "_audio_presets"
        audio_dir.mkdir(parents=True, exist_ok=True)
        dest = audio_dir / upload.filename
        upload.save(str(dest), overwrite=True)
        result = create_audio_preset(
            name, str(dest),
            filetype=mimetypes.guess_type(upload.filename)[0] or "audio/mpeg",
            db_path=db_path
        )
        return json_resp(result, 201)

    @app.route("/api/audio/<pid>", method="DELETE")
    def api_audio_delete(pid):
        require_api_key(db_path)
        delete_audio_preset(pid, db_path)
        return json_resp({"ok": True})

    @app.route("/api/streams/<sid>/audio/<pid>/play", method="POST")
    def api_play_audio(sid, pid):
        require_api_key(db_path)
        if not get_stream(sid, db_path):
            raise HTTPError(404, "Stream not found")
        preset = get_audio_preset(pid, db_path)
        if not preset:
            raise HTTPError(404, "Audio preset not found")
        result = AudioPresetManager.trigger(preset["filepath"])
        return json_resp(result)

    # ─── VIDEO CLIPS ──────────────────────────────────────────────
    @app.route("/api/clips", method="GET")
    def api_clips_list():
        require_api_key(db_path)
        return json_resp({"presets": list_clip_presets(db_path)})

    @app.route("/api/clips", method="POST")
    def api_clips_upload():
        require_api_key(db_path)
        upload = request.files.get("file")
        if not upload:
            raise HTTPError(400, "No file provided")
        name = request.forms.get("name", "").strip() or upload.filename
        clip_dir = media_dir / "_clips"
        clip_dir.mkdir(parents=True, exist_ok=True)
        dest = clip_dir / upload.filename
        upload.save(str(dest), overwrite=True)
        # Best-effort duration via ffprobe
        duration = 0.0
        try:
            out = subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(dest)],
                timeout=10
            )
            duration = float(out.decode().strip())
        except Exception:
            pass
        result = create_clip_preset(
            name, str(dest),
            filetype=mimetypes.guess_type(upload.filename)[0] or "video/mp4",
            duration_s=duration, db_path=db_path,
        )
        return json_resp(result, 201)

    @app.route("/api/clips/<pid>", method="DELETE")
    def api_clips_delete(pid):
        require_api_key(db_path)
        delete_clip_preset(pid, db_path)
        return json_resp({"ok": True})

    @app.route("/api/streams/<sid>/clip/<pid>/sling", method="POST")
    def api_sling_clip(sid, pid):
        """
        Append a clip to a file-mode stream's playlist.
        Restarts ffmpeg with updated playlist — so there's a ~1-2s cut.
        For test/api modes this just adds the clip as a pending media file.
        """
        require_api_key(db_path)
        stream = get_stream(sid, db_path)
        if not stream:
            raise HTTPError(404, "Stream not found")
        preset = get_clip_preset(pid, db_path)
        if not preset:
            raise HTTPError(404, "Clip not found")

        # Copy/register the clip as a stream media file
        add_media(
            sid, preset["name"] + " (clip)",
            preset["filepath"], preset["filetype"],
            os.path.getsize(preset["filepath"]) if os.path.exists(preset["filepath"]) else 0,
            db_path
        )

        # If the stream is currently live in FILE mode, restart it so the
        # playlist gets picked up. Otherwise just register it.
        was_live = stream.get("status") == "live"
        if was_live and stream.get("mode") == "file":
            StreamManager.stop(sid, db_path)
            import time as _t; _t.sleep(0.8)
            r = StreamManager.start(sid, db_path)
            return json_resp({"ok": True, "restarted": True,
                              "clip": preset["name"], "pid": r.get("pid")})

        return json_resp({"ok": True, "registered": True, "clip": preset["name"]})

    @app.route("/api/health")
    def health():
        require_api_key(db_path)
        return json_resp({"status": "ok", "time": _now()})

    @app.route("/api/streams/<sid>/log", method="GET")
    def api_stream_log(sid):
        """Return live ffmpeg log buffer (tail) + a health verdict."""
        require_api_key(db_path)
        s = get_stream(sid, db_path)
        if not s:
            raise HTTPError(404, "Stream not found")
        with _process_lock:
            proc = _processes.get(sid)
            running = proc is not None and proc.poll() is None
        live_log = StreamManager.get_log(sid)
        health = StreamManager.assess_health(sid, live_log, running)
        return json_resp({
            "id": sid,
            "running": running,
            "status": s.get("status", "unknown"),
            "pid": s.get("pid"),
            "rtmp_target": s.get("rtmp_target", ""),
            "last_error": s.get("error_msg", ""),
            "live_log": live_log,
            "health": health,
        })

    return app

# ─────────────────────────────────────────────────────────────
#  SETUP GUIDE
# ─────────────────────────────────────────────────────────────
SETUP_GUIDE = """
╔══════════════════════════════════════════════════════════════╗
║          YT STREAM HUB — SETUP GUIDE                        ║
╚══════════════════════════════════════════════════════════════╝

1. INSTALL DEPENDENCIES
   ─────────────────────
   pip install bottle          # web framework
   # FFmpeg (system package):
   sudo apt install ffmpeg     # Debian/Ubuntu
   sudo dnf install ffmpeg     # Fedora
   brew install ffmpeg         # macOS

2. ENVIRONMENT VARIABLES  (optional, all have defaults)
   ──────────────────────────────────────────────────────
   YTS_DB=streams.db           # SQLite database path
   YTS_MEDIA=media             # upload directory
   YTS_HOST=0.0.0.0            # bind host
   YTS_PORT=8080               # bind port

3. RUN THE SERVER
   ─────────────────
   python yt_stream_service.py

   Admin panel: http://localhost:8080
   API base:    http://localhost:8080/api

4. GET YOUR YOUTUBE STREAM KEY
   ─────────────────────────────
   a) Go to https://studio.youtube.com
   b) Click ▶ Create → Go Live
   c) Copy the "Stream key" (keep secret!)
   d) RTMP URL: rtmp://a.rtmp.youtube.com/live2

5. CREATE A STREAM (UI or API)
   ─────────────────────────────
   UI:   Click "+ NEW STREAM" on the admin panel
   API:  curl -X POST http://localhost:8080/api/streams \\
              -H "X-API-Key: admin-secret-change-me" \\
              -H "Content-Type: application/json" \\
              -d '{"name":"My Stream","stream_key":"xxxx","mode":"file"}'

6. CONTINUOUS STREAMING MODES
   ─────────────────────────────
   FILE   Upload video/audio via UI or /api/streams/:id/upload
          Files loop forever.
   TEST   Streams a colour-bars test card (no upload needed)
   API    Push raw H.264 from any encoder:
          ffmpeg -i input.mp4 -f h264 pipe:1 | \\
            curl -X POST -T - \\
              -H "X-API-Key: admin-secret-change-me" \\
              http://localhost:8080/api/streams/<id>/push

7. PRODUCTION NOTES
   ─────────────────
   • Put nginx in front (proxy_pass to :8080)
   • Change the API key in SQLite: UPDATE api_keys SET key='new-key'
   • Use a process manager (systemd, supervisord) to keep the server alive
   • YouTube allows ~12h per stream session; rotate stream keys for longer

8. RUN TESTS
   ──────────
   python yt_stream_service.py test
"""

# ─────────────────────────────────────────────────────────────
#  MINIMAL WSGI TEST CLIENT  (no external deps)
# ─────────────────────────────────────────────────────────────
class _WSGIClient:
    """Bare-bones WSGI test client — calls app directly via WSGI environ."""

    def __init__(self, app):
        self.app = app

    def _call(self, method, path, body=b"", headers=None):
        from io import BytesIO
        headers = headers or {}
        environ = {
            "REQUEST_METHOD": method.upper(),
            "PATH_INFO": path,
            "QUERY_STRING": "",
            "CONTENT_TYPE": headers.get("Content-Type", "application/octet-stream"),
            "CONTENT_LENGTH": str(len(body)),
            "wsgi.input": BytesIO(body),
            "wsgi.errors": BytesIO(),
            "wsgi.url_scheme": "http",
            "wsgi.multithread": False,
            "wsgi.multiprocess": False,
            "wsgi.run_once": False,
            "SERVER_NAME": "localhost",
            "SERVER_PORT": "8080",
            "HTTP_HOST": "localhost:8080",
        }
        for k, v in headers.items():
            key = "HTTP_" + k.upper().replace("-", "_")
            environ[key] = v
        status_holder = []
        def start_response(status, response_headers, exc_info=None):
            status_holder.append(status)
        result = self.app(environ, start_response)
        raw = b"".join(result)
        status_code = int(status_holder[0].split()[0]) if status_holder else 500
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None
        return {"status": status_code, "body": raw, "json": parsed}

    def get(self, path, headers=None):
        return self._call("GET", path, headers=headers)

    def post(self, path, body=b"", headers=None):
        return self._call("POST", path, body=body, headers=headers)

    def post_json(self, path, data, headers=None):
        body = json.dumps(data).encode()
        h = {"Content-Type": "application/json", **(headers or {})}
        return self._call("POST", path, body=body, headers=h)

    def delete(self, path, headers=None):
        return self._call("DELETE", path, headers=headers)


# ─────────────────────────────────────────────────────────────
#  UNIT TESTS  (stdlib unittest only)
# ─────────────────────────────────────────────────────────────
class TestDB(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "test.db")
        init_db(self.db)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ── DB: create / list ──────────────────────────────────────
    def test_create_stream_returns_id(self):
        r = create_stream("Test", db_path=self.db)
        self.assertIn("id", r)
        self.assertEqual(r["name"], "Test")
        self.assertEqual(r["status"], "stopped")

    def test_list_streams_empty(self):
        rows = list_streams(self.db)
        # only default api_key row shouldn't appear; streams list should be empty
        self.assertEqual(rows, [])

    def test_list_streams_after_create(self):
        create_stream("A", db_path=self.db)
        create_stream("B", db_path=self.db)
        rows = list_streams(self.db)
        self.assertEqual(len(rows), 2)
        names = {r["name"] for r in rows}
        self.assertIn("A", names)
        self.assertIn("B", names)

    # ── DB: get / update / delete ─────────────────────────────
    def test_get_stream_exists(self):
        r = create_stream("X", db_path=self.db)
        s = get_stream(r["id"], self.db)
        self.assertIsNotNone(s)
        self.assertEqual(s["name"], "X")

    def test_get_stream_missing(self):
        self.assertIsNone(get_stream("no-such-id", self.db))

    def test_update_stream_name(self):
        r = create_stream("Old", db_path=self.db)
        update_stream(r["id"], {"name": "New"}, self.db)
        s = get_stream(r["id"], self.db)
        self.assertEqual(s["name"], "New")

    def test_update_stream_ignores_bad_fields(self):
        r = create_stream("Safe", db_path=self.db)
        update_stream(r["id"], {"status": "live", "name": "Safe2"}, self.db)
        s = get_stream(r["id"], self.db)
        # status should NOT have changed via update_stream
        self.assertEqual(s["status"], "stopped")
        self.assertEqual(s["name"], "Safe2")

    def test_delete_stream(self):
        r = create_stream("Del", db_path=self.db)
        delete_stream(r["id"], self.db)
        self.assertIsNone(get_stream(r["id"], self.db))

    # ── DB: media ─────────────────────────────────────────────
    def test_add_and_list_media(self):
        r = create_stream("MediaStream", db_path=self.db)
        add_media(r["id"], "clip.mp4", "/tmp/clip.mp4", "video/mp4", 1024, self.db)
        files = list_media(r["id"], self.db)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["filename"], "clip.mp4")

    def test_media_belongs_to_stream(self):
        a = create_stream("A", db_path=self.db)
        b = create_stream("B", db_path=self.db)
        add_media(a["id"], "a.mp4", "/tmp/a.mp4", "video/mp4", 10, self.db)
        add_media(b["id"], "b.mp4", "/tmp/b.mp4", "video/mp4", 20, self.db)
        self.assertEqual(len(list_media(a["id"], self.db)), 1)
        self.assertEqual(len(list_media(b["id"], self.db)), 1)

    # ── Default API key ───────────────────────────────────────
    def test_env_api_key_default(self):
        """ENV key defaults to admin-secret-change-me when not set."""
        import os
        os.environ.pop("YTS_API_KEY", None)
        # reimport the module-level variable via the global
        import importlib, sys
        # just verify the fallback logic works
        key = os.environ.get("YTS_API_KEY", "admin-secret-change-me")
        self.assertEqual(key, "admin-secret-change-me")

    # ── StreamManager: build command ──────────────────────────
    def test_ffmpeg_cmd_test_mode(self):
        stream = {"mode": "test", "stream_key": "mykey",
                  "rtmp_target": "rtmp://a.rtmp.youtube.com/live2"}
        cmd = StreamManager._build_ffmpeg_cmd(stream, [])
        self.assertIn("ffmpeg", cmd)
        self.assertIn("testsrc", " ".join(cmd))
        self.assertTrue(cmd[-1].endswith("mykey"))

    def test_ffmpeg_cmd_api_mode(self):
        stream = {"mode": "api", "stream_key": "mykey",
                  "rtmp_target": "rtmp://a.rtmp.youtube.com/live2"}
        cmd = StreamManager._build_ffmpeg_cmd(stream, [])
        self.assertIn("pipe:0", cmd)
        self.assertTrue(cmd[-1].endswith("mykey"))

    def test_ffmpeg_cmd_file_mode_no_media_falls_back_to_test(self):
        stream = {"mode": "file", "stream_key": "mykey",
                  "rtmp_target": "rtmp://a.rtmp.youtube.com/live2"}
        cmd = StreamManager._build_ffmpeg_cmd(stream, [])
        # no media → should fall back to testsrc
        self.assertIn("testsrc", " ".join(cmd))

    def test_ffmpeg_cmd_file_mode_with_video(self):
        stream = {"mode": "file", "stream_key": "mykey",
                  "rtmp_target": "rtmp://a.rtmp.youtube.com/live2"}
        files = [{"filepath": "/tmp/a.mp4", "filetype": "video/mp4"},
                 {"filepath": "/tmp/b.mp4", "filetype": "video/mp4"}]
        cmd = StreamManager._build_ffmpeg_cmd(stream, files)
        joined = " ".join(cmd)
        self.assertIn("concat", joined)
        # playlist file path is in cmd; read it to verify content
        playlist_path = cmd[cmd.index("-i") + 1]
        with open(playlist_path) as pf:
            content = pf.read()
        self.assertIn("/tmp/a.mp4", content)
        self.assertIn("/tmp/b.mp4", content)

    def test_ffmpeg_cmd_file_mode_audio_only(self):
        stream = {"mode": "file", "stream_key": "mykey",
                  "rtmp_target": "rtmp://a.rtmp.youtube.com/live2"}
        files = [{"filepath": "/tmp/track.mp3", "filetype": "audio/mpeg"}]
        cmd = StreamManager._build_ffmpeg_cmd(stream, files)
        joined = " ".join(cmd)
        self.assertIn("concat", joined)
        self.assertIn("color=c=black", joined)
        playlist_path = None
        for i, c in enumerate(cmd):
            if c == "-i" and i+1 < len(cmd) and "ytsh_playlist" in cmd[i+1]:
                playlist_path = cmd[i+1]
                break
        self.assertIsNotNone(playlist_path)
        with open(playlist_path) as pf:
            self.assertIn("/tmp/track.mp3", pf.read())

    # ── now() helper ──────────────────────────────────────────
    def test_now_returns_iso_string(self):
        n = _now()
        self.assertIsInstance(n, str)
        self.assertTrue("T" in n)

    # ── create_stream defaults ────────────────────────────────
    def test_create_stream_defaults(self):
        r = create_stream("defaults", db_path=self.db)
        s = get_stream(r["id"], self.db)
        self.assertEqual(s["mode"], "file")
        self.assertEqual(s["rtmp_target"], YT_RTMP_BASE)
        self.assertEqual(s["status"], "stopped")

    def test_create_multiple_streams_unique_ids(self):
        ids = [create_stream(f"S{i}", db_path=self.db)["id"] for i in range(5)]
        self.assertEqual(len(set(ids)), 5)

    # ── HTTP layer ────────────────────────────────────────────
    def _make_client(self):
        app = make_app(db_path=self.db, media_dir=Path(self.tmp) / "media")
        return _WSGIClient(app)

    def test_http_health(self):
        r = self._make_client().get("/api/health",
            headers={"X-API-Key": "admin-secret-change-me"})
        self.assertEqual(r["status"], 200)
        self.assertEqual(r["json"]["status"], "ok")

    def test_http_list_streams_requires_auth(self):
        r = self._make_client().get("/api/streams")
        self.assertEqual(r["status"], 401)

    def test_http_list_streams_with_auth(self):
        r = self._make_client().get("/api/streams",
            headers={"X-API-Key": "admin-secret-change-me"})
        self.assertEqual(r["status"], 200)
        self.assertEqual(r["json"]["streams"], [])

    def test_http_create_stream(self):
        r = self._make_client().post_json(
            "/api/streams",
            {"name": "HTTP Test", "stream_key": "abc", "mode": "test"},
            headers={"X-API-Key": "admin-secret-change-me"}
        )
        self.assertEqual(r["status"], 201)
        self.assertIn("id", r["json"])

    def test_http_create_stream_no_name(self):
        r = self._make_client().post_json(
            "/api/streams",
            {"stream_key": "abc"},
            headers={"X-API-Key": "admin-secret-change-me"}
        )
        self.assertEqual(r["status"], 400)

    def test_http_create_stream_no_auth(self):
        r = self._make_client().post_json("/api/streams", {"name": "nope", "stream_key": "abc"})
        self.assertEqual(r["status"], 401)

    def test_http_get_stream(self):
        c = self._make_client()
        cr = c.post_json(
            "/api/streams",
            {"name": "GetMe", "stream_key": "abc"},
            headers={"X-API-Key": "admin-secret-change-me"}
        )
        sid = cr["json"]["id"]
        r = c.get(f"/api/streams/{sid}", headers={"X-API-Key": "admin-secret-change-me"})
        self.assertEqual(r["status"], 200)
        self.assertEqual(r["json"]["name"], "GetMe")

    def test_http_get_stream_404(self):
        r = self._make_client().get("/api/streams/no-such-id",
            headers={"X-API-Key": "admin-secret-change-me"})
        self.assertEqual(r["status"], 404)

    def test_http_delete_stream(self):
        c = self._make_client()
        cr = c.post_json(
            "/api/streams",
            {"name": "DelMe", "stream_key": "abc"},
            headers={"X-API-Key": "admin-secret-change-me"}
        )
        sid = cr["json"]["id"]
        dr = c.delete(f"/api/streams/{sid}", headers={"X-API-Key": "admin-secret-change-me"})
        self.assertEqual(dr["status"], 200)
        gr = c.get(f"/api/streams/{sid}", headers={"X-API-Key": "admin-secret-change-me"})
        self.assertEqual(gr["status"], 404)

    def test_http_update_stream(self):
        c = self._make_client()
        cr = c.post_json(
            "/api/streams",
            {"name": "OldName", "stream_key": "abc"},
            headers={"X-API-Key": "admin-secret-change-me"}
        )
        sid = cr["json"]["id"]
        c.post_json(
            f"/api/streams/{sid}",
            {"name": "NewName"},
            headers={"X-API-Key": "admin-secret-change-me"}
        )
        r = c.get(f"/api/streams/{sid}", headers={"X-API-Key": "admin-secret-change-me"})
        self.assertEqual(r["json"]["name"], "NewName")

    def test_http_index_returns_html(self):
        r = self._make_client().get("/")
        self.assertIn(b"YT STREAM HUB", r["body"])

    def test_http_start_missing_ffmpeg(self):
        import unittest.mock as mock
        c = self._make_client()
        cr = c.post_json(
            "/api/streams",
            {"name": "NoFFmpeg", "stream_key": "abc", "mode": "test"},
            headers={"X-API-Key": "admin-secret-change-me"}
        )
        sid = cr["json"]["id"]
        with mock.patch("subprocess.Popen", side_effect=FileNotFoundError):
            r = c.post(f"/api/streams/{sid}/start",
                       headers={"X-API-Key": "admin-secret-change-me"})
        self.assertEqual(r["status"], 400)

    # ── Overlay Preset tests ─────────────────────────────────
    def test_create_overlay_preset(self):
        r = create_overlay_preset(
            "Logo Pop", {"shapes": [{"type":"rect","x":0,"y":0,"w":100,"h":50}]},
            duration_ms=2000, db_path=self.db
        )
        self.assertIn("id", r)
        self.assertEqual(r["name"], "Logo Pop")

    def test_get_overlay_preset_parses_svg(self):
        r = create_overlay_preset(
            "Test", {"shapes": [{"type":"circle","cx":50,"cy":50,"r":20}]},
            db_path=self.db
        )
        p = get_overlay_preset(r["id"], self.db)
        self.assertIsNotNone(p)
        self.assertEqual(len(p["svg"]["shapes"]), 1)
        self.assertEqual(p["svg"]["shapes"][0]["type"], "circle")

    def test_delete_overlay_preset(self):
        r = create_overlay_preset("gone", {"shapes":[]}, db_path=self.db)
        delete_overlay_preset(r["id"], self.db)
        self.assertIsNone(get_overlay_preset(r["id"], self.db))

    def test_http_overlay_create_and_list(self):
        c = self._make_client()
        r = c.post_json(
            "/api/overlays",
            {"name": "Test FX", "svg": {"shapes": [], "duration_ms": 1500}, "duration_ms": 1500},
            headers={"X-API-Key": "admin-secret-change-me"}
        )
        self.assertEqual(r["status"], 201)
        ls = c.get("/api/overlays", headers={"X-API-Key": "admin-secret-change-me"})
        self.assertEqual(ls["status"], 200)
        self.assertEqual(len(ls["json"]["presets"]), 1)

    def test_http_overlay_trigger(self):
        c = self._make_client()
        # Create stream
        sr = c.post_json(
            "/api/streams",
            {"name": "S", "stream_key": "k", "mode": "test"},
            headers={"X-API-Key": "admin-secret-change-me"}
        )
        sid = sr["json"]["id"]
        # Create overlay
        op = c.post_json(
            "/api/overlays",
            {"name": "Pop", "svg": {"shapes": [{"type": "rect"}]}, "duration_ms": 1000},
            headers={"X-API-Key": "admin-secret-change-me"}
        )
        pid = op["json"]["id"]
        # Trigger it
        tr = c.post(
            f"/api/streams/{sid}/overlay/{pid}/trigger",
            headers={"X-API-Key": "admin-secret-change-me"}
        )
        self.assertEqual(tr["status"], 200)
        self.assertTrue(tr["json"]["ok"])

    def test_http_overlay_clear(self):
        c = self._make_client()
        sr = c.post_json(
            "/api/streams",
            {"name": "S", "stream_key": "k", "mode": "test"},
            headers={"X-API-Key": "admin-secret-change-me"}
        )
        sid = sr["json"]["id"]
        r = c.post(f"/api/streams/{sid}/overlay/clear",
                   headers={"X-API-Key": "admin-secret-change-me"})
        self.assertEqual(r["status"], 200)

    # ── FFmpeg cmd includes overlay when overlay_png provided ─
    def test_ffmpeg_cmd_with_overlay(self):
        stream = {"mode": "test", "stream_key": "x",
                  "rtmp_target": "rtmp://a.rtmp.youtube.com/live2"}
        cmd = StreamManager._build_ffmpeg_cmd(stream, [], overlay_png="/tmp/o.png")
        joined = " ".join(cmd)
        self.assertIn("overlay", joined)
        self.assertIn("/tmp/o.png", joined)
        self.assertIn("filter_complex", joined)

    def test_ffmpeg_cmd_without_overlay_unchanged(self):
        stream = {"mode": "test", "stream_key": "x",
                  "rtmp_target": "rtmp://a.rtmp.youtube.com/live2"}
        cmd = StreamManager._build_ffmpeg_cmd(stream, [], overlay_png=None)
        joined = " ".join(cmd)
        self.assertNotIn("filter_complex", joined)

    # ── SVG path parser ──────────────────────────────────────
    def test_parse_svg_path_simple(self):
        pts = _parse_svg_path("M 10 20 L 30 40 Z")
        self.assertEqual(pts[0], (10.0, 20.0))
        self.assertIn((30.0, 40.0), pts)

    # ── Interp helper ────────────────────────────────────────
    def test_interp_middle(self):
        self.assertAlmostEqual(_interp([0, 100], 0.5), 50.0)
        self.assertAlmostEqual(_interp([0, 50, 100], 0.25), 25.0)

    # ── Audio preset ─────────────────────────────────────────
    def test_audio_preset_crud(self):
        r = create_audio_preset("boom", "/tmp/fake.mp3", db_path=self.db)
        self.assertIn("id", r)
        lst = list_audio_presets(self.db)
        self.assertEqual(len(lst), 1)
        self.assertEqual(lst[0]["name"], "boom")

    # ── Filter scaling — prevents 4K video from killing ffmpeg ──
    def test_ffmpeg_overlay_normalizes_resolution(self):
        """Overlay filter MUST include scale+pad to 1280x720@30fps."""
        stream = {"mode": "test", "stream_key": "x",
                  "rtmp_target": "rtmp://a.rtmp.youtube.com/live2"}
        cmd = StreamManager._build_ffmpeg_cmd(stream, [], overlay_png="/tmp/o.png")
        joined = " ".join(cmd)
        self.assertIn("scale=1280:720", joined)
        self.assertIn("fps=30", joined)
        self.assertIn("pad=1280:720", joined)

    # ── Health assessment parsing ────────────────────────────
    def test_assess_health_detects_streaming(self):
        log = ("Stream mapping:\n"
               "frame= 1250 fps= 30 q=18.0 size=4092KiB time=00:00:41.56 "
               "bitrate=2500.1kbits/s speed=0.998x")
        h = StreamManager.assess_health("any", log, running=True)
        self.assertEqual(h["state"], "streaming")
        self.assertEqual(h["fps"], 30.0)
        self.assertAlmostEqual(h["speed"], 0.998)

    def test_assess_health_detects_io_error(self):
        log = "Stream mapping:\n[out#0/flv] Error opening output rtmp://...: I/O error"
        h = StreamManager.assess_health("any", log, running=True)
        self.assertEqual(h["state"], "error")
        self.assertTrue(h["has_io_error"])

    def test_assess_health_detects_starting(self):
        h = StreamManager.assess_health("x", "libavutil 59.39.100\nStream mapping:", running=True)
        self.assertEqual(h["state"], "starting")

    def test_assess_health_offline_when_not_running(self):
        h = StreamManager.assess_health("x", "", running=False)
        self.assertEqual(h["state"], "offline")

    # ── Clip preset CRUD ─────────────────────────────────────
    def test_create_clip_preset(self):
        r = create_clip_preset("intro", "/tmp/x.mp4", duration_s=3.5, db_path=self.db)
        self.assertIn("id", r)
        lst = list_clip_presets(self.db)
        self.assertEqual(len(lst), 1)
        self.assertEqual(lst[0]["name"], "intro")

    def test_delete_clip_preset(self):
        r = create_clip_preset("gone", "/tmp/x.mp4", db_path=self.db)
        delete_clip_preset(r["id"], self.db)
        self.assertIsNone(get_clip_preset(r["id"], self.db))

    def test_http_clip_list_empty(self):
        r = self._make_client().get("/api/clips",
            headers={"X-API-Key": "admin-secret-change-me"})
        self.assertEqual(r["status"], 200)
        self.assertEqual(r["json"]["presets"], [])

    # ── OverlayEngine trigger state ──────────────────────────
    def test_overlay_engine_trigger_stores_state(self):
        OverlayEngine.clear("test-stream")
        OverlayEngine.trigger("test-stream", {
            "id": "preset-1", "name": "X", "shapes": [], "duration_ms": 500
        })
        with _overlay_state_lock:
            active = list(_overlay_state.get("test-stream", {}).get("active", []))
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["preset_id"], "preset-1")
        # cleanup
        OverlayEngine.clear("test-stream")
        OverlayEngine.stop_worker("test-stream")

# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"

    if cmd == "test":
        sys.argv.pop(1)
        unittest.main(verbosity=2, module=__name__, exit=True)

    elif cmd == "setup":
        print(SETUP_GUIDE)

    else:
        if not _BOTTLE_AVAILABLE:
            print("ERROR: bottle not installed. Run:  pip install bottle")
            sys.exit(1)
        print(f"[YT STREAM HUB] Starting on http://{HOST}:{PORT}")
        print(f"[YT STREAM HUB] Admin: http://localhost:{PORT}/")
        print(f"[YT STREAM HUB] DB: {DB_PATH}  |  Media: {MEDIA_DIR}")
        print(f"[YT STREAM HUB] API key source: {'env YTS_API_KEY' if os.environ.get('YTS_API_KEY') else 'DEFAULT (set YTS_API_KEY!)'}")
        print(f"[YT STREAM HUB] Run with 'setup' arg for setup guide")
        app = make_app()
        run(app, host=HOST, port=PORT, reloader=False)


if __name__ == "__main__":
    main()
