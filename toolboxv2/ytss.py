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
DB_PATH   = os.environ.get("YTS_DB", "../streams.db")
MEDIA_DIR = Path(os.environ.get("YTS_MEDIA", "../media"))
HOST      = os.environ.get("YTS_HOST", "0.0.0.0")
PORT      = int(os.environ.get("YTS_PORT", "8080"))

YT_RTMP_BASE = "rtmp://a.rtmp.youtube.com/live2"

# ─────────────────────────────────────────────────────────────
#  DATABASE
# ─────────────────────────────────────────────────────────────
def get_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(path=DB_PATH):
    with get_db(path) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS streams (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            description TEXT DEFAULT '',
            stream_key  TEXT NOT NULL,
            rtmp_target TEXT NOT NULL,
            status      TEXT DEFAULT 'stopped',
            mode        TEXT DEFAULT 'file',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            pid         INTEGER DEFAULT NULL,
            error_msg   TEXT DEFAULT ''
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
        """)
    # bootstrap a default admin API key if none exist
    with get_db(path) as db:
        if not db.execute("SELECT 1 FROM api_keys LIMIT 1").fetchone():
            db.execute(
                "INSERT INTO api_keys VALUES (?,?,?)",
                ("admin-secret-change-me", "default", _now())
            )
            db.commit()

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
    def _build_ffmpeg_cmd(stream: dict, files: list[dict]) -> list[str]:
        """
        Build FFmpeg command.
        mode='file'  → loop a playlist of uploaded video/audio files
        mode='api'   → read from stdin pipe (caller pushes raw h264)
        mode='test'  → generate a colour-bars test card (no input needed)
        """
        target = f"{stream['rtmp_target']}/{stream['stream_key']}"
        common_out = [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-g", "50",
            "-b:v", "2500k",
            "-maxrate", "2500k",
            "-bufsize", "5000k",
            "-c:a", "aac",
            "-ar", "44100",
            "-b:a", "128k",
            "-f", "flv",
            target
        ]

        mode = stream.get("mode", "file")

        if mode == "test":
            return [
                "ffmpeg", "-re",
                "-f", "lavfi", "-i", "testsrc=duration=86400:size=1280x720:rate=30",
                "-f", "lavfi", "-i", "sine=frequency=1000:duration=86400",
                *common_out
            ]

        if mode == "api":
            return [
                "ffmpeg",
                "-f", "h264", "-i", "pipe:0",
                "-f", "s16le", "-ar", "44100", "-ac", "2", "-i", "/dev/zero",
                *common_out
            ]

        # mode == 'file' — build concat playlist
        video_files = [f for f in files if f["filetype"].startswith("video")]
        audio_files = [f for f in files if f["filetype"].startswith("audio")]

        if not video_files and not audio_files:
            # fallback to test card when no media uploaded
            return StreamManager._build_ffmpeg_cmd({**stream, "mode": "test"}, [])

        if video_files:
            # concat all video files in a loop
            file_list = "|".join(f["filepath"] for f in video_files)
            input_spec = ["-stream_loop", "-1",
                          "-i", f"concat:{file_list}"]
        else:
            # audio-only → silent video + audio loop
            file_list = "|".join(f["filepath"] for f in audio_files)
            input_spec = [
                "-f", "lavfi", "-i", "color=c=black:size=1280x720:rate=30",
                "-stream_loop", "-1", "-i", f"concat:{file_list}",
            ]

        return ["ffmpeg", "-re", *input_spec, *common_out]

    @staticmethod
    def start(stream_id: str, db_path=DB_PATH) -> dict:
        with get_db(db_path) as db:
            row = db.execute("SELECT * FROM streams WHERE id=?", (stream_id,)).fetchone()
            if not row:
                return {"ok": False, "error": "Stream not found"}
            stream = dict(row)
            files = [dict(r) for r in db.execute(
                "SELECT * FROM media_files WHERE stream_id=? ORDER BY created_at",
                (stream_id,)
            ).fetchall()]

        with _process_lock:
            if stream_id in _processes and _processes[stream_id].poll() is None:
                return {"ok": False, "error": "Already running"}

            cmd = StreamManager._build_ffmpeg_cmd(stream, files)
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE if stream["mode"] == "api" else subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                _processes[stream_id] = proc
            except FileNotFoundError:
                return {"ok": False, "error": "ffmpeg not found — run setup first"}

        with get_db(db_path) as db:
            db.execute(
                "UPDATE streams SET status='live', pid=?, updated_at=? WHERE id=?",
                (proc.pid, _now(), stream_id)
            )
            db.commit()

        # background watcher — resets status when ffmpeg exits
        def _watch():
            proc.wait()
            with _process_lock:
                _processes.pop(stream_id, None)
            stderr_output = ""
            if proc.stderr:
                try:
                    stderr_output = proc.stderr.read(2000).decode(errors="replace")
                except Exception:
                    pass
            with get_db(db_path) as db2:
                db2.execute(
                    "UPDATE streams SET status='stopped', pid=NULL, error_msg=?, updated_at=? WHERE id=?",
                    (stderr_output[-500:] if stderr_output else "", _now(), stream_id)
                )
                db2.commit()

        threading.Thread(target=_watch, daemon=True).start()
        return {"ok": True, "pid": proc.pid, "cmd": cmd}

    @staticmethod
    def stop(stream_id: str, db_path=DB_PATH) -> dict:
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
            proc.terminate()
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
#  CRUD helpers
# ─────────────────────────────────────────────────────────────
def _stream_row_to_dict(row) -> dict:
    d = dict(row)
    d.pop("stream_key", None)  # don't expose key in list endpoints
    return d

def create_stream(name, description="", stream_key="", rtmp_target=YT_RTMP_BASE,
                  mode="file", db_path=DB_PATH) -> dict:
    sid = str(uuid.uuid4())
    now = _now()
    with get_db(db_path) as db:
        db.execute(
            "INSERT INTO streams VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (sid, name, description, stream_key, rtmp_target,
             "stopped", mode, now, now, None, "")
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
    allowed = {"name", "description", "stream_key", "rtmp_target", "mode"}
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
  .logo-dot {
    width: 10px; height: 10px;
    background: var(--cut1);
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.4); opacity: 0.7; }
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
  .status-live   { background: var(--cut1); }
  .status-stopped { background: var(--cream); }
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
        <div class="field-note">Get it from YouTube Studio → Go Live → Stream Setup</div>
      </div>
      <div>
        <label>RTMP TARGET</label>
        <input type="text" id="fRtmp" value="rtmp://a.rtmp.youtube.com/live2">
        <div class="field-note">Default is YouTube. Change for Twitch, custom RTMP, etc.</div>
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

<script>
const API = '/api';
let _currentMediaStreamId = null;

// ── API helpers ──────────────────────────────────────────────
const apiKey = () => {
  let k = localStorage.getItem('apiKey');
  if (!k) {
    k = prompt('Enter API Key (default: admin-secret-change-me):', 'admin-secret-change-me');
    if (k) localStorage.setItem('apiKey', k);
  }
  return k || 'admin-secret-change-me';
};

async function apiFetch(path, opts = {}) {
  const headers = { 'X-API-Key': apiKey(), ...(opts.headers || {}) };
  if (!(opts.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }
  const res = await fetch(API + path, { ...opts, headers });
  let data;
  try { data = await res.json(); } catch { data = {}; }
  if (!res.ok) throw new Error(data.error || 'Request failed');
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
    return `<div class="card" id="card-${s.id}">
      <div class="card-head">
        <div class="card-title">${escHtml(s.name)}</div>
        <div class="status-badge ${isLive ? 'status-live' : 'status-stopped'}">
          ${isLive ? '⬤ LIVE' : '◯ STOPPED'}
        </div>
      </div>
      <div class="card-body">
        <div class="card-desc">${escHtml(s.description || '—')}</div>
        <div class="meta-row">
          <span class="tag tag-mode">${modeLabel}</span>
          <span class="tag">${created}</span>
          ${s.pid ? `<span class="tag">PID ${s.pid}</span>` : ''}
        </div>
        ${errorHtml}
        <div class="card-actions">
          ${isLive
            ? `<button class="btn btn-live btn-sm" onclick="stopStream('${s.id}')">■ STOP</button>`
            : `<button class="btn btn-primary btn-sm" onclick="startStream('${s.id}')">▶ START</button>`
          }
          <button class="btn btn-sm" onclick="openEditModal(${JSON.stringify(s)})">✎ EDIT</button>
          <button class="btn btn-sm" onclick="openMediaModal('${s.id}', '${escHtml(s.name)}')">⊕ MEDIA</button>
          <button class="btn btn-danger btn-sm" onclick="deleteStream('${s.id}')">✕</button>
        </div>
      </div>
    </div>`;
  }).join('');
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
  document.getElementById('streamModal').classList.add('open');
}

function openEditModal(s) {
  document.getElementById('modalTitle').textContent = 'EDIT STREAM';
  document.getElementById('editStreamId').value = s.id;
  document.getElementById('fName').value = s.name || '';
  document.getElementById('fDesc').value = s.description || '';
  document.getElementById('fKey').value = ''; // don't pre-fill key
  document.getElementById('fRtmp').value = s.rtmp_target || 'rtmp://a.rtmp.youtube.com/live2';
  document.getElementById('fMode').value = s.mode || 'file';
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
refreshAll();
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────
#  AUTH MIDDLEWARE
# ─────────────────────────────────────────────────────────────
def require_api_key(db_path=DB_PATH):
    """Check X-API-Key header. Returns None if OK, else abort."""
    if not _BOTTLE_AVAILABLE:
        return
    key = request.headers.get("X-API-Key", "")
    if not key:
        key = request.query.get("api_key", "")
    with get_db(db_path) as db:
        row = db.execute("SELECT 1 FROM api_keys WHERE key=?", (key,)).fetchone()
    if not row:
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
        return json_resp({"streams": list_streams(db_path)})

    @app.route("/api/streams", method="POST")
    def api_streams_create():
        require_api_key(db_path)
        data = request.json or {}
        name = data.get("name", "").strip()
        if not name:
            raise HTTPError(400, "name is required")
        result = create_stream(
            name=name,
            description=data.get("description", ""),
            stream_key=data.get("stream_key", ""),
            rtmp_target=data.get("rtmp_target", YT_RTMP_BASE),
            mode=data.get("mode", "file"),
            db_path=db_path,
        )
        return json_resp(result, 201)

    # ── SINGLE STREAM ─────────────────────────────────────────
    @app.route("/api/streams/<sid>", method="GET")
    def api_stream_get(sid):
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
    @app.route("/health")
    def health():
        return json_resp({"status": "ok", "time": _now()})

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
    def test_default_api_key_created(self):
        with get_db(self.db) as db:
            row = db.execute("SELECT key FROM api_keys LIMIT 1").fetchone()
        self.assertIsNotNone(row)

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
        self.assertIn("/tmp/a.mp4", joined)

    def test_ffmpeg_cmd_file_mode_audio_only(self):
        stream = {"mode": "file", "stream_key": "mykey",
                  "rtmp_target": "rtmp://a.rtmp.youtube.com/live2"}
        files = [{"filepath": "/tmp/track.mp3", "filetype": "audio/mpeg"}]
        cmd = StreamManager._build_ffmpeg_cmd(stream, files)
        joined = " ".join(cmd)
        self.assertIn("concat", joined)
        # should add black video for audio-only
        self.assertIn("color=c=black", joined)

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
        r = self._make_client().get("/health")
        self.assertEqual(r["status"], 200)
        self.assertEqual(r["json"]["status"], "ok")

    def test_http_list_streams_empty(self):
        r = self._make_client().get("/api/streams")
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
        r = c.get(f"/api/streams/{sid}")
        self.assertEqual(r["status"], 200)
        self.assertEqual(r["json"]["name"], "GetMe")

    def test_http_get_stream_404(self):
        r = self._make_client().get("/api/streams/no-such-id")
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
        gr = c.get(f"/api/streams/{sid}")
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
        r = c.get(f"/api/streams/{sid}")
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

# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"

    if cmd == "test":
        # remove "test" from argv so unittest doesn't choke
        sys.argv.pop(1)
        unittest.main(verbosity=2)

    elif cmd == "setup":
        print(SETUP_GUIDE)

    else:
        if not _BOTTLE_AVAILABLE:
            print("ERROR: bottle not installed. Run:  pip install bottle")
            sys.exit(1)
        print(f"[YT STREAM HUB] Starting on http://{HOST}:{PORT}")
        print(f"[YT STREAM HUB] Admin: http://localhost:{PORT}/")
        print(f"[YT STREAM HUB] DB: {DB_PATH}  |  Media: {MEDIA_DIR}")
        print(f"[YT STREAM HUB] Run with 'setup' arg for setup guide")
        app = make_app()
        run(app, host=HOST, port=PORT, reloader=False)

if __name__ == "__main__":
    main()
