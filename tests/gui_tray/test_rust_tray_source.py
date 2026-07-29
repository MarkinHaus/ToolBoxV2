"""Statische Invarianten für simple-core/src-tauri/src/lib.rs.

Ein echter ``cargo build`` braucht webkit2gtk + Netzwerk und läuft in CI/Sandbox
nicht. Diese Tests sichern stattdessen genau die Eigenschaften ab, die zuletzt
kaputt waren: Deklarationsreihenfolge im Tray-Menü, Tray-ID, Close-Verhalten
und die konfigurierbare Tray-Basis-URL.
"""
from __future__ import annotations

import re


def _index(src: str, needle: str) -> int:
    pos = src.find(needle)
    assert pos != -1, f"nicht gefunden: {needle!r}"
    return pos


def test_menu_wird_erst_nach_allen_items_gebaut(rust_lib_source):
    """`Menu::with_items` darf erst kommen, wenn separator und quit gebunden sind."""
    src = rust_lib_source
    menu_pos = _index(src, "let menu = Menu::with_items")
    assert _index(src, 'let separator = MenuItem::with_id') < menu_pos
    assert _index(src, 'let quit = MenuItem::with_id') < menu_pos


def test_nur_ein_menu_binding(rust_lib_source):
    assert rust_lib_source.count("let menu = Menu::with_items") == 1


def test_open_cli_item_haengt_im_menue(rust_lib_source):
    """Der Terminal-CLI-Eintrag hat einen Handler - er muss auch im Menü hängen."""
    src = rust_lib_source
    block_start = _index(src, "let menu = Menu::with_items")
    block = src[block_start : src.index("?;", block_start)]
    for item in ("open_app", "app_mode", "hud_mode", "open_cli", "separator", "quit"):
        assert f"&{item}" in block, f"{item} fehlt im Tray-Menü"


def test_tray_hat_id_main(rust_lib_source):
    """apply_sse_frame greift per tray_by_id("main") zu -> ID muss gesetzt sein."""
    assert 'TrayIconBuilder::with_id("main")' in rust_lib_source
    assert 'tray_by_id("main")' in rust_lib_source


def test_close_stoppt_worker_nur_beim_hauptfenster(rust_lib_source):
    src = rust_lib_source
    handler = src[_index(src, ".on_window_event(") :]
    handler = handler[: handler.find(".run(tauri::generate_context")]
    assert 'window.label() != "main"' in handler, "HUD-Close darf den Worker nicht killen"


def test_tray_urls_kommen_aus_dem_endpoint_helper(rust_lib_source):
    """Rust und Python müssen dieselbe Quelle nutzen (local_ui.json), keinen Hardcode."""
    src = rust_lib_source
    assert "worker_manager::local_ui_url()" in src
    assert '"http://127.0.0.1:5000/tray/state"' not in src
    assert '"http://127.0.0.1:5000"' not in src


def test_tray_setup_paniced_nicht_ohne_icon(rust_lib_source):
    assert "default_window_icon().unwrap()" not in rust_lib_source
    assert "default_window_icon()\n        .cloned()\n        .ok_or(" in rust_lib_source
