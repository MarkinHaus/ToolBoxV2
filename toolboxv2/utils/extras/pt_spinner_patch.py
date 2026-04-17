# toolboxv2/utils/extras/pt_spinner_patch.py

import threading
import time
from typing import Optional
from prompt_toolkit.application import Application

_PT_SPINNER_STATE: dict = {"active": False, "line": ""}
_STATE_LOCK = threading.Lock()
_APP_REF: Optional[Application] = None


def register_app(app: Application) -> None:
    global _APP_REF
    _APP_REF = app


def get_spinner_toolbar_fragment() -> str:
    with _STATE_LOCK:
        if _PT_SPINNER_STATE["active"]:
            return _PT_SPINNER_STATE["line"]
    return ""


def _set_state(active: bool, line: str = "") -> None:
    with _STATE_LOCK:
        _PT_SPINNER_STATE["active"] = active
        _PT_SPINNER_STATE["line"]   = line
    _try_invalidate()


def _try_invalidate() -> None:
    app = _APP_REF
    if app is not None:
        try:
            app.invalidate()
        except Exception:
            pass


def apply_prompt_toolkit_patch_safe() -> None:
    try:
        from toolboxv2.utils.extras.Style import Spinner, SpinnerManager
    except ImportError as e:
        print(f"[pt_spinner_patch] Import failed: {e}")
        return

    if getattr(SpinnerManager, "_pt_patched", False):
        return

    # ── _render_loop: kein stdout mehr ───────────────────────────────────────
    def _pt_render_loop(self: SpinnerManager) -> None:
        while self._should_run:
            with self._lock:
                if not self._spinners:
                    self._should_run = False
                    _set_state(False)
                    break

                primary = next(
                    (s for s in self._spinners if s._is_primary), None
                )
                if primary and primary.running:
                    line = primary._generate_render_line()
                    if len(self._spinners) > 1:
                        secondary = " | ".join(
                            s._generate_secondary_info()
                            for s in self._spinners
                            if s is not primary and s.running
                        )
                        line += f"  [{secondary}]"
                    _set_state(True, line)

            time.sleep(0.08)

        _set_state(False)

    # ── _signal_handler: kein stdout mehr ────────────────────────────────────
    import sys, signal as _signal

    def _pt_signal_handler(self: SpinnerManager, signum, frame) -> None:
        with self._lock:
            for spinner in self._spinners:
                spinner.running = False
            self._spinners.clear()
        self._should_run = False
        _set_state(False)   # Toolbar leeren statt \r\033[K
        #sys.exit(0)

    # ── __exit__: kein stdout mehr ───────────────────────────────────────────
    def _pt_spinner_exit(self: Spinner, exc_type, exc_val, exc_tb) -> None:
        self.running = False
        self.manager.unregister_spinner(self)
        if self._is_primary:
            _set_state(False)   # statt \r\033[K

    # ── __aenter__ / __aexit__ für async with ────────────────────────────────
    async def _pt_spinner_aenter(self: Spinner):
        return self.__enter__()

    async def _pt_spinner_aexit(self: Spinner, exc_type, exc_val, exc_tb) -> None:
        _pt_spinner_exit(self, exc_type, exc_val, exc_tb)

    # ── Patch anwenden ────────────────────────────────────────────────────────
    SpinnerManager._render_loop    = _pt_render_loop
    SpinnerManager._signal_handler = _pt_signal_handler
    Spinner.__exit__               = _pt_spinner_exit
    Spinner.__aenter__             = _pt_spinner_aenter
    Spinner.__aexit__              = _pt_spinner_aexit
    SpinnerManager._pt_patched     = True
    print("[pt_spinner_patch] Patch applied.")
