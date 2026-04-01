# toolboxv2/utils/extras/pt_spinner_patch.py

import threading
import time
from prompt_toolkit.application import get_app

# ── Shared state ─────────────────────────────────────────────────────────────
_PT_SPINNER_STATE: dict = {
    "active": False,
    "line":   "",
}
_STATE_LOCK = threading.Lock()


def get_spinner_toolbar_fragment() -> str:
    """Gibt den aktuellen Spinner-Text zurück (leer wenn kein Spinner aktiv)."""
    with _STATE_LOCK:
        if _PT_SPINNER_STATE["active"]:
            return _PT_SPINNER_STATE["line"]
    return ""


def _set_state(active: bool, line: str = "") -> None:
    with _STATE_LOCK:
        _PT_SPINNER_STATE["active"] = active
        _PT_SPINNER_STATE["line"] = line
    _try_invalidate()


def _try_invalidate() -> None:
    """Trigger prompt_toolkit re-render ohne Exception wenn kein App aktiv."""
    try:
        get_app().invalidate()
    except Exception:
        pass


# ── Patch ─────────────────────────────────────────────────────────────────────
def apply_prompt_toolkit_patch_safe() -> None:
    """
    Patcht SpinnerManager + Spinner so dass:
      - kein direktes stdout-Writing mehr passiert
      - Spinner-State in _PT_SPINNER_STATE landet
      - prompt_toolkit via invalidate() neu rendert
    Idempotent – mehrfaches Aufrufen ist safe.
    """
    try:
        from toolboxv2.utils.extras.Style import Spinner, SpinnerManager
    except ImportError:
        return

    # Bereits gepatcht?
    if getattr(SpinnerManager, "_pt_patched", False):
        return

    # ── SpinnerManager._render_loop ──────────────────────────────────────────
    def _pt_render_loop(self: SpinnerManager) -> None:  # type: ignore[override]
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

            time.sleep(0.08)  # etwas flüssiger als original 0.1

        _set_state(False)

    # ── Spinner.__exit__ ─────────────────────────────────────────────────────
    def _pt_spinner_exit(
        self: Spinner,  # type: ignore[override]
        exc_type,
        exc_value,
        exc_traceback,
    ) -> None:
        self.running = False
        self.manager.unregister_spinner(self)
        if self._is_primary:
            _set_state(False)   # kein \r\033[K mehr

    SpinnerManager._render_loop = _pt_render_loop   # type: ignore[method-assign]
    Spinner.__exit__ = _pt_spinner_exit              # type: ignore[method-assign]
    SpinnerManager._pt_patched = True                # type: ignore[attr-defined]
