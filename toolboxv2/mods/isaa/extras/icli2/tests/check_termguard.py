"""termguard.log_to_file: Terminal-Handler werden fuer die Laufzeit umgelenkt und danach wiederhergestellt."""
import io, logging, sys, tempfile
from pathlib import Path
import stub  # noqa: F401
from toolboxv2.mods.isaa.extras.icli2.termguard import log_to_file

tmp = Path(tempfile.mkdtemp()) / "sub" / "icli2.log"
term = logging.StreamHandler(sys.stderr)
other = io.StringIO(); keep = logging.StreamHandler(other)
filed = logging.FileHandler(Path(tempfile.mkdtemp()) / "eigen.log")
lg = logging.getLogger("icli2.termguard.test"); lg.propagate = False; lg.setLevel(logging.INFO)
for h in (term, keep, filed): lg.addHandler(h)

with log_to_file(tmp) as moved:
    assert term in moved and keep not in moved and filed not in moved, moved   # nur Terminal-Handler
    lg.warning("waehrend der TUI")
    assert term.stream is not sys.stderr and keep.stream is other
assert term.stream is sys.stderr and keep.stream is other  # wiederhergestellt
assert "waehrend der TUI" in tmp.read_text() and "waehrend der TUI" in other.getvalue()
for h in (term, keep, filed): lg.removeHandler(h)
filed.close()
print("termguard ok")
