"""termguard.quiet_terminal: Terminal-Handler schweigen waehrend der TUI, alle anderen Handler
(Datei des ToolBox-Loggers usw.) bekommen jede Meldung; danach ist alles wie vorher."""
import io, logging, sys, tempfile
from pathlib import Path
import stub  # noqa: F401
from toolboxv2.mods.isaa.extras.icli2.termguard import quiet_terminal

term_out = io.StringIO()
term = logging.StreamHandler(sys.stderr); term.stream = sys.stderr
other = io.StringIO(); keep = logging.StreamHandler(other)
path = Path(tempfile.mkdtemp()) / "tb.log"; filed = logging.FileHandler(path)
last = logging.lastResort                                # Fall aus Windows: _StderrHandler, stream ohne Setter
lg = logging.getLogger("icli2.termguard.test"); lg.propagate = False; lg.setLevel(logging.INFO)
for h in (term, keep, filed, last): lg.addHandler(h)
seen = []
term.emit = lambda record: seen.append(record.getMessage())        # was im Terminal landen wuerde
term.handle = lambda record, _h=term.handle: _h(record)

with quiet_terminal() as muted:
    assert muted == [term], muted                         # nur der Terminal-Handler
    lg.warning("waehrend der TUI")
assert not seen, seen                                     # nichts ins Terminal
filed.flush()
assert "waehrend der TUI" in path.read_text() and "waehrend der TUI" in other.getvalue()
lg.warning("danach")
assert seen == ["danach"], seen                           # wiederhergestellt
for h in (term, keep, filed, last): lg.removeHandler(h)
filed.close()
print("termguard ok")
