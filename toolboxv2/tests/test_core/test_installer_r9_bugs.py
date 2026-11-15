import re, sys, unittest
from pathlib import Path

PS1 = Path(__file__).resolve().parents[3] / "installer.ps1"


class TestInstallerBugfixes(unittest.TestCase):
    def test_line444_guarded_against_strictmode(self):
        # R9-BUG1: Zeile ~444 liest $script:tbFeaturesCfg, das nur im
        # Config-Zweig (285) initialisiert wird -> StrictMode-Crash ohne Config.
        src = PS1.read_text(encoding="utf-8")
        m = re.search(r'if \(\$AssumeYes -eq "yes" -and -not \$script:tbFeaturesCfg\)', src)
        self.assertIsNotNone(m, "Guard-Zeile fehlt/veraendert")
        # Init muss VOR dem ersten Les existieren (top-level oder Funktionskopf)
        first_read = m.start()
        init = src.find("$script:tbFeaturesCfg = \"\"")
        self.assertGreaterEqual(init, 0, "keine Init-Zuweisung gefunden")
        # Init vor dem Guard ODER im Config-Zweig + top-level Default:
        top_default = re.search(r'^\$script:tbFeaturesCfg\s*=\s*""', src, re.M)
        self.assertIsNotNone(top_default, "Default-Init fehlt auf Top-Level (Spalte 0)")

    def test_explicit_path_wins_over_existing(self):
        # R9-BUG2: -Path <dir> + assume=yes + existing installs ->
        # INSTALL_PATH darf nicht still auf foundInstalls[0] umgebogen werden.
        src = PS1.read_text(encoding="utf-8")
        block = re.search(
            r'if \(\$ACTION -eq "install"\) \{(.*?)\n    \}', src, re.S)
        self.assertIsNotNone(block, "install/update-Block nicht gefunden")
        b = block.group(1)
        self.assertIn(
            "$INSTALL_PATH",
            b,
            "INSTALL_PATH-Referenz fehlt im Block",
        )
        self.assertRegex(
            b,
            r'if \(\$INSTALL_PATH\)',
            "Explicit-Path-Guard fehlt im assume=yes-Zweig",
        )


if __name__ == "__main__":
    unittest.main()
