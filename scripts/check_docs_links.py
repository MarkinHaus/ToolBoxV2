#!/usr/bin/env python3
"""Docs-Link-Gate: bricht bei toten Links/nav-Referenzen im mkdocs-Build.
Kein --strict, weil strict auch an ~137 griffe/autorefs-Warnings aus
Code-Docstrings und 2 AliasResolutionErrors (TruthSeeker/arXivCrawler.py:22,
isaa/CodingAgent/coder.py:36) abbricht — Code-Probleme, nicht Docs-Probleme."""
import re, subprocess, sys
log = subprocess.run(["mkdocs","build","--clean"], capture_output=True, text=True).stderr
bad = re.findall(r"^(?:WARNING|ERROR)\s+-\s+(Doc file .*|A reference to .*)$", log, re.M)
if bad:
    print(f"::error::{len(bad)} tote Links / nav-Referenzen")
    [print(" ", b) for b in bad]
    sys.exit(1)
print("Docs-Links OK.")
