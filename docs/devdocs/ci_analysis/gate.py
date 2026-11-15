#!/usr/bin/env python
"""Gate for WP2 fix orders: no red test ID outside catalog.csv.

Usage (repo root, Windows or POSIX):

    python docs/devdocs/ci_analysis/gate.py                 # run the whole suite
    python docs/devdocs/ci_analysis/gate.py toolboxv2/tests/test_clis   # a subset
    python docs/devdocs/ci_analysis/gate.py --card K-MISS-01            # status of one card's IDs

Exit code 0: no new red ID. Exit code 1: at least one red ID that is not in
catalog.csv (a regression), listed on stdout.
"""
import csv
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_catalog():
    rows = {}
    with open(os.path.join(HERE, "catalog.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=";"):
            rows[r["test_id"]] = r
    return rows


def main(argv):
    card = None
    if "--card" in argv:
        i = argv.index("--card")
        card = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    catalog = load_catalog()
    targets = argv or ["toolboxv2/tests/"]
    if card and not argv:
        targets = sorted({t.split("::")[0] for t, r in catalog.items() if r["karte"] == card})
    env = dict(os.environ, MOCK_EXTERNAL_SERVICES="true", IS_OFFLINE_DB="true")
    cmd = [sys.executable, "-m", "pytest", *targets, "-q", "-rfE", "--tb=no", "-p", "no:cacheprovider"]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    red = set()
    for line in out.splitlines():
        m = re.match(r"^(FAILED|ERROR) (\S+)", line)
        if m:
            red.add(m.group(2).replace("\\", "/"))
    tail = [l for l in out.splitlines() if re.search(r"\d+ (passed|failed|error)", l)]
    print(tail[-1] if tail else out[-400:])
    new = sorted(red - set(catalog))
    if card:
        ids = sorted(t for t, r in catalog.items() if r["karte"] == card)
        still = [t for t in ids if t in red]
        print(f"Karte {card}: {len(ids) - len(still)} von {len(ids)} IDs gruen")
        for t in still:
            print("  noch rot:", t)
    print(f"rote IDs: {len(red)}, davon im Katalog: {len(red & set(catalog))}")
    print(f"neue rote IDs: {len(new)}")
    for t in new:
        print("  NEU:", t)
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
