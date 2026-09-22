# ORCHESTRATOR-HANDOFF — Rust Agent Core Harness

Stand: nach Ticket 2 (committet). Dieser Kurszustand ist minimal nötig, um jeden
weiteren Ticket-Lauf ohne Kontextverlust fortzusetzen.

## Projekt

- Ziel: „Functional Core, Imperative Async Shell" Agent-Harness in Rust.
  Master-Spec: 8 Tickets, sequentiell, jedes Ticket = Gate vor dem nächsten.
- Workspace-Root (Cargo): `mods/isaa/core/` in diesem Repo (= `C:/Users/Markin/Workspace/ToolBoxV2/toolboxv2/mods/isaa/core`)
- Crates: `crates/core` (isaa-core), `crates/tui` (isaa-tui, Stub), `crates/harness` (isaa-harness, Stub)
- Referenz-Altcode (NICHT anfassen, NICHT committen): `isaa_agent_ref/` (Python)

## Git

- Repo-Root ist `C:/Users/Markin/Workspace/ToolBoxV2` (Branch `master`) — NICHT der Cargo-Root!
- **Pflicht**: Commits gescoped aus dem Cargo-Root heraus:
  ```bash
  git add Cargo.toml Cargo.lock crates .handoff/<spec>.md .gitignore
  git commit -m "feat(core): complete ticket X - <kurz>"
  ```
  Niemals `git add .` am Repo-Root (fremde Dateien: backup_installers/, docs/, logs/ ...).
- `.gitignore` (im Cargo-Root): `target/`, `.handoff/logs/`
- Commits bisher:
  - `4034cb48` feat(core): ticket 1 - domain, pure reducer, determinism proptests
  - `249f25d5` feat(core): complete ticket 2 - WAL with CRC32 framing, postcard checkpoints, tail recovery

## Sub-Agent-Dispatch (erprobt, funktioniert)

```bash
timeout 840 pi --provider 9router --model fast --no-session -p "$(cat .handoff/ticket_N_spec.md)" > .handoff/logs/tN.log 2>&1
echo "EXIT=$?"
```

- **Wichtig**: `--provider 9router --model fast` ist PFLICHT. Nacktes `pi -p` → 401
  (pi-9router-Ext lädt headless nicht: „ctx stale", pi fällt auf Direct-Provider zurück).
- `claude -p` hängt auf diesem System (Timeout, kein Output) — nicht nutzen.
- Logs: `.handoff/logs/tN.log`. Specs: `.handoff/ticket_N_spec.md` (müssen in sich
  geschlossen sein: Sub-Agent kennt den Chat nicht).

## Gatekeeper-Prozedur (nach jedem Worker-Lauf, Worte des Workers gelten nicht)

1. `cargo test --workspace` → Exit 0
2. `cargo clippy --workspace --all-targets -- -D warnings` → Exit 0
3. `grep -nE "cfg\(windows\)|cfg\(unix\)|std::os|libc|winapi" crates/core/src/*.rs` → nur `#[cfg(test)]` erlaubt
4. `git status --short -- .` + `git diff -w` → Worker darf nur vereinbarte Dateien ändern
5. Erst dann gescopeter Commit (s. o.)

Bei rotem Gate: Compiler-/Test-Fehler-Auszug an Worker zurückgeben (gleicher Dispatch-Befehl, Spec + Fehlerlog als Prompt).

## System-State nach Ticket 2

- domain.rs: Domain-Modelle, `State::hash()` (postcard+BLAKE3), `reduce()` pure.
  Achtung: `im::OrdMap` statt `im::HashMap` (HashMap seedet prozess-random → bricht
  State-Hash-Determinismus; NICHT zurückändern).
- wal.rs: Frame `[u8 marker][u32 len LE][u32 crc32 LE][payload]`, Marker 0x01 Event /
  0x02 Checkpoint `{version, state}`, Batch-Flush alle 64 Events, `recover()` verwirft
  Torn-Tail still, `replay()` via purem Reducer.
- Tests: `crates/core/tests/determinism.rs` (3 Proptests), WAL-Tests in wal.rs (4).
- Ticket 3–8 offen: 3 Workspace-Jail+4 Core-Tools, 4 WASM+Process-Runner,
  5 BM25++, 6 S1/S2-Dispatcher, 7 Telemetry+DPO-Harvester, 8 TUI.

## Bekannte Fallstricke

1. Worker lief `cargo fmt` workspace-weit → Reformattings in fremden Dateien.
   In jede Spec: „`cargo fmt` NUR auf neu angelegten Dateien, bestehende nicht anfassen."
2. `git add .` am Repo-Root ist verboten (fremde Files im Repo, s. o.).
3. Worker-Berichte sind geschönt — immer selbst gate-keepern.
4. Bash-Tool-Timeout beim Dispatch ≥ 900s setzen; Worker-Timeout 840s (User-Vorgabe: Timeout gegen Hängen).
5. TUI (Ticket 8) parallel im Worktree bauen, erst wenn Ticket 3 läuft:
   ```bash
   git worktree add ../isaa-tui -b feat/tui
   ```
   Merge am Ende: `git merge feat/tui` (trennte Crates, keine Konflikte erwartet).
   Kein verschachteltes cargo: erst verifizieren, wenn der jeweilige andere Prozess beendet ist.

## Nächster Schritt

Ticket 3: Spec `.handoff/ticket_3_spec.md` schreiben (fs_read/fs_write/fs_patch/
proc_exec, Path-Kanonisierung, SecurityError bei Path-Traversal, proc ohne Shell,
Tests für `../../etc/passwd`-Abweisung), dispatchen, gaten, committen.
