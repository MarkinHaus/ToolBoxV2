# TICKET 3: Workspace Jail & 4 Primitive Core Tools (Pi-Style)

Branch `master`. Ticket 1 (domain/reducer) und Ticket 2 (wal) sind fertig und
committet. Du ergänzt NUR Ticket 3. Bestehende Module (domain.rs, reducer.rs,
wal.rs, bestehende Tests) werden NICHT verändert — `crates/core/src/lib.rs` darf
nur um `pub mod tools;` erweitert werden.

## Ziel

Sicherheits-Isolation (`WorkspaceJail`) und 4 primitive, Pi-inspirierte Core-Tools
in `crates/core`:

1. `fs_read` (Bounded File Chunks)
2. `fs_write` (Atomares Schreiben)
3. `fs_patch` (Deterministisches Unified-Diff/Hunk-Patching)
4. `proc_exec` (Direkte Prozess-Ausführung ohne Zwischen-Shell)

Neue Dateien:
- `crates/core/src/tools/mod.rs` (Re-Exports)
- `crates/core/src/tools/error.rs`
- `crates/core/src/tools/jail.rs`
- `crates/core/src/tools/core_pi.rs`
- `crates/core/tests/tools_jail.rs`

## 1. Architektur & Sicherheits-Invarianten

### A. Das Workspace Jail (`jail.rs`)
Jeder Dateizugriff MUSS strikt auf `WorkspacePolicy.root` begrenzt sein.
- **Kanonisierung**: Root und Zielpfad via `std::fs::canonicalize` kanonisieren.
  (`dunce` o. ä. ist KEINE erlaubte Dependency — siehe Regeln.)
- **Path-Traversal-Schutz**: Zugriffe via `../`, relative Tricks, absolute
  Fremdpfade oder Symlink-Breakouts MÜSSEN mit
  `ToolError::SecurityViolation(String)` abgebrochen werden.
- **Root-Invariante**: `target_canonical.starts_with(&root_canonical)` muss
  zwingend `true` sein. Existiert die Zieldatei noch nicht (z. B. bei
  `fs_write`), wird der nächstliegende existierende Parent-Ordner kanonisiert
  und geprüft.
- API-Vorschlag: `jail.resolve(&WorkspacePolicy, &Path) -> Result<PathBuf, ToolError>`.

### B. Die 4 Core-Tools (`core_pi.rs`)
Alle Tools arbeiten nicht-blockierend (Tokio) und liefern strukturierte
Ergebnisse oder `ToolError`:

1. `fs_read(path: &Path, offset: Option<usize>, limit: Option<usize>, policy) -> Result<String, ToolError>`
   - Liest Textdateien sicher ein (über Jail).
   - `limit` begrenzt Zeilen ab `offset` (Zeilen-Semantik, 0-basiert; Schutz vor
     Context-Explosion).
   - Ungültiges UTF-8 → `ToolError::InvalidEncoding`.

2. `fs_write(path: &Path, content: &str, policy) -> Result<WriteResult, ToolError>`
   - **Atomar**: Temp-Datei im selben Verzeichnis (`.tmp.<pid/nanos-frei, aber
     deterministisch wählbar>`), `sync_data()`, dann `std::fs::rename` über das
     Ziel. Verhindert unvollständige Dateien bei Abstürzen.
   - `WriteResult { bytes_written: usize, path: PathBuf }` o. ä. strukturiert.

3. `fs_patch(path: &Path, patch_content: &str, policy) -> Result<PatchResult, ToolError>`
   - Wendet Unified-Diff deterministisch an. Handgeschriebener minimaler Parser
     für Hunks (`@@ -start,len +start,len @@`) mit Kontext-/`-`/`+`-Zeilen.
     KEINE Diff-Crate als Dependency.
   - Exact-Match-Pflicht: Kontext/entfernte Zeilen müssen exakt im File
     stehen, sonst `ToolError::PatchFailed(String)` — keine unscharfen
     Anpassungen.
   - Anwendung über fs_write-Mechanik (atomar).
   - `PatchResult { hunks_applied: usize, bytes_written: usize }` o. ä.

4. `proc_exec(cmd: &str, args: &[String], timeout_ms: u32, policy) -> Result<ExecResult, ToolError>`
   - Via `tokio::process::Command`. **Niemals** eine Shell als Zwischenschicht;
     `cmd` wird direkt als Executable aufgerufen.
   - Working Directory immer `WorkspacePolicy.root`.
   - `policy.allowed_subprocesses` prüfen, falls Liste nicht leer ist → sonst
     `ToolError::PolicyViolation(String)`.
   - Paralleles Draining von stdout und stderr (Schutz vor Pipe-Buffer-Deadlocks).
   - Hartes Timeout via `tokio::time::timeout` → `ToolError::Timeout(u32)`.
   - `ExecResult { exit_code: Option<i32>, stdout: String, stderr: String }` o. ä.

## 2. Fehlertypen (`error.rs`) — exakt

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ToolError {
    #[error("Security violation: path {0:?} is outside workspace root")]
    SecurityViolation(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Patch failed: context mismatch or invalid hunk")]
    PatchFailed(String),
    #[error("Command timed out after {0} ms")]
    Timeout(u32),
    #[error("Subprocess forbidden by policy: {0}")]
    PolicyViolation(String),
    #[error("Encoding error: non-UTF-8 payload")]
    InvalidEncoding,
}
```

## 3. Integrationstests (`crates/core/tests/tools_jail.rs`) — Pflicht

- `test_jail_blocks_parent_traversal`: Zugriff auf `../../Cargo.toml` → `SecurityViolation`.
- `test_jail_blocks_absolute_outside_path`: absoluter Pfad außerhalb des
  Test-Temp-Dirs → `SecurityViolation`.
- `test_atomic_write_and_read`: `fs_write` schreibt, `fs_read` liest identisch zurück.
- `test_fs_patch_unified_diff`: Datei anlegen, 1-Hunk-Diff anwenden, Ergebnis prüfen.
- `test_proc_exec_isolated`: simples Kommando im Workspace ausführen; Timeout
  bricht Endlosläufe zuverlässig ab.
- Mindestens ein Test für `PolicyViolation` (leerer vs. gefüllter Allowed-Liste).

**Portabilität ohne `#[cfg]` (Pflicht-Lösung)**:
- Positiv-Test-Kommando: `rustc --version` (existiert überall dort, wo
  `cargo test` läuft; null Plattform-Verzweigung).
- Endlos-Prozess für Timeout-Test: das Test-Binary selbst re-spawnen
  (`std::env::current_exe()`) mit Sentinel-Argument `__isaa_hang__`; oben in
  `main` der Tests: wenn `std::env::args()` den Sentinel enthält →
  Endlos-Sleep-Schleife, sonst normale Tests ausführen. Kein `#[cfg]`, kein
  Plattform-Binary-Raten.

## 4. Regeln & Definition of Done

1. Keine OS-spezifischen `#[cfg(...)]`-Blöcke in der Applikationslogik (nur
   plattformneutrale Tokio/Std-Funktionen; `#[cfg(test)]` zum Test-Gating bleibt ok).
2. **Keine neuen Dependencies.** Vorhandene Workspace-Dependencies genügen
   (tokio, tokio-util, thiserror, bytes, serde, ...). Kein `dunce`, keine Diff-Crate.
3. **Kein pauschales `cargo fmt` über unberührte Dateien!** Formatieren
   ausschließlich: `crates/core/src/tools/*`, `tests/tools_jail.rs` sowie
   die Ein-Zeilen-Änderung an `lib.rs`.
4. domain.rs, reducer.rs, wal.rs und deren Tests unverändert lassen.
5. `cargo test --workspace` muss alle Tests bestehen (Ticket 1 + 2 + 3).
6. `cargo clippy --workspace --all-targets -- -D warnings` = 0 Warnungen.

Berichte am Ende: angelegte/geänderte Dateien + Kurzfassung der Testausgabe.
