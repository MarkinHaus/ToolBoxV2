# TICKET 2: Persistent Journal (WAL) & Postcard Checkpoints

Du arbeitest an einem bestehenden Rust-Workspace (crates/core = `isaa-core`).
Ticket 1 (domain.rs, reducer.rs) ist fertig und grün. Füge NUR Ticket 2 hinzu.
Ändere KEINE bestehenden Module (domain.rs, reducer.rs, lib.rs nur um `pub mod wal;` ergänzen).

## Ziel

`crates/core/src/wal.rs`: Framed Append-Only-Write-Ahead-Log für `Event`s mit
CRC32-Prüfsummen und O(1)-Checkpoints des `State`.

## Exaktes Frame-Format (Record)

Jeder Record in der Datei:

```
[u8 marker] [u32 len (LE)] [u32 crc32 (LE)] [payload: len Bytes]
```

- `marker = 0x01`: payload = postcard-serialisiertes `Event` (aus `isaa_core::domain::Event`)
- `marker = 0x02`: payload = postcard-serialisierter Checkpoint-Envelope:
  `struct Checkpoint { pub version: u64, pub state: State }` (State ist postcard-serialisierbar, `State::hash()` existiert)

## API (Minimal, keine Abstraktionen mehr)

```rust
pub struct Wal { /* BufWriter<std::fs::File>, Pfad, Zähler */ }

impl Wal {
    /// Öffnet/erstellt die WAL-Datei (append).
    pub fn open(path: &std::path::Path) -> std::io::Result<Self>;

    /// Hängt Event gepuffert an; flush() spätestens jedes 64. Event
    /// (Batch-Commit 64 Events, siehe Master-Spec).
    pub fn append(&mut self, event: &Event) -> std::io::Result<()>;

    /// Flush des Puffers + fsync-äquivalent (std: flush genügt; kein OS-Branching).
    pub fn flush(&mut self) -> std::io::Result<()>;

    /// Schreibt Checkpoint-Record (marker 0x02) und flusht sofort.
    pub fn checkpoint(&mut self, state: &State) -> std::io::Result<()>;
}

/// Recovery: letzten gültigen Checkpoint lesen, danach alle gültigen
/// Event-Records sammeln. Beschädigter/unvollständiger Tail (CRC-Fehler,
/// abgebrochener Frame) wird STILL verworfen; valide Prefix bleibt.
pub fn recover(path: &std::path::Path) -> std::io::Result<(Option<State>, Vec<Event>)>;
```

Replay-Funktion (separat, nutzt den puren Reducer, verwirft Effects):

```rust
/// Spielt Events auf einen State via isaa_core::reduce; Rückgabe: finaler State.
pub fn replay(state: Option<State>, events: &[Event]) -> State;
```

`replay(None, events)` startet bei `State::new(WorkspacePolicy::default())`.

## Harte constraints

1. **Kein OS-Branching**: verboten sind `#[cfg(windows)]`, `#[cfg(unix)]`,
   `std::os::*`, `libc`, `winapi`. Nur `std::fs`, `std::io`, stdlib.
2. Nur bestehende Workspace-Dependencies verwenden (postcard, crc32fast, serde
   sind schon in crates/core/Cargo.toml). KEINE neuen Dependencies.
3. Kein RNG, keine Systemuhr im Code-Pfad (Determinismus).
4. `cargo clippy --workspace --all-targets -- -D warnings` muss 0 Fehler melden.
5. `cargo fmt` auf den neuen Dateien.

## Tests (in crates/core/tests/wal.rs oder #[cfg(test)] in wal.rs)

1. **Roundtrip**: 100 beliebiger Events appenden, flush, `recover` →
   identische Event-Sequenz zurück (assert_eq über postcard-Bytes).
2. **Checkpoint-Recovery**: 30 Events, checkpoint(state), 30 weitere Events,
   recover → Checkpoint-State + danach nur die 30 neuen Events.
3. **Tail-Corruption (PFICHT)**: WAL schreiben, flush, dann die Datei-Bytes
   kürzen (z. B. letzte 3 Bytes abschneiden → halber Frame). `recover` darf
   NICHT panic-en, muss den defekten Tail verwerfen und den State exakt so
   rekonstruieren, wie vor dem defekten Event (Vergleich über
   `State::hash()` gegen Referenz-Replay ohne das letzte Event).
4. **Replay-Determinismus**: `replay` über dieselben Events zweimal →
   gleicher `State::hash()`.

## Definition of Done

- `cargo test --workspace` Exit 0
- `cargo clippy --workspace --all-targets -- -D warnings` Exit 0
- Keine Änderung an domain.rs/reducer.rs außer nichts (lib.rs: `pub mod wal;`)

Berichte am Ende: geänderte/angelegte Dateien + Testausgabe (Kurzfassung).
