# TASK: Ticket 4 – Dual Runtime: WASM Sandbox & Stdio-RPC Process Runner

Du arbeitest auf dem Branch `master` im Cargo-Root (Repo-Root liegt höher — nur
innerhalb des Cargo-Roots arbeiten). Ticket 1, 2 und 3 sind vollständig getestet
und committet. Bestehende Module (domain.rs, reducer.rs, wal.rs, tools/jail.rs,
tools/core_pi.rs, tools/error.rs bestehende Varianten) NICHT verändern; nur
Erweiterungen wie unten.

## ZIEL VON TICKET 4

Beide Runtime-Treiber in `crates/core/src/tools/`:
1. `wasm.rs`: In-Process WASM-Sandbox via `wasmtime` mit striktem CPU-Fuel- und Memory-Limit.
2. `process_rpc.rs`: Out-of-Process NDJSON-RPC 2.0 Prozess-Runner über Stdio
   (für Python Toolbox v2 & Binaries) mit Request-ID-Korrelation und parallelem
   Stderr-Drain.
3. Integration beider Treiber in `crates/core/src/tools/mod.rs`.

## 1. SPEZIFIKATION DER TREIBER

### A. In-Process WASM Sandbox (`wasm.rs`)
- **Engine-Setup**: Wiederverwendbare `wasmtime::Engine` mit `async_support(true)`
  und `consume_fuel(true)`.
- **WASI-Integration**: `wasmtime_wasi::preview1` für grundlegendes Sandboxed I/O
  (stdin = übergebene Input-Bytes, stdout = gesammelter Output).
- **CPU-Schutz (Fuel)**: `store.set_fuel(fuel_limit)`. Endlosschleife → Wasmtime-Trap
  → mappe auf `ToolError::FuelExhausted`.
- **Memory-Begrenzung**: Resource-Limiter auf dem Store (max. 64 MiB), damit ein
  Modul den Host-RAM nicht flutet. Überschreitung → Trap → `WasmExecution`/`FuelExhausted`.
- **Aufruf-Konvention**:
  ```rust
  pub async fn execute_wasm(
      &self,
      module_bytes: &[u8],
      input: &[u8],
      fuel_limit: u64,
  ) -> Result<Vec<u8>, ToolError>;
  ```
  Konvention: Modul erhält Input via WASI-Stdin (oder Memory), schreibt Resultat
  nach stdout (oder Memory). Dokumentiere im Code, welche Konvention gilt.
- Tests dürfen WAT-Text direkt an `Module::new` geben (wasmtime parsst Text-Format
  selbst — KEINE `wat`-Dependency nötig).

### B. Out-of-Process Stdio-RPC Runner (`process_rpc.rs`)
Für langlebige/warm gehaltene Worker:
- **Lifecycle**: `tokio::process::Command` mit **`kill_on_drop(true)`**; Working
  Directory strikt im kanonisierten Workspace-Root (via `jail::resolve`).
- **Protokoll**: NDJSON RPC 2.0 über Stdio:
  - Request: `{"jsonrpc":"2.0","id":1,"method":"execute","params":{...}}\n`
  - Response: `{"jsonrpc":"2.0","id":1,"result":{...}}\n` oder `{"error":{...}}\n`
  - Nutze `tokio_util::codec::{FramedRead, FramedWrite, LinesCodec}` (strippt
    `\r` automatisch cross-platform).
- **Deadlock-Schutz**: stderr wird kontinuierlich im Hintergrund gelesen
  (`tokio::spawn`, ringförmiger Puffer, z. B. letzte 64 KiB), damit der
  OS-Pipe-Puffer nie vollläuft.
- **Call-Semantik**:
  ```rust
  pub async fn call(
      &mut self,
      method: &str,
      params: serde_json::Value,
      timeout_ms: u32,
  ) -> Result<serde_json::Value, ToolError>;
  ```
  - Request-ID (`u64`-Zähler) korrelieren; Antwort mit falscher/fehlender ID →
    `RpcProtocol`.
  - Timeout bricht den Call ab, reappt/kills den Prozess und gibt `ToolError::Timeout` zurück.
  - `pub async fn shutdown(self)` o. ä. für sauberes Beenden.

## 2. FEHLERTYPEN-ERWEITERUNG (`error.rs`) — exakt diese Varianten ergänzen

```rust
#[error("WASM fuel exhausted (CPU limit reached)")]
FuelExhausted,
#[error("WASM execution failed: {0}")]
WasmExecution(String),
#[error("RPC protocol error: {0}")]
RpcProtocol(String),
#[error("Process crashed with code: {0:?}")]
ProcessCrash(Option<i32>),
```

## 3. INTEGRATIONSTESTS (`crates/core/tests/tools_runtime.rs`)

100 % ohne externe Tools/Dateien lauffähig:
1. `test_wasm_execution_success`: minimales WASM-Modul (WAT-Text), das Input
   transformiert (z. B. Bytes invertieren/addieren) → Output validieren.
2. `test_wasm_fuel_exhaustion`: Modul mit Endlosschleife (`(loop (br 0))`) →
   deterministisch `FuelExhausted`.
3. `test_process_rpc_self_respawn`: portabler Self-Respawn-Sentinel wie in
   `tests/tools_jail.rs` (MUSTER DORT ÜBERNEHMEN — `std::env::current_exe()`):
   - Stub-Subprozess (Test mit Sentinel-Filter) liest NDJSON von stdin,
     beantwortet Ping/Echo, schreibt NDJSON auf stdout.
   - Parent sendet 5 Requests sequenziell, validiert Antworten.
   - Timeout-Fall: Stub-Methode schläft → bricht nach X ms mit `Timeout` ab.
   - Zusätzlich: Prozess-Crash des Stubs → `ProcessCrash`.

## 4. Regeln & DoD

1. Keine neuen Dependencies (wasmtime, wasmtime-wasi, tokio-util, serde_json sind
   bereits in crates/core/Cargo.toml).
2. Keine `#[cfg(windows)]`-Verzweigungen in der Logik (`#[cfg(test)]` ok).
3. **Kein globales `cargo fmt`** — nur die neuen/bearbeiteten Dateien formatieren
   (`tools/wasm.rs`, `tools/process_rpc.rs`, `tools/error.rs`, `tools/mod.rs`,
   `tests/tools_runtime.rs`).
4. domain.rs, reducer.rs, wal.rs, jail.rs, core_pi.rs und deren Tests unverändert.
5. `cargo test --workspace` grün (Tickets 1+2+3+4).
6. `cargo clippy --workspace --all-targets -- -D warnings` = 0.

Berichte am Ende: Dateien + Kurzfassung Testausgabe.
