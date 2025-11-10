
Phase 1: Bereinigung und Entfernung von überflüssigen Features

In dieser Phase entfernen wir Code, der fehlerhaft, ungenutzt oder inkonsistent ist, um eine saubere Basis für die weiteren Änderungen zu schaffen.

1.1. Entfernung von "Blob Storage" und "Encryption"

Da diese Features nicht implementiert sind, werden alle Verweise darauf entfernt, um die Codebasis und die Dokumentation in Einklang zu bringen.

tb-builtins/README.md:

Entfernen Sie den gesamten Abschnitt "Blob storage" unter "Features".

Entfernen Sie die Codebeispiele für blob_init, blob_create etc.

Entfernen Sie die Beschreibung von "Blob Storage" im Abschnitt "Implementation Details".

tb-builtins/Cargo.toml:

Entfernen Sie die auskommentierten Zeilen für aes-gcm und argon2.

tb-builtins/src/file_io.rs:

Entfernen Sie den ungenutzten Parameter _is_blob: bool aus den Signaturen aller relevanten Funktionen (read_file, write_file, file_exists etc.).

tb-builtins/src/error.rs:

Entfernen Sie die Varianten BlobStorage(String) und Encryption(String) aus dem BuiltinError-Enum.

1.2. Entfernung der redundanten task_runtime.rs

Der fehlerhafte Mini-Interpreter wird entfernt. Seine Funktionalität wird später nativ im JIT-Executor und Code-Generator implementiert.

Dateisystem: Löschen Sie die Datei tb-builtins/src/task_runtime.rs.

tb-builtins/src/lib.rs:

Entfernen Sie die Moduldeklaration pub mod task_runtime;.

tb-builtins/src/builtins_impl.rs:

Entfernen Sie alle use crate::task_runtime::TaskExecutor; Anweisungen. Dies wird zu Kompilierungsfehlern in den Funktionen spawn, map, filter etc. führen, was der nächste Schritt beheben wird.

1.3. Entfernung der nutzlosen open() Funktion

Da die open()-Funktion keine nutzbare Funktionalität bietet, wird sie entfernt.

tb-builtins/src/builtins_impl.rs:

Entfernen Sie die gesamte Funktion builtin_open.

tb-builtins/src/file_io.rs:

Entfernen Sie die Funktion open_file und die globale Variable FILE_HANDLES.

tb-builtins/src/lib.rs:

Entfernen Sie die Registrierung von open: builtins.push(("open", builtin_open as BuiltinFn));.

tb-builtins/README.md:

Entfernen Sie alle Verweise oder Beispiele, die open() verwenden.

1.4. Entfernung der benutzerdefinierten UUID-Implementierung

Die redundante und nicht standardkonforme UUID-Implementierung wird durch die bereits vorhandene uuid-Crate ersetzt.

tb-builtins/src/file_io.rs:

Entfernen Sie das gesamte mod uuid { ... } am Ende der Datei.

Fügen Sie am Anfang der Datei use uuid::Uuid; hinzu.

Suchen Sie die Zeile let handle_id = format!("file_{}", uuid::Uuid::new_v4()); und stellen Sie sicher, dass sie Uuid::new_v4().to_string() aus der uuid-Crate verwendet (was bereits in anderen Teilen des Codes der Fall ist, z.B. in builtins_impl.rs).

Phase 2: Architekturelle Verbesserungen und Fehlerbehebungen

Hier werden die Kernprobleme in der JIT-Engine und im Build-Prozess behoben.

2.1. Native Implementierung von map, filter und spawn

Diese Funktionen werden direkt im JIT-Executor und im Code-Generator implementiert, um task_runtime.rs zu ersetzen.

Für den JIT-Modus (tb-jit):

Analyse: Die BuiltinFn-Signatur fn(Vec<Value>) -> Result<Value, TBError> hat keinen Zugriff auf den Executor (&mut self). Die beste Lösung ist, diese Funktionen als Sonderfälle direkt in der eval_expression innerhalb des Expression::Call-Blocks zu behandeln, wo &mut self verfügbar ist.

Implementierung in tb-jit/src/executor.rs (innerhalb von eval_expression):

Im Expression::Call-Match-Arm, bevor die generische Funktionsaufruflogik ausgeführt wird, prüfen Sie, ob callee ein Ident mit dem Namen "map", "filter" oder "spawn" ist.

Für map und filter:

Evaluieren Sie das zweite Argument (die Liste).

Iterieren Sie über die Liste.

Rufen Sie für jedes Element die Funktion (das erste Argument) über self.call_function(...) auf. Dies stellt sicher, dass der Haupt-Executor verwendet wird und der korrekte Scope erhalten bleibt.

Sammeln Sie die Ergebnisse und geben Sie eine neue Liste zurück.

Für spawn:

Extrahieren Sie die Funktion und die Argumente.

Klonen Sie die aktuelle Umgebung: let captured_env = self.env.clone();. im::HashMap macht dies zu einer sehr schnellen Operation.

Verwenden Sie tokio::spawn, um eine neue asynchrone Aufgabe zu starten.

Innerhalb des async-Blocks:

Erstellen Sie einen neuen JitExecutor.

Setzen Sie seine Umgebung auf captured_env.

Führen Sie die Funktion mit executor.call_function(...) aus.

Für den kompilierten Modus (tb-codegen):

Analyse: Der Code-Generator kann die nativen, hoch-performanten Iterator-Methoden von Rust nutzen.

Implementierung in tb-codegen/src/rust_codegen.rs (innerhalb von generate_expression):

Im Expression::Call-Match-Arm, wenn ein Aufruf von map, filter oder reduce erkannt wird:

Generieren Sie anstelle eines Funktionsaufrufs den entsprechenden Rust-Code:

map(f, list) -> list.iter().map(|item| f(item)).collect::<Vec<_>>()

filter(f, list) -> list.iter().filter(|&item| f(item)).cloned().collect::<Vec<_>>()

Die bestehende Logik für die Generierung von Closures für Lambdas kann hier wiederverwendet und verfeinert werden.

Für spawn:

Der Aufruf von spawn(f, args) muss in einen Aufruf einer Helper-Funktion in tb-runtime übersetzt werden, z.B. tb_runtime::spawn_task(...).

Diese spawn_task-Funktion im Runtime-Crate muss Tokio (als optionales Feature) verwenden, um die Aufgabe auszuführen.

2.2. Korrektur des Scopings im JIT-Executor

Die inkonsistente Behandlung von Geltungsbereichen wird vereinheitlicht.

Analyse: Das Problem liegt darin, dass for- und while-Schleifen Variablen in den äußeren Geltungsbereich "leaken" lassen, während if-Blöcke dies korrekt verhindern. break und continue müssen weiterhin funktionieren.

Implementierung in tb-jit/src/executor.rs:

Gehen Sie zu den eval_statement-Implementierungen für Statement::For und Statement::While.

Anstatt die Variablen manuell zu entfernen, verwenden Sie die im::HashMap-Klon-Funktion, um den Scope für jede Iteration zu isolieren.

Logik für die Schleifenimplementierung:

code
Rust
download
content_copy
expand_less
// Vor der Schleife
let saved_loop_var = self.env.get(variable).cloned();

for item in items.iter() {
    let old_env_before_iteration = self.env.clone(); // Klon vor der Iteration
    self.env.insert(Arc::clone(variable), item.clone());

    self.eval_block(body)?; // Führt den Schleifenkörper aus

    // WICHTIG: break/continue/return prüfen
    if self.should_break || self.return_value.is_some() { break; }
    if self.should_continue { self.should_continue = false; }

    self.env = old_env_before_iteration; // Stellt die Umgebung von vor der Iteration wieder her
}

// Nach der Schleife: Wiederherstellen oder Entfernen der Schleifenvariable
if let Some(original_value) = saved_loop_var {
    self.env.insert(Arc::clone(variable), original_value);
} else {
    self.env.remove(variable);
}

Diese Logik stellt sicher, dass in der Schleife deklarierte Variablen nicht nach außen dringen, aber Änderungen an äußeren Variablen erhalten bleiben.

2.3. Analyse des Scopings im kompilierten Modus

Analyse in tb-codegen/src/rust_codegen.rs:

Der Code-Generator übersetzt if, for, while etc. in nativen Rust-Code (if { ... }, for item in iter { ... }).

Ergebnis: Rust selbst verfügt über ein strenges lexisches Scoping. Jede Variable, die innerhalb eines {}-Blocks deklariert wird, ist außerhalb dieses Blocks nicht sichtbar.

Fazit: Der kompilierte Modus ist von diesem Fehler nicht betroffen. Es sind keine Änderungen erforderlich.

Phase 3: Finale Korrekturen und Ergänzungen
3.1. Behebung der fest kodierten Build-Pfade

Die Build-Logik wird robust gemacht, indem sie sich auf das Cargo-Ökosystem verlässt.

Implementierung:

Erstellen Sie eine build.rs-Datei im tb-cli-Verzeichnis (tb-cli/build.rs).

Fügen Sie in tb-cli/Cargo.toml im [package]-Abschnitt build = "build.rs" hinzu.

Inhalt von build.rs:

code
Rust
download
content_copy
expand_less
use std::env;
use std::path::PathBuf;

fn main() {
    // Finde das Workspace-Root relativ zum 'OUT_DIR'
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // Der Pfad von 'OUT_DIR' ist typischerweise .../target/{profile}/build/tb-cli-xxxx/out
    // Wir müssen mehrere Ebenen nach oben gehen, um das Workspace-Root zu finden.
    let workspace_root = out_dir.ancestors()
        .find(|p| p.join("Cargo.lock").exists())
        .expect("Konnte das Workspace-Root nicht finden");

    let runtime_path = workspace_root.join("tb-exc/src/crates/tb-runtime");

    if !runtime_path.exists() {
        panic!("tb-runtime Verzeichnis nicht gefunden unter: {}", runtime_path.display());
    }

    // Setze eine Umgebungsvariable, die im Code verwendet werden kann
    println!("cargo:rustc-env=TB_RUNTIME_PATH={}", runtime_path.display());
}

In tb-cli/src/runner.rs:

Ersetzen Sie die manuelle Pfad-Logik durch:

code
Rust
download
content_copy
expand_less
let tb_runtime_path_str = env!("TB_RUNTIME_PATH");

Dies macht den Build-Prozess robust und unabhängig von der Verzeichnisstruktur, solange er über Cargo ausgeführt wird.

3.2. Korrektur der ignorierten connect_to-Argumente

Die Callback-Funktionen werden implementiert.

Implementierung:

tb-builtins/src/networking.rs: Erweitern Sie die ServerHandle-Struktur (oder eine ähnliche Client-Struktur), um die tb_core::Function-Callbacks zu speichern.

tb-builtins/src/builtins_impl.rs: In builtin_connect_to, parsen Sie die Callback-Funktionen aus den Argumenten.

tb-builtins/src/networking.rs: In den tokio::spawn-Tasks, die die Netzwerkverbindungen verwalten, rufen Sie die gespeicherten Callbacks auf, wenn Ereignisse eintreten (Verbindung, Nachricht, Trennung).

Zum Aufrufen eines Callbacks muss eine temporäre Executor-Instanz erstellt werden (ähnlich der spawn-Logik), um die TB-Funktion in der asynchronen Rust-Umgebung auszuführen.

3.3. Implementierung von clear() im Artifact Cache

Die fehlende Funktionalität wird hinzugefügt.

Implementierung:

tb-cache/src/artifact_cache.rs:

code
Rust
download
content_copy
expand_less
impl ArtifactCache {
    // ... bestehende Methoden ...

    pub fn clear(&self) -> Result<()> {
        // Iteriere und lösche die gecachten Dateien vom Dateisystem
        for entry in self.artifacts.iter() {
            if entry.value().path.exists() {
                fs::remove_file(&entry.value().path)?;
            }
        }
        self.artifacts.clear();
        Ok(())
    }
}

tb-cache/src/cache_manager.rs: In CacheManager::clear, rufen Sie die neue Methode auf:

code
Rust
download
content_copy
expand_less
pub fn clear(&self) -> Result<()> {
    self.import_cache().clear()?;
    self.artifact_cache().clear()?; // Jetzt aufrufen
    self.clear_hot_cache();
    Ok(())
}

tb-cli/src/main.rs: Passen Sie den CacheAction::Clear-Block an, um den Result zu behandeln:

code
Rust
download
content_copy
expand_less
CacheAction::Clear => {
    cache_manager.clear()?;
    println!("{}", "Cache cleared".bright_green());
}
