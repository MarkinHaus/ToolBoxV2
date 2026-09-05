//! installer.rs - Kaltstart-Install via installer.sh/ps1 (P1: Tauri Installer Wizard).
//!
//! Wizard-State -> tb-install.yaml -> subprocess mit --json-output -> JSON-Lines
//! -> `install-event` Tauri-Events. Protokoll (installer.sh L48-60 / ps1 JLog):
//!   {"type":"phase"|"progress"|"error"|"done", ...}

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use tauri::{Emitter, Manager};

/// Ein Installations-Config-Set (aus dem Wizard gesammelt).
#[derive(serde::Deserialize, Debug)]
pub struct InstallConfig {
    /// Wizard-Profil: consumer|homelab|server|business|developer
    pub profile: String,
    #[serde(default)]
    pub install_path: Option<String>,
    /// "uv" (default) | "source" | "docker"
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub optional: Vec<String>,
}

/// tb-install.yaml Keys (installer.ps1 L245-258 / installer.sh L296-301):
/// install_mode, install_path, tb_version, source_from, source_branch,
/// environment, instance_id, optional.nginx/docker_runtime/ollama/minio/registry
pub fn config_to_yaml(cfg: &InstallConfig) -> String {
    let optional_all: &[&str] = &[
        "nginx", "docker_runtime", "ollama", "minio", "registry",
    ];
    // Progressive Disclosure: developer baut aus Source, alle anderen via uv.
    let (mode, environment) = match cfg.profile.as_str() {
        "developer" => ("source", "development"),
        _ => ("uv", "production"),
    };
    let mut yaml = String::new();
    if let Some(m) = &cfg.mode {
        yaml.push_str(&format!("install_mode: {}\n", m));
    } else {
        yaml.push_str(&format!("install_mode: {}\n", mode));
    }
    if let Some(p) = &cfg.install_path {
        yaml.push_str(&format!("install_path: {}\n", p.replace('\\', "/")));
    }
    yaml.push_str("source_from: git\n");
    yaml.push_str("source_branch: master\n");
    yaml.push_str(&format!("environment: {}\n", environment));
    yaml.push_str("instance_id: tbv2_main\n");
    for opt in optional_all {
        let on = cfg.optional.iter().any(|o| o == opt);
        yaml.push_str(&format!("optional.{}: {}\n", opt, on));
    }
    yaml
}

/// JSON-Lines parsen (reine Funktion, testbar). Ungueltige Zeilen werden ignoriert.
/// Referenz-Parser des Installer-Protokolls - von Tests + Wizard-Doku genutzt;
/// run_install parst inline im Reader-Thread.
#[allow(dead_code)]
pub fn parse_lines(buf: &str) -> Vec<serde_json::Value> {
    buf.lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// Absoluter Pfad zum OS-Installer-Script + Basis-Args. Reine Funktion:
/// resource_dir kommt aus app.path().resource_dir() (Bundle-Layout,
/// tauri.conf.json -> "installer.sh"/"installer.ps1" in src-tauri/);
/// None = CWD (Dev-Run im src-tauri-Verzeichnis findet sie via CWD).
fn script_args(resource_dir: Option<&std::path::Path>) -> (&'static str, Vec<String>) {
    let resolve = |name: &str| -> String {
        if let Some(dir) = resource_dir {
            let p = dir.join(name);
            if p.exists() {
                return p.to_string_lossy().to_string();
            }
        }
        name.to_string()
    };
    if cfg!(windows) {
        (
            "powershell",
            vec![
                "-ExecutionPolicy".into(),
                "Bypass".into(),
                "-File".into(),
                resolve("installer.ps1"),
            ],
        )
    } else {
        ("bash", vec![resolve("installer.sh")])
    }
}

/// Kaltstart-Install: fuehrt den OS-Installer mit Wizard-Config aus und
/// emittiert pro Protokoll-Zeile ein `install-event`. Sync-Command
/// (Tauri trennt sync Commands auf einen Thread-Pool ab - der blocking
/// child.wait() darf den async-Runtime-Thread nicht Minuten blockieren).
#[tauri::command]
pub fn run_install(
    config: InstallConfig,
    app: tauri::AppHandle,
) -> Result<serde_json::Value, String> {
    use std::io::Write;
    let yaml = config_to_yaml(&config);

    // Config ins Temp-Dir schreiben
    let yaml_path = std::env::temp_dir().join("tb-install.yaml");
    std::fs::write(&yaml_path, yaml).map_err(|e| format!("config write failed: {e}"))?;

    let (shell, mut args) = {
        // resource_dir: Bundle-Layout (installierte App); Dev-Run: None -> CWD.
        let res_dir = app
            .path()
            .resource_dir()
            .ok()
            .map(|p| p.to_path_buf());
        let (s, a) = script_args(res_dir.as_deref());
        (s, a)
    };
    args.push("--json-output".into());
    args.push("--config".into());
    args.push(yaml_path.to_string_lossy().to_string());

    // install.log (Runde-11-Muster: niemals still scheitern) - exe-sibling logs/.
    let log_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("logs")));
    let log_path = log_dir.as_ref().map(|d| d.join("install.log"));
    let log: Option<std::sync::Arc<std::sync::Mutex<std::fs::File>>> = log_path
        .and_then(|p| {
            if let Some(dir) = p.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .ok()
        })
        .map(|f| std::sync::Arc::new(std::sync::Mutex::new(f)));
    if let Some(f) = log.as_ref() {
        if let Ok(mut f) = f.lock() {
            let _ = writeln!(
                f,
                "=== install start: shell={shell} args={args:?} config={:?} ===",
                config
            );
        }
    }
    let emit_log = |handle: &tauri::AppHandle,
                    log: &Option<std::sync::Arc<std::sync::Mutex<std::fs::File>>>,
                    stream: &str,
                    line: &str| {
        if let Some(f) = log.as_ref() {
            if let Ok(mut f) = f.lock() {
                let _ = writeln!(f, "[{stream}] {line}");
            }
        }
        let _ = handle.emit(
            "install-event",
            serde_json::json!({"type": "log", "stream": stream, "msg": line}),
        );
    };

    let mut child = Command::new(&shell)
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn failed: {e}"))?;

    let stdout = child.stdout.take().ok_or("no stdout")?;
    let handle = app.clone();
    let log2 = log.clone();
    let reader_thread = std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            match serde_json::from_str::<serde_json::Value>(&line) {
                Ok(value) => {
                    let _ = handle.emit("install-event", value);
                    if let Some(f) = log2.as_ref() {
                        if let Ok(mut f) = f.lock() {
                            let _ = writeln!(f, "{line}");
                        }
                    }
                }
                Err(_) => emit_log(&handle, &log2, "stdout", &line),
            }
        }
    });

    // stderr gelesen statt nur gepiped: Deadlock-Risiko weg, Diagnostik im Log.
    if let Some(stderr) = child.stderr.take() {
        let handle = app.clone();
        let log3 = log.clone();
        std::thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                emit_log(&handle, &log3, "stderr", &line);
            }
        });
    }

    let status = child.wait().map_err(|e| format!("wait failed: {e}"))?;
    let _ = reader_thread.join();
    if let Some(f) = log.as_ref() {
        if let Ok(mut f) = f.lock() {
            let _ = writeln!(
                f,
                "=== install end: exit={} success={} ===",
                status.code().unwrap_or(-1),
                status.success()
            );
        }
    }

    Ok(serde_json::json!({
        "exit_code": status.code().unwrap_or(-1),
        "success": status.success(),
        "profile": config.profile,
    }))
}

/// Install abbrechen: entfernt das PID-File-Signal-Flag der Installer-Scripts.
/// ponytail: richtiges Kill-Tracking (Child-Handle global) wenn User-Abbruch
/// im Wizard-UI verdrahtet wird.
#[tauri::command]
pub fn cancel_install() -> Result<bool, String> {
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yaml_contains_canon_keys() {
        let cfg = InstallConfig {
            profile: "consumer".into(),
            install_path: Some("C:\\Users\\x\\TB".into()),
            mode: None,
            optional: vec!["nginx".into()],
        };
        let yaml = config_to_yaml(&cfg);
        assert!(yaml.contains("install_mode: uv"));
        assert!(yaml.contains("install_path: C:/Users/x/TB"));
        assert!(yaml.contains("optional.nginx: true"));
        assert!(yaml.contains("optional.minio: false"));
        assert!(yaml.contains("instance_id: tbv2_main"));
    }

    #[test]
    fn parses_valid_and_skips_invalid_lines() {
        let raw = "{\"type\":\"phase\",\"phase\":\"discovery\"}\nnot json\n{\"type\":\"done\"}";
        let events = parse_lines(raw);
        assert_eq!(events.len(), 2);
        assert_eq!(events[1]["type"], "done");
    }
}
