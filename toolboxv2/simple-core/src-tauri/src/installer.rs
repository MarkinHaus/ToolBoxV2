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
    /// Wizard-Profil: local|server|developer (legacy consumer|homelab|business accepted)
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
/// Legacy wizard names -> ToolBoxV2 ProfileType (manifest schema is the source).
pub fn canon_profile(p: &str) -> &'static str {
    match p {
        "server" | "business" => "server",
        "developer" => "developer",
        _ => "local",
    }
}

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
    // installer.sh/ps1 run `tb -init <profile>` after install -> no CLI questions.
    yaml.push_str(&format!("profile: {}\n", canon_profile(&cfg.profile)));
    // T5/F1: Feature-Set fuer uv-Installationen; developer baut aus Source.
    if canon_profile(&cfg.profile) != "developer" {
        yaml.push_str("tb_features: cli,web,isaa,desktop\n");
    }
    // Tauri shows its own UI afterwards; never open a browser/CLI onboarding.
    yaml.push_str("ui: none\n");
    // Tauri laeuft headless (kein stdin) -> Installer-Fragen sind unantwortbar.
    // assume: yes + action machen sh/ps1 voll non-interactive.
    yaml.push_str("assume: yes\n");
    yaml.push_str("action: install\n");
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

/// uv sidecar (bundle.externalBin "binaries/uv"): Tauri places it next to the
/// app executable with the target triple stripped. None -> installer bootstraps uv.
pub fn bundled_uv(exe: &std::path::Path) -> Option<std::path::PathBuf> {
    let name = if cfg!(windows) { "uv.exe" } else { "uv" };
    let p = exe.parent()?.join(name);
    p.is_file().then_some(p)
}

/// T6/F4: gebuendeltes Kern-Wheel unter <resource_dir>/wheels finden
/// (groesster Name gewinnt => neueste Version).
fn bundled_wheel(resource_dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let mut best: Option<std::path::PathBuf> = None;
    let entries = std::fs::read_dir(resource_dir.join("wheels")).ok()?;
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if name.starts_with("toolboxv2-") && name.ends_with(".whl") {
            let better = best
                .as_ref()
                .map_or(true, |b| name > b.file_name().map_or_else(String::new, |f| f.to_string_lossy().to_string()));
            if better {
                best = Some(e.path());
            }
        }
    }
    best
}

/// Build id of the bundled core: <resource_dir>/wheels/BUILD_ID (sha256 of the
/// wheel, written by scripts/build_wheel_for_tauri.sh). None in dev runs.
pub fn bundled_build_id(resource_dir: &std::path::Path) -> Option<String> {
    let id = std::fs::read_to_string(resource_dir.join("wheels").join("BUILD_ID")).ok()?;
    let id = id.trim();
    (!id.is_empty()).then(|| id.to_string())
}

/// `core_build:` of an installation: <install_dir>/install.manifest.
/// None = no manifest (not installed). Some("") = installed, id unknown.
pub fn installed_build_id(install_dir: &std::path::Path) -> Option<String> {
    let txt = std::fs::read_to_string(install_dir.join("install.manifest")).ok()?;
    for line in txt.lines() {
        if let Some(v) = line.strip_prefix("core_build:") {
            return Some(v.trim().trim_matches('"').to_string());
        }
    }
    Some(String::new())
}

/// The core is updated when the app brings a core build that differs from the
/// installed one. Version numbers are not compared: every build of the same
/// version would look equal.
pub fn core_update_needed(bundled: Option<&str>, installed: Option<&str>) -> bool {
    match (bundled, installed) {
        (Some(b), Some(i)) => !b.is_empty() && b != i,
        _ => false, // nothing bundled (dev run) or nothing installed (cold start installs)
    }
}

/// Arguments for an in-place core update: the installers' own --update/--path
/// switches (no config file, so no profile is applied and `tb -init` never runs).
pub fn update_args(install_dir: &std::path::Path) -> Vec<String> {
    let dir = install_dir.to_string_lossy().to_string();
    if cfg!(windows) {
        vec!["-JsonOutput".into(), "-Update".into(), "-AssumeYes".into(), "yes".into(),
             "-Path".into(), dir]
    } else {
        vec!["--json-output".into(), "--update".into(), "--yes".into(), "--path".into(), dir]
    }
}

/// Run the OS installer script with `extra` arguments. Every protocol line on
/// stdout becomes an `install-event`; everything is appended to install.log
/// next to the app executable. Blocking - call from a sync command or a
/// blocking thread, never from the async runtime.
fn run_installer_script(
    app: &tauri::AppHandle,
    extra: Vec<String>,
    what: &str,
) -> Result<std::process::ExitStatus, String> {
    use std::io::Write;
    // resource_dir: Bundle-Layout (installierte App); Dev-Run: None -> CWD.
    let res_dir = app
        .path()
        .resource_dir()
        .ok()
        .map(|p| p.to_path_buf());
    let (shell, mut args) = script_args(res_dir.as_deref());
    args.extend(extra);

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
            let _ = writeln!(f, "=== {what} start: shell={shell} args={args:?} ===");
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

    let mut cmd = Command::new(&shell);
    cmd.args(&args);
    // Bundled uv -> no network bootstrap of uv/Python tooling (installer.sh/ps1 TB_UV_BIN).
    if let Some(uv) = std::env::current_exe().ok().as_deref().and_then(bundled_uv) {
        cmd.env("TB_UV_BIN", &uv);
    }
    // Bundled core wheel + its build id (installer.sh/ps1 pkg_spec / core_build).
    if let Some(res) = res_dir.as_ref() {
        if let Some(wheel) = bundled_wheel(res) {
            cmd.env("TB_WHEEL", wheel);
        }
        if let Some(id) = bundled_build_id(res) {
            cmd.env("TB_CORE_BUILD", id);
        }
    }
    let mut child = cmd
        .stdin(Stdio::null())
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
                "=== {what} end: exit={} success={} ===",
                status.code().unwrap_or(-1),
                status.success()
            );
        }
    }
    Ok(status)
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
    let yaml = config_to_yaml(&config);

    // Config ins Temp-Dir schreiben
    let yaml_path = std::env::temp_dir().join("tb-install.yaml");
    std::fs::write(&yaml_path, yaml).map_err(|e| format!("config write failed: {e}"))?;

    // R16-Fix: OS-spezifische Flags. PowerShell kennt NUR -JsonOutput/-Config
    // (Doppeldash = "Parameter nicht gefunden" -> exit 1, install.log-Beweis).
    let (json_flag, config_flag) = if cfg!(windows) {
        ("-JsonOutput", "-Config")
    } else {
        ("--json-output", "--config")
    };
    let status = run_installer_script(
        &app,
        vec![json_flag.into(), config_flag.into(), yaml_path.to_string_lossy().to_string()],
        "install",
    )?;

    Ok(serde_json::json!({
        "exit_code": status.code().unwrap_or(-1),
        "success": status.success(),
        "profile": config.profile,
    }))
}

/// Core update in place (blocking): reinstall the bundled wheel into
/// `install_dir`. The caller has stopped the workers. Profile, .env and data
/// are untouched.
pub fn run_core_update(app: &tauri::AppHandle, install_dir: &std::path::Path) -> Result<bool, String> {
    let status = run_installer_script(app, update_args(install_dir), "core-update")?;
    Ok(status.success())
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
        assert!(yaml.contains("profile: local"));
        assert!(yaml.contains("ui: none"));
        assert!(yaml.contains("tb_features: cli,web,isaa,desktop"));
    }

    #[test]
    fn bundled_wheel_found_in_resources() {
        let d = std::env::temp_dir().join("tb_wheel_unit");
        let _ = std::fs::remove_dir_all(&d);
        // layout as bundled: <resource_dir>/wheels/*.whl
        let w = d.join("wheels");
        std::fs::create_dir_all(&w).unwrap();
        assert!(bundled_wheel(&d).is_none());
        assert!(bundled_build_id(&d).is_none());
        std::fs::write(w.join("toolboxv2-0.1.28-py3-none-any.whl"), b"x").unwrap();
        std::fs::write(w.join("toolboxv2-0.1.29-py3-none-any.whl"), b"x").unwrap();
        std::fs::write(w.join("other.whl"), b"x").unwrap();
        std::fs::write(w.join("BUILD_ID"), b"abc123\n").unwrap();
        let found = bundled_wheel(&d).unwrap();
        assert!(found.ends_with("toolboxv2-0.1.29-py3-none-any.whl"));
        assert_eq!(bundled_build_id(&d).as_deref(), Some("abc123"));
    }

    #[test]
    fn installed_build_id_reads_manifest() {
        let d = std::env::temp_dir().join("tb_core_build_unit");
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        assert_eq!(installed_build_id(&d), None, "no manifest = not installed");
        std::fs::write(d.join("install.manifest"), "tb_version: \"0.1.29\"\n").unwrap();
        assert_eq!(installed_build_id(&d).as_deref(), Some(""), "installed, id unknown");
        std::fs::write(
            d.join("install.manifest"),
            "tb_version: \"0.1.29\"\ncore_build: \"abc123\"\n",
        )
        .unwrap();
        assert_eq!(installed_build_id(&d).as_deref(), Some("abc123"));
    }

    #[test]
    fn core_update_only_when_bundled_build_differs() {
        // same version number, different build -> update (the nightly case)
        assert!(core_update_needed(Some("new"), Some("old")));
        // installation from before build ids existed -> update once
        assert!(core_update_needed(Some("new"), Some("")));
        assert!(!core_update_needed(Some("same"), Some("same")));
        // dev run without a bundled wheel never touches an installation
        assert!(!core_update_needed(None, Some("old")));
        assert!(!core_update_needed(Some(""), Some("old")));
        // not installed: the cold-start installer is responsible, not the updater
        assert!(!core_update_needed(Some("new"), None));
    }

    #[test]
    fn update_args_use_update_and_path_and_no_config() {
        let a = update_args(std::path::Path::new("/opt/my tb"));
        let joined = a.join(" ").to_lowercase();
        assert!(joined.contains("update"));
        assert!(joined.contains("path"));
        assert!(!joined.contains("config"), "a config would re-apply a profile");
        assert_eq!(a.last().map(String::as_str), Some("/opt/my tb"));
    }

    #[test]
    fn bundled_uv_found_next_to_exe() {
        let d = std::env::temp_dir().join("tb_uv_sidecar_unit");
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let exe = d.join("SimpleCore");
        assert!(bundled_uv(&exe).is_none());
        let name = if cfg!(windows) { "uv.exe" } else { "uv" };
        std::fs::write(d.join(name), b"x").unwrap();
        assert_eq!(bundled_uv(&exe), Some(d.join(name)));
    }

    #[test]
    fn legacy_profiles_map_to_schema() {
        assert_eq!(canon_profile("consumer"), "local");
        assert_eq!(canon_profile("homelab"), "local");
        assert_eq!(canon_profile("business"), "server");
        assert_eq!(canon_profile("developer"), "developer");
        assert_eq!(canon_profile("server"), "server");
    }

    #[test]
    fn parses_valid_and_skips_invalid_lines() {
        let raw = "{\"type\":\"progress\",\"phase\":\"discovery\",\"progress\":0.02}\nnot json\n{\"type\":\"done\"}";
        let events = parse_lines(raw);
        assert_eq!(events.len(), 2);
        assert_eq!(events[1]["type"], "done");
    }
}
