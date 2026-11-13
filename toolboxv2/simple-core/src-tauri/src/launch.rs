//! launch.rs - Bootstrap-Routing beim Tauri-Start (W1a).
//!
//! Kette: health-probe 8467 -> Running | tb vorhanden -> InstalledStopped
//! (start + poll) | sonst NotInstalled (installer.html).
//! Eine Onboarding-UI fuer alle Wege: installer.html (Tauri-Bundle, Kaltstart)
//! == web/installer/index.html, von der local-ui unter /setup ausgeliefert.

use std::time::Duration;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

/// GET {base}/health mit kurzem Timeout. Reine Netzwerk-Funktion.
pub async fn probe_health(base: &str) -> bool {
    let url = format!("{}/health", base);
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_millis(700))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    matches!(
        client.get(&url).send().await,
        Ok(resp) if resp.status().is_success()
    )
}

use std::path::PathBuf;
use std::process::{Command, Stdio};

/// Moegliche Orte einer tb-Installation, kanonisch laut installer.
///
/// Hintergrund (S6-Bug): GUI-Start (Explorer/Doppelklick) erbt den
/// User-PATH aus der Registry, NICHT den PATH einer Dev-Shell. `where tb`
/// schlaegt dann fehl, obwohl TB installiert ist. Diese Kandidaten decken
/// den installer-Kanon (%LOCALAPPDATA%\toolboxv2) und exe-nahe Layouts ab.
/// TB_TB_PATH dient als expliziter Override (auch fuer Tests).
/// ponytail: tb.cmd (src-Install-Modus) bewusst nicht abgedeckt - der
/// braeuchte einen Shell-Spawn; Upgrade-Pfad: Kandidat + `cmd /C call`.
fn tb_candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("TB_TB_PATH") {
        if !p.trim().is_empty() {
            v.push(PathBuf::from(p));
        }
    }
    if let Ok(la) = std::env::var("LOCALAPPDATA") {
        let la = PathBuf::from(la);
        v.push(la.join("toolboxv2").join("bin").join("tb.exe"));
        v.push(la.join("toolboxv2").join(".venv").join("Scripts").join("tb.exe"));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            v.push(dir.join("tb.exe"));
        }
    }
    v
}

/// Findet ein tb-Executable: erst Installer-Kandidaten (is_file),
/// dann PATH (`where`/`which`). None = wirklich nicht auffindbar.
pub fn discover_tb() -> Option<PathBuf> {
    for p in tb_candidates() {
        if p.is_file() {
            return Some(p);
        }
    }
    #[cfg(target_os = "windows")]
    let mut probe = {
        use std::os::windows::process::CommandExt;
        let mut c = Command::new("where");
        c.arg("tb").creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        c
    };
    #[cfg(not(target_os = "windows"))]
    let mut probe = {
        let mut c = Command::new("which");
        c.arg("tb");
        c
    };
    probe.stdout(Stdio::null()).stderr(Stdio::null());
    probe
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|_| PathBuf::from("tb")) // PATH-Hit: direkter Spawn reicht
}

/// Ist ein tb-Executable auffindbar? (PATH oder Installer-Kandidaten.)
/// Args-Vec statt Shell-String -> kein Injektionsvektor.
pub fn tb_on_path() -> bool {
    discover_tb().is_some()
}

/// TB im Hintergrund starten (detached): `<tb> workers start`.
///
/// S6-Fix: nimmt den per discover_tb() gefundenen Pfad statt
/// `cmd /C start /B tb` (PATHEXT-Gluecksspiel unter GUI-PATH). Output
/// geht in ein Logfile - stiller Stdio::null() machte jeden Startfehler
/// unsichtbar. Kein Warten hier - der Poll passiert in wait_until_running().
pub fn start_background() -> Result<(), String> {
    let tb = discover_tb()
        .ok_or_else(|| "tb not found (PATH + %LOCALAPPDATA%\\toolboxv2 checked)".to_string())?;

    let log_dir = std::env::var("LOCALAPPDATA")
        .map(|la| PathBuf::from(la).join("ToolBoxV2").join("app").join("logs"))
        .unwrap_or_else(|_| std::env::temp_dir());
    std::fs::create_dir_all(&log_dir)
        .map_err(|e| format!("log dir {} unwritable: {e}", log_dir.display()))?;
    let log_file = log_dir.join("workers_start.log");
    let out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file)
        .map_err(|e| format!("log file {} unwritable: {e}", log_file.display()))?;
    let err = out
        .try_clone()
        .map_err(|e| format!("log file clone failed: {e}"))?;

    #[cfg(target_os = "windows")]
    let mut cmd = {
        use std::os::windows::process::CommandExt;
        let mut c = Command::new(&tb);
        // DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP: ueberlebt Tauri-Exit.
        c.args(["workers", "start"])
            .creation_flags(0x0000_0008 | 0x0000_0200);
        c
    };
    #[cfg(not(target_os = "windows"))]
    let mut cmd = {
        let mut c = Command::new("nohup");
        c.arg(&tb).arg("workers").arg("start");
        c
    };

    cmd.stdout(out).stderr(err);
    cmd.spawn().map(|_| ()).map_err(|e| {
        format!(
            "{} workers start failed: {e} (log: {})",
            tb.display(),
            log_file.display()
        )
    })
}

/// Pollt health bis max_secs. True sobald Running.
pub async fn wait_until_running(base: &str, max_secs: u64) -> bool {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(max_secs);
    while tokio::time::Instant::now() < deadline {
        if probe_health(base).await {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(1000)).await;
    }
    false
}

/// Where a core installation can be: the dir the core published itself
/// (custom path), then the installers' defaults (installer.sh os_default_path /
/// installer.ps1 Get-DefaultPath).
pub fn install_dir_candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(p) = crate::worker_manager::published_install_dir() {
        v.push(p);
    }
    if let Ok(p) = std::env::var("TB_INSTALL_DIR") {
        if !p.trim().is_empty() {
            v.push(PathBuf::from(p));
        }
    }
    if cfg!(target_os = "windows") {
        if let Ok(la) = std::env::var("LOCALAPPDATA") {
            v.push(PathBuf::from(la).join("toolboxv2"));
        }
    } else if let Some(home) = dirs::home_dir() {
        if cfg!(target_os = "macos") {
            v.push(home.join("Library").join("Application Support").join("toolboxv2"));
        }
        v.push(home.join(".local").join("share").join("toolboxv2"));
        v.push(PathBuf::from("/opt/toolboxv2"));
    }
    v
}

/// First candidate that really is an installation (has install.manifest).
pub fn find_installation_in(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates
        .iter()
        .find(|p| p.join("install.manifest").is_file())
        .cloned()
}

pub fn find_installation() -> Option<PathBuf> {
    find_installation_in(&install_dir_candidates())
}

/// `<install_dir>/bin/tb workers stop` (blocking, output discarded). Falls back
/// to the discovered tb when the install dir has no bin/tb.
pub fn stop_workers(install_dir: &std::path::Path) {
    let name = if cfg!(target_os = "windows") { "tb.exe" } else { "tb" };
    let local = install_dir.join("bin").join(name);
    let tb = if local.is_file() { Some(local) } else { discover_tb() };
    let Some(tb) = tb else {
        log::warn!("[Launch] stop_workers: tb not found");
        return;
    };
    let mut cmd = Command::new(&tb);
    cmd.args(["workers", "stop"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    if let Err(e) = cmd.status() {
        log::warn!("[Launch] {} workers stop failed: {e}", tb.display());
    }
}

/// Pollt health bis max_secs. True sobald NICHT mehr erreichbar.
pub async fn wait_until_stopped(base: &str, max_secs: u64) -> bool {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(max_secs);
    while tokio::time::Instant::now() < deadline {
        if !probe_health(base).await {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(1000)).await;
    }
    false
}

/// Ziel-Route auf der local-ui bestimmen: first_run -> /setup, sonst /.
///
/// Bei JEGLICHER Unsicherheit (Timeout, 4xx/5xx, JSON-Fehler, fehlendes Feld)
/// -> "/" (nie eine Wizard-Schleife). Die local-ui zeigt auf "/" bei echtem
/// first_run selbst das Onboarding (Quelle: manifest app.profile).
pub async fn target_route(base: &str) -> &'static str {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(3000))
        .build()
        .ok();
    let Some(client) = client else {
        return "/";
    };
    let Ok(resp) = client
        .get(format!("{}/api/wizard/state", base))
        .send()
        .await
    else {
        return "/";
    };
    if !resp.status().is_success() {
        return "/";
    }
    let Ok(v) = resp.json::<serde_json::Value>().await else {
        return "/";
    };
    // Nur ein explizites first_run:true oeffnet den Wizard.
    if v.get("first_run").and_then(|f| f.as_bool()) == Some(true) {
        "/setup"
    } else {
        "/"
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_installation_needs_the_manifest() {
        let base = std::env::temp_dir().join("tb_find_install_unit");
        let _ = std::fs::remove_dir_all(&base);
        let empty = base.join("empty");
        let real = base.join("custom dir");
        std::fs::create_dir_all(&empty).unwrap();
        std::fs::create_dir_all(&real).unwrap();
        assert_eq!(find_installation_in(&[empty.clone(), real.clone()]), None);
        std::fs::write(real.join("install.manifest"), "install_mode: \"uv\"\n").unwrap();
        assert_eq!(find_installation_in(&[empty, real.clone()]), Some(real));
    }

    #[test]
    fn tb_on_path_does_not_panic() {
        // Ergebnis systemabhaengig, aber Aufruf muss sauber sein.
        let _ = tb_on_path();
    }

    #[test]
    fn start_background_reports_missing_tb_cleanly() {
        // In CI ist tb typischerweise nicht auf PATH -> Err, kein Panic.
        let _ = start_background();
    }

    #[test]
    fn discover_tb_honors_override() {
        // Contract: gueltiger Override gewinnt, ungueltiger wird NIE geliefert.
        // WAS stattdessen gefunden wird (PATH/Candidates/None) ist Umgebungssache:
        // der CI-Job installiert TB vorab per pip -> discover_tb findet das dann.
        // (assert None war PATH-abhaengig -> S8-Run#3-Linux-Fail, launch.rs:218)
        let fake = std::env::temp_dir().join("tb_does_not_exist_9f3a.exe");
        std::env::set_var("TB_TB_PATH", &fake);
        let d = discover_tb();
        assert_ne!(d.as_ref(), Some(&fake), "ungueltiger Override darf nie gewinnen");
        let real = std::env::temp_dir().join("tb_override_test_9f3a.exe");
        std::fs::write(&real, b"stub").unwrap();
        std::env::set_var("TB_TB_PATH", &real);
        assert_eq!(discover_tb(), Some(real));
        std::env::remove_var("TB_TB_PATH");
        let _ = std::fs::remove_file(
            std::env::temp_dir().join("tb_override_test_9f3a.exe"),
        );
    }
}
