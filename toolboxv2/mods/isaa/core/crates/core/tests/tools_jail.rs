//! Integrationstests Ticket 3: Workspace Jail & 4 primitive Core-Tools.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use isaa_core::domain::WorkspacePolicy;
use isaa_core::tools::{fs_patch, fs_read, fs_write, proc_exec, ToolError};

/// Eindeutiges Temp-Root pro Test (keine Extra-Dependency).
fn unique_root(tag: &str) -> PathBuf {
    static N: AtomicU32 = AtomicU32::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let d = std::env::temp_dir().join(format!(
        "isaa_t3_{tag}_{pid}_{n}_{nanos}",
        pid = std::process::id()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Re-Spawn-Ziel für den Timeout-Test: Das Test-Binary wird mit dem
/// Sentinel-Argument `__isaa_hang__` neu gestartet; libtest filtert auf
/// diesen Test (Name == Sentinel) und dieser Endlos-Sleep blockiert, bis der
/// Parent ihn per Timeout killed. Ohne Sentinel: No-op (normaler Testlauf).
#[test]
fn __isaa_hang__() {
    if std::env::args().any(|a| a == "__isaa_hang__") {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(3600));
        }
    }
}

#[tokio::test]
async fn test_jail_blocks_parent_traversal() {
    let policy = WorkspacePolicy {
        root: unique_root("traversal"),
        ..Default::default()
    };

    let err = fs_read(Path::new("../../Cargo.toml"), None, None, &policy)
        .await
        .unwrap_err();
    assert!(
        matches!(err, ToolError::SecurityViolation(_)),
        "got {err:?}"
    );

    let err = fs_write(Path::new("sub/../../../evil.txt"), "x", &policy)
        .await
        .unwrap_err();
    assert!(
        matches!(err, ToolError::SecurityViolation(_)),
        "got {err:?}"
    );
}

#[tokio::test]
async fn test_jail_blocks_absolute_outside_path() {
    let policy = WorkspacePolicy {
        root: unique_root("abs"),
        ..Default::default()
    };
    let outside = std::env::temp_dir()
        .join("isaa_t3_definitely_outside_root")
        .join("evil.txt");
    assert!(outside.is_absolute());

    let err = fs_write(&outside, "x", &policy).await.unwrap_err();
    assert!(
        matches!(err, ToolError::SecurityViolation(_)),
        "got {err:?}"
    );

    let err = fs_read(&outside, None, None, &policy).await.unwrap_err();
    assert!(
        matches!(err, ToolError::SecurityViolation(_)),
        "got {err:?}"
    );
}

#[tokio::test]
async fn test_atomic_write_and_read() {
    let policy = WorkspacePolicy {
        root: unique_root("rw"),
        ..Default::default()
    };
    let content = "hello jail\nsecond line\nthird";

    let wr = fs_write(Path::new("notes/a.txt"), content, &policy)
        .await
        .unwrap();
    assert_eq!(wr.bytes_written, content.len());

    let back = fs_read(Path::new("notes/a.txt"), None, None, &policy)
        .await
        .unwrap();
    assert_eq!(back, content);

    // Zeilen-Fenster: offset=1, limit=1
    let window = fs_read(Path::new("notes/a.txt"), Some(1), Some(1), &policy)
        .await
        .unwrap();
    assert_eq!(window, "second line");
}

#[tokio::test]
async fn test_fs_patch_unified_diff() {
    let policy = WorkspacePolicy {
        root: unique_root("patch"),
        ..Default::default()
    };
    fs_write(Path::new("greet.txt"), "alpha\nbeta\ngamma\n", &policy)
        .await
        .unwrap();

    let patch = "\
--- a/greet.txt
+++ b/greet.txt
@@ -2,1 +2,1 @@
-beta
+BETA
";
    let res = fs_patch(Path::new("greet.txt"), patch, &policy)
        .await
        .unwrap();
    assert_eq!(res.hunks_applied, 1);

    let back = fs_read(Path::new("greet.txt"), None, None, &policy)
        .await
        .unwrap();
    assert_eq!(back, "alpha\nBETA\ngamma");

    // Exact-Match-Pflicht: falscher Kontext -> PatchFailed, keine Fuzzy-Suche.
    let bad = "\
@@ -1,1 +1,1 @@
-ALPHA
+X
";
    let err = fs_patch(Path::new("greet.txt"), bad, &policy)
        .await
        .unwrap_err();
    assert!(matches!(err, ToolError::PatchFailed(_)), "got {err:?}");
    // Datei unverändert nach fehlgeschlagenem Patch (atomar).
    let back = fs_read(Path::new("greet.txt"), None, None, &policy)
        .await
        .unwrap();
    assert_eq!(back, "alpha\nBETA\ngamma");
}

#[tokio::test]
async fn test_proc_exec_isolated() {
    let policy = WorkspacePolicy {
        root: unique_root("exec"),
        allowed_subprocesses: vec!["rustc".to_string()],
        ..Default::default()
    };
    // rustc existiert überall dort, wo cargo test läuft — null Plattform-#cfg.
    let res = proc_exec("rustc", &["--version".to_string()], 30_000, &policy)
        .await
        .unwrap();
    assert_eq!(res.exit_code, Some(0));
    assert!(res.stdout.contains("rustc"), "stdout: {}", res.stdout);
}

#[tokio::test]
async fn test_proc_exec_timeout_kills_endless() {
    let exe = std::env::current_exe().unwrap();
    let policy = WorkspacePolicy {
        root: unique_root("timeout"),
        allowed_subprocesses: vec![exe.to_string_lossy().into_owned()],
        ..Default::default()
    };
    // Endloser Prozess = Test-Binary selbst, re-gespawnt mit Sentinel-Arg
    // (siehe `__isaa_hang__`). Kein Plattform-Binary-Raten, kein #[cfg].
    let err = proc_exec(
        exe.to_str().unwrap(),
        &["__isaa_hang__".to_string()],
        1_500,
        &policy,
    )
    .await
    .unwrap_err();
    assert!(matches!(err, ToolError::Timeout(1_500)), "got {err:?}");
}

#[tokio::test]
async fn test_policy_violation_allowed_list() {
    let args = ["--version".to_string()];

    // Leere Allowed-Liste = Policy greift nicht -> Ausführung erlaubt.
    let open = WorkspacePolicy {
        root: unique_root("pol_empty"),
        allowed_subprocesses: Vec::new(),
        ..Default::default()
    };
    proc_exec("rustc", &args, 30_000, &open).await.unwrap();

    // Gefüllte Liste ohne rustc -> PolicyViolation.
    let strict = WorkspacePolicy {
        root: unique_root("pol_strict"),
        allowed_subprocesses: vec!["python".to_string()],
        ..Default::default()
    };
    let err = proc_exec("rustc", &args, 30_000, &strict)
        .await
        .unwrap_err();
    assert!(matches!(err, ToolError::PolicyViolation(_)), "got {err:?}");
}
