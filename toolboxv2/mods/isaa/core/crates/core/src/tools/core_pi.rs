//! 4 primitive, Pi-inspirierte Core-Tools: `fs_read`, `fs_write`, `fs_patch`,
//! `proc_exec`. Alle nicht-blockierend (Tokio), alle Datei-Zugriffe über das
//! Workspace Jail.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;

use super::error::ToolError;
use super::jail;
use crate::domain::WorkspacePolicy;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteResult {
    pub bytes_written: usize,
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchResult {
    pub hunks_applied: usize,
    pub bytes_written: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecResult {
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

/// Liest eine Textdatei (über Jail) als UTF-8. `offset`/`limit` sind
/// 0-basierte Zeilen-Semantik und schützen vor Context-Explosion.
pub async fn fs_read(
    path: &Path,
    offset: Option<usize>,
    limit: Option<usize>,
    policy: &WorkspacePolicy,
) -> Result<String, ToolError> {
    let p = jail::resolve(policy, path)?;
    let bytes = tokio::fs::read(p).await?;
    let text = String::from_utf8(bytes).map_err(|_| ToolError::InvalidEncoding)?;
    let lines: Vec<&str> = text.lines().collect();
    let start = offset.unwrap_or(0).min(lines.len());
    let end = start
        .saturating_add(limit.unwrap_or(usize::MAX))
        .min(lines.len());
    Ok(lines[start..end].join("\n"))
}

/// Schreibt `content` atomar nach `path` (über Jail): Temp-Datei im selben
/// Verzeichnis, `sync_data()`, dann `rename` übers Ziel.
pub async fn fs_write(
    path: &Path,
    content: &str,
    policy: &WorkspacePolicy,
) -> Result<WriteResult, ToolError> {
    let target = jail::resolve(policy, path)?;
    let bytes_written = atomic_write(&target, content).await?;
    Ok(WriteResult {
        bytes_written,
        path: target,
    })
}

/// Wendet einen Unified-Diff deterministisch an (handgeschriebener
/// Hunk-Parser, Exact-Match-Pflicht). Anwendung atomar über die
/// fs_write-Mechanik.
pub async fn fs_patch(
    path: &Path,
    patch_content: &str,
    policy: &WorkspacePolicy,
) -> Result<PatchResult, ToolError> {
    let target = jail::resolve(policy, path)?;
    let bytes = tokio::fs::read(&target).await?;
    let text = String::from_utf8(bytes).map_err(|_| ToolError::InvalidEncoding)?;
    let hunks = parse_hunks(patch_content)?;

    // ponytail: sauberer Roundtrip nur für \n-only Dateien; CRLF wird beim
    // Patch auf \n normalisiert (Upgrade: split_inclusive-Zeilenerhaltung).
    let had_trailing_nl = text.ends_with('\n');
    let lines: Vec<&str> = if text.is_empty() {
        Vec::new()
    } else {
        text.strip_suffix('\n')
            .unwrap_or(&text)
            .split('\n')
            .collect()
    };

    // Cursor-Verfahren: Hunks beziehen sich auf Original-Zeilennummern,
    // Ausgabe wird in einem Durchgang gebaut — deterministisch, kein Fuzzy.
    let mut out: Vec<&str> = Vec::with_capacity(lines.len());
    let mut cursor = 0usize;
    for h in &hunks {
        let pos = h.old_start.saturating_sub(1);
        if pos < cursor || pos + h.old.len() > lines.len() {
            return Err(ToolError::PatchFailed(format!(
                "hunk @{} out of range",
                h.old_start
            )));
        }
        out.extend_from_slice(&lines[cursor..pos]);
        if lines[pos..pos + h.old.len()] != h.old[..] {
            return Err(ToolError::PatchFailed(format!(
                "context mismatch at original line {}",
                pos + 1
            )));
        }
        out.extend(h.new.iter().map(|s| s.as_str()));
        cursor = pos + h.old.len();
    }
    out.extend_from_slice(&lines[cursor..]);

    let mut new_text = out.join("\n");
    if had_trailing_nl {
        new_text.push('\n');
    }
    let bytes_written = atomic_write(&target, &new_text).await?;
    Ok(PatchResult {
        hunks_applied: hunks.len(),
        bytes_written,
    })
}

/// Führt `cmd` direkt (ohne Zwischen-Shell) mit `args` aus, cwd = Workspace
/// Root. Paralleles Draining von stdout/stderr, hartes Timeout.
pub async fn proc_exec(
    cmd: &str,
    args: &[String],
    timeout_ms: u32,
    policy: &WorkspacePolicy,
) -> Result<ExecResult, ToolError> {
    if !policy.allowed_subprocesses.is_empty()
        && !subprocess_allowed(cmd, &policy.allowed_subprocesses)
    {
        return Err(ToolError::PolicyViolation(cmd.to_string()));
    }
    let cwd = jail::resolve(policy, Path::new("."))?;

    let mut child = Command::new(cmd)
        .args(args)
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    // Beide Pipes parallel drainen — verhindert Pipe-Buffer-Deadlocks.
    let stdout_pipe = child.stdout.take().expect("stdout is piped");
    let stderr_pipe = child.stderr.take().expect("stderr is piped");
    fn drain<R>(mut pipe: R) -> tokio::task::JoinHandle<String>
    where
        R: tokio::io::AsyncRead + Unpin + Send + 'static,
    {
        tokio::spawn(async move {
            let mut buf = Vec::new();
            let _ = pipe.read_to_end(&mut buf).await;
            String::from_utf8_lossy(&buf).into_owned()
        })
    }
    let so = drain(stdout_pipe);
    let se = drain(stderr_pipe);

    match tokio::time::timeout(Duration::from_millis(u64::from(timeout_ms)), child.wait()).await {
        Ok(status) => {
            let status = status?;
            Ok(ExecResult {
                exit_code: status.code(),
                stdout: so.await.unwrap_or_default(),
                stderr: se.await.unwrap_or_default(),
            })
        }
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            let _ = so.await;
            let _ = se.await;
            Err(ToolError::Timeout(timeout_ms))
        }
    }
}

// --- interne Helfer ---

struct Hunk {
    /// 1-basierte Original-Startzeile wie im Diff-Header (0 = leeres Original).
    old_start: usize,
    old: Vec<String>,
    new: Vec<String>,
}

/// Temp-Datei im Zielverzeichnis, sync_data, rename — atomar.
async fn atomic_write(target: &Path, content: &str) -> Result<usize, ToolError> {
    let dir = target
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .ok_or_else(|| ToolError::SecurityViolation(target.display().to_string()))?;
    tokio::fs::create_dir_all(dir).await?;
    let name = target
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".to_string());
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = dir.join(format!(".{name}.{}.{}.tmp", std::process::id(), nanos));
    {
        let mut f = tokio::fs::File::create(&tmp).await?;
        f.write_all(content.as_bytes()).await?;
        f.sync_data().await?;
    }
    tokio::fs::rename(&tmp, target).await?;
    Ok(content.len())
}

fn subprocess_allowed(cmd: &str, allowed: &[String]) -> bool {
    let base = Path::new(cmd)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| cmd.to_string());
    allowed.iter().any(|a| a == cmd || a == &base)
}

fn parse_range(s: &str) -> Option<(usize, usize)> {
    let s = s.trim().trim_start_matches('-');
    let (a, b) = match s.split_once(',') {
        Some((a, b)) => (a, b),
        None => (s, "1"),
    };
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

fn parse_hunks(patch: &str) -> Result<Vec<Hunk>, ToolError> {
    let mut hunks: Vec<Hunk> = Vec::new();
    let mut cur: Option<Hunk> = None;
    for line in patch.lines() {
        if let Some(rest) = line.strip_prefix("@@") {
            if let Some(h) = cur.take() {
                hunks.push(h);
            }
            let ranges = rest.split("@@").next().unwrap_or("");
            let mut nums = ranges.split_whitespace();
            let old = nums.next().and_then(parse_range);
            let new = nums.next().and_then(parse_range);
            let (old_start, _old_len) =
                old.ok_or_else(|| ToolError::PatchFailed(format!("invalid hunk header: {line}")))?;
            let _ =
                new.ok_or_else(|| ToolError::PatchFailed(format!("invalid hunk header: {line}")))?;
            cur = Some(Hunk {
                old_start,
                old: Vec::new(),
                new: Vec::new(),
            });
            continue;
        }
        if let Some(h) = cur.as_mut() {
            let mut chars = line.chars();
            match chars.next() {
                Some(' ') | Some('-') => h.old.push(chars.as_str().to_string()),
                Some('+') => h.new.push(chars.as_str().to_string()),
                Some('\\') => {} // "\ No newline at end of file"
                None => {
                    h.old.push(String::new());
                    h.new.push(String::new());
                }
                Some(_) => {
                    return Err(ToolError::PatchFailed(format!("invalid hunk line: {line}")))
                }
            }
        }
        // Header-Zeilen (---, +++, diff --git) außerhalb von Hunks: ignorieren.
    }
    if let Some(h) = cur.take() {
        hunks.push(h);
    }
    if hunks.is_empty() {
        return Err(ToolError::PatchFailed("no hunks in patch".to_string()));
    }
    Ok(hunks)
}
