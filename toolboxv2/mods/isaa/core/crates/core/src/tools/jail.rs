//! Workspace Jail: erzwingt, dass jeder Dateizugriff strikt innerhalb von
//! `WorkspacePolicy.root` bleibt.

use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

use super::error::ToolError;
use crate::domain::WorkspacePolicy;

/// Kanonisiert `path` (via `std::fs::canonicalize`) und erzwingt die
/// Root-Invariante `target.starts_with(root)`. Traversierungsversuche über
/// `..`-Komponenten, absolute Fremdpfade und Symlink-Breakouts werden mit
/// [`ToolError::SecurityViolation`] abgelehnt.
///
/// Existiert das Ziel (noch) nicht (z. B. bei `fs_write`), wird der
/// nächstliegende existierende Parent-Ordner kanonisiert und geprüft; die
/// verbleibenden (nicht existierenden) Komponenten werden angehängt.
pub fn resolve(policy: &WorkspacePolicy, path: &Path) -> Result<PathBuf, ToolError> {
    let root = std::fs::canonicalize(&policy.root)
        .map_err(|_| ToolError::SecurityViolation(format!("{:?}", policy.root)))?;

    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };

    // `..`-Tricks werden grundsätzlich abgelehnt (auch wenn sie im Root landen
    // würden) — kein weiches Normalisieren.
    if joined.components().any(|c| c == Component::ParentDir) {
        return Err(ToolError::SecurityViolation(path.display().to_string()));
    }

    match std::fs::canonicalize(&joined) {
        Ok(target) => {
            if !target.starts_with(&root) {
                return Err(ToolError::SecurityViolation(path.display().to_string()));
            }
            Ok(target)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Nächstliegenden existierenden Ancestor kanonisieren und Rest
            // (der keine Symlinks enthalten kann) deterministisch anhängen.
            let mut suffix: Vec<OsString> = Vec::new();
            let mut anc: &Path = joined.as_path();
            loop {
                let (Some(parent), Some(name)) = (anc.parent(), anc.file_name()) else {
                    return Err(ToolError::SecurityViolation(path.display().to_string()));
                };
                if parent.as_os_str().is_empty() {
                    return Err(ToolError::SecurityViolation(path.display().to_string()));
                }
                let owned = name.to_os_string();
                anc = parent;
                suffix.push(owned);
                if let Ok(canon_parent) = std::fs::canonicalize(anc) {
                    if !canon_parent.starts_with(&root) {
                        return Err(ToolError::SecurityViolation(path.display().to_string()));
                    }
                    let mut out = canon_parent;
                    for comp in suffix.iter().rev() {
                        out.push(comp);
                    }
                    return Ok(out);
                }
            }
        }
        Err(e) => Err(e.into()),
    }
}
