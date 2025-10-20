use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;
use tb_core::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactEntry {
    pub hash: String,
    pub artifact_type: ArtifactType,
    pub path: PathBuf,
    pub created_at: SystemTime,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactType {
    NativeBinary,
    RustSource,
    SharedLibrary,
    Plugin,
}

/// Cache for compiled artifacts (binaries, libraries, etc.)
pub struct ArtifactCache {
    #[allow(dead_code)]
    cache_dir: PathBuf,
    artifacts: DashMap<String, ArtifactEntry>,
}

impl ArtifactCache {
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&cache_dir)?;

        Ok(Self {
            cache_dir,
            artifacts: DashMap::new(),
        })
    }

    pub fn get(&self, hash: &str) -> Option<PathBuf> {
        self.artifacts.get(hash).map(|e| e.value().path.clone())
    }

    pub fn put(&self, hash: String, artifact_type: ArtifactType, path: PathBuf) -> Result<()> {
        let size_bytes = fs::metadata(&path)?.len();

        let entry = ArtifactEntry {
            hash: hash.clone(),
            artifact_type,
            path,
            created_at: SystemTime::now(),
            size_bytes,
        };

        self.artifacts.insert(hash, entry);
        Ok(())
    }
}

