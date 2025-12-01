// Nuitka Builder - Kompiliert Python Module mit Nuitka und verwaltet Build Cache

use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use chrono::{DateTime, Utc};
use tracing::{info, warn, debug};

/// Build Cache Eintrag für ein Modul
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildCacheEntry {
    /// Name des Moduls
    pub module_name: String,
    /// SHA256 Hash der Source-Datei
    pub source_hash: String,
    /// Zeitpunkt des letzten Builds
    pub build_time: DateTime<Utc>,
    /// Pfad zur kompilierten .pyd Datei
    pub output_path: PathBuf,
}

/// Nuitka Builder - Verwaltet Kompilierung von Python Modulen
pub struct NuitkaBuilder {
    /// Pfad zur .build_cache.json Datei
    cache_file: PathBuf,
    /// Build Cache (module_name -> BuildCacheEntry)
    cache: HashMap<String, BuildCacheEntry>,
    /// Pfad zum Nuitka Executable (optional, nutzt "nuitka" aus PATH wenn None)
    nuitka_path: Option<PathBuf>,
}

impl NuitkaBuilder {
    /// Erstellt einen neuen NuitkaBuilder
    pub fn new(cache_file: PathBuf) -> Result<Self> {
        let mut builder = Self {
            cache_file,
            cache: HashMap::new(),
            nuitka_path: None,
        };

        // Lade existierenden Cache
        builder.load_cache()?;

        Ok(builder)
    }

    /// Setzt den Pfad zum Nuitka Executable
    pub fn with_nuitka_path(mut self, path: PathBuf) -> Self {
        self.nuitka_path = Some(path);
        self
    }

    /// Lädt den Build Cache von Disk
    fn load_cache(&mut self) -> Result<()> {
        if !self.cache_file.exists() {
            info!("Build cache file not found, starting with empty cache");
            return Ok(());
        }

        let content = fs::read_to_string(&self.cache_file)
            .context("Failed to read build cache file")?;

        let entries: Vec<BuildCacheEntry> = serde_json::from_str(&content)
            .context("Failed to parse build cache JSON")?;

        self.cache = entries.into_iter()
            .map(|entry| (entry.module_name.clone(), entry))
            .collect();

        info!("Loaded {} entries from build cache", self.cache.len());
        Ok(())
    }

    /// Speichert den Build Cache auf Disk
    fn save_cache(&self) -> Result<()> {
        let entries: Vec<&BuildCacheEntry> = self.cache.values().collect();
        let json = serde_json::to_string_pretty(&entries)
            .context("Failed to serialize build cache")?;

        fs::write(&self.cache_file, json)
            .context("Failed to write build cache file")?;

        debug!("Saved {} entries to build cache", entries.len());
        Ok(())
    }

    /// Berechnet SHA256 Hash einer Datei
    pub fn calculate_hash(file_path: &Path) -> Result<String> {
        let content = fs::read(file_path)
            .with_context(|| format!("Failed to read file: {:?}", file_path))?;

        let mut hasher = Sha256::new();
        hasher.update(&content);
        let hash = hasher.finalize();

        Ok(format!("{:x}", hash))
    }

    /// Prüft ob ein Modul neu gebaut werden muss
    pub fn needs_rebuild(&self, module_name: &str, source_path: &Path) -> Result<bool> {
        // Wenn kein Cache-Eintrag existiert, muss gebaut werden
        let Some(cache_entry) = self.cache.get(module_name) else {
            info!("Module '{}' not in cache, needs rebuild", module_name);
            return Ok(true);
        };

        // Wenn Output-Datei nicht existiert, muss gebaut werden
        if !cache_entry.output_path.exists() {
            warn!("Output file {:?} missing, needs rebuild", cache_entry.output_path);
            return Ok(true);
        }

        // Berechne aktuellen Hash
        let current_hash = Self::calculate_hash(source_path)?;

        // Vergleiche mit Cache
        if current_hash != cache_entry.source_hash {
            info!("Module '{}' hash changed, needs rebuild", module_name);
            info!("  Old: {}", cache_entry.source_hash);
            info!("  New: {}", current_hash);
            return Ok(true);
        }

        debug!("Module '{}' up-to-date, no rebuild needed", module_name);
        Ok(false)
    }

    /// Kompiliert ein Python Modul mit Nuitka
    pub fn build_module(
        &mut self,
        module_name: &str,
        source_path: &Path,
        output_dir: &Path,
    ) -> Result<PathBuf> {
        info!("Building module '{}' from {:?}", module_name, source_path);

        // Output-Verzeichnis erstellen
        fs::create_dir_all(output_dir)
            .context("Failed to create output directory")?;

        // Nuitka Command Cross-Platform Logik
        let (cmd_name, cmd_args) = if let Some(path) = &self.nuitka_path {
            // Benutzerdefinierter Pfad
            (path.to_string_lossy().to_string(), vec![])
        } else if cfg!(target_os = "windows") {
            // Windows: Oft ist 'nuitka' ein Batch-File oder nicht direkt ausführbar.
            // Sicherer ist der Aufruf über Python.
            ("python".to_string(), vec!["-m", "nuitka"])
        } else {
            // Linux/Mac: 'nuitka' oder 'nuitka3' ist meist im PATH
            ("nuitka3".to_string(), vec![]) // Versuche nuitka3, fallback logik nötig wenn nur nuitka da ist
        };

        // Fallback: Wenn 'nuitka3' nicht da ist, nimm 'nuitka', oder 'python3 -m nuitka'
        // Für dieses Beispiel nehmen wir an, der User hat "python -m nuitka" als sichersten Weg
        let (final_cmd, initial_args) = if self.nuitka_path.is_some() {
            (cmd_name, cmd_args)
        } else {
            // Cross-Platform Safe-Bet: Python Modul Aufruf
            // Voraussetzung: python/python3 ist im PATH
            #[cfg(target_os = "windows")]
            let py = "python";
            #[cfg(not(target_os = "windows"))]
            let py = "python3";

            (py.to_string(), vec!["-m", "nuitka"])
        };

        let mut cmd = Command::new(&final_cmd);

        // Initiale Argumente (z.B. ["-m", "nuitka"])
        for arg in initial_args {
            cmd.arg(arg);
        }

        // Nuitka Argumente
        cmd.arg("--module")
            .arg(source_path)
            .arg("--output-dir")
            .arg(output_dir)
            .arg("--remove-output")
            .arg("--assume-yes-for-downloads");

        // Auf Linux/Mac ist --clang oft besser, auf Windows msvc
        #[cfg(not(target_os = "windows"))]
        cmd.arg("--lto=yes"); // Link Time Optimization auf Unix oft gut

        info!("Running: {:?}", cmd);

        let output = cmd.output()
            .context(format!("Failed to execute compilation command: {:?}", final_cmd))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            anyhow::bail!("Nuitka compilation failed.\nSTDERR:\n{}\nSTDOUT:\n{}", stderr, stdout);
        }

        // Finde die generierte Datei (OS-agnostisch durch find_output_file)
        let output_file = self.find_output_file(output_dir, module_name)?;

        // Cache updaten
        let source_hash = Self::calculate_hash(source_path)?;
        let cache_entry = BuildCacheEntry {
            module_name: module_name.to_string(),
            source_hash,
            build_time: Utc::now(),
            output_path: output_file.clone(),
        };

        self.cache.insert(module_name.to_string(), cache_entry);
        self.save_cache()?;

        info!("Successfully built module '{}' -> {:?}", module_name, output_file);
        Ok(output_file)
    }

    /// Findet die generierte .pyd Datei im Output-Verzeichnis
    fn find_output_file(&self, output_dir: &Path, module_name: &str) -> Result<PathBuf> {
        // Bestimme die Dateiendung basierend auf dem Betriebssystem
        #[cfg(target_os = "windows")]
        let ext = ".pyd";
        #[cfg(not(target_os = "windows"))]
        let ext = ".so";

        debug!("Looking for file starting with '{}' and ending with '{}' in {:?}", module_name, ext, output_dir);

        let entries = fs::read_dir(output_dir)
            .context("Failed to read output directory")?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if let Some(filename) = path.file_name() {
                let filename_str = filename.to_string_lossy();
                // Nuitka generiert oft Namen wie "module.cpython-39-x86_64-linux-gnu.so"
                // Daher prüfen wir start und end.
                if filename_str.starts_with(module_name) && filename_str.ends_with(ext) {
                    return Ok(path);
                }
            }
        }

        anyhow::bail!("Could not find output file (extension '{}') for module '{}'", ext, module_name)
    }

    /// Baut ein Modul nur wenn nötig (Hash-basiert)
    pub fn build_if_needed(
        &mut self,
        module_name: &str,
        source_path: &Path,
        output_dir: &Path,
    ) -> Result<PathBuf> {
        if self.needs_rebuild(module_name, source_path)? {
            self.build_module(module_name, source_path, output_dir)
        } else {
            // Gib existierenden Output-Pfad zurück
            let cache_entry = self.cache.get(module_name)
                .context("Module should be in cache")?;
            Ok(cache_entry.output_path.clone())
        }
    }

    /// Gibt den Build Cache zurück
    pub fn get_cache(&self) -> &HashMap<String, BuildCacheEntry> {
        &self.cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_calculate_hash() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.py");

        fs::write(&file_path, "print('hello')").unwrap();

        let hash1 = NuitkaBuilder::calculate_hash(&file_path).unwrap();
        let hash2 = NuitkaBuilder::calculate_hash(&file_path).unwrap();

        // Gleicher Inhalt = gleicher Hash
        assert_eq!(hash1, hash2);

        // Ändere Datei
        fs::write(&file_path, "print('world')").unwrap();
        let hash3 = NuitkaBuilder::calculate_hash(&file_path).unwrap();

        // Unterschiedlicher Inhalt = unterschiedlicher Hash
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_cache_load_save() {
        let temp_dir = TempDir::new().unwrap();
        let cache_file = temp_dir.path().join(".build_cache.json");

        // Erstelle Builder und füge Eintrag hinzu
        let mut builder = NuitkaBuilder::new(cache_file.clone()).unwrap();

        let entry = BuildCacheEntry {
            module_name: "test_module".to_string(),
            source_hash: "abc123".to_string(),
            build_time: Utc::now(),
            output_path: PathBuf::from("/tmp/test.pyd"),
        };

        builder.cache.insert("test_module".to_string(), entry.clone());
        builder.save_cache().unwrap();

        // Lade Cache in neuem Builder
        let builder2 = NuitkaBuilder::new(cache_file).unwrap();

        assert_eq!(builder2.cache.len(), 1);
        assert_eq!(builder2.cache.get("test_module").unwrap().source_hash, "abc123");
    }
}

