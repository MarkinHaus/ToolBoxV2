// =================== App Singleton Wrapper ===================
// Lädt und verwaltet den Nuitka-kompilierten App Singleton
// app_singleton.rs
use crate::python_ffi::PythonFFI;
use anyhow::{Result, bail};
use serde_json::Value;
use std::sync::Arc;
use tracing::{debug, info};

/// Wrapper für den global App Singleton
pub struct AppSingleton {
    ffi: Arc<PythonFFI>,
    module_name: String, // Name des Moduls
    module_obj: Option<*mut std::ffi::c_void>, // CACHED PyObject für das Modul (WICHTIG: Nicht neu importieren!)
    app_instance: Option<*mut std::ffi::c_void>, // PyObject für App instance
}

unsafe impl Send for AppSingleton {}
unsafe impl Sync for AppSingleton {}

impl AppSingleton {
    /// Erstellt neuen AppSingleton und lädt das Nuitka-Modul
    pub fn new(ffi: Arc<PythonFFI>) -> Result<Self> {

        let module = ffi.import_module("app_singleton")?;

        let app_singleton = Self {
            ffi,
            module_name: "app_singleton".to_string(),
            module_obj: Some(module), // Cache the module PyObject
            app_instance: None,
        };

        Ok(app_singleton)
    }

    /// Initialisiert den globalen App Singleton
    pub fn init_app(&mut self, instance_id: &str) -> Result<Value> {

        // Use cached module (NICHT neu importieren!)
        let module = self.module_obj.ok_or_else(|| anyhow::anyhow!("Module not loaded"))?;
        let init_app_fn = self.ffi.get_attr(module, "init_app")?;

        let args = self.ffi.create_tuple(1)?;
        let id_str = self.ffi.string_from_str(instance_id)?;

        self.ffi.tuple_set_item(args, 0, id_str)?;

        let _result = self.ffi.call_function(init_app_fn, args, None)?;

        info!("App singleton initialized successfully");

        Ok(serde_json::json!({
            "status": "initialized",
            "instance_id": instance_id
        }))
    }

    /// Holt den globalen App Singleton
    pub fn get_app(&mut self) -> Result<*mut std::ffi::c_void> {
        if let Some(app) = self.app_instance {
            return Ok(app);
        }

        debug!("Getting App singleton instance...");

        // Use cached module (NICHT neu importieren!)
        let module = self.module_obj.ok_or_else(|| anyhow::anyhow!("Module not loaded"))?;

        // Hole get_app Funktion
        let get_app_fn = self.ffi.get_attr(module, "get_app")?;

        // Rufe get_app() auf (keine Args)
        let empty_args = self.ffi.create_tuple(0)?;
        let app = self.ffi.call_function(get_app_fn, empty_args, None)?;

        self.app_instance = Some(app);

        debug!("App singleton instance retrieved");

        Ok(app)
    }

    /// Ruft eine Modul-Funktion über den App Singleton auf
    pub fn call_module_function(
        &mut self,
        module: &str,
        function: &str,
        kwargs: Value,
    ) -> Result<Value> {
        info!("Calling module function: {}::{}", module, function);

        // Use cached module (NICHT neu importieren!)
        let app_singleton_module = self.module_obj.ok_or_else(|| anyhow::anyhow!("Module not loaded"))?;

        // Hole json_call Funktion (einfacher als manuelle Konvertierung!)
        let json_call_fn = self.ffi.get_attr(app_singleton_module, "json_call")?;

        // Erstelle JSON-String mit {"module": ..., "function": ..., "kwargs": ...}
        let request_json = serde_json::json!({
            "module": module,
            "function": function,
            "kwargs": kwargs
        });
        let request_str = serde_json::to_string(&request_json)?;

        // Konvertiere zu Python-String
        let py_request_str = self.ffi.string_from_str(&request_str)?;

        // Erstelle Args tuple mit (json_str,)
        let args = self.ffi.create_tuple(1)?;
        self.ffi.tuple_set_item(args, 0, py_request_str)?;

        // Rufe json_call(json_str) auf
        let py_result = self.ffi.call_function(json_call_fn, args, None)?;

        // Konvertiere Python-String zurück zu Rust-String
        let result_str = self.ffi.unicode_as_utf8(py_result)?;

        // Parse JSON-String zu Value
        let result: Value = serde_json::from_str(&result_str)?;

        Ok(result)
    }

    /// Ruft eine Funktion im app_singleton Modul auf (ohne Module/Function-Routing).
    ///
    /// Args:
    ///     function_name: Name der Funktion in app_singleton.py
    ///     kwargs: JSON-Argumente für die Funktion
    pub fn call_function_json(&self, function_name: &str, kwargs: &Value) -> Result<Value> {

        // Use cached module
        let module = self.module_obj.ok_or_else(|| anyhow::anyhow!("Module not loaded"))?;

        // Hole die Funktion
        let function = self.ffi.get_attr(module, function_name)?;

        // Erstelle Args tuple (leer, da wir nur kwargs verwenden)
        let args = self.ffi.create_tuple(0)?;

        // Konvertiere kwargs zu Python-Dict
        let kwargs_str = serde_json::to_string(kwargs)?;
        let py_kwargs_str = self.ffi.string_from_str(&kwargs_str)?;

        // Parse JSON-String zu Python-Dict
        let json_module = self.ffi.import_module("json")?;
        let json_loads = self.ffi.get_attr(json_module, "loads")?;
        let loads_args = self.ffi.create_tuple(1)?;
        self.ffi.tuple_set_item(loads_args, 0, py_kwargs_str)?;
        let py_kwargs = self.ffi.call_function(json_loads, loads_args, None)?;

        // Rufe Funktion auf
        let py_result = self.ffi.call_function(function, args, Some(py_kwargs))?;

        // Konvertiere Ergebnis zu JSON
        let json_dumps = self.ffi.get_attr(json_module, "dumps")?;
        let dumps_args = self.ffi.create_tuple(1)?;
        self.ffi.tuple_set_item(dumps_args, 0, py_result)?;
        let py_result_str = self.ffi.call_function(json_dumps, dumps_args, None)?;

        // Konvertiere zu Rust-String
        let result_str = self.ffi.unicode_as_utf8(py_result_str)?;

        // Parse JSON
        let result: Value = serde_json::from_str(&result_str)?;

        Ok(result)
    }

    /// Health Check
    pub fn health_check(&self) -> Result<Value> {
        debug!("Running health check...");

        // Use cached module (NICHT neu importieren!)
        let module = self.module_obj.ok_or_else(|| anyhow::anyhow!("Module not loaded"))?;

        // Hole health_check Funktion
        let health_fn = self.ffi.get_attr(module, "health_check")?;

        // Rufe health_check() auf
        let empty_args = self.ffi.create_tuple(0)?;
        let result = self.ffi.call_function(health_fn, empty_args, None)?;

        // TODO: Konvertiere Result zu JSON

        Ok(serde_json::json!({
            "status": "healthy",
            "app_singleton": "loaded"
        }))
    }
}

impl Drop for AppSingleton {
    fn drop(&mut self) {
        debug!("Dropping AppSingleton");
        // Python objects werden automatisch durch GC freigegeben
    }
}

// =================== Tests ===================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Deaktiviert bis Python FFI stabil ist
    fn test_load_app_singleton() {
        let ffi = Arc::new(PythonFFI::new().expect("Failed to create FFI"));
        let app = AppSingleton::new(ffi);
        assert!(app.is_ok(), "Should load app_singleton module");
    }

    #[test]
    #[ignore]
    fn test_init_app() {
        let ffi = Arc::new(PythonFFI::new().expect("Failed to create FFI"));
        let mut app = AppSingleton::new(ffi).expect("Failed to load app_singleton");

        let result = app.init_app("test_instance");
        assert!(result.is_ok(), "Should initialize app");
    }

    #[test]
    #[ignore]
    fn test_health_check() {
        let ffi = Arc::new(PythonFFI::new().expect("Failed to create FFI"));
        let app = AppSingleton::new(ffi).expect("Failed to load app_singleton");

        let result = app.health_check();
        assert!(result.is_ok(), "Health check should succeed");
    }
}
