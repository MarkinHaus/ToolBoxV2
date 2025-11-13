// =================== App Singleton Wrapper ===================
// Lädt und verwaltet den Nuitka-kompilierten App Singleton

use crate::python_ffi::PythonFFI;
use anyhow::{Result, bail};
use serde_json::Value;
use std::sync::Arc;
use tracing::{debug, info};

/// Wrapper für den global App Singleton
pub struct AppSingleton {
    ffi: Arc<PythonFFI>,
    module_name: String, // Name des Moduls (wird bei Bedarf re-importiert)
    app_instance: Option<*mut std::ffi::c_void>, // PyObject für App instance
}

unsafe impl Send for AppSingleton {}
unsafe impl Sync for AppSingleton {}

impl AppSingleton {
    /// Erstellt neuen AppSingleton und lädt das Nuitka-Modul
    pub fn new(ffi: Arc<PythonFFI>) -> Result<Self> {
        info!("Loading app_singleton module...");

        // Verify module can be imported (but don't store the PyObject)
        info!("Verifying app_singleton module can be imported...");
        let _module = ffi.import_module("app_singleton")?;
        info!("app_singleton module verified successfully!");

        info!("Creating AppSingleton struct...");
        let app_singleton = Self {
            ffi,
            module_name: "app_singleton".to_string(),
            app_instance: None,
        };
        info!("AppSingleton struct created successfully!");

        info!("Returning AppSingleton...");
        Ok(app_singleton)
    }

    /// Initialisiert den globalen App Singleton
    pub fn init_app(&mut self, instance_id: &str) -> Result<Value> {
        info!("Initializing App singleton with instance_id: {}", instance_id);

        // Re-import module to get a fresh PyObject
        info!("Re-importing {} module...", self.module_name);
        let module = self.ffi.import_module(&self.module_name)?;
        info!("Module re-imported successfully");

        // Hole init_app Funktion
        info!("Getting init_app function from module...");
        let init_app_fn = self.ffi.get_attr(module, "init_app")?;
        info!("Got init_app function successfully");

        // Erstelle Args tuple mit instance_id
        info!("Creating args tuple...");
        let args = self.ffi.create_tuple(1)?;
        let id_str = self.ffi.string_from_str(instance_id)?;
        info!("Created args tuple successfully");

        // Setze args[0] = instance_id
        info!("Setting args[0] = instance_id...");
        self.ffi.tuple_set_item(args, 0, id_str)?;
        info!("Set args[0] successfully");

        // Rufe init_app(instance_id) auf
        info!("Calling init_app(instance_id) - THIS MAY TAKE 1-2 MINUTES...");
        let _result = self.ffi.call_function(init_app_fn, args, None)?;
        info!("init_app() returned successfully!");

        // Konvertiere Result zu JSON
        // TODO: Implementiere py_to_json Konvertierung

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

        // Re-import module to get a fresh PyObject
        let module = self.ffi.import_module(&self.module_name)?;

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

        // Re-import module to get a fresh PyObject
        let app_singleton_module = self.ffi.import_module(&self.module_name)?;

        // Hole call_module_function
        let call_fn = self.ffi.get_attr(app_singleton_module, "call_module_function")?;

        // Erstelle Args tuple (module, function)
        let args = self.ffi.create_tuple(2)?;
        // TODO: Setze args[0] = module, args[1] = function

        // Erstelle kwargs dict
        let kwargs_dict = self.ffi.create_dict()?;
        // TODO: Konvertiere JSON kwargs zu Python dict

        // Rufe call_module_function(module, function, **kwargs) auf
        let result = self.ffi.call_function(call_fn, args, Some(kwargs_dict))?;

        // TODO: Konvertiere Result zu JSON

        Ok(serde_json::json!({
            "status": "success",
            "module": module,
            "function": function
        }))
    }

    /// Health Check
    pub fn health_check(&self) -> Result<Value> {
        debug!("Running health check...");

        // Re-import module to get a fresh PyObject
        let module = self.ffi.import_module(&self.module_name)?;

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

