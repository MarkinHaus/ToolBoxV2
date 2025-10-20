use im::HashMap as ImHashMap;
use std::sync::Arc;
use tb_core::Type;

/// Type environment using persistent data structures for O(1) cloning
#[derive(Debug, Clone)]
pub struct TypeEnvironment {
    // Persistent HashMap - structural sharing
    bindings: ImHashMap<Arc<String>, Type>,
    parent: Option<Box<TypeEnvironment>>,
}

impl TypeEnvironment {
    pub fn new() -> Self {
        Self {
            bindings: ImHashMap::new(),
            parent: None,
        }
    }

    pub fn with_parent(parent: TypeEnvironment) -> Self {
        Self {
            bindings: ImHashMap::new(),
            parent: Some(Box::new(parent)),
        }
    }

    /// O(log n) insert with structural sharing
    pub fn define(&mut self, name: Arc<String>, ty: Type) {
        self.bindings.insert(name, ty);
    }

    /// Lookup with parent chain traversal
    pub fn lookup(&self, name: &str) -> Option<&Type> {
        self.bindings.get(&Arc::new(name.to_string()))
            .or_else(|| self.parent.as_ref().and_then(|p| p.lookup(name)))
    }

    /// Get all bindings (for debugging)
    pub fn bindings(&self) -> &ImHashMap<Arc<String>, Type> {
        &self.bindings
    }

    /// Create child environment (O(1) due to structural sharing)
    pub fn child(&self) -> TypeEnvironment {
        TypeEnvironment::with_parent(self.clone())
    }
}

impl Default for TypeEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

