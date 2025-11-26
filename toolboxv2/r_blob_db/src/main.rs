use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{delete, get, post, put},
    Json, Router,
};
use bytes::Bytes;
use dashmap::{DashMap, DashSet};
use reed_solomon_erasure::galois_8::Field;
use reed_solomon_erasure::ReedSolomon;
use serde::{Deserialize, Serialize};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    net::SocketAddr,
    path::PathBuf,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    fs::{self, File},
    sync::broadcast,
};
use tracing::{error, info, instrument, warn};
use uuid::Uuid;

// --- CONFIGURATION ---
const STORAGE_ENV: &str = "R_BLOB_DB_DATA_DIR";
const DEFAULT_STORAGE: &str = "./data_blobs";

// Persistence file names
const PERMISSIONS_FILE: &str = "sys_permissions.json";
const USER_KEYS_FILE: &str = "sys_user_keys.json";
const KEY_USERS_FILE: &str = "sys_key_users.json";
const BLOB_PERMISSIONS_FILE: &str = "sys_blob_permissions.json";
const BLOB_LISTENERS_FILE: &str = "sys_blob_listeners.json";

// --- DATA STRUCTURES ---

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BlobMeta {
    pub version: u64,
    pub size: u64,
    pub data_shards: usize,
    pub parity_shards: usize,
    pub created_at: u64,
    pub shard_locations: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ShardMeta {
    pub original_blob_id: String,
    pub shard_index: usize,
    pub version: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AccessLevel {
    ReadOnly,
    ReadWrite,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BlobPermission {
    pub api_key: String,
    pub user_id: String,
    pub access_level: AccessLevel,
    pub granted_by: String,
    pub granted_at: u64,
}

#[derive(Deserialize, Debug)]
struct PutRequestParams {
    #[serde(default)]
    peers: Vec<String>,
    #[serde(default = "default_ds")]
    data_shards: usize,
    #[serde(default = "default_ps")]
    parity_shards: usize,
}

fn default_ds() -> usize { 4 }
fn default_ps() -> usize { 2 }

// In-Memory State
struct AppState {
    storage_path: PathBuf,

    // Blob-based permissions
    blob_permissions: DashMap<String, Vec<BlobPermission>>,

    // User <-> Key mappings (NOW PERSISTED)
    user_keys: DashMap<String, String>,
    key_users: DashMap<String, String>,

    // Legacy permissions
    permissions: DashMap<String, DashSet<String>>,

    // Notification system
    blob_listeners: DashMap<String, DashSet<String>>,
    notify_channels: DashMap<String, broadcast::Sender<String>>,

    http_client: reqwest::Client,
}

// --- ERROR HANDLING ---
#[derive(thiserror::Error, Debug)]
enum AppError {
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Not Found")]
    NotFound,
    #[error("Unauthorized")]
    Unauthorized,
    #[error("Forbidden: Insufficient Permissions")]
    Forbidden,
    #[error("Conflict: Version Mismatch")]
    Conflict,
    #[error("Bad Request: {0}")]
    BadRequest(String),
    #[error("ReedSolomon Error: {0}")]
    ReedSolomon(String),
    #[error("Reqwest Error: {0}")]
    Reqwest(#[from] reqwest::Error),
    #[error("Serialization Error: {0}")]
    Serde(#[from] serde_json::Error),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, msg) = match self {
            AppError::NotFound => (StatusCode::NOT_FOUND, "Not Found".to_string()),
            AppError::Unauthorized => (StatusCode::UNAUTHORIZED, "Invalid API Key".to_string()),
            AppError::Forbidden => (StatusCode::FORBIDDEN, "Insufficient Permissions".to_string()),
            AppError::Conflict => (StatusCode::CONFLICT, "Version Conflict - Reload Data".to_string()),
            AppError::BadRequest(e) => (StatusCode::BAD_REQUEST, e),
            _ => {
                error!("Internal Error: {:?}", self);
                (StatusCode::INTERNAL_SERVER_ERROR, "Internal Error".to_string())
            }
        };
        (status, msg).into_response()
    }
}

// --- HELPER FUNCTIONS ---

fn get_api_key(headers: &HeaderMap) -> Result<String, AppError> {
    headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .ok_or(AppError::Unauthorized)
}

fn generate_user_id(device_name: &str) -> String {
    let mut hasher = DefaultHasher::new();
    device_name.hash(&mut hasher);
    let hash = hasher.finish();
    format!("user_{:x}", hash)
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

// --- PERSISTENCE ---

/// Load a DashMap<String, DashSet<String>> from JSON
async fn load_dashmap_dashset(path: &PathBuf) -> DashMap<String, DashSet<String>> {
    if !path.exists() {
        return DashMap::new();
    }
    match fs::read(path).await {
        Ok(data) => {
            let parsed: Result<std::collections::HashMap<String, Vec<String>>, _> =
                serde_json::from_slice(&data);
            match parsed {
                Ok(map) => {
                    let dm = DashMap::new();
                    for (k, v) in map {
                        let ds = DashSet::new();
                        for item in v {
                            ds.insert(item);
                        }
                        dm.insert(k, ds);
                    }
                    dm
                }
                Err(e) => {
                    warn!("Failed to parse {:?}: {}", path, e);
                    DashMap::new()
                }
            }
        }
        Err(e) => {
            warn!("Failed to read {:?}: {}", path, e);
            DashMap::new()
        }
    }
}

/// Save a DashMap<String, DashSet<String>> to JSON
async fn save_dashmap_dashset(path: &PathBuf, map: &DashMap<String, DashSet<String>>) -> Result<(), AppError> {
    let mut serializable: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    for entry in map.iter() {
        let key = entry.key().clone();
        let values: Vec<String> = entry.value().iter().map(|r| r.key().clone()).collect();
        serializable.insert(key, values);
    }
    let json = serde_json::to_vec_pretty(&serializable)?;
    fs::write(path, json).await?;
    Ok(())
}

/// Load a DashMap<String, String> from JSON
async fn load_dashmap_string(path: &PathBuf) -> DashMap<String, String> {
    if !path.exists() {
        return DashMap::new();
    }
    match fs::read(path).await {
        Ok(data) => {
            let parsed: Result<std::collections::HashMap<String, String>, _> =
                serde_json::from_slice(&data);
            match parsed {
                Ok(map) => {
                    let dm = DashMap::new();
                    for (k, v) in map {
                        dm.insert(k, v);
                    }
                    dm
                }
                Err(e) => {
                    warn!("Failed to parse {:?}: {}", path, e);
                    DashMap::new()
                }
            }
        }
        Err(e) => {
            warn!("Failed to read {:?}: {}", path, e);
            DashMap::new()
        }
    }
}

/// Save a DashMap<String, String> to JSON
async fn save_dashmap_string(path: &PathBuf, map: &DashMap<String, String>) -> Result<(), AppError> {
    let serializable: std::collections::HashMap<String, String> =
        map.iter().map(|r| (r.key().clone(), r.value().clone())).collect();
    let json = serde_json::to_vec_pretty(&serializable)?;
    fs::write(path, json).await?;
    Ok(())
}

/// Load blob permissions from JSON
async fn load_blob_permissions(path: &PathBuf) -> DashMap<String, Vec<BlobPermission>> {
    if !path.exists() {
        return DashMap::new();
    }
    match fs::read(path).await {
        Ok(data) => {
            let parsed: Result<std::collections::HashMap<String, Vec<BlobPermission>>, _> =
                serde_json::from_slice(&data);
            match parsed {
                Ok(map) => {
                    let dm = DashMap::new();
                    for (k, v) in map {
                        dm.insert(k, v);
                    }
                    dm
                }
                Err(e) => {
                    warn!("Failed to parse {:?}: {}", path, e);
                    DashMap::new()
                }
            }
        }
        Err(e) => {
            warn!("Failed to read {:?}: {}", path, e);
            DashMap::new()
        }
    }
}

/// Save blob permissions to JSON
async fn save_blob_permissions(path: &PathBuf, map: &DashMap<String, Vec<BlobPermission>>) -> Result<(), AppError> {
    let serializable: std::collections::HashMap<String, Vec<BlobPermission>> =
        map.iter().map(|r| (r.key().clone(), r.value().clone())).collect();
    let json = serde_json::to_vec_pretty(&serializable)?;
    fs::write(path, json).await?;
    Ok(())
}

/// Initialize notify channels for all known API keys
fn init_notify_channels(
    permissions: &DashMap<String, DashSet<String>>,
    key_users: &DashMap<String, String>,
) -> DashMap<String, broadcast::Sender<String>> {
    let channels = DashMap::new();

    // Add channels for all keys in legacy permissions
    for entry in permissions.iter() {
        let key = entry.key().clone();
        if !channels.contains_key(&key) {
            let (tx, _) = broadcast::channel(100);
            channels.insert(key, tx);
        }
    }

    // Add channels for all keys in key_users
    for entry in key_users.iter() {
        let key = entry.key().clone();
        if !channels.contains_key(&key) {
            let (tx, _) = broadcast::channel(100);
            channels.insert(key, tx);
        }
    }

    info!("Initialized {} notification channels", channels.len());
    channels
}

/// Ensure a notify channel exists for a key, creating one if needed
fn ensure_notify_channel(state: &Arc<AppState>, key: &str) {
    if !state.notify_channels.contains_key(key) {
        let (tx, _) = broadcast::channel(100);
        state.notify_channels.insert(key.to_string(), tx);
        info!("Created notify channel for key: {}...", &key[..8.min(key.len())]);
    }
}

/// Save all persistent state
async fn save_all_state(state: &Arc<AppState>) -> Result<(), AppError> {
    let storage_path = &state.storage_path;

    // Save permissions
    save_dashmap_dashset(&storage_path.join(PERMISSIONS_FILE), &state.permissions).await?;

    // Save user_keys
    save_dashmap_string(&storage_path.join(USER_KEYS_FILE), &state.user_keys).await?;

    // Save key_users
    save_dashmap_string(&storage_path.join(KEY_USERS_FILE), &state.key_users).await?;

    // Save blob_permissions
    save_blob_permissions(&storage_path.join(BLOB_PERMISSIONS_FILE), &state.blob_permissions).await?;

    // Save blob_listeners
    save_dashmap_dashset(&storage_path.join(BLOB_LISTENERS_FILE), &state.blob_listeners).await?;

    Ok(())
}

// --- MAIN SERVER LOGIC ---

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let path_str = std::env::var(STORAGE_ENV).unwrap_or(DEFAULT_STORAGE.into());
    let storage_path = PathBuf::from(path_str);
    fs::create_dir_all(&storage_path).await?;
    info!("Mounting storage at: {:?}", storage_path);

    // Load all persistent state
    info!("Loading persistent state...");
    let permissions = load_dashmap_dashset(&storage_path.join(PERMISSIONS_FILE)).await;
    let user_keys = load_dashmap_string(&storage_path.join(USER_KEYS_FILE)).await;
    let key_users = load_dashmap_string(&storage_path.join(KEY_USERS_FILE)).await;
    let blob_permissions = load_blob_permissions(&storage_path.join(BLOB_PERMISSIONS_FILE)).await;
    let blob_listeners = load_dashmap_dashset(&storage_path.join(BLOB_LISTENERS_FILE)).await;

    info!("Loaded {} permissions, {} users, {} blob permissions",
          permissions.len(), user_keys.len(), blob_permissions.len());

    // Initialize notify channels for ALL known keys
    let notify_channels = init_notify_channels(&permissions, &key_users);

    let state = Arc::new(AppState {
        storage_path,
        blob_permissions,
        user_keys,
        key_users,
        permissions,
        blob_listeners,
        notify_channels,
        http_client: reqwest::Client::new(),
    });

    // Auto-Save Task (every 30s)
    let state_clone = state.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(30)).await;
            if let Err(e) = save_all_state(&state_clone).await {
                error!("Failed to save state: {}", e);
            }
        }
    });

    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        // Key Management
        .route("/keys", post(create_key))
        .route("/keys/{key}", delete(revoke_key))
        .route("/keys/validate", get(validate_key))  // NEW: Key validation endpoint
        .route("/permissions/{blob_id}", post(grant_permission))
        // Sharing API
        .route("/share/{blob_id}", post(share_blob).get(list_shares))
        .route("/share/{blob_id}/{user_id}", delete(revoke_share))
        // Blob Operations
        .route("/blob/{id}", put(upload_blob).get(read_blob).delete(delete_blob))
        .route("/blob/{id}/meta", get(read_meta))
        // Shard Handling
        .route("/shard/{id}/{index}", post(receive_shard).get(read_shard))
        // Realtime
        .route("/watch", get(watch_changes))
        .with_state(state);

    let port = std::env::var("R_BLOB_DB_PORT").unwrap_or("3000".into());
    let addr: SocketAddr = format!("0.0.0.0:{}", port).parse()?;
    info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// --- HANDLERS ---

// 1. API Key Management
#[derive(Serialize)]
struct KeyResponse {
    key: String,
    user_id: String,
}

#[derive(Deserialize)]
struct CreateKeyRequest {
    device_name: Option<String>,
}

async fn create_key(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateKeyRequest>
) -> Json<KeyResponse> {
    let key = Uuid::new_v4().to_string();

    let user_id = if let Some(device_name) = payload.device_name {
        generate_user_id(&device_name)
    } else {
        format!("user_{}", &key[..8])
    };

    // Store mappings
    state.user_keys.insert(user_id.clone(), key.clone());
    state.key_users.insert(key.clone(), user_id.clone());

    // Legacy permission system
    state.permissions.insert(key.clone(), DashSet::new());

    // Initialize notification channel
    let (tx, _) = broadcast::channel(100);
    state.notify_channels.insert(key.clone(), tx);

    // Save state immediately
    let state_clone = state.clone();
    tokio::spawn(async move {
        if let Err(e) = save_all_state(&state_clone).await {
            error!("Failed to save state after key creation: {}", e);
        }
    });

    info!("Created API key for user_id: {}", user_id);
    Json(KeyResponse { key, user_id })
}

/// NEW: Validate an existing API key and re-register it if needed
async fn validate_key(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, AppError> {
    let key = get_api_key(&headers)?;

    // Check if key exists in any of our systems
    let exists_in_permissions = state.permissions.contains_key(&key);
    let exists_in_key_users = state.key_users.contains_key(&key);

    if !exists_in_permissions && !exists_in_key_users {
        return Err(AppError::Unauthorized);
    }

    // Ensure notify channel exists (self-healing)
    ensure_notify_channel(&state, &key);

    // Ensure key is in permissions if not already
    if !exists_in_permissions {
        state.permissions.insert(key.clone(), DashSet::new());
    }

    let user_id = state.key_users.get(&key)
        .map(|r| r.value().clone())
        .unwrap_or_else(|| format!("user_{}", &key[..8]));

    Ok(Json(serde_json::json!({
        "valid": true,
        "user_id": user_id
    })))
}

async fn revoke_key(
    State(state): State<Arc<AppState>>,
    Path(key): Path<String>,
    headers: HeaderMap
) -> Result<StatusCode, AppError> {
    let _requester = get_api_key(&headers)?;

    // Remove from all maps
    if let Some((_, user_id)) = state.key_users.remove(&key) {
        state.user_keys.remove(&user_id);
    }
    state.permissions.remove(&key);
    state.notify_channels.remove(&key);

    // Remove from blob_permissions
    for mut entry in state.blob_permissions.iter_mut() {
        entry.value_mut().retain(|p| p.api_key != key);
    }

    // Remove from blob_listeners
    for entry in state.blob_listeners.iter() {
        entry.value().remove(&key);
    }

    // Save state
    let state_clone = state.clone();
    tokio::spawn(async move {
        if let Err(e) = save_all_state(&state_clone).await {
            error!("Failed to save state after key revocation: {}", e);
        }
    });

    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
struct GrantReq { key_to_grant: String }

async fn grant_permission(
    State(state): State<Arc<AppState>>,
    Path(blob_id): Path<String>,
    headers: HeaderMap,
    Json(payload): Json<GrantReq>
) -> Result<StatusCode, AppError> {
    let requester = get_api_key(&headers)?;
    check_access(&state, &requester, &blob_id).await?;

    // Ensure notify channel exists for target key
    ensure_notify_channel(&state, &payload.key_to_grant);

    if let Some(set) = state.permissions.get(&payload.key_to_grant) {
        set.insert(blob_id.clone());

        let listener_entry = state.blob_listeners.entry(blob_id.clone()).or_insert_with(DashSet::new);
        listener_entry.insert(payload.key_to_grant.clone());
    } else {
        return Err(AppError::BadRequest("Target Key does not exist".into()));
    }

    let state_clone = state.clone();
    tokio::spawn(async move {
        if let Err(e) = save_all_state(&state_clone).await {
            error!("Failed to save state: {}", e);
        }
    });

    Ok(StatusCode::OK)
}

// 2. Sharing API

#[derive(Deserialize)]
struct ShareRequest {
    user_id: String,
    access_level: AccessLevel,
}

#[derive(Serialize)]
struct ShareResponse {
    blob_id: String,
    user_id: String,
    access_level: AccessLevel,
    granted_at: u64,
}

async fn share_blob(
    State(state): State<Arc<AppState>>,
    Path(blob_id): Path<String>,
    headers: HeaderMap,
    Json(payload): Json<ShareRequest>
) -> Result<Json<ShareResponse>, AppError> {
    let requester_key = get_api_key(&headers)?;

    let requester_user_id = state.key_users.get(&requester_key)
        .map(|r| r.value().clone())
        .ok_or(AppError::Unauthorized)?;

    if requester_user_id == payload.user_id {
        return Err(AppError::BadRequest("Cannot share with yourself".into()));
    }

    check_write_access(&state, &requester_key, &blob_id).await?;

    let target_key = state.user_keys.get(&payload.user_id)
        .map(|r| r.value().clone())
        .ok_or(AppError::BadRequest("Target user does not exist".into()))?;

    // Ensure target has a notify channel
    ensure_notify_channel(&state, &target_key);

    let permission = BlobPermission {
        api_key: target_key.clone(),
        user_id: payload.user_id.clone(),
        access_level: payload.access_level.clone(),
        granted_by: requester_user_id,
        granted_at: now(),
    };

    let mut perms = state.blob_permissions.entry(blob_id.clone())
        .or_insert_with(Vec::new);
    perms.value_mut().retain(|p| p.user_id != payload.user_id);
    perms.value_mut().push(permission.clone());

    let listener_entry = state.blob_listeners.entry(blob_id.clone())
        .or_insert_with(DashSet::new);
    listener_entry.insert(target_key);

    let state_clone = state.clone();
    tokio::spawn(async move {
        if let Err(e) = save_all_state(&state_clone).await {
            error!("Failed to save state: {}", e);
        }
    });

    info!("Shared blob '{}' with user '{}' ({:?})",
          blob_id, payload.user_id, payload.access_level);

    Ok(Json(ShareResponse {
        blob_id,
        user_id: payload.user_id,
        access_level: payload.access_level,
        granted_at: permission.granted_at,
    }))
}

#[derive(Serialize)]
struct ListSharesResponse {
    blob_id: String,
    shares: Vec<ShareInfo>,
}

#[derive(Serialize)]
struct ShareInfo {
    user_id: String,
    access_level: AccessLevel,
    granted_by: String,
    granted_at: u64,
}

async fn list_shares(
    State(state): State<Arc<AppState>>,
    Path(blob_id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<ListSharesResponse>, AppError> {
    let requester_key = get_api_key(&headers)?;
    check_access(&state, &requester_key, &blob_id).await?;

    let shares = if let Some(perms) = state.blob_permissions.get(&blob_id) {
        perms.value().iter().map(|p| ShareInfo {
            user_id: p.user_id.clone(),
            access_level: p.access_level.clone(),
            granted_by: p.granted_by.clone(),
            granted_at: p.granted_at,
        }).collect()
    } else {
        Vec::new()
    };

    Ok(Json(ListSharesResponse {
        blob_id,
        shares,
    }))
}

async fn revoke_share(
    State(state): State<Arc<AppState>>,
    Path((blob_id, user_id)): Path<(String, String)>,
    headers: HeaderMap,
) -> Result<StatusCode, AppError> {
    let requester_key = get_api_key(&headers)?;
    check_write_access(&state, &requester_key, &blob_id).await?;

    if let Some(mut perms) = state.blob_permissions.get_mut(&blob_id) {
        let before_len = perms.len();
        perms.retain(|p| p.user_id != user_id);

        if perms.len() == before_len {
            return Err(AppError::NotFound);
        }

        if let Some(target_key) = state.user_keys.get(&user_id) {
            if let Some(listeners) = state.blob_listeners.get(&blob_id) {
                listeners.remove(target_key.value());
            }
        }
    } else {
        return Err(AppError::NotFound);
    }

    let state_clone = state.clone();
    tokio::spawn(async move {
        if let Err(e) = save_all_state(&state_clone).await {
            error!("Failed to save state: {}", e);
        }
    });

    info!("Revoked share for blob '{}' from user '{}'", blob_id, user_id);
    Ok(StatusCode::NO_CONTENT)
}

// 3. Blob Operations

async fn check_access(state: &Arc<AppState>, key: &str, blob_id: &str) -> Result<(), AppError> {
    // Ensure key has a channel (self-healing)
    ensure_notify_channel(state, key);

    // Check new permission system first
    if let Some(perms) = state.blob_permissions.get(blob_id) {
        for perm in perms.value() {
            if perm.api_key == key {
                return Ok(());
            }
        }
    }

    // Fallback to legacy permission system
    match state.permissions.get(key) {
        Some(blobs) => {
            if blobs.contains(blob_id) {
                Ok(())
            } else {
                Err(AppError::Unauthorized)
            }
        },
        None => Err(AppError::Unauthorized),
    }
}

async fn check_write_access(state: &Arc<AppState>, key: &str, blob_id: &str) -> Result<(), AppError> {
    // Ensure key has a channel (self-healing)
    ensure_notify_channel(state, key);

    // Check new permission system
    if let Some(perms) = state.blob_permissions.get(blob_id) {
        for perm in perms.value() {
            if perm.api_key == key {
                if perm.access_level == AccessLevel::ReadWrite {
                    return Ok(());
                } else {
                    return Err(AppError::Forbidden);
                }
            }
        }
    }

    // Fallback to legacy system
    match state.permissions.get(key) {
        Some(blobs) => {
            if blobs.contains(blob_id) {
                Ok(())
            } else {
                Err(AppError::Unauthorized)
            }
        },
        None => Err(AppError::Unauthorized),
    }
}

async fn notify_subscribers(state: &Arc<AppState>, blob_id: &str) {
    if let Some(listeners) = state.blob_listeners.get(blob_id) {
        for key in listeners.iter() {
            if let Some(sender) = state.notify_channels.get(key.key()) {
                let _ = sender.send(blob_id.to_string());
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct ShardConfig {
    data_shards: usize,
    parity_shards: usize,
    #[serde(default)]
    peers: Vec<String>,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            data_shards: 3,
            parity_shards: 2,
            peers: vec![],
        }
    }
}

#[instrument(skip(state, body))]
async fn upload_blob(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Json<BlobMeta>, AppError> {
    let key = get_api_key(&headers)?;

    // Ensure key has a channel
    ensure_notify_channel(&state, &key);

    let shard_config: ShardConfig = if let Some(config_header) = headers.get("x-shard-config") {
        match config_header.to_str() {
            Ok(config_str) => {
                serde_json::from_str(config_str).unwrap_or_else(|e| {
                    warn!("Failed to parse shard config: {}, using defaults", e);
                    ShardConfig::default()
                })
            }
            Err(e) => {
                warn!("Invalid shard config header: {}, using defaults", e);
                ShardConfig::default()
            }
        }
    } else {
        ShardConfig::default()
    };

    let is_new = !state.storage_path.join(format!("{}.meta", id)).exists();

    if !is_new {
        check_write_access(&state, &key, &id).await?;

        if let Some(if_match) = headers.get("if-match") {
            let client_ver: u64 = if_match.to_str().unwrap_or("0").parse().unwrap_or(0);
            let current_meta = load_meta(&state.storage_path, &id).await?;
            if current_meta.version > client_ver {
                return Err(AppError::Conflict);
            }
        }
    } else {
        // Grant full access to creator
        if let Some(user_id) = state.key_users.get(&key) {
            let permission = BlobPermission {
                api_key: key.clone(),
                user_id: user_id.value().clone(),
                access_level: AccessLevel::ReadWrite,
                granted_by: user_id.value().clone(),
                granted_at: now(),
            };

            state.blob_permissions.entry(id.clone())
                .or_insert_with(Vec::new)
                .push(permission);
        }

        // Legacy system
        if let Some(set) = state.permissions.get(&key) {
            set.insert(id.clone());
            let listener_entry = state.blob_listeners.entry(id.clone()).or_insert_with(DashSet::new);
            listener_entry.insert(key.clone());
        }
    }

    let version = now();
    let data_shards_count = shard_config.data_shards;
    let parity_shards_count = shard_config.parity_shards;

    let encoder = ReedSolomon::<Field>::new(data_shards_count, parity_shards_count)
        .map_err(|e| AppError::ReedSolomon(e.to_string()))?;

    // Write Data
    let data_path = state.storage_path.join(format!("{}.data", id));
    fs::write(&data_path, &body).await?;

    // Prepare Shards
    let data_vec = body.to_vec();
    let shard_len = (data_vec.len() + data_shards_count - 1) / data_shards_count;
    let mut padded_data = data_vec.clone();
    padded_data.resize(shard_len * data_shards_count, 0);

    let mut master_shards: Vec<Vec<u8>> = padded_data
        .chunks(shard_len)
        .map(|c| c.to_vec())
        .collect();

    for _ in 0..parity_shards_count {
        master_shards.push(vec![0u8; shard_len]);
    }

    encoder.encode(&mut master_shards)
        .map_err(|e| AppError::ReedSolomon(e.to_string()))?;

    // Write Metadata
    let meta = BlobMeta {
        version,
        size: body.len() as u64,
        data_shards: data_shards_count,
        parity_shards: parity_shards_count,
        created_at: version,
        shard_locations: shard_config.peers.clone(),
    };
    let meta_path = state.storage_path.join(format!("{}.meta", id));
    fs::write(&meta_path, serde_json::to_vec(&meta)?).await?;

    // Distribute Shards
    if !shard_config.peers.is_empty() {
        let client = state.http_client.clone();
        let peers = shard_config.peers.clone();
        let blob_id = id.clone();

        tokio::spawn(async move {
            for (i, shard) in master_shards.iter().enumerate() {
                if peers.is_empty() { break; }
                let peer = &peers[i % peers.len()];
                let url = format!("{}/shard/{}/{}", peer, blob_id, i);

                let shard_meta = ShardMeta {
                    original_blob_id: blob_id.clone(),
                    shard_index: i,
                    version,
                };

                let _res = client.post(&url)
                    .body(shard.clone())
                    .header("x-shard-meta", serde_json::to_string(&shard_meta).unwrap_or_default())
                    .send()
                    .await;
            }
        });
    }

    notify_subscribers(&state, &id).await;

    // Save state
    let state_clone = state.clone();
    tokio::spawn(async move {
        if let Err(e) = save_all_state(&state_clone).await {
            error!("Failed to save state: {}", e);
        }
    });

    Ok(Json(meta))
}

async fn load_meta(path: &PathBuf, id: &str) -> Result<BlobMeta, AppError> {
    let p = path.join(format!("{}.meta", id));
    if !p.exists() { return Err(AppError::NotFound); }
    let d = fs::read(p).await?;
    Ok(serde_json::from_slice(&d)?)
}

#[instrument(skip(state))]
async fn read_blob(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, AppError> {
    let path = state.storage_path.join(format!("{}.data", id));
    if !path.exists() { return Err(AppError::NotFound); }

    let key = get_api_key(&headers)?;
    check_access(&state, &key, &id).await?;

    let file = File::open(path).await?;
    let stream = tokio_util::io::ReaderStream::new(file);
    let body = Body::from_stream(stream);

    let meta = load_meta(&state.storage_path, &id).await?;
    let mut headers = HeaderMap::new();
    headers.insert("ETag", meta.version.to_string().parse().unwrap());

    Ok((headers, body))
}

async fn read_meta(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<BlobMeta>, AppError> {
    let meta_path = state.storage_path.join(format!("{}.meta", id));
    if !meta_path.exists() { return Err(AppError::NotFound); }

    let key = get_api_key(&headers)?;
    check_access(&state, &key, &id).await?;

    let meta = load_meta(&state.storage_path, &id).await?;
    Ok(Json(meta))
}

#[instrument(skip(state))]
async fn delete_blob(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<StatusCode, AppError> {
    let key = get_api_key(&headers)?;
    check_write_access(&state, &key, &id).await?;

    let data_path = state.storage_path.join(format!("{}.data", id));
    if data_path.exists() {
        fs::remove_file(&data_path).await?;
    }

    let meta_path = state.storage_path.join(format!("{}.meta", id));
    if meta_path.exists() {
        fs::remove_file(&meta_path).await?;
    }

    state.blob_permissions.remove(&id);

    for entry in state.permissions.iter() {
        entry.value().remove(&id);
    }

    state.blob_listeners.remove(&id);
    notify_subscribers(&state, &id).await;

    let state_clone = state.clone();
    tokio::spawn(async move {
        if let Err(e) = save_all_state(&state_clone).await {
            error!("Failed to save state: {}", e);
        }
    });

    info!("Deleted blob: {}", id);
    Ok(StatusCode::NO_CONTENT)
}

// 4. Shard Management

async fn receive_shard(
    State(state): State<Arc<AppState>>,
    Path((id, index)): Path<(String, usize)>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<StatusCode, AppError> {
    let shard_filename = format!("{}.shard.{}", id, index);
    let data_path = state.storage_path.join(&shard_filename);
    fs::write(&data_path, &body).await?;

    if let Some(meta_json) = headers.get("x-shard-meta") {
        let meta_path = state.storage_path.join(format!("{}.meta", shard_filename));
        fs::write(&meta_path, meta_json.as_bytes()).await?;
    }

    Ok(StatusCode::OK)
}

async fn read_shard(
    State(state): State<Arc<AppState>>,
    Path((id, index)): Path<(String, usize)>,
) -> Result<Body, AppError> {
    let shard_filename = format!("{}.shard.{}", id, index);
    let path = state.storage_path.join(&shard_filename);
    if !path.exists() { return Err(AppError::NotFound); }

    let file = File::open(path).await?;
    let stream = tokio_util::io::ReaderStream::new(file);
    Ok(Body::from_stream(stream))
}

// 5. Watch / Notifications

async fn watch_changes(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<String, AppError> {
    let key = get_api_key(&headers)?;

    // Self-healing: ensure channel exists if key is valid
    let key_exists = state.permissions.contains_key(&key) || state.key_users.contains_key(&key);

    if !key_exists {
        return Err(AppError::Unauthorized);
    }

    // Create channel if it doesn't exist (self-healing after restart)
    ensure_notify_channel(&state, &key);

    let mut rx = if let Some(sender) = state.notify_channels.get(&key) {
        sender.subscribe()
    } else {
        // This should not happen after ensure_notify_channel, but handle gracefully
        return Err(AppError::Unauthorized);
    };

    match tokio::time::timeout(Duration::from_secs(60), rx.recv()).await {
        Ok(Ok(blob_id)) => Ok(blob_id),
        Ok(Err(_)) => Err(AppError::BadRequest("Lagged".into())),
        Err(_) => Ok("timeout".to_string()),
    }
}
