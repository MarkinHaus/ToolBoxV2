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
    sync::{Arc},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    fs::{self, File},
    sync::{broadcast},
};
use tracing::{error, info, instrument};
use tracing::log::warn;
use uuid::Uuid;

// --- KONFIGURATION ---
const STORAGE_ENV: &str = "R_BLOB_DB_DATA_DIR";
const DEFAULT_STORAGE: &str =  "./data_blobs";

// --- DATENSTRUKTUREN ---

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BlobMeta {
    pub version: u64,
    pub size: u64,
    pub data_shards: usize,
    pub parity_shards: usize,
    pub created_at: u64,
    // Liste der Server, wo Shards liegen (zur Info für den Client bei Recovery)
    pub shard_locations: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ShardMeta {
    pub original_blob_id: String,
    pub shard_index: usize,
    pub version: u64,
}

// Access Level für Blob-Permissions
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AccessLevel {
    ReadOnly,
    ReadWrite,
}

// Permission Entry für einen Blob
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BlobPermission {
    pub api_key: String,
    pub user_id: String,  // Public User ID (hash of device name)
    pub access_level: AccessLevel,
    pub granted_by: String,  // User ID who granted access
    pub granted_at: u64,  // Unix timestamp
}

// Request Body für Upload
#[derive(Deserialize, Debug)]
struct PutRequestParams {
    // Wenn gesetzt, wird Sharding aktiviert
    #[serde(default)]
    peers: Vec<String>,
    #[serde(default = "default_ds")]
    data_shards: usize,
    #[serde(default = "default_ps")]
    parity_shards: usize,
}
fn default_ds() -> usize { 4 }
fn default_ps() -> usize { 2 }

// In-Memory State des Servers
struct AppState {
    storage_path: PathBuf,

    // NEW: Blob-based permissions with access levels
    // BlobID -> Vec<BlobPermission>
    blob_permissions: DashMap<String, Vec<BlobPermission>>,

    // NEW: User ID <-> API Key mappings
    // UserID -> API Key
    user_keys: DashMap<String, String>,
    // API Key -> UserID (reverse index)
    key_users: DashMap<String, String>,

    // LEGACY: Old permission system (kept for backward compatibility)
    // API Key -> Set<BlobID>
    permissions: DashMap<String, DashSet<String>>,

    // BlobID -> Set<KeyID> (Reverse Index für Notifications)
    blob_listeners: DashMap<String, DashSet<String>>,
    // Notification Channels: KeyID -> Sender
    notify_channels: DashMap<String, broadcast::Sender<String>>,
    // HTTP Client für Replikation
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

// --- HELPER FUNKTIONEN ---

fn get_api_key(headers: &HeaderMap) -> Result<String, AppError> {
    headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .ok_or(AppError::Unauthorized)
}

// Generate Public User ID from device name
fn generate_user_id(device_name: &str) -> String {
    let mut hasher = DefaultHasher::new();
    device_name.hash(&mut hasher);
    let hash = hasher.finish();
    format!("user_{:x}", hash)
}

// Get current Unix timestamp
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

// --- MAIN SERVER LOGIC ---

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let path_str = std::env::var(STORAGE_ENV).unwrap_or(DEFAULT_STORAGE.into());
    let storage_path = PathBuf::from(path_str);
    fs::create_dir_all(&storage_path).await?;
    info!("Mounting storage at: {:?}", storage_path);

    // Persistenz laden (Permissions)
    let permissions_path = storage_path.join("sys_permissions.json");
    let permissions = if permissions_path.exists() {
        let data = fs::read(&permissions_path).await?;
        serde_json::from_slice(&data).unwrap_or_default()
    } else {
        DashMap::new()
    };

    let state = Arc::new(AppState {
        storage_path,
        blob_permissions: DashMap::new(),
        user_keys: DashMap::new(),
        key_users: DashMap::new(),
        permissions,
        blob_listeners: DashMap::new(),
        notify_channels: DashMap::new(),
        http_client: reqwest::Client::new(),
    });

    // Auto-Save Task für Permissions (alle 30s)
    let state_clone = state.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(30)).await;
            save_permissions(&state_clone).await.ok();
        }
    });

    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        // Key Management
        .route("/keys", post(create_key))
        .route("/keys/{key}", delete(revoke_key))
        .route("/permissions/{blob_id}", post(grant_permission))
        // NEW: Sharing API
        .route("/share/{blob_id}", post(share_blob).get(list_shares))
        .route("/share/{blob_id}/{user_id}", delete(revoke_share))
        // Blob Operationen
        .route("/blob/{id}", put(upload_blob).get(read_blob).delete(delete_blob))
        .route("/blob/{id}/meta", get(read_meta))
        // Shard Handling (Server-zu-Server)
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

async fn save_permissions(state: &Arc<AppState>) -> Result<(), AppError> {
    let path = state.storage_path.join("sys_permissions.json");
    let json = serde_json::to_vec(&state.permissions)?;
    fs::write(path, json).await?;
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

    // Generate User ID from device name
    let user_id = if let Some(device_name) = payload.device_name {
        generate_user_id(&device_name)
    } else {
        // Fallback: use key as user_id
        format!("user_{}", &key[..8])
    };

    // Store mappings
    state.user_keys.insert(user_id.clone(), key.clone());
    state.key_users.insert(key.clone(), user_id.clone());

    // Legacy permission system
    state.permissions.insert(key.clone(), DashSet::new());

    // Initialisiere Channel für Notifications
    let (tx, _) = broadcast::channel(100);
    state.notify_channels.insert(key.clone(), tx);

    save_permissions(&state).await.ok();

    info!("Created API key for user_id: {}", user_id);
    Json(KeyResponse { key, user_id })
}

async fn revoke_key(
    State(state): State<Arc<AppState>>,
    Path(key): Path<String>,
    headers: HeaderMap
) -> Result<StatusCode, AppError> {
    // Einfache Admin Logik: Nur wer Admin Key hat darf löschen?
    // Hier vereinfacht: Jeder mit gültigem Key darf seinen löschen oder andere (Vorsicht in Prod!)
    let _requester = get_api_key(&headers)?;
    state.permissions.remove(&key);
    state.notify_channels.remove(&key);
    save_permissions(&state).await.ok();
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

    // Darf der Requester überhaupt Berechtigungen vergeben?
    // Wir nehmen an: Wer Zugriff auf den Blob hat, darf teilen.
    check_access(&state, &requester, &blob_id).await?;

    if let Some(set) = state.permissions.get(&payload.key_to_grant) {
        set.insert(blob_id.clone());

        // Reverse Index Update für Notifications
        let listener_entry = state.blob_listeners.entry(blob_id.clone()).or_insert_with(DashSet::new);
        listener_entry.insert(payload.key_to_grant.clone());
    } else {
        return Err(AppError::BadRequest("Target Key does not exist".into()));
    }

    save_permissions(&state).await.ok();
    Ok(StatusCode::OK)
}

// 2. NEW: Sharing API

#[derive(Deserialize)]
struct ShareRequest {
    user_id: String,  // Public User ID to share with
    access_level: AccessLevel,  // read_only or read_write
}

#[derive(Serialize)]
struct ShareResponse {
    blob_id: String,
    user_id: String,
    access_level: AccessLevel,
    granted_at: u64,
}

/// Share a blob with another user
/// POST /share/:blob_id
/// Body: { "user_id": "user_abc123", "access_level": "read_only" | "read_write" }
async fn share_blob(
    State(state): State<Arc<AppState>>,
    Path(blob_id): Path<String>,
    headers: HeaderMap,
    Json(payload): Json<ShareRequest>
) -> Result<Json<ShareResponse>, AppError> {
    let requester_key = get_api_key(&headers)?;

    // Get requester's user_id
    let requester_user_id = state.key_users.get(&requester_key)
        .map(|r| r.value().clone())
        .ok_or(AppError::Unauthorized)?;

    // Security: Prevent sharing with yourself
    if requester_user_id == payload.user_id {
        return Err(AppError::BadRequest("Cannot share with yourself".into()));
    }

    // Check if requester has write access to this blob
    check_write_access(&state, &requester_key, &blob_id).await?;

    // Get target user's API key
    let target_key = state.user_keys.get(&payload.user_id)
        .map(|r| r.value().clone())
        .ok_or(AppError::BadRequest("Target user does not exist".into()))?;

    // Security: Ensure one party doesn't have both keys
    // This is enforced by the user_id system - each device has unique user_id

    // Create permission entry
    let permission = BlobPermission {
        api_key: target_key.clone(),
        user_id: payload.user_id.clone(),
        access_level: payload.access_level.clone(),
        granted_by: requester_user_id,
        granted_at: now(),
    };

    // Add to blob permissions
    let mut perms = state.blob_permissions.entry(blob_id.clone())
        .or_insert_with(Vec::new);

    // Remove existing permission for this user if any
    perms.value_mut().retain(|p| p.user_id != payload.user_id);

    // Add new permission
    perms.value_mut().push(permission.clone());

    // Update notification listeners
    let listener_entry = state.blob_listeners.entry(blob_id.clone())
        .or_insert_with(DashSet::new);
    listener_entry.insert(target_key);

    save_permissions(&state).await.ok();

    info!("Shared blob '{}' with user '{}' ({:?})",
          blob_id, payload.user_id, payload.access_level);

    Ok(Json(ShareResponse {
        blob_id,
        user_id: payload.user_id,
        access_level: payload.access_level,
        granted_at: permission.granted_at,
    }))
}

/// List all shares for a blob
/// GET /share/:blob_id
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

    // Check if requester has access to this blob
    check_access(&state, &requester_key, &blob_id).await?;

    // Get all permissions for this blob
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

/// Revoke share for a user
/// DELETE /share/:blob_id/:user_id
async fn revoke_share(
    State(state): State<Arc<AppState>>,
    Path((blob_id, user_id)): Path<(String, String)>,
    headers: HeaderMap,
) -> Result<StatusCode, AppError> {
    let requester_key = get_api_key(&headers)?;

    // Check if requester has write access
    check_write_access(&state, &requester_key, &blob_id).await?;

    // Remove permission
    if let Some(mut perms) = state.blob_permissions.get_mut(&blob_id) {
        let before_len = perms.len();
        perms.retain(|p| p.user_id != user_id);

        if perms.len() == before_len {
            return Err(AppError::NotFound);
        }

        // Remove from listeners if no longer has access
        if let Some(target_key) = state.user_keys.get(&user_id) {
            if let Some(listeners) = state.blob_listeners.get(&blob_id) {
                listeners.remove(target_key.value());
            }
        }
    } else {
        return Err(AppError::NotFound);
    }

    save_permissions(&state).await.ok();

    info!("Revoked share for blob '{}' from user '{}'", blob_id, user_id);

    Ok(StatusCode::NO_CONTENT)
}

// 3. Blob Operations

/// Check if user has any access (read or write) to a blob
async fn check_access(state: &Arc<AppState>, key: &str, blob_id: &str) -> Result<(), AppError> {
    // Check new permission system first
    if let Some(perms) = state.blob_permissions.get(blob_id) {
        for perm in perms.value() {
            if &perm.api_key == key {
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

/// Check if user has write access to a blob
async fn check_write_access(state: &Arc<AppState>, key: &str, blob_id: &str) -> Result<(), AppError> {
    // Check new permission system
    if let Some(perms) = state.blob_permissions.get(blob_id) {
        for perm in perms.value() {
            if &perm.api_key == key {
                if perm.access_level == AccessLevel::ReadWrite {
                    return Ok(());
                } else {
                    return Err(AppError::Forbidden);
                }
            }
        }
    }

    // Fallback to legacy system (assumes full access)
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

// Helper: Notify listeners
async fn notify_subscribers(state: &Arc<AppState>, blob_id: &str) {
    if let Some(listeners) = state.blob_listeners.get(blob_id) {
        for key in listeners.iter() {
            if let Some(sender) = state.notify_channels.get(key.key()) {
                // Sendet "BlobID:Version" oder einfach BlobID
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

    // Parse shard config from header (defaults if not provided)
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

    // Auto-Grant Permission if new, else check
    let is_new = !state.storage_path.join(format!("{}.meta", id)).exists();
    if !is_new {
        // For updates, check write access
        check_write_access(&state, &key, &id).await?;

        // Concurrency Check (Optimistic Locking)
        if let Some(if_match) = headers.get("if-match") {
            let client_ver: u64 = if_match.to_str().unwrap_or("0").parse().unwrap_or(0);
            let current_meta = load_meta(&state.storage_path, &id).await?;
            if current_meta.version > client_ver {
                return Err(AppError::Conflict);
            }
        }
    } else {
        // Grant full access to creator (NEW permission system)
        if let Some(user_id) = state.key_users.get(&key) {
            let permission = BlobPermission {
                api_key: key.clone(),
                user_id: user_id.value().clone(),
                access_level: AccessLevel::ReadWrite,
                granted_by: user_id.value().clone(),  // Self-granted
                granted_at: now(),
            };

            state.blob_permissions.entry(id.clone())
                .or_insert_with(Vec::new)
                .push(permission);
        }

        // Legacy permission system
        if let Some(set) = state.permissions.get(&key) {
            set.insert(id.clone());
            let listener_entry = state.blob_listeners.entry(id.clone()).or_insert_with(DashSet::new);
            listener_entry.insert(key.clone());
        }
    }

    let version = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();

    // 1. Sharding Logic (Reed Solomon)
    let data_shards_count = shard_config.data_shards;
    let parity_shards_count = shard_config.parity_shards;

    // FIX: Added ::<Field> to specify we are using 8-bit Galois Field
    let encoder = ReedSolomon::<Field>::new(data_shards_count, parity_shards_count)
        .map_err(|e| AppError::ReedSolomon(e.to_string()))?;

    // Write Data (Original)
    let data_path = state.storage_path.join(format!("{}.data", id));
    fs::write(&data_path, &body).await?;

    // Prepare Shards
    let data_vec = body.to_vec();

    // Calculate shard length (round up)
    // IMPORTANT: RS requires equal length shards. We pad the data.
    let shard_len = (data_vec.len() + data_shards_count - 1) / data_shards_count;
    let mut padded_data = data_vec.clone();

    // Pad with zeros to fit perfect rectangle
    padded_data.resize(shard_len * data_shards_count, 0);

    let mut master_shards: Vec<Vec<u8>> = padded_data
        .chunks(shard_len)
        .map(|c| c.to_vec())
        .collect();

    // Add empty parity shards containers
    for _ in 0..parity_shards_count {
        master_shards.push(vec![0u8; shard_len]);
    }

    // Perform encoding
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

    // Distribute Shards Concurrently
    if !shard_config.peers.is_empty() {
        let client = state.http_client.clone();
        let peers = shard_config.peers.clone();
        let blob_id = id.clone();

        // Background task to push shards to other servers
        tokio::spawn(async move {
            for (i, shard) in master_shards.iter().enumerate() {
                if peers.is_empty() { break; }
                // Round robin distribution
                let peer = &peers[i % peers.len()];
                let url = format!("{}/shard/{}/{}", peer, blob_id, i);

                let shard_meta = ShardMeta {
                    original_blob_id: blob_id.clone(),
                    shard_index: i,
                    version,
                };

                // Send Shard + Meta headers
                let _res = client.post(&url)
                    .body(shard.clone())
                    .header("x-shard-meta", serde_json::to_string(&shard_meta).unwrap_or_default())
                    .send()
                    .await;

                // In production: Retry logic and error logging goes here
            }
        });
    }

    notify_subscribers(&state, &id).await;

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
    // Check if blob exists BEFORE checking access
    // This ensures we return 404 (not found) instead of 401 (unauthorized) when blob doesn't exist
    let path = state.storage_path.join(format!("{}.data", id));
    if !path.exists() { return Err(AppError::NotFound); }

    // Now check access permissions
    let key = get_api_key(&headers)?;
    check_access(&state, &key, &id).await?;

    let file = File::open(path).await?;
    let stream = tokio_util::io::ReaderStream::new(file);
    let body = Body::from_stream(stream);

    // Add ETag header
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
    // Check if blob metadata exists BEFORE checking access
    // This ensures we return 404 (not found) instead of 401 (unauthorized) when blob doesn't exist
    let meta_path = state.storage_path.join(format!("{}.meta", id));
    if !meta_path.exists() { return Err(AppError::NotFound); }

    // Now check access permissions
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

    // Check if user has write access to delete
    check_write_access(&state, &key, &id).await?;

    // Delete data file
    let data_path = state.storage_path.join(format!("{}.data", id));
    if data_path.exists() {
        fs::remove_file(&data_path).await?;
    }

    // Delete metadata file
    let meta_path = state.storage_path.join(format!("{}.meta", id));
    if meta_path.exists() {
        fs::remove_file(&meta_path).await?;
    }

    // Remove permissions
    state.blob_permissions.remove(&id);

    // Remove from legacy permissions
    for entry in state.permissions.iter() {
        entry.value().remove(&id);
    }

    // Remove listeners
    state.blob_listeners.remove(&id);

    // Notify subscribers about deletion
    notify_subscribers(&state, &id).await;

    info!("Deleted blob: {}", id);

    Ok(StatusCode::NO_CONTENT)
}

// 3. Shard Management (No Auth check needed usually, or specific Cluster Key)
// For simplicity: We assume peers are trusted or use a specific "Cluster-Key"
// Prompt: "Server doesn't need to know about others". So open or basic validation.

async fn receive_shard(
    State(state): State<Arc<AppState>>,
    Path((id, index)): Path<(String, usize)>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<StatusCode, AppError> {
    // 1. Save .shard.N
    let shard_filename = format!("{}.shard.{}", id, index);
    let data_path = state.storage_path.join(&shard_filename);
    fs::write(&data_path, &body).await?;

    // 2. Save .shard.N.meta
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

// 4. Notifications (Long Polling / Server Sent Events alternative)

async fn watch_changes(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<String, AppError> {
    let key = get_api_key(&headers)?;

    let mut rx = if let Some(sender) = state.notify_channels.get(&key) {
        sender.subscribe()
    } else {
        return Err(AppError::Unauthorized);
    };

    // Wait for notification or timeout
    match tokio::time::timeout(Duration::from_secs(60), rx.recv()).await {
        Ok(Ok(blob_id)) => Ok(blob_id), // Return ID of changed blob
        Ok(Err(_)) => Err(AppError::BadRequest("Lagged".into())),
        Err(_) => Ok("timeout".to_string()), // Client connects again
    }
}
