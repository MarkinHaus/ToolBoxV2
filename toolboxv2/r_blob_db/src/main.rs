// file: src/main.rs

// FIX: Import the http_body_util crate for StreamBody
use axum::{
    body::{Body},
    response::{Response},
};
use once_cell::sync::Lazy;
use std::{env, sync::Arc};
// FIX: Replace std::sync::RwLock with tokio::sync::RwLock for async safety
use tokio::sync::RwLock;
use tokio_util::io::ReaderStream;
use tracing::instrument;
// --- Main Application Entry Point ---
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(env::var("RUST_LOG").unwrap_or_else(|_| "info,tower_http=debug,blob_storage=trace".into()))
        .with_target(true)
        .with_timer(tracing_subscriber::fmt::time::UtcTime::rfc_3339())
        .init();

    if env::var("R_BLOB_DB_CLEAN").map_or(false, |v| v == "true") {
        let storage_dir = blob_storage::get_storage_dir();
        if storage_dir.exists() {
            tokio::fs::remove_dir_all(&storage_dir).await?;
            tracing::info!("Cleared old blob storage at '{}'.", storage_dir.display());
        }
    }

    server::run_server().await
}

// --- Global Blob Storage Singleton ---
// FIX: Use tokio's async-aware RwLock.
pub static GLOBAL_STORAGE: Lazy<Arc<RwLock<blob_storage::BlobStorage>>> = Lazy::new(|| {
    let data_shards = 4;
    let parity_shards = 2;
    tracing::info!("Initializing global blob storage with {} data shards and {} parity shards.", data_shards, parity_shards);
    let storage = blob_storage::BlobStorage::new(data_shards, parity_shards).expect("Failed to initialize blob storage");
    Arc::new(RwLock::new(storage))
});

// --- Custom Error Types ---
mod errors {
    use axum::{http::StatusCode, response::{IntoResponse, Response}};
    use thiserror::Error;

    #[derive(Error, Debug)]
    pub enum StorageError {
        // FIX: Consolidate IO errors into one variant. `tokio::io::Error` is the standard in async code.
        #[error("IO Error: {0}")]
        Io(#[from] tokio::io::Error),
        #[error("JSON serialization error: {0}")]
        Serialization(#[from] serde_json::Error),
        #[error("Reed-Solomon error: {0}")]
        ReedSolomon(#[from] reed_solomon_erasure::Error),
        #[error("Blob not found: {0}")]
        NotFound(String),
        #[error("Not enough shards to reconstruct data")]
        NotEnoughShards,
        #[error("Shard data has an invalid length")]
        InvalidShardLength,
        #[error("Invalid operation: {0}")]
        InvalidOperation(String),
        #[error("Data is corrupt or has invalid format: {0}")]
        InvalidDataFormat(String),
    }

    impl IntoResponse for StorageError {
        fn into_response(self) -> Response {
            let (status, error_message) = match &self {
                StorageError::NotFound(id) => (StatusCode::NOT_FOUND, format!("Blob '{}' not found", id)),
                StorageError::InvalidOperation(_) => (StatusCode::BAD_REQUEST, self.to_string()),
                StorageError::InvalidDataFormat(_) => (StatusCode::BAD_REQUEST, self.to_string()),
                _ => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            };
            tracing::error!("Responding with error: status={}, message='{}'", status, error_message);
            (status, error_message).into_response()
        }
    }
}

// --- Blob Storage Core Logic ---
mod blob_storage {
    use super::errors::StorageError;
    use reed_solomon_erasure::{galois_8::Field, ReedSolomon};
    use serde::{Deserialize, Serialize};
    use std::{collections::HashMap, path::PathBuf, io::SeekFrom};
    use tokio::{
        fs::{self, File},
        io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufReader},
    };
    use tracing::instrument;

    const MAGIC_BYTES: &[u8; 4] = b"BLOB";
    const FORMAT_VERSION: u8 = 1;

    #[derive(Serialize, Deserialize, Debug, Clone, Default)]
    pub struct BlobMetadata {
        pub links: HashMap<String, Vec<u8>>,
    }

    pub fn get_storage_dir() -> PathBuf {
        let dir = std::env::var("R_BLOB_DB_DATA_DIR").unwrap_or_else(|_| "./rust_blobs".to_string());
        PathBuf::from(dir)
    }

    pub struct BlobStorage {
        storage_directory: PathBuf,
        rs_codec: ReedSolomon<Field>,
        data_shards: usize,
        parity_shards: usize,
    }

    impl BlobStorage {
        pub fn new(data_shards: usize, parity_shards: usize) -> Result<Self, StorageError> {
            std::fs::create_dir_all(get_storage_dir())?;
            Ok(Self {
                storage_directory: get_storage_dir(),
                rs_codec: ReedSolomon::new(data_shards, parity_shards)?,
                data_shards,
                parity_shards,
            })
        }

        #[instrument(skip_all, fields(blob_id, data_len = data.len()))]
        pub async fn put_blob(&self, blob_id: &str, data: &[u8]) -> Result<(), StorageError> {
            // FIX: Make this operation idempotent (create or update).
            // Try to read the metadata of the blob.
            let metadata = match self.read_blob_metadata(blob_id).await {
                // If the blob exists, use its current metadata to preserve links.
                Ok(meta) => meta,
                // If it's a NotFound error, it means we are CREATING the blob.
                // In this case, we create new, empty default metadata.
                Err(StorageError::NotFound(_)) => BlobMetadata::default(),
                // If any other error occurs (e.g., disk I/O), propagate it.
                Err(e) => return Err(e),
            };

            // Proceed to write the blob file with either the old or the new metadata.
            self.write_blob_file(blob_id, &metadata, data).await
        }

        #[instrument(skip(self))]
        pub async fn read_blob_stream(&self, blob_id: &str) -> Result<File, StorageError> {
            let (mut file, metadata_len) = self.open_blob_file_for_read(blob_id).await?;
            let data_start_pos = (MAGIC_BYTES.len() + 1 + 8 + metadata_len as usize) as u64;
            file.seek(SeekFrom::Start(data_start_pos)).await?;
            Ok(file)
        }

        #[instrument(skip(self))]
        pub async fn delete_blob(&self, blob_id: &str) -> Result<(), StorageError> {
            let path = self.get_blob_path(blob_id);
            if path.exists() {
                Ok(fs::remove_file(path).await?)
            } else {
                Err(StorageError::NotFound(blob_id.to_string()))
            }
        }

        #[instrument(skip_all)]
        pub async fn share_and_link_blobs(&self, blob_ids: &[String]) -> Result<(), StorageError> {
            if blob_ids.len() < self.data_shards + 1 {
                return Err(StorageError::InvalidOperation(format!("Sharing requires at least {} helper blobs.", self.data_shards + 1)));
            }

            let mut all_blob_data = Vec::with_capacity(blob_ids.len());
            for id in blob_ids {
                all_blob_data.push((
                    id.clone(),
                    self.read_blob_metadata(id).await?,
                    self.read_full_blob_data(id).await?,
                ));
            }

            let mut updated_metadata_map: HashMap<String, BlobMetadata> = all_blob_data.iter().map(|(id, meta, _)| (id.clone(), meta.clone())).collect();

            for (source_id, _, source_data) in &all_blob_data {
                if source_data.is_empty() { continue; }
                let shards = self.encode_data_to_shards(source_data)?;
                let helper_blob_ids: Vec<_> = blob_ids.iter().filter(|&id| id != source_id).collect();

                for (i, shard_data) in shards.iter().enumerate() {
                    let target_id = &helper_blob_ids[i % helper_blob_ids.len()];
                    // FIX: Borrow the key (`&String`) which can be compared to the HashMap's stored key (`String`).
                    let target_meta = updated_metadata_map.get_mut(target_id.as_str()).unwrap();
                    let (shard_type, type_index) = if i < self.data_shards { ("data", i) } else { ("parity", i - self.data_shards) };
                    let link_key = format!("{}_for_{}_shard_{}", shard_type, source_id, type_index);
                    target_meta.links.insert(link_key, shard_data.clone());
                }
            }

            for (id, metadata) in updated_metadata_map {
                self.write_blob_metadata(&id, &metadata).await?;
            }
            Ok(())
        }

        #[instrument(skip(self))]
        pub async fn recover_blob(&self, lost_blob_id: &str) -> Result<Vec<u8>, StorageError> {
            let helper_blob_ids = self.get_all_blob_ids().await?.into_iter().filter(|id| id != lost_blob_id);
            let mut shards: Vec<Option<Vec<u8>>> = vec![None; self.data_shards + self.parity_shards];
            let mut shards_present_count = 0;
            let mut shard_len = 0;

            for helper_id in helper_blob_ids {
                if let Ok(helper_meta) = self.read_blob_metadata(&helper_id).await {
                    for i in 0..(self.data_shards + self.parity_shards) {
                        let (shard_type, type_index) = if i < self.data_shards { ("data", i) } else { ("parity", i - self.data_shards) };
                        let link_key = format!("{}_for_{}_shard_{}", shard_type, lost_blob_id, type_index);

                        if let Some(shard_data) = helper_meta.links.get(&link_key) {
                            if shards[i].is_none() {
                                if shard_len == 0 { shard_len = shard_data.len(); }
                                if shard_data.len() != shard_len { return Err(StorageError::InvalidShardLength); }
                                shards[i] = Some(shard_data.clone());
                                shards_present_count += 1;
                            }
                        }
                    }
                }
            }
            if shards_present_count < self.data_shards { return Err(StorageError::NotEnoughShards); }
            self.rs_codec.reconstruct_data(&mut shards)?;
            let mut padded_data = Vec::with_capacity(self.data_shards * shard_len);
            for i in 0..self.data_shards {
                padded_data.extend_from_slice(shards[i].as_ref().ok_or_else(|| StorageError::InvalidOperation("Reconstruction failed.".to_string()))?);
            }
            Self::unpad_data(&padded_data)
        }

        pub async fn get_all_blob_ids(&self) -> Result<Vec<String>, StorageError> {
            let mut ids = Vec::new();
            if !self.storage_directory.exists() { return Ok(ids); }
            let mut entries = fs::read_dir(&self.storage_directory).await?;
            while let Some(entry) = entries.next_entry().await? {
                if let Some(filename) = entry.file_name().to_str() {
                    if let Some(id) = filename.strip_suffix(".blob") {
                        ids.push(id.to_string());
                    }
                }
            }
            Ok(ids)
        }

        fn get_blob_path(&self, blob_id: &str) -> PathBuf {
            self.storage_directory.join(format!("{}.blob", blob_id))
        }

        async fn open_blob_file_for_read(&self, blob_id: &str) -> Result<(File, u64), StorageError> {
            let path = self.get_blob_path(blob_id);
            if !path.exists() { return Err(StorageError::NotFound(blob_id.to_string())); }
            let mut file = File::open(path).await?;
            let mut magic_buf = [0u8; 4];
            file.read_exact(&mut magic_buf).await?;
            if magic_buf != *MAGIC_BYTES { return Err(StorageError::InvalidDataFormat("Bad magic bytes".to_string())); }
            let version = file.read_u8().await?;
            if version != FORMAT_VERSION { return Err(StorageError::InvalidDataFormat(format!("Unsupported version {}", version))); }
            let metadata_len = file.read_u64_le().await?;
            Ok((file, metadata_len))
        }

        async fn read_blob_metadata(&self, blob_id: &str) -> Result<BlobMetadata, StorageError> {
            let (file, metadata_len) = self.open_blob_file_for_read(blob_id).await?;
            let mut metadata_reader = BufReader::new(file.take(metadata_len));
            let mut metadata_bytes = Vec::new();
            metadata_reader.read_to_end(&mut metadata_bytes).await?;
            Ok(serde_json::from_slice(&metadata_bytes)?)
        }

        async fn read_full_blob_data(&self, blob_id: &str) -> Result<Vec<u8>, StorageError> {
            let mut stream = self.read_blob_stream(blob_id).await?;
            let mut buffer = Vec::new();
            stream.read_to_end(&mut buffer).await?;
            Ok(buffer)
        }

        async fn write_blob_file(&self, blob_id: &str, metadata: &BlobMetadata, data: &[u8]) -> Result<(), StorageError> {
            let path = self.get_blob_path(blob_id);
            let mut file = File::create(path).await?;
            let metadata_bytes = serde_json::to_vec(metadata)?;
            file.write_all(MAGIC_BYTES).await?;
            file.write_u8(FORMAT_VERSION).await?;
            file.write_u64_le(metadata_bytes.len() as u64).await?;
            file.write_all(&metadata_bytes).await?;
            file.write_all(data).await?;
            Ok(())
        }

        async fn write_blob_metadata(&self, blob_id: &str, metadata: &BlobMetadata) -> Result<(), StorageError> {
            let existing_data = self.read_full_blob_data(blob_id).await?;
            self.write_blob_file(blob_id, metadata, &existing_data).await
        }

        fn encode_data_to_shards(&self, data: &[u8]) -> Result<Vec<Vec<u8>>, StorageError> {
            let padded_data = Self::pad_data(data, self.data_shards)?;
            let shard_len = padded_data.len() / self.data_shards;
            let mut shards: Vec<Vec<u8>> = padded_data.chunks(shard_len).map(|c| c.to_vec()).collect();
            let mut parity_shards = vec![vec![0u8; shard_len]; self.parity_shards];
            self.rs_codec.encode_sep(&shards, &mut parity_shards)?;
            shards.append(&mut parity_shards);
            Ok(shards)
        }

        fn pad_data(data: &[u8], n: usize) -> Result<Vec<u8>, StorageError> {
            let header_size = 8;
            let total_len = header_size + data.len();
            let required_padding = (n - (total_len % n)) % n;
            let mut buffer = Vec::with_capacity(total_len + required_padding);
            buffer.extend_from_slice(&(data.len() as u64).to_le_bytes());
            buffer.extend_from_slice(data);
            buffer.resize(buffer.capacity(), 0);
            Ok(buffer)
        }

        fn unpad_data(padded_data: &[u8]) -> Result<Vec<u8>, StorageError> {
            if padded_data.len() < 8 { return Err(StorageError::InvalidDataFormat("Padded data too short for header".to_string())); }
            let mut len_bytes = [0u8; 8];
            len_bytes.copy_from_slice(&padded_data[0..8]);
            let original_len = u64::from_le_bytes(len_bytes) as usize;
            if (8 + original_len) > padded_data.len() { return Err(StorageError::InvalidDataFormat("Original length exceeds padded data size".to_string())); }
            Ok(padded_data[8..(8 + original_len)].to_vec())
        }
    }
}

// --- Web Server (API) ---
mod server {
    // FIX: Import the async RwLock from tokio
    use super::{blob_storage, errors::StorageError, GLOBAL_STORAGE, instrument, ReaderStream, Body, Response, RwLock};
    use axum::{
        body::Bytes, extract::{Path, State}, http::StatusCode, routing::{get, post}, Json, Router, response::{IntoResponse},
    };
    use serde::{Deserialize, Serialize};
    use std::{net::SocketAddr, sync::Arc};

    type AppState = Arc<RwLock<blob_storage::BlobStorage>>;

    pub async fn run_server() -> anyhow::Result<()> {
        let port = std::env::var("R_BLOB_DB_PORT").unwrap_or_else(|_| "3000".to_string()).parse::<u16>()?;
        let app = Router::new()
            .route("/blob/{id}", get(read_blob).put(put_blob).delete(delete_blob))
            .route("/share", post(share_blobs))
            .route("/recover", post(recover_blob))
            .route("/health", get(health_check))
            .with_state(GLOBAL_STORAGE.clone());
        // .layer(TraceLayer::new_for_http()); // This can be added back if needed

        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        tracing::info!("API server listening on {}", addr);
        axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
        Ok(())
    }

    #[derive(Serialize)]
    struct HealthStatus { status: &'static str, version: &'static str, blobs_managed: usize }

    // FIX: All handlers now correctly handle the async RwLock, which is Send-safe.
    #[instrument(skip(state))]
    async fn health_check(State(state): State<AppState>) -> (StatusCode, Json<HealthStatus>) {
        let blob_count = state.read().await.get_all_blob_ids().await.unwrap_or_default().len();
        (StatusCode::OK, Json(HealthStatus { status: "OK", version: env!("CARGO_PKG_VERSION"), blobs_managed: blob_count }))
    }

    #[instrument(skip(state))]
    async fn read_blob(State(state): State<AppState>, Path(id): Path<String>) -> Result<Response, StorageError> {
        let file_stream = state.read().await.read_blob_stream(&id).await?;
        let stream = ReaderStream::new(file_stream);
        let body = Body::from_stream(stream);
        Ok(body.into_response())
    }

    #[instrument(skip(state, body), fields(bytes = body.len()))]
    async fn put_blob(State(state): State<AppState>, Path(id): Path<String>, body: Bytes) -> Result<StatusCode, StorageError> {
        state.write().await.put_blob(&id, &body).await?;
        Ok(StatusCode::CREATED)
    }

    #[instrument(skip(state))]
    async fn delete_blob(State(state): State<AppState>, Path(id): Path<String>) -> Result<StatusCode, StorageError> {
        state.write().await.delete_blob(&id).await?;
        Ok(StatusCode::NO_CONTENT)
    }

    #[derive(Deserialize, Debug)]
    struct ShareRequest { blob_ids: Vec<String> }

    #[instrument(skip(state, payload))]
    async fn share_blobs(State(state): State<AppState>, Json(payload): Json<ShareRequest>) -> Result<StatusCode, StorageError> {
        state.write().await.share_and_link_blobs(&payload.blob_ids).await?;
        Ok(StatusCode::OK)
    }

    #[derive(Deserialize, Debug)]
    struct RecoverRequest { blob_id: String }

    #[instrument(skip(state, payload))]
    async fn recover_blob(State(state): State<AppState>, Json(payload): Json<RecoverRequest>) -> Result<Vec<u8>, StorageError> {
        state.read().await.recover_blob(&payload.blob_id).await
    }
}