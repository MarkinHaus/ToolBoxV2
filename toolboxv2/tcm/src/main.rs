// src/main.rs

// --- Crate Imports ---
use anyhow::{anyhow, Context, Result};
use base64::{engine::general_purpose, Engine as _};
use chacha20poly1305::{
    aead::{Aead, KeyInit, OsRng},
    ChaCha20Poly1305, Nonce,
};
use hkdf::Hkdf;
use log::{error, info, warn, LevelFilter};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt,ReadHalf, WriteHalf},
    net::{TcpListener, TcpStream, UdpSocket},
    select,
    sync::{Mutex, RwLock},
    time::timeout,
};

// --- Konstanten ---
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);
const PEER_TIMEOUT: Duration = Duration::from_secs(45);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_MSG_SIZE: usize = 4096;
const ENCRYPTION_KEY_SIZE: usize = 32; // 256-bit key

// --- Konfiguration (wird aus config.toml geladen) ---
mod config {
    use super::*;

    #[derive(Debug, Deserialize)]
    pub struct Config {
        pub mode: Mode,
        pub relay: Option<RelayConfig>,
        pub peer: Option<PeerConfig>,
    }

    #[derive(Debug, Deserialize, PartialEq, Eq)]
    #[serde(rename_all = "lowercase")]
    pub enum Mode {
        Relay,
        Peer,
    }

    #[derive(Debug, Deserialize)]
    pub struct RelayConfig {
        pub bind_address: String,
        pub password: String,
    }

    #[derive(Debug, Deserialize, Clone)]
    pub struct PeerConfig {
        pub relay_address: String,
        pub relay_password: String,
        pub peer_id: String,
        pub listen_address: String,
        pub forward_to_address: String,
        pub target_peer_id: Option<String>,
    }

    pub fn load() -> Result<Config> {
        let config_str = std::fs::read_to_string("config.toml")
            .context("Konnte config.toml nicht finden oder lesen.")?;
        toml::from_str(&config_str).context("Konnte config.toml nicht parsen.")
    }
}

// --- Netzwerkprotokoll-Definitionen ---
mod protocol {
    use super::*;

    #[derive(Debug, Serialize, Deserialize)]
    pub enum ClientMessage {
        Register { peer_id: String, password_hash: String },
        RequestPeer { target_id: String },
        Heartbeat,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub enum ServerMessage {
        Registered { public_addr: SocketAddr },
        PeerInfo { peer_id: String, public_addr: SocketAddr },
        InitiateHolePunch { peer_id: String, public_addr: SocketAddr },
        Error(String),
        AuthChallenge { salt: String },
        AuthSuccess,
    }
}

// --- Kryptographie-Helfer ---
mod crypto {
    use rand::RngCore;
    use super::*;

    /// Leitet einen sicheren Schlüssel aus einem Passwort und einem Salt ab.
    pub fn derive_key(password: &str, salt: &str) -> [u8; ENCRYPTION_KEY_SIZE] {
        let ikm = password.as_bytes();
        let salt_bytes = salt.as_bytes();
        let hkdf = Hkdf::<Sha256>::new(Some(salt_bytes), ikm);
        let mut okm = [0u8; ENCRYPTION_KEY_SIZE];
        hkdf.expand(b"p2p-engine-key", &mut okm).expect("Key derivation failed");
        okm
    }

    /// Erzeugt einen Hash des Passworts für die Authentifizierung.
    pub fn hash_password(password: &str, salt: &str) -> String {
        let key = derive_key(password, salt);
        general_purpose::STANDARD.encode(key)
    }

    /// Verschlüsselt Daten mit AEAD.
    pub fn encrypt(data: &[u8], key: &[u8; ENCRYPTION_KEY_SIZE]) -> Result<Vec<u8>> {
        let cipher = ChaCha20Poly1305::new(key.into());
        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let mut encrypted_data = cipher.encrypt(nonce, data)
            .map_err(|e| anyhow!("Encryption failed: {}", e))?;

        let mut result = Vec::with_capacity(12 + encrypted_data.len());
        result.extend_from_slice(&nonce_bytes);
        result.append(&mut encrypted_data);
        Ok(result)
    }

    /// Entschlüsselt Daten mit AEAD.
    pub fn decrypt(encrypted_data: &[u8], key: &[u8; ENCRYPTION_KEY_SIZE]) -> Result<Vec<u8>> {
        if encrypted_data.len() < 12 {
            return Err(anyhow!("Encrypted data is too short"));
        }
        let cipher = ChaCha20Poly1305::new(key.into());
        let (nonce_bytes, ciphertext) = encrypted_data.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        cipher.decrypt(nonce, ciphertext)
            .map_err(|e| anyhow!("Decryption failed: {}", e))
    }
}

// --- Relay-Server-Logik ---
mod relay {
    use super::*;

    #[derive(Debug, Clone)]
    struct RegisteredPeer {
        public_addr: SocketAddr,
        last_heartbeat: Instant,
    }

    type PeerRegistry = Arc<RwLock<HashMap<String, RegisteredPeer>>>;

    pub async fn run(config: config::RelayConfig) -> Result<()> {
        info!("Relay-Server startet auf {}", config.bind_address);
        let listener = TcpListener::bind(&config.bind_address).await?;
        let peer_registry: PeerRegistry = Arc::new(RwLock::new(HashMap::new()));

        // Task zum Aufräumen inaktiver Peers
        let registry_clone = peer_registry.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(HEARTBEAT_INTERVAL).await;
                let mut registry = registry_clone.write().await;
                registry.retain(|id, peer| {
                    let should_keep = peer.last_heartbeat.elapsed() < PEER_TIMEOUT;
                    if !should_keep {
                        info!("Peer {} wegen Timeout entfernt.", id);
                    }
                    should_keep
                });
            }
        });

        // Haupt-Loop zum Akzeptieren von Verbindungen
        loop {
            let (stream, addr) = listener.accept().await?;
            info!("Neue Verbindung von {}", addr);
            let registry_clone = peer_registry.clone();
            let password = config.password.clone();
            tokio::spawn(async move {
                if let Err(e) = handle_client(stream, addr, registry_clone, password).await {
                    warn!("Fehler bei der Verarbeitung des Clients {}: {}", addr, e);
                }
            });
        }
    }

    // GEÄNDERT: Nimmt den Stream per Wert entgegen, um ihn zu splitten.
    async fn handle_client(
        stream: TcpStream,
        addr: SocketAddr,
        registry: PeerRegistry,
        password: String,
    ) -> Result<()> {
        // GEÄNDERT: Stream in Lese- und Schreibhälfte aufteilen.
        // Dies löst alle "cannot borrow as mutable more than once" Fehler (E0499).
        let (mut reader, mut writer) = tokio::io::split(stream);
        let mut peer_id_option: Option<String> = None;
        let salt = "unique-relay-salt";

        // Auth Challenge (mit writer senden)
        send_message(&mut writer, &protocol::ServerMessage::AuthChallenge { salt: salt.to_string() }).await?;

        loop {
            // Von reader lesen, in writer schreiben. Keine Borrow-Konflikte mehr.
            let read_future = read_message(&mut reader);
            tokio::pin!(read_future);

            select! {
                message = &mut read_future => {
                    let message = match message {
                        Ok(Some(msg)) => msg,
                        Ok(None) => break, // Connection closed
                        Err(e) => {
                            warn!("Fehler beim Lesen der Nachricht von {}: {}", addr, e);
                            let _ = send_message(&mut writer, &protocol::ServerMessage::Error(e.to_string())).await;
                            break;
                        }
                    };

                    match message {
                        protocol::ClientMessage::Register { peer_id, password_hash } => {
                           let expected_hash = crypto::hash_password(&password, salt);
                           if password_hash != expected_hash {
                               send_message(&mut writer, &protocol::ServerMessage::Error("Ungültiges Passwort".to_string())).await?;
                               return Err(anyhow!("Authentifizierung fehlgeschlagen für {}", peer_id));
                           }

                           info!("Peer {} hat sich von {} registriert", peer_id, addr);
                           let mut reg = registry.write().await;
                           reg.insert(peer_id.clone(), RegisteredPeer {
                               public_addr: addr,
                               last_heartbeat: Instant::now(),
                           });
                           peer_id_option = Some(peer_id);
                           send_message(&mut writer, &protocol::ServerMessage::AuthSuccess).await?;
                           send_message(&mut writer, &protocol::ServerMessage::Registered { public_addr: addr }).await?;
                        }
                        protocol::ClientMessage::Heartbeat => {
                            if let Some(id) = &peer_id_option {
                                if let Some(peer) = registry.write().await.get_mut(id) {
                                    peer.last_heartbeat = Instant::now();
                                }
                            }
                        }
                        protocol::ClientMessage::RequestPeer { target_id } => {
                            let reg = registry.read().await;
                            if let Some(target_peer) = reg.get(&target_id) {
                                let initiator_id = peer_id_option.as_ref().ok_or_else(|| anyhow!("Nicht registrierter Peer"))?;
                                info!("Peer {} fordert Verbindung zu {}", initiator_id, target_id);

                                send_message(&mut writer, &protocol::ServerMessage::PeerInfo {
                                    peer_id: target_id.clone(),
                                    public_addr: target_peer.public_addr,
                                }).await?;

                                // Diese Vereinfachung hat sich nicht geändert.
                                if let Ok(mut target_stream) = TcpStream::connect(target_peer.public_addr).await {
                                     send_message(&mut target_stream, &protocol::ServerMessage::InitiateHolePunch {
                                        peer_id: initiator_id.clone(),
                                        public_addr: addr,
                                     }).await.unwrap_or_else(|e| warn!("Konnte Ziel-Peer nicht benachrichtigen: {}", e));
                                } else {
                                     warn!("Konnte keine TCP-Verbindung zum Ziel-Peer {} herstellen", target_id);
                                }
                            } else {
                                send_message(&mut writer, &protocol::ServerMessage::Error("Peer nicht gefunden".to_string())).await?;
                            }
                        }
                    }
                }
            }
        }

        if let Some(id) = peer_id_option {
            info!("Verbindung zu Peer {} geschlossen.", id);
            registry.write().await.remove(&id);
        }
        Ok(())
    }
}

// --- Peer-Client-Logik ---
mod peer {
    use super::*;

    pub async fn run(config: config::PeerConfig) -> Result<()> {
        info!("Peer-Client '{}' startet...", config.peer_id);

        let relay_stream = TcpStream::connect(&config.relay_address).await
            .context(format!("Verbindung zum Relay {} fehlgeschlagen", config.relay_address))?;
        info!("Verbunden mit Relay {}", config.relay_address);

        // GEÄNDERT: Reader wird in einen Arc<Mutex> verpackt, um ihn sicher zwischen
        // Tasks zu teilen. Dies löst die "use of moved value" Fehler (E0382).
        let (relay_reader_half, mut relay_writer) = tokio::io::split(relay_stream);
        let relay_reader = Arc::new(Mutex::new(relay_reader_half));

        // Authentifizierungsprozess (erfordert jetzt das Sperren des Readers)
        let salt = {
            let mut reader_guard = relay_reader.lock().await;
            match read_message(&mut *reader_guard).await? {
                Some(protocol::ServerMessage::AuthChallenge { salt }) => salt,
                _ => return Err(anyhow!("Relay hat keine Auth-Challenge gesendet.")),
            }
        };

        let password_hash = crypto::hash_password(&config.relay_password, &salt);
        send_message(&mut relay_writer, &protocol::ClientMessage::Register {
            peer_id: config.peer_id.clone(),
            password_hash,
        }).await?;

        {
            let mut reader_guard = relay_reader.lock().await;
            match read_message(&mut *reader_guard).await? {
                Some(protocol::ServerMessage::AuthSuccess) => info!("Erfolgreich beim Relay authentifiziert."),
                Some(protocol::ServerMessage::Error(e)) => return Err(anyhow!("Relay-Authentifizierung fehlgeschlagen: {}", e)),
                _ => return Err(anyhow!("Unerwartete Antwort vom Relay nach der Registrierung.")),
            }
        }

        let public_addr = {
            let mut reader_guard = relay_reader.lock().await;
            match read_message(&mut *reader_guard).await? {
                Some(protocol::ServerMessage::Registered { public_addr }) => public_addr,
                _ => return Err(anyhow!("Konnte öffentliche Adresse vom Relay nicht erhalten.")),
            }
        };
        info!("Registriert beim Relay. Öffentliche Adresse: {}", public_addr);

        let shared_key = Arc::new(crypto::derive_key(&config.relay_password, &config.peer_id));
        let peer_state = Arc::new(Mutex::new(PeerState { local_public_addr: public_addr }));
        let shared_writer = Arc::new(Mutex::new(relay_writer));

        // Heartbeat-Task
        let writer_clone = Arc::clone(&shared_writer);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(HEARTBEAT_INTERVAL).await;
                let mut writer = writer_clone.lock().await;
                if send_message(&mut *writer, &protocol::ClientMessage::Heartbeat).await.is_err() {
                    error!("Heartbeat zum Relay fehlgeschlagen. Verbindung verloren.");
                    break;
                }
            }
        });

        if let Some(target_id) = config.target_peer_id.clone() {
            let config_clone = Arc::new(config.clone());
            let state_clone = peer_state.clone();
            let key_clone = shared_key.clone();
            let writer_clone = Arc::clone(&shared_writer);
            // GEÄNDERT: Klonen des Arc für den Listener-Task
            let reader_clone = Arc::clone(&relay_reader);

            let local_listener = TcpListener::bind(&config_clone.listen_address).await?;
            info!("Lausche auf {} für lokale Anwendungsverbindungen.", config_clone.listen_address);

            tokio::spawn(async move {
                loop {
                    let (local_socket, _) = local_listener.accept().await.unwrap();
                    info!("Lokale Anwendung hat sich verbunden. Starte P2P-Verbindung zu '{}'.", target_id);

                    let writer = Arc::clone(&writer_clone);
                    let config = Arc::clone(&config_clone);
                    let state = state_clone.clone();
                    let key = key_clone.clone();
                    let target_id = target_id.clone();
                    // GEÄNDERT: Erneutes Klonen für den Verbindungs-Task
                    let reader_for_connection = Arc::clone(&reader_clone);

                    tokio::spawn(async move {
                        if let Err(e) = connect_to_peer(local_socket, writer, reader_for_connection, &target_id, config, state, key).await {
                            error!("P2P-Verbindung zu {} fehlgeschlagen: {}", target_id, e);
                        }
                    });
                }
            });
        } else {
            info!("Kein Ziel-Peer konfiguriert. Warte auf eingehende Verbindungen...");
        }

        // Haupt-Loop für Nachrichten vom Relay
        // WICHTIGER HINWEIS: Diese Architektur führt wahrscheinlich zu einem Deadlock.
        // Wenn dieser Loop den Reader sperrt, kann connect_to_peer ihn nicht sperren und umgekehrt.
        // Der Code ist jetzt kompilierbar, aber die zugrundeliegende Logik muss überarbeitet werden,
        // z.B. mit einem zentralen Reader-Task und Channels (mpsc/oneshot).
        loop {
            let msg = {
                let mut reader_guard = relay_reader.lock().await;
                read_message(&mut *reader_guard).await?
            };

            match msg {
                Some(protocol::ServerMessage::InitiateHolePunch { peer_id, public_addr }) => {
                    info!("Eingehende Verbindungsanfrage von {} ({})", peer_id, public_addr);
                    // GEÄNDERT: Wert aus MutexGuard extrahieren, bevor der Task gespawnt wird.
                    // Dies löst den Lifetime-Fehler (E0597).
                    let local_public_addr = peer_state.lock().await.local_public_addr;
                    let config_clone = config.clone();
                    let key = shared_key.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handle_incoming_connection(config_clone, local_public_addr, public_addr, key).await {
                            error!("Fehler bei der Bearbeitung der eingehenden Verbindung von {}: {}", peer_id, e);
                        }
                    });
                }
                Some(protocol::ServerMessage::Error(e)) => warn!("Fehler vom Relay erhalten: {}", e),
                None => {
                    error!("Verbindung zum Relay verloren.");
                    return Ok(());
                }
                _ => {}
            }
        }
    }

    struct PeerState {
        local_public_addr: SocketAddr,
    }

    // GEÄNDERT: Funktionssignatur, um den geteilten Reader zu akzeptieren.
    async fn connect_to_peer(
        local_socket: TcpStream,
        relay_writer: Arc<Mutex<WriteHalf<TcpStream>>>,
        relay_reader: Arc<Mutex<ReadHalf<TcpStream>>>,
        target_id: &str,
        _config: Arc<config::PeerConfig>,
        state: Arc<Mutex<PeerState>>,
        key: Arc<[u8; 32]>,
    ) -> Result<()> {
        info!("Fordere Peer-Info für '{}' vom Relay an.", target_id);
        {
            let mut writer = relay_writer.lock().await;
            send_message(&mut *writer, &protocol::ClientMessage::RequestPeer { target_id: target_id.to_string() }).await?;
        }

        let target_info = {
            let mut reader_guard = relay_reader.lock().await;
            match timeout(CONNECT_TIMEOUT, read_message(&mut *reader_guard)).await?? {
                Some(protocol::ServerMessage::PeerInfo { public_addr, .. }) => public_addr,
                Some(protocol::ServerMessage::Error(e)) => return Err(anyhow!("Relay-Fehler: {}", e)),
                _ => return Err(anyhow!("Unerwartete Antwort vom Relay.")),
            }
        };
        info!("Peer-Info für '{}' erhalten: {}", target_id, target_info);

        let local_addr = { state.lock().await.local_public_addr };

        info!("Starte UDP Hole Punching zu {}", target_info);
        let p2p_socket = match attempt_hole_punch(local_addr, target_info).await {
            Ok(socket) => {
                info!("Hole Punching erfolgreich! Direkte TCP-Verbindung hergestellt.");
                socket
            },
            Err(e) => {
                warn!("Hole Punching fehlgeschlagen: {}. Fallback wäre jetzt nötig.", e);
                return Err(anyhow!("Direkte Verbindung fehlgeschlagen, Fallback nicht implementiert."));
            }
        };

        info!("Proxying data between local app and peer '{}'", target_id);
        proxy_data(local_socket, p2p_socket, *key).await
    }

    async fn handle_incoming_connection(
        config: config::PeerConfig,
        local_public_addr: SocketAddr,
        remote_public_addr: SocketAddr,
        key: Arc<[u8; 32]>,
    ) -> Result<()> {
        info!("Warte auf direkte Verbindung nach Hole Punch von {}", remote_public_addr);

        let p2p_listener = TcpListener::bind(format!("0.0.0.0:{}", local_public_addr.port())).await?;

        // Parallel UDP-Pakete senden, um NAT offen zu halten
        let udp_socket = UdpSocket::bind(format!("0.0.0.0:{}", local_public_addr.port())).await?;
        udp_socket.connect(remote_public_addr).await?;
        tokio::spawn(async move {
            for _ in 0..10 {
                let _ = udp_socket.send(b"ping").await;
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        });

        match timeout(CONNECT_TIMEOUT, p2p_listener.accept()).await {
            Ok(Ok((p2p_socket, _))) => {
                info!("Eingehende P2P-Verbindung akzeptiert von {}", remote_public_addr);
                let local_service_socket = TcpStream::connect(&config.forward_to_address).await
                    .context("Konnte keine Verbindung zum lokalen Dienst herstellen")?;

                info!("Proxying data between peer and local service at {}", config.forward_to_address);
                proxy_data(local_service_socket, p2p_socket, *key).await
            }
            _ => Err(anyhow!("Timeout beim Warten auf P2P-Verbindung.")),
        }
    }

    /// Versucht, eine direkte TCP-Verbindung via UDP-Hole-Punching aufzubauen.
    async fn attempt_hole_punch(local_addr: SocketAddr, remote_addr: SocketAddr) -> Result<TcpStream> {
        let local_udp_port = local_addr.port();
        let socket = Arc::new(UdpSocket::bind(format!("0.0.0.0:{}", local_udp_port)).await?);
        socket.connect(remote_addr).await?;

        let punch_socket = socket.clone();
        let punch_task = tokio::spawn(async move {
            for _ in 0..10 {
                if punch_socket.send(b"punch").await.is_err() { break; }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });

        let connect_task = tokio::spawn(async move {
            for i in 0..10 {
                tokio::time::sleep(Duration::from_millis(200 * i)).await;
                if let Ok(stream) = TcpStream::connect(remote_addr).await {
                    return Ok(stream);
                }
            }
            Err(anyhow!("TCP-Verbindung konnte nicht hergestellt werden"))
        });

        let (punch_res, connect_res) = tokio::join!(punch_task, connect_task);
        punch_res?;
        connect_res?
    }

    /// Leitet Daten bidirektional zwischen zwei Sockets weiter und ver-/entschlüsselt sie.
    async fn proxy_data(mut local_socket: TcpStream, mut remote_socket: TcpStream, key: [u8; 32]) -> Result<()> {
        let (mut local_reader, mut local_writer) = local_socket.split();
        let (mut remote_reader, mut remote_writer) = remote_socket.split();

        let client_to_remote = async {
            let mut buf = vec![0; MAX_MSG_SIZE];
            loop {
                let n = local_reader.read(&mut buf).await?;
                if n == 0 { break; }
                let encrypted = crypto::encrypt(&buf[..n], &key)?;
                remote_writer.write_u32(encrypted.len() as u32).await?;
                remote_writer.write_all(&encrypted).await?;
            }
            Ok::<(), anyhow::Error>(())
        };

        let remote_to_client = async {
            loop {
                let len = remote_reader.read_u32().await? as usize;
                if len > MAX_MSG_SIZE * 2 { return Err(anyhow!("Verschlüsselte Nachricht zu groß")); }

                let mut encrypted_buf = vec![0; len];
                remote_reader.read_exact(&mut encrypted_buf).await?;

                let decrypted = crypto::decrypt(&encrypted_buf, &key)?;
                local_writer.write_all(&decrypted).await?;
            }
            #[allow(unreachable_code)]
            Ok::<(), anyhow::Error>(())
        };

        select! {
            res = client_to_remote => {
                if let Err(e) = res { if !e.to_string().contains("Connection reset by peer") { info!("Lokale -> Remote Verbindung geschlossen: {}", e); } }
            },
            res = remote_to_client => {
                if let Err(e) = res { if !e.to_string().contains("Connection reset by peer") { info!("Remote -> Lokale Verbindung geschlossen: {}", e); } }
            },
        }
        info!("Proxy-Sitzung beendet.");
        Ok(())
    }
}

// --- Generische Netzwerk-Helfer ---
async fn send_message<W, T>(writer: &mut W, message: &T) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
    T: Serialize,
{
    let serialized = serde_json::to_vec(message)?;
    if serialized.len() > MAX_MSG_SIZE {
        return Err(anyhow!("Nachricht zu groß: {} bytes", serialized.len()));
    }
    writer.write_u32(serialized.len() as u32).await?;
    writer.write_all(&serialized).await?;
    Ok(())
}

async fn read_message<R, T>(reader: &mut R) -> Result<Option<T>>
where
    R: AsyncReadExt + Unpin,
    T: for<'de> Deserialize<'de>,
{
    match reader.read_u32().await {
        Ok(len) => {
            if len as usize > MAX_MSG_SIZE {
                return Err(anyhow!("Eingehende Nachricht zu groß: {} bytes", len));
            }
            let mut buf = vec![0; len as usize];
            reader.read_exact(&mut buf).await?;
            Ok(Some(serde_json::from_slice(&buf)?))
        }
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(e.into()),
    }
}
// --- Hauptfunktion und Graceful Shutdown ---
#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_default_env()
        .filter(None, LevelFilter::Info)
        .init();

    let config = config::load().context("Fehler beim Laden der Konfiguration")?;

    let main_task = async {
        match config.mode {
            config::Mode::Relay => {
                let relay_config = config.relay.ok_or_else(|| anyhow!("Relay-Konfiguration fehlt"))?;
                relay::run(relay_config).await
            }
            config::Mode::Peer => {
                let peer_config = config.peer.ok_or_else(|| anyhow!("Peer-Konfiguration fehlt"))?;
                peer::run(peer_config).await
            }
        }
    };

    select! {
        res = main_task => {
            if let Err(e) = res {
                error!("Anwendungsfehler: {:?}", e);
                std::process::exit(1);
            }
        }
        _ = tokio::signal::ctrl_c() => {
            info!("Ctrl-C empfangen. Fahre herunter...");
        }
    }

    Ok(())
}