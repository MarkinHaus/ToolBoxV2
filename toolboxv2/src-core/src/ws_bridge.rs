use crate::{WsMessage, GLOBAL_WS_BROADCASTER, ACTIVE_CONNECTIONS};
use tracing::{info, warn, error};

/// Rust-Bridge für WebSocket-Kommunikation von Python aus.
/// Diese Struktur stellt statische Methoden bereit, die von Python über
/// die app_singleton.py Bridge aufgerufen werden können.
pub struct RustWsBridge;

impl RustWsBridge {
    /// Sendet eine Nachricht an eine einzelne WebSocket-Verbindung.
    pub fn send_message(conn_id: String, payload: String) {
        info!("RustWsBridge::send_message: conn_id={}, payload_len={}", conn_id, payload.len());

        if let Some(conn) = ACTIVE_CONNECTIONS.get(&conn_id) {
            conn.value().do_send(WsMessage {
                source_conn_id: "python_direct".to_string(),
                content: payload,
                target_conn_id: Some(conn_id.clone()),
                target_channel_id: None,
            });
            info!("Message sent to connection: {}", conn_id);
        } else {
            warn!("RustWsBridge: Connection ID '{}' not found for sending.", conn_id);
        }
    }

    /// Sendet eine Nachricht an alle Clients in einem Kanal.
    pub fn broadcast_message(channel_id: String, payload: String, source_conn_id: String) {
        info!("RustWsBridge::broadcast_message: channel_id={}, source_conn_id={}, payload_len={}",
              channel_id, source_conn_id, payload.len());

        let msg = WsMessage {
            source_conn_id,
            content: payload,
            target_conn_id: None,
            target_channel_id: Some(channel_id.clone()),
        };

        if let Err(e) = GLOBAL_WS_BROADCASTER.send(msg) {
            error!("RustWsBridge: Failed to send broadcast message to channel {}: {}", channel_id, e);
        } else {
            info!("Broadcast message sent to channel: {}", channel_id);
        }
    }
}

