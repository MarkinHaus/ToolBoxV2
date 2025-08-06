#!/usr/bin/env python3
import re

# Read the main.rs file
with open('/home/daytona/ToolBoxV2/toolboxv2/src-core/src/main.rs', 'r') as f:
    content = f.read()

# Add WebSocket imports
imports_pattern = r'(use actix_web::\{web, App, HttpRequest, HttpServer, HttpResponse, middleware, FromRequest\};)'
imports_replacement = r'''\1
use actix_web_actors::ws;
use actix::{Actor, StreamHandler, ActorContext, AsyncContext, Handler, Message};'''

content = re.sub(imports_pattern, imports_replacement, content)

# Add WebSocket actor and message structures before the configuration structs
websocket_code = '''
// WebSocket message types
#[derive(Message)]
#[rtype(result = "()")]
struct WsMessage(String);

// WebSocket actor
struct WebSocketActor {
    session_id: String,
    session_manager: web::Data<SessionManager>,
    toolbox_client: Arc<ToolboxClient>,
}

impl WebSocketActor {
    fn new(session_id: String, session_manager: web::Data<SessionManager>, toolbox_client: Arc<ToolboxClient>) -> Self {
        Self {
            session_id,
            session_manager,
            toolbox_client,
        }
    }

    async fn forward_to_python(&self, message: &str) -> Result<String, String> {
        // Parse the incoming message
        let parsed_message: serde_json::Value = match serde_json::from_str(message) {
            Ok(msg) => msg,
            Err(e) => {
                error!("Failed to parse WebSocket message: {}", e);
                return Err(format!("Invalid JSON: {}", e));
            }
        };

        // Create message data for Python
        let mut kwargs = std::collections::HashMap::new();
        kwargs.insert("message".to_string(), parsed_message);
        kwargs.insert("session_id".to_string(), serde_json::json!(self.session_id));

        // Forward to Python WebSocketManager
        match self.toolbox_client.run_function(
            "WebSocketManager",
            "handle_incoming_message",
            "",
            vec![],
            kwargs,
        ).await {
            Ok(response) => {
                match serde_json::to_string(&response) {
                    Ok(json_str) => Ok(json_str),
                    Err(e) => {
                        error!("Failed to serialize Python response: {}", e);
                        Err(format!("Serialization error: {}", e))
                    }
                }
            },
            Err(e) => {
                error!("Failed to forward message to Python: {}", e);
                Err(format!("Python error: {}", e))
            }
        }
    }

    async fn handle_connect(&self) {
        info!("WebSocket connection established for session: {}", self.session_id);
        
        // Notify Python about connection
        let mut kwargs = std::collections::HashMap::new();
        kwargs.insert("session_id".to_string(), serde_json::json!(self.session_id));
        
        if let Err(e) = self.toolbox_client.run_function(
            "WebSocketManager",
            "on_connect",
            "",
            vec![],
            kwargs,
        ).await {
            warn!("Failed to notify Python about WebSocket connection: {}", e);
        }
    }

    async fn handle_disconnect(&self) {
        info!("WebSocket connection closed for session: {}", self.session_id);
        
        // Notify Python about disconnection
        let mut kwargs = std::collections::HashMap::new();
        kwargs.insert("session_id".to_string(), serde_json::json!(self.session_id));
        
        if let Err(e) = self.toolbox_client.run_function(
            "WebSocketManager",
            "on_disconnect",
            "",
            vec![],
            kwargs,
        ).await {
            warn!("Failed to notify Python about WebSocket disconnection: {}", e);
        }
    }
}

impl Actor for WebSocketActor {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("WebSocket actor started for session: {}", self.session_id);
        
        // Handle connection in async context
        let session_id = self.session_id.clone();
        let toolbox_client = self.toolbox_client.clone();
        
        ctx.spawn(async move {
            let mut kwargs = std::collections::HashMap::new();
            kwargs.insert("session_id".to_string(), serde_json::json!(session_id));
            
            if let Err(e) = toolbox_client.run_function(
                "WebSocketManager",
                "on_connect",
                "",
                vec![],
                kwargs,
            ).await {
                warn!("Failed to notify Python about WebSocket connection: {}", e);
            }
        }.into_actor(self));
    }

    fn stopped(&mut self, ctx: &mut Self::Context) {
        info!("WebSocket actor stopped for session: {}", self.session_id);
        
        // Handle disconnection in async context
        let session_id = self.session_id.clone();
        let toolbox_client = self.toolbox_client.clone();
        
        ctx.spawn(async move {
            let mut kwargs = std::collections::HashMap::new();
            kwargs.insert("session_id".to_string(), serde_json::json!(session_id));
            
            if let Err(e) = toolbox_client.run_function(
                "WebSocketManager",
                "on_disconnect",
                "",
                vec![],
                kwargs,
            ).await {
                warn!("Failed to notify Python about WebSocket disconnection: {}", e);
            }
        }.into_actor(self));
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WebSocketActor {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                debug!("WebSocket ping received for session: {}", self.session_id);
                ctx.pong(&msg);
            }
            Ok(ws::Message::Pong(_)) => {
                debug!("WebSocket pong received for session: {}", self.session_id);
            }
            Ok(ws::Message::Text(text)) => {
                debug!("WebSocket text message received for session: {}: {}", self.session_id, text);
                
                // Forward message to Python and handle response
                let session_id = self.session_id.clone();
                let toolbox_client = self.toolbox_client.clone();
                let message_text = text.to_string();
                
                ctx.spawn(async move {
                    // Create message data for Python
                    let mut kwargs = std::collections::HashMap::new();
                    
                    // Parse the incoming message
                    let parsed_message: serde_json::Value = match serde_json::from_str(&message_text) {
                        Ok(msg) => msg,
                        Err(e) => {
                            error!("Failed to parse WebSocket message: {}", e);
                            serde_json::json!({"error": format!("Invalid JSON: {}", e)})
                        }
                    };
                    
                    kwargs.insert("message".to_string(), parsed_message);
                    kwargs.insert("session_id".to_string(), serde_json::json!(session_id));

                    // Forward to Python WebSocketManager
                    match toolbox_client.run_function(
                        "WebSocketManager",
                        "handle_incoming_message",
                        "",
                        vec![],
                        kwargs,
                    ).await {
                        Ok(response) => {
                            match serde_json::to_string(&response) {
                                Ok(json_str) => json_str,
                                Err(e) => {
                                    error!("Failed to serialize Python response: {}", e);
                                    format!(r#"{{"error": "Serialization error: {}"}}"#, e)
                                }
                            }
                        },
                        Err(e) => {
                            error!("Failed to forward message to Python: {}", e);
                            format!(r#"{{"error": "Python error: {}"}}"#, e)
                        }
                    }
                }.into_actor(self).map(|response, _actor, ctx| {
                    ctx.text(response);
                }));
            }
            Ok(ws::Message::Binary(bin)) => {
                debug!("WebSocket binary message received for session: {} ({} bytes)", self.session_id, bin.len());
                // For now, we'll just echo binary messages back
                ctx.binary(bin);
            }
            Ok(ws::Message::Close(reason)) => {
                info!("WebSocket close message received for session: {}: {:?}", self.session_id, reason);
                ctx.stop();
            }
            Err(e) => {
                error!("WebSocket protocol error for session: {}: {}", self.session_id, e);
                ctx.stop();
            }
        }
    }
}

// WebSocket handler function
async fn websocket_handler(
    req: HttpRequest,
    stream: web::Payload,
    path: web::Path<String>,
    session_manager: web::Data<SessionManager>,
) -> Result<HttpResponse, actix_web::Error> {
    let session_id = path.into_inner();
    
    info!("WebSocket connection attempt for session: {}", session_id);
    
    // Validate session
    if !session_manager.validate_session(&session_id).await {
        warn!("Invalid session ID for WebSocket connection: {}", session_id);
        return Ok(HttpResponse::Unauthorized().json(serde_json::json!({
            "error": "Invalid session ID"
        })));
    }
    
    // Get toolbox client
    let toolbox_client = match get_toolbox_client() {
        Ok(client) => Arc::new(client),
        Err(e) => {
            error!("Failed to get toolbox client for WebSocket: {}", e);
            return Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "Internal server error"
            })));
        }
    };
    
    info!("Starting WebSocket actor for session: {}", session_id);
    
    // Create and start WebSocket actor
    let actor = WebSocketActor::new(session_id, session_manager, toolbox_client);
    let resp = ws::start(actor, &req, stream)?;
    
    Ok(resp)
}

'''

# Insert the WebSocket code before the configuration structs
config_pattern = r'(// Configuration struct\n#\[derive\(Debug, Deserialize, Clone\)\]\nstruct ServerConfig \{)'
content = re.sub(config_pattern, websocket_code + r'\1', content)

# Write the modified content back
with open('/home/daytona/ToolBoxV2/toolboxv2/src-core/src/main.rs', 'w') as f:
    f.write(content)

print('Successfully added WebSocket handler and forwarder implementation to main.rs')
