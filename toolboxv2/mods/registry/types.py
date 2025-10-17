# toolboxv2/mods/registry/types.py

from typing import Any

from pydantic import BaseModel, Field

# --- Pydantic Models for WebSocket Payloads ---

class AgentRegistration(BaseModel):
    """Client -> Server: Payload to register a new agent."""
    public_name: str = Field(..., description="A user-friendly name for the agent.")
    description: str | None = Field(None, description="Optional description of the agent's capabilities.")

class AgentRegistered(BaseModel):
    """Server -> Client: Response after successful registration."""
    public_name: str
    public_agent_id: str = Field(..., description="The unique public ID for the agent.")
    public_api_key: str = Field(..., description="The secret API key for public access.")
    public_url: str = Field(..., description="The full public URL to run the agent.")

class RunRequest(BaseModel):
    """Server -> Client: Request to execute an agent."""
    request_id: str = Field(..., description="A unique ID for this specific execution request.")
    public_agent_id: str = Field(..., description="The ID of the agent to run.")
    query: str = Field(..., description="The main input/query for the agent.")
    session_id: str | None = Field(None, description="Session ID for maintaining context.")
    kwargs: dict[str, Any] = Field({}, description="Additional keyword arguments for the a_run method.")

class ExecutionResult(BaseModel):
    """Client -> Server: A chunk of the execution result (for streaming)."""
    request_id: str
    payload: dict[str, Any] = Field(..., description="The ProgressEvent or final result as a dictionary.")
    is_final: bool = Field(False, description="True if this is the last message for this request.")

class ExecutionError(BaseModel):
    """Client -> Server: Reports an error during execution."""
    request_id: str
    error: str

class WsMessage(BaseModel):
    """A generic wrapper for all WebSocket messages."""
    event: str
    data: dict[str, Any]
