# manager.py
# Enhanced P2P Manager with Interactive Chat, File Transfer & Voice Support
# Modern ToolBox-style interface with E2E encryption

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import hashlib
import base64
import select
import socket
from enum import Enum

# ToolBox Integration
try:
    from toolboxv2 import get_app, App, Result
    from toolboxv2.utils.extras.Style import Spinner, Style
    from toolboxv2 import tb_root_dir
except ImportError:
    print("ERROR: ToolBoxV2 not found. Please install it first.")
    sys.exit(1)

# Required dependencies
try:
    import psutil
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import serialization
    try:
        import pyaudio
        VOICE_ENABLED = True
    except ImportError:
        pyaudio = lambda: None
        VOICE_ENABLED = False
    import wave
except ImportError as e:
    print(f"{Style.RED('FATAL:')} Missing required library: {e}")
    print(f"{Style.YELLOW('Install with:')} pip install psutil cryptography pyaudio")
    sys.exit(1)

# Configuration
EXECUTABLE_NAME = "tcm.exe" if platform.system() == "Windows" else "tcm"
INSTANCES_ROOT_DIR = tb_root_dir / ".info" / "p2p_instances"
CHAT_ROOMS_DIR = tb_root_dir / ".info" / "p2p_chat_rooms"
FILE_TRANSFER_DIR = tb_root_dir / ".info" / "p2p_files"
VOICE_CACHE_DIR = tb_root_dir / ".info" / "p2p_voice"

# Constants
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
CHUNK_SIZE = 8192
VOICE_CHUNK = 1024
VOICE_FORMAT = pyaudio.paInt16 if VOICE_ENABLED else None
VOICE_CHANNELS = 1
VOICE_RATE = 44100


# =================== Data Models ===================

class MessageType(Enum):
    TEXT = "text"
    FILE = "file"
    VOICE = "voice"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """Represents a chat message with encryption support."""
    sender: str
    content: str
    timestamp: datetime
    room_id: str
    message_type: MessageType = MessageType.TEXT
    encrypted: bool = True
    file_name: Optional[str] = None
    file_size: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'message_type': self.message_type.value
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ChatMessage':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['message_type'] = MessageType(data['message_type'])
        return cls(**data)


@dataclass
class ChatRoom:
    """Represents a P2P chat room with E2E encryption."""
    room_id: str
    name: str
    owner: str
    participants: Set[str]
    is_locked: bool
    is_private: bool
    created_at: datetime
    encryption_key: str
    max_participants: int = 10
    voice_enabled: bool = False
    file_transfer_enabled: bool = True

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            'participants': list(self.participants),
            'created_at': self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ChatRoom':
        data['participants'] = set(data['participants'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class P2PConnection:
    """Represents a P2P connection configuration."""
    name: str
    mode: str  # relay, peer-provider, peer-consumer
    status: str  # active, stopped, error
    pid: Optional[int] = None
    config: dict = field(default_factory=dict)
    chat_room: Optional[str] = None


# =================== Crypto Manager ===================

class CryptoManager:
    """Handles all E2E encryption operations."""

    @staticmethod
    def generate_room_key(room_id: str, password: str) -> bytes:
        """Generate encryption key for room."""
        salt = room_id.encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    @staticmethod
    def encrypt_message(message: str, key: bytes) -> str:
        """Encrypt message content."""
        f = Fernet(key)
        return f.encrypt(message.encode()).decode()

    @staticmethod
    def decrypt_message(encrypted_message: str, key: bytes) -> str:
        """Decrypt message content."""
        f = Fernet(key)
        return f.decrypt(encrypted_message.encode()).decode()

    @staticmethod
    def encrypt_file(file_path: Path, key: bytes) -> bytes:
        """Encrypt file content."""
        f = Fernet(key)
        with open(file_path, 'rb') as file:
            return f.encrypt(file.read())

    @staticmethod
    def decrypt_file(encrypted_data: bytes, key: bytes, output_path: Path):
        """Decrypt file content."""
        f = Fernet(key)
        decrypted_data = f.decrypt(encrypted_data)
        with open(output_path, 'wb') as file:
            file.write(decrypted_data)

    @staticmethod
    def encrypt_bytes(data: bytes, key: bytes) -> bytes:
        """Encrypt binary data directly (for audio/files)."""
        f = Fernet(key)
        return f.encrypt(data)

    @staticmethod
    def decrypt_bytes(encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt binary data directly (for audio/files)."""
        f = Fernet(key)
        return f.decrypt(encrypted_data)

# =================== File Transfer Manager ===================

class FileTransferManager:
    """Manages P2P file transfers with E2E encryption."""

    def __init__(self, room_id: str, encryption_key: bytes):
        self.room_id = room_id
        self.encryption_key = encryption_key
        self.transfer_dir = FILE_TRANSFER_DIR / room_id
        self.transfer_dir.mkdir(parents=True, exist_ok=True)

    def prepare_file(self, file_path: Path) -> Tuple[str, int]:
        """Prepare file for transfer (encrypt and chunk)."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File too large: {file_size} bytes (max: {MAX_FILE_SIZE})")

        # Encrypt file
        encrypted_data = CryptoManager.encrypt_file(file_path, self.encryption_key)

        # Save encrypted file
        transfer_id = hashlib.sha256(f"{file_path.name}{time.time()}".encode()).hexdigest()[:16]
        encrypted_file_path = self.transfer_dir / f"{transfer_id}.enc"

        with open(encrypted_file_path, 'wb') as f:
            f.write(encrypted_data)

        return transfer_id, len(encrypted_data)

    def receive_file(self, transfer_id: str, file_name: str) -> Path:
        """Receive and decrypt file."""
        encrypted_file_path = self.transfer_dir / f"{transfer_id}.enc"

        if not encrypted_file_path.exists():
            raise FileNotFoundError(f"Transfer file not found: {transfer_id}")

        # Read encrypted data
        with open(encrypted_file_path, 'rb') as f:
            encrypted_data = f.read()

        # Decrypt and save
        output_path = FILE_TRANSFER_DIR / "received" / file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        CryptoManager.decrypt_file(encrypted_data, self.encryption_key, output_path)

        return output_path


# =================== Voice Chat Manager ===================

class VoiceChatManager:
    """Manages P2P voice chat with live streaming and speaker detection."""

    def __init__(self, room_id: str, encryption_key: bytes, username: str):
        self.room_id = room_id
        self.encryption_key = encryption_key
        self.username = username
        self.is_recording = False
        self.is_playing = False
        self.current_speaker = None
        self.voice_server_port = None

        if not VOICE_ENABLED:
            return
            raise RuntimeError("pyaudio not installed. Install with: pip install pyaudio")

        self.audio = pyaudio.PyAudio()
        self.voice_dir = VOICE_CACHE_DIR / room_id
        self.voice_dir.mkdir(parents=True, exist_ok=True)

        # Voice activity detection
        self.voice_threshold = 500  # Audio level threshold
        self.speaking = False

        # Network
        self.server_socket = None
        self.clients = {}  # {addr: socket}
        self.running = False

    def calculate_rms(self, audio_data):
        """Calculate RMS (Root Mean Square) for voice activity detection."""
        import array
        count = len(audio_data) / 2
        format_str = "%dh" % count
        shorts = array.array('h', audio_data)
        sum_squares = sum((sample ** 2 for sample in shorts))
        rms = (sum_squares / count) ** 0.5
        return rms

    def start_voice_server(self, port: int = 0):
        """Start voice relay server for this room."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(0.5)

        self.voice_server_port = self.server_socket.getsockname()[1]
        self.running = True

        # Start accepting clients
        accept_thread = threading.Thread(target=self._accept_clients, daemon=True)
        accept_thread.start()

        return self.voice_server_port

    def _accept_clients(self):
        """Accept incoming voice client connections."""
        while self.running:
            try:
                client_sock, addr = self.server_socket.accept()
                client_sock.settimeout(1.0)
                self.clients[addr] = client_sock
                print(f"\r{Style.GREEN('🎤 New voice participant connected')}{' ' * 20}")

                # Start receiving thread for this client
                recv_thread = threading.Thread(
                    target=self._receive_from_client,
                    args=(client_sock, addr),
                    daemon=True
                )
                recv_thread.start()

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"\r{Style.RED(f'Voice server error: {e}')}{' ' * 20}")

    def _receive_from_client(self, client_sock, addr):
        """Receive audio from client and broadcast to others."""
        try:
            while self.running:
                try:
                    # Receive packet size
                    size_data = client_sock.recv(4)
                    if not size_data or len(size_data) < 4:
                        break

                    packet_size = int.from_bytes(size_data, 'big')

                    # Sanity check
                    if packet_size > 1024 * 1024:  # 1MB max
                        break

                    # Receive full packet
                    packet = b''
                    while len(packet) < packet_size:
                        remaining = packet_size - len(packet)
                        chunk = client_sock.recv(min(remaining, 4096))
                        if not chunk:
                            break
                        packet += chunk

                    if len(packet) != packet_size:
                        break

                    # Parse packet header to update speaker
                    if len(packet) >= 3:
                        username_len = int.from_bytes(packet[:2], 'big')
                        if len(packet) >= 2 + username_len + 1:
                            username = packet[2:2 + username_len].decode('utf-8')
                            is_speaking = packet[2 + username_len] == 1

                            # Update current speaker
                            if is_speaking:
                                self.current_speaker = username
                            elif self.current_speaker == username:
                                self.current_speaker = None

                    # Broadcast to all other clients
                    self._broadcast_audio(packet, addr)

                except socket.timeout:
                    continue
                except Exception:
                    break

        except Exception:
            pass
        finally:
            if addr in self.clients:
                del self.clients[addr]
                print(f"\r{Style.YELLOW('Voice participant disconnected')}{' ' * 20}")
            try:
                client_sock.close()
            except:
                pass

    def _broadcast_audio(self, packet, exclude_addr):
        """Broadcast audio packet to all clients except sender."""
        dead_clients = []
        for addr, client_sock in self.clients.items():
            if addr == exclude_addr:
                continue
            try:
                # Send packet size then packet
                size_bytes = len(packet).to_bytes(4, 'big')
                client_sock.sendall(size_bytes + packet)
            except:
                dead_clients.append(addr)

        # Remove dead clients
        for addr in dead_clients:
            if addr in self.clients:
                del self.clients[addr]

    def connect_to_voice_server(self, host: str, port: int):
        """Connect to voice relay server."""
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))
        self.client_socket.settimeout(1.0)
        self.running = True

        # Start playback thread
        playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        playback_thread.start()

    def _playback_loop(self):
        """Receive and play audio from server."""
        stream = self.audio.open(
            format=VOICE_FORMAT,
            channels=VOICE_CHANNELS,
            rate=VOICE_RATE,
            output=True,
            frames_per_buffer=VOICE_CHUNK
        )

        print(f"{Style.GREEN('🔊 Playback active - Listening...')}")

        try:
            while self.running:
                try:
                    # Receive packet size
                    size_data = self.client_socket.recv(4)
                    if not size_data or len(size_data) < 4:
                        break

                    packet_size = int.from_bytes(size_data, 'big')

                    # Sanity check
                    if packet_size > 1024 * 1024:  # 1MB max
                        print(f"\r{Style.RED('Invalid packet size')}{' ' * 20}")
                        break

                    # Receive full packet
                    packet = b''
                    while len(packet) < packet_size:
                        remaining = packet_size - len(packet)
                        chunk = self.client_socket.recv(min(remaining, 4096))
                        if not chunk:
                            break
                        packet += chunk

                    if len(packet) != packet_size:
                        break

                    # Parse packet
                    if len(packet) < 3:
                        continue

                    username_len = int.from_bytes(packet[:2], 'big')
                    if len(packet) < 2 + username_len + 1:
                        continue

                    username = packet[2:2 + username_len].decode('utf-8')
                    is_speaking = packet[2 + username_len] == 1
                    audio_data = packet[2 + username_len + 1:]

                    # Update speaker
                    if is_speaking:
                        self.current_speaker = username
                    elif self.current_speaker == username:
                        self.current_speaker = None

                    # Decrypt and play if there's audio data
                    if len(audio_data) > 0:
                        try:
                            decrypted_audio = CryptoManager.decrypt_bytes(
                                audio_data,
                                self.encryption_key
                            )
                            stream.write(decrypted_audio)
                        except Exception as e:
                            # Decryption failed, skip this packet
                            pass

                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"\r{Style.RED(f'Playback error: {e}')}{' ' * 20}")
                    break

        finally:
            stream.stop_stream()
            stream.close()
            print(f"\n{Style.YELLOW('🔇 Playback stopped')}")

    def start_recording_stream(self):
        """Start streaming microphone input to server."""
        self.is_recording = True

        stream = self.audio.open(
            format=VOICE_FORMAT,
            channels=VOICE_CHANNELS,
            rate=VOICE_RATE,
            input=True,
            frames_per_buffer=VOICE_CHUNK
        )

        print(f"{Style.GREEN('🎤 Microphone active - Start speaking!')}")

        try:
            silence_counter = 0
            while self.is_recording:
                try:
                    # Read audio
                    audio_data = stream.read(VOICE_CHUNK, exception_on_overflow=False)

                    # Voice activity detection
                    rms = self.calculate_rms(audio_data)
                    is_speaking = rms > self.voice_threshold

                    if is_speaking:
                        self.speaking = True
                        silence_counter = 0

                        # Encrypt audio directly as bytes
                        encrypted_bytes = CryptoManager.encrypt_bytes(
                            audio_data,
                            self.encryption_key
                        )

                        # Build packet: [username_len(2)][username][speaker_flag(1)][audio_data]
                        username_bytes = self.username.encode('utf-8')
                        username_len = len(username_bytes).to_bytes(2, 'big')
                        speaker_flag = b'\x01'

                        packet = username_len + username_bytes + speaker_flag + encrypted_bytes

                        # Send to server
                        try:
                            size_bytes = len(packet).to_bytes(4, 'big')
                            self.client_socket.sendall(size_bytes + packet)
                        except Exception as e:
                            print(f"\r{Style.RED(f'Send error: {e}')}{' ' * 20}")
                            break
                    else:
                        silence_counter += 1

                        # Send stop-speaking packet after 3 consecutive silent chunks
                        if self.speaking and silence_counter > 3:
                            username_bytes = self.username.encode('utf-8')
                            username_len = len(username_bytes).to_bytes(2, 'big')
                            packet = username_len + username_bytes + b'\x00'

                            try:
                                size_bytes = len(packet).to_bytes(4, 'big')
                                self.client_socket.sendall(size_bytes + packet)
                            except:
                                pass

                            self.speaking = False

                except Exception as e:
                    if self.is_recording:
                        print(f"\r{Style.RED(f'Recording error: {e}')}{' ' * 20}")
                    break

        finally:
            stream.stop_stream()
            stream.close()
            print(f"\n{Style.YELLOW('🔇 Microphone stopped')}")

    def stop_recording(self):
        """Stop recording stream."""
        self.is_recording = False

    def get_current_speaker(self):
        """Get username of current speaker."""
        return self.current_speaker

    def cleanup(self):
        """Cleanup voice resources."""
        self.running = False
        self.is_recording = False

        if hasattr(self, 'client_socket'):
            try:
                self.client_socket.close()
            except:
                pass

        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

        # Close all client connections
        for client_sock in self.clients.values():
            try:
                client_sock.close()
            except:
                pass

        self.clients.clear()

        try:
            self.audio.terminate()
        except:
            pass


# =================== Chat Manager ===================

class P2PChatManager:
    """Manages E2E encrypted chat rooms with file and voice support."""

    def __init__(self, app: App, voice_server_info: Dict[str, Tuple[str, int]]):
        self.app = app
        self.rooms: Dict[str, ChatRoom] = {}
        self.current_room: Optional[str] = None
        self.username = app.get_username() or "anonymous"
        CHAT_ROOMS_DIR.mkdir(parents=True, exist_ok=True)
        self._load_rooms()
        self.file_managers: Dict[str, FileTransferManager] = {}
        self.voice_manager: Optional[VoiceChatManager] = None
        self.voice_server_info = voice_server_info

    def create_room(self, name: str, password: str, max_participants: int = 10,
                    voice_enabled: bool = False, private: bool = False) -> Result:
        """Create a new chat room."""
        room_id = hashlib.sha256(f"{name}_{self.username}_{time.time()}".encode()).hexdigest()[:12]
        encryption_key = CryptoManager.generate_room_key(room_id, password)

        room = ChatRoom(
            room_id=room_id,
            name=name,
            owner=self.username,
            participants={self.username},
            is_locked=False,
            is_private=private,
            created_at=datetime.now(),
            encryption_key=encryption_key.decode(),
            max_participants=max_participants,
            voice_enabled=voice_enabled,
            file_transfer_enabled=True
        )

        self.rooms[room_id] = room
        self._save_room(room)
        # Start voice server if voice enabled
        if voice_enabled:
            try:
                voice_mgr = VoiceChatManager(room_id, encryption_key, self.username)
                port = voice_mgr.start_voice_server()
                self.voice_server_info[room_id] = ('127.0.0.1', port)
                print(f"   {Style.GREEN('Voice server started on port:')} {port}")
            except Exception as e:
                print(f"   {Style.YELLOW(f'Warning: Could not start voice server: {e}')}")
        # Send system message
        self._send_system_message(room_id, f"Room '{name}' created by {self.username}")

        return Result.ok(data={
            'room_id': room_id,
            'name': name,
            'message': f'Room "{name}" created successfully'
        })

    def join_room(self, room_id: str, password: str) -> Result:
        """Join an existing chat room."""
        if room_id not in self.rooms:
            return Result.default_user_error("Room not found")

        room = self.rooms[room_id]

        if room.is_locked:
            return Result.default_user_error("Room is locked")

        if len(room.participants) >= room.max_participants:
            return Result.default_user_error("Room is full")

        # Verify password
        try:
            key = CryptoManager.generate_room_key(room_id, password)
            if key.decode() != room.encryption_key:
                return Result.default_user_error("Invalid password")
        except Exception:
            return Result.default_user_error("Invalid password")

        room.participants.add(self.username)
        self.current_room = room_id
        self._save_room(room)

        # Initialize file manager
        self.file_managers[room_id] = FileTransferManager(room_id, key)

        # Initialize voice manager if enabled
        # Get voice server info if available
        if room.voice_enabled:
            # Ask for voice server details if not already known
            if room_id not in self.voice_server_info:
                print(f"\n{Style.CYAN('Voice chat is enabled. Enter server details:')}")
                voice_host = input(
                    f"  {Style.WHITE('Voice server host (default: 127.0.0.1):')} ").strip() or "127.0.0.1"
                voice_port = input(f"  {Style.WHITE('Voice server port:')} ").strip()

                if voice_port and voice_port.isdigit():
                    self.voice_server_info[room_id] = (voice_host, int(voice_port))

        # Send system message
        self._send_system_message(room_id, f"{self.username} joined the room")

        return Result.ok(data={
            'room_id': room_id,
            'name': room.name,
            'participants': list(room.participants),
            'voice_enabled': room.voice_enabled,
            'file_transfer_enabled': room.file_transfer_enabled
        })

    def leave_room(self, room_id: str) -> Result:
        """Leave a chat room."""
        if room_id not in self.rooms:
            return Result.default_user_error("Room not found")

        room = self.rooms[room_id]
        if self.username not in room.participants:
            return Result.default_user_error("You are not in this room")

        # Send system message before leaving
        self._send_system_message(room_id, f"{self.username} left the room")

        room.participants.remove(self.username)

        # If owner leaves, transfer ownership or delete room
        if room.owner == self.username:
            if len(room.participants) > 0:
                room.owner = list(room.participants)[0]
                self._send_system_message(room_id, f"Room ownership transferred to {room.owner}")
            else:
                # Delete empty room
                self._delete_room(room_id)
                return Result.ok(data="Room deleted (no participants)")

        self._save_room(room)

        if self.current_room == room_id:
            self.current_room = None

        # Cleanup managers
        if room_id in self.file_managers:
            del self.file_managers[room_id]
        if self.voice_manager:
            self.voice_manager.cleanup()
            self.voice_manager = None

        return Result.ok(data="Left room successfully")

    def lock_room(self, room_id: str) -> Result:
        """Lock a room to prevent new participants."""
        if room_id not in self.rooms:
            return Result.default_user_error("Room not found")

        room = self.rooms[room_id]

        if room.owner != self.username:
            return Result.default_user_error("Only room owner can lock the room")

        room.is_locked = True
        room.is_private = True
        self._save_room(room)

        self._send_system_message(room_id, f"Room locked by {self.username}")

        return Result.ok(data=f'Room "{room.name}" is now locked and private')

    def send_message(self, room_id: str, content: str, password: str) -> Result:
        """Send encrypted text message to room."""
        if room_id not in self.rooms:
            return Result.default_user_error("Room not found")

        room = self.rooms[room_id]
        if self.username not in room.participants:
            return Result.default_user_error("You are not in this room")

        try:
            key = CryptoManager.generate_room_key(room_id, password)
            encrypted_content = CryptoManager.encrypt_message(content, key)

            message = ChatMessage(
                sender=self.username,
                content=encrypted_content,
                timestamp=datetime.now(),
                room_id=room_id,
                message_type=MessageType.TEXT,
                encrypted=True
            )

            self._save_message(message)
            return Result.ok(data="Message sent")

        except Exception as e:
            return Result.default_internal_error(f"Failed to send message: {e}")

    def send_file(self, room_id: str, file_path: Path, password: str) -> Result:
        """Send encrypted file to room."""
        if room_id not in self.rooms:
            return Result.default_user_error("Room not found")

        room = self.rooms[room_id]
        if not room.file_transfer_enabled:
            return Result.default_user_error("File transfer disabled in this room")

        if self.username not in room.participants:
            return Result.default_user_error("You are not in this room")

        try:
            # Prepare file for transfer
            file_manager = self.file_managers.get(room_id)
            if not file_manager:
                key = CryptoManager.generate_room_key(room_id, password)
                file_manager = FileTransferManager(room_id, key)
                self.file_managers[room_id] = file_manager

            transfer_id, file_size = file_manager.prepare_file(file_path)

            # Create file message
            key = CryptoManager.generate_room_key(room_id, password)
            encrypted_content = CryptoManager.encrypt_message(transfer_id, key)

            message = ChatMessage(
                sender=self.username,
                content=encrypted_content,
                timestamp=datetime.now(),
                room_id=room_id,
                message_type=MessageType.FILE,
                encrypted=True,
                file_name=file_path.name,
                file_size=file_size
            )

            self._save_message(message)

            return Result.ok(data={
                'transfer_id': transfer_id,
                'file_name': file_path.name,
                'file_size': file_size
            })

        except Exception as e:
            return Result.default_internal_error(f"Failed to send file: {e}")

    def receive_file(self, room_id: str, transfer_id: str, file_name: str) -> Result:
        """Receive and decrypt file from room."""
        if room_id not in self.rooms:
            return Result.default_user_error("Room not found")

        try:
            file_manager = self.file_managers.get(room_id)
            if not file_manager:
                return Result.default_user_error("File manager not initialized")

            output_path = file_manager.receive_file(transfer_id, file_name)

            return Result.ok(data={
                'file_path': str(output_path),
                'file_name': file_name
            })

        except Exception as e:
            return Result.default_internal_error(f"Failed to receive file: {e}")

    def get_messages(self, room_id: str, password: str, limit: int = 50) -> Result:
        """Get decrypted messages from room."""
        if room_id not in self.rooms:
            return Result.default_user_error("Room not found")

        room = self.rooms[room_id]
        if self.username not in room.participants:
            return Result.default_user_error("You are not in this room")

        try:
            key = CryptoManager.generate_room_key(room_id, password)
            messages = self._load_messages(room_id, limit)

            decrypted_messages = []
            for msg in messages:
                if msg.encrypted and msg.message_type != MessageType.SYSTEM:
                    try:
                        decrypted_content = CryptoManager.decrypt_message(msg.content, key)
                        decrypted_messages.append({
                            'sender': msg.sender,
                            'content': decrypted_content,
                            'timestamp': msg.timestamp.strftime('%H:%M:%S'),
                            'message_type': msg.message_type.value,
                            'is_own': msg.sender == self.username,
                            'file_name': msg.file_name,
                            'file_size': msg.file_size
                        })
                    except Exception:
                        continue
                else:
                    decrypted_messages.append({
                        'sender': msg.sender,
                        'content': msg.content,
                        'timestamp': msg.timestamp.strftime('%H:%M:%S'),
                        'message_type': msg.message_type.value,
                        'is_own': False
                    })

            return Result.ok(data=decrypted_messages)

        except Exception as e:
            return Result.default_internal_error(f"Failed to get messages: {e}")

    def list_rooms(self, show_all: bool = False) -> Result:
        """List available rooms for user."""
        user_rooms = []
        for room in self.rooms.values():
            # Show only user's rooms unless show_all is True
            if show_all or self.username in room.participants:
                # Don't show private/locked rooms to non-participants
                if room.is_private and self.username not in room.participants:
                    continue

                user_rooms.append({
                    'room_id': room.room_id,
                    'name': room.name,
                    'owner': room.owner,
                    'participants_count': len(room.participants),
                    'max_participants': room.max_participants,
                    'is_locked': room.is_locked,
                    'is_private': room.is_private,
                    'voice_enabled': room.voice_enabled,
                    'file_transfer_enabled': room.file_transfer_enabled,
                    'created_at': room.created_at.strftime('%Y-%m-%d %H:%M'),
                    'is_member': self.username in room.participants
                })

        return Result.ok(data=user_rooms)

    def _send_system_message(self, room_id: str, content: str):
        """Send a system message (not encrypted)."""
        message = ChatMessage(
            sender="SYSTEM",
            content=content,
            timestamp=datetime.now(),
            room_id=room_id,
            message_type=MessageType.SYSTEM,
            encrypted=False
        )
        self._save_message(message)

    def _save_room(self, room: ChatRoom):
        """Save room to storage."""
        room_file = CHAT_ROOMS_DIR / f"room_{room.room_id}.json"

        with open(room_file, 'w') as f:
            json.dump(room.to_dict(), f, indent=2)

    def _load_rooms(self):
        """Load rooms from storage."""
        if not CHAT_ROOMS_DIR.exists():
            return

        for room_file in CHAT_ROOMS_DIR.glob("room_*.json"):
            try:
                with open(room_file) as f:
                    room_data = json.load(f)
                    room = ChatRoom.from_dict(room_data)
                    self.rooms[room.room_id] = room
            except Exception as e:
                print(f"Warning: Failed to load room {room_file}: {e}")

    def _delete_room(self, room_id: str):
        """Delete a room and its messages."""
        if room_id in self.rooms:
            del self.rooms[room_id]

        # Delete room file
        room_file = CHAT_ROOMS_DIR / f"room_{room_id}.json"
        if room_file.exists():
            room_file.unlink()

        # Delete messages file
        messages_file = CHAT_ROOMS_DIR / f"messages_{room_id}.jsonl"
        if messages_file.exists():
            messages_file.unlink()

    def _save_message(self, message: ChatMessage):
        """Save message to storage."""
        messages_file = CHAT_ROOMS_DIR / f"messages_{message.room_id}.jsonl"
        with open(messages_file, 'a') as f:
            f.write(json.dumps(message.to_dict()) + '\n')

    def _load_messages(self, room_id: str, limit: int = 50) -> List[ChatMessage]:
        """Load messages from storage."""
        messages_file = CHAT_ROOMS_DIR / f"messages_{room_id}.jsonl"
        if not messages_file.exists():
            return []

        messages = []
        with open(messages_file) as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                try:
                    message_data = json.loads(line.strip())
                    messages.append(ChatMessage.from_dict(message_data))
                except Exception:
                    continue

        return messages


# =================== Chat Helper ===================

class ChatListener(threading.Thread):
    """Background thread to listen for new chat messages."""

    def __init__(self, chat_manager, room_id, password, callback):
        super().__init__(daemon=True)
        self.chat_manager = chat_manager
        self.room_id = room_id
        self.password = password
        self.callback = callback
        self.running = True
        self.last_message_count = 0

    def run(self):
        while self.running:
            try:
                result = self.chat_manager.get_messages(self.room_id, self.password, 50)
                if result.is_ok():
                    messages = result.get()
                    if len(messages) > self.last_message_count:
                        # New messages arrived
                        new_messages = messages[self.last_message_count:]
                        for msg in new_messages:
                            if not msg['is_own']:  # Only show messages from others
                                self.callback(msg)
                        self.last_message_count = len(messages)
            except Exception:
                pass
            time.sleep(1)  # Poll every second

    def stop(self):
        self.running = False

# =================== Instance Manager ===================

class EnhancedInstanceManager:
    """Enhanced instance manager with chat integration."""

    def __init__(self, name: str, app: App):
        self.name = name
        self.app = app
        self.instance_dir = INSTANCES_ROOT_DIR / self.name
        self.state_file = self.instance_dir / "state.json"
        self.config_file = self.instance_dir / "config.toml"
        self.log_file = self.instance_dir / "instance.log"

    def read_state(self) -> dict:
        """Read instance state."""
        if not self.state_file.exists():
            return {}
        try:
            with open(self.state_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def write_state(self, state_data: dict):
        """Write instance state."""
        self.instance_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state_data, f, indent=2)

    def is_running(self) -> bool:
        """Check if instance is running."""
        pid = self.read_state().get('pid')
        return psutil.pid_exists(pid) if pid else False

    def generate_config(self, mode: str, config_data: dict):
        """Generate config.toml for instance."""
        content = f'mode = "{mode}"\n\n'

        if mode == "relay":
            content += "[relay]\n"
            content += f'bind_address = "{config_data.get("bind_address", "0.0.0.0:9000")}"\n'
            content += f'password = "{config_data.get("password", "")}"\n'

        elif mode == "peer":
            content += "[peer]\n"
            content += f'relay_address = "{config_data.get("relay_address", "127.0.0.1:9000")}"\n'
            content += f'relay_password = "{config_data.get("relay_password", "")}"\n'
            content += f'peer_id = "{config_data.get("peer_id", "default-peer")}"\n'
            content += f'listen_address = "{config_data.get("listen_address", "127.0.0.1:8000")}"\n'
            content += f'forward_to_address = "{config_data.get("forward_to_address", "127.0.0.1:3000")}"\n'
            if config_data.get("target_peer_id"):
                content += f'target_peer_id = "{config_data.get("target_peer_id")}"\n'

        self.instance_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w") as f:
            f.write(content)

    def start(self, executable_path: Path, mode: str, config_data: dict, chat_room: Optional[str] = None) -> bool:
        """Start instance."""
        if self.is_running():
            print(Style.YELLOW(f"Instance '{self.name}' is already running"))
            return True

        self.generate_config(mode, config_data)
        log_handle = open(self.log_file, 'a')

        try:
            with Spinner(f"Starting '{self.name}'", symbols="d"):
                process = subprocess.Popen(
                    [str(executable_path)],
                    cwd=str(self.instance_dir),
                    stdout=log_handle,
                    stderr=log_handle,
                    creationflags=subprocess.DETACHED_PROCESS if platform.system() == "Windows" else 0
                )
                time.sleep(1.5)

            if process.poll() is not None:
                print(f"\n{Style.RED2('❌')} Instance failed to start")
                return False

            state = {'pid': process.pid, 'mode': mode, 'config': config_data}
            if chat_room:
                state['chat_room'] = chat_room
            self.write_state(state)

            print(f"\n{Style.GREEN2('✅')} Instance '{Style.Bold(self.name)}' started (PID: {process.pid})")
            if chat_room:
                print(f"   {Style.BLUE('Chat Room:')} {Style.CYAN(chat_room)}")
            return True

        except Exception as e:
            print(f"\n{Style.RED2('❌')} Failed to start: {e}")
            return False

    def stop(self, timeout: int = 10) -> bool:
        """Stop instance."""
        if not self.is_running():
            self.write_state({})
            return True

        pid = self.read_state().get('pid')

        try:
            with Spinner(f"Stopping '{self.name}'", symbols="+", time_in_s=timeout, count_down=True):
                proc = psutil.Process(pid)
                proc.terminate()
                proc.wait(timeout)
        except psutil.TimeoutExpired:
            proc.kill()
        except (psutil.NoSuchProcess, Exception):
            pass

        self.write_state({})
        print(f"\n{Style.VIOLET2('⏹️')} Instance '{Style.Bold(self.name)}' stopped")
        return True


# =================== Interactive CLI ===================

class InteractiveP2PCLI:
    """Interactive P2P CLI with modern ToolBox-style interface."""

    def __init__(self):
        self.app = get_app("P2P_Interactive_CLI")

        self.voice_server_info: Dict[str, Tuple[str, int]] = {}
        self.chat_manager = P2PChatManager(self.app, self.voice_server_info)
        self.instances: Dict[str, EnhancedInstanceManager] = {}
        self.current_chat_room = None
        self.current_chat_password = None
        self.running = True
        self._load_instances()

        self.file_managers: Dict[str, FileTransferManager] = {}
        self.voice_manager: Optional[VoiceChatManager] = None

    def _load_instances(self):
        """Load existing instances."""
        if INSTANCES_ROOT_DIR.exists():
            for instance_dir in INSTANCES_ROOT_DIR.iterdir():
                if instance_dir.is_dir():
                    self.instances[instance_dir.name] = EnhancedInstanceManager(instance_dir.name, self.app)

    def clear_screen(self):
        """Clear terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self):
        """Print main header."""
        print(f"""
{Style.CYAN('╔══════════════════════════════════════════════════════════════════════╗')}
{Style.CYAN('║')} {Style.Bold(Style.WHITE('🌐 ToolBox P2P Manager'))} {Style.CYAN('v2.0')} {Style.GREY('- Interactive Mode')} {self._current_room_name() or '':<21} {Style.CYAN('║')}
{Style.CYAN('║')} {Style.GREY('E2E Encrypted Chat • File Transfer • Voice Chat • P2P Tunnels')} {' ' * 6} {Style.CYAN('║')}
{Style.CYAN('╚══════════════════════════════════════════════════════════════════════╝')}
""")

    def print_menu(self):
        """Print main menu."""
        print(f"""
{Style.Bold(Style.WHITE('┌─ 🎯 MAIN MENU ───────────────────────────────────────────────────────┐'))}
{Style.WHITE('│')}                                                                      {Style.WHITE('│')}
{Style.WHITE('│')}  {Style.CYAN('1.')} {Style.WHITE('💬 Chat Mode')}          - Start interactive E2E encrypted chat     {Style.WHITE('│')}
{Style.WHITE('│')}  {Style.CYAN('2.')} {Style.WHITE('🔧 P2P Configuration')}  - Configure P2P connections                {Style.WHITE('│')}
{Style.WHITE('│')}  {Style.CYAN('3.')} {Style.WHITE('📊 Status & Monitoring')} - View connections and rooms              {Style.WHITE('│')}
{Style.WHITE('│')}  {Style.CYAN('4.')} {Style.WHITE('⚙️  Settings')}           - Manage configuration                    {Style.WHITE('│')}
{Style.WHITE('│')}  {Style.CYAN('0.')} {Style.WHITE('🚪 Exit')}               - Quit application                         {Style.WHITE('│')}
{Style.WHITE('│')}                                                                      {Style.WHITE('│')}
{Style.Bold(Style.WHITE('└──────────────────────────────────────────────────────────────────────┘'))}
""")

    def chat_menu(self):
        """Interactive chat menu."""
        try:
            while True:
                self.clear_screen()
                self.print_header()

                # Show current room info
                if self.current_chat_room:
                    room = self.chat_manager.rooms.get(self.current_chat_room)
                    if room:
                        print(f"""
    {Style.GREEN('╔══ Current Room ════════════════════════════════════════════════════╗')}
    {Style.GREEN('║')} {Style.WHITE('Name:')} {Style.YELLOW(room.name):<30} {Style.WHITE('ID:')} {Style.CYAN(room.room_id):<15} {" "*22+Style.GREEN('║')}
    {Style.GREEN('║')} {Style.WHITE('Participants:')} {', '.join(list(room.participants)[:10]):<50}{'...' if len(room.participants) > 3 else '':<30}
    {Style.GREEN('╚════════════════════════════════════════════════════════════════════╝')}
    """)

                print(f"""
    {Style.Bold(Style.WHITE('┌─ 💬 CHAT MENU ───────────────────────────────────────────────────────┐'))}
    {Style.WHITE('│')}                                                                      {Style.WHITE('│')}
    {Style.WHITE('│')}  {Style.CYAN('1.')} {Style.WHITE('Create Room')}         - Create new E2E encrypted chat room         {Style.WHITE('│')}
    {Style.WHITE('│')}  {Style.CYAN('2.')} {Style.WHITE('Join Room')}           - Join existing room by ID                   {Style.WHITE('│')}
    {Style.WHITE('│')}  {Style.CYAN('3.')} {Style.WHITE('List Rooms')}          - Show available chat rooms                  {Style.WHITE('│')}
    {Style.WHITE('│')}  {Style.CYAN('4.')} {Style.WHITE('Interactive Chat')}    - Start live chat (current room)             {Style.WHITE('│')}
    {Style.WHITE('│')}  {Style.CYAN('5.')} {Style.WHITE('Send File')}           - Transfer file (E2E encrypted)              {Style.WHITE('│')}
    {Style.WHITE('│')}  {Style.CYAN('6.')} {Style.WHITE('Voice Chat')}          - Start voice chat (beta)                    {Style.WHITE('│')}
    {Style.WHITE('│')}  {Style.CYAN('7.')} {Style.WHITE('Lock Room')}           - Lock current room (owner only)             {Style.WHITE('│')}
    {Style.WHITE('│')}  {Style.CYAN('8.')} {Style.WHITE('Leave Room')}          - Leave current chat room                    {Style.WHITE('│')}
    {Style.WHITE('│')}  {Style.CYAN('0.')} {Style.WHITE('Back')}                - Return to main menu                        {Style.WHITE('│')}
    {Style.WHITE('│')}                                                                      {Style.WHITE('│')}
    {Style.Bold(Style.WHITE('└──────────────────────────────────────────────────────────────────────┘'))}
    """)

                choice = input(f"\n{Style.CYAN('❯')} {Style.WHITE('Select option:')} ").strip()

                if choice == '0':
                    break
                elif choice == '1':
                    self._create_chat_room()
                elif choice == '2':
                    self._join_chat_room()
                elif choice == '3':
                    self._list_chat_rooms()
                elif choice == '4':
                    self._interactive_chat()
                elif choice == '5':
                    self._send_file()
                elif choice == '6':
                    self._voice_chat()
                elif choice == '7':
                    self._lock_room()
                elif choice == '8':
                    self._leave_room()
                else:
                    print(f"{Style.RED('Invalid option')}")
                    time.sleep(1)
        finally:
            if self._current_room_name() is not None:
                self._leave_room(auto=True)

    def _create_chat_room(self):
        """Create new chat room."""
        print(f"\n{Style.Bold(Style.CYAN('Create New Chat Room'))}")
        print(Style.GREY('─' * 70))

        name = input(f"{Style.WHITE('Room name:')} ").strip()
        if not name:
            return

        password = input(f"{Style.WHITE('Room password:')} ").strip()
        if not password:
            return

        max_participants = input(f"{Style.WHITE('Max participants (default 10):')} ").strip()
        max_participants = int(max_participants) if max_participants.isdigit() else 10

        voice_enabled = input(f"{Style.WHITE('Enable voice chat? (y/N):')} ").strip().lower() == 'y'
        private = input(f"{Style.WHITE('Make private? (y/N):')} ").strip().lower() == 'y'

        result = self.chat_manager.create_room(name, password, max_participants, voice_enabled, private)

        if result.is_ok():
            data = result.get()
            print(f"\n{Style.GREEN2('✅ Room created successfully!')}")
            print(f"   {Style.WHITE('Room ID:')} {Style.CYAN(data['room_id'])}")
            print(f"   {Style.WHITE('Name:')} {Style.YELLOW(data['name'])}")

            # Auto-join created room
            self.current_chat_room = data['room_id']
            self.current_chat_password = password
        else:
            print(f"{Style.RED2('❌ Failed:')} {result.info}")

        input(f"\n{Style.GREY('Press Enter to continue...')}")

    def _join_chat_room(self):
        """Join existing chat room."""
        print(f"\n{Style.Bold(Style.CYAN('Join Chat Room'))}")
        print(Style.GREY('─' * 70))

        # First show available rooms
        result = self.chat_manager.list_rooms(show_all=True)
        if result.is_ok():
            rooms = result.get()
            if rooms:
                print(f"\n{Style.WHITE('Available Rooms:')}")
                for i, room in enumerate(rooms, 1):
                    status = "🔒" if room['is_locked'] else "🔓"
                    member = "✓" if room['is_member'] else " "
                    print(
                        f"  {i}. [{member}] {status} {Style.YELLOW(room['name'][:20])} - {Style.CYAN(room['room_id'])}")
                print()

        room_id = input(f"{Style.WHITE('Room ID:')} ").strip()
        if not room_id:
            return

        password = input(f"{Style.WHITE('Password:')} ").strip()
        if not password:
            return

        result = self.chat_manager.join_room(room_id, password)

        if result.is_ok():
            data = result.get()
            self.current_chat_room = room_id
            self.current_chat_password = password

            print(f"\n{Style.GREEN2('✅ Joined room successfully!')}")
            print(f"   {Style.WHITE('Room:')} {Style.YELLOW(data['name'])}")
            print(f"   {Style.WHITE('Participants:')} {', '.join(data['participants'])}")
            if data['voice_enabled']:
                print(f"   {Style.WHITE('Voice chat:')} {Style.GREEN('Enabled')}")
        else:
            print(f"{Style.RED2('❌ Failed:')} {result.info}")

        input(f"\n{Style.GREY('Press Enter to continue...')}")

    def _list_chat_rooms(self):
        """List available chat rooms."""
        print(f"\n{Style.Bold(Style.CYAN('Chat Rooms'))}")
        print(Style.GREY('═' * 90))

        result = self.chat_manager.list_rooms(show_all=True)

        if result.is_ok():
            rooms = result.get()
            if not rooms:
                print(Style.YELLOW("\n  No chat rooms available"))
            else:
                print(
                    f"\n{Style.Underline('NAME'):<22} {Style.Underline('ROOM ID'):<14} {Style.Underline('OWNER'):<12} {Style.Underline('PARTICIPANTS'):<15} {Style.Underline('STATUS'):<12} {Style.Underline('FEATURES')}")
                print(Style.GREY('─' * 90))

                for room in rooms:
                    name = Style.YELLOW(room['name'][:20])
                    room_id = Style.CYAN(room['room_id'])
                    owner = Style.BLUE(room['owner'][:10])
                    participants = f"{room['participants_count']}/{room['max_participants']}"

                    status_parts = []
                    if room['is_locked']:
                        status_parts.append(Style.RED('🔒 Locked'))
                    if room['is_private']:
                        status_parts.append(Style.YELLOW('🔐 Private'))
                    if not status_parts:
                        status_parts.append(Style.GREEN('🔓 Open'))
                    status = ' '.join(status_parts)[:11]

                    features = []
                    if room['voice_enabled']:
                        features.append('🎤')
                    if room['file_transfer_enabled']:
                        features.append('📁')
                    if room['is_member']:
                        features.append('✓')
                    features_str = ' '.join(features)

                    print(f"{name:<22} {room_id:<14} {owner:<12} {participants:<15} {status:<12} {features_str}")
        else:
            print(f"{Style.RED2('❌ Failed:')} {result.info}")

        input(f"\n{Style.GREY('Press Enter to continue...')}")

    def _interactive_chat(self):
        """Start interactive chat mode."""
        if not self.current_chat_room:
            print(f"{Style.RED2('❌ No active chat room. Join a room first.')}")
            input(f"\n{Style.GREY('Press Enter to continue...')}")
            return

        room = self.chat_manager.rooms.get(self.current_chat_room)
        if not room:
            print(f"{Style.RED2('❌ Room not found')}")
            input(f"\n{Style.GREY('Press Enter to continue...')}")
            return

        self.clear_screen()
        print(f"""
    {Style.CYAN('╔══════════════════════════════════════════════════════════════════════╗')}
    {Style.CYAN('║')} {Style.Bold(Style.WHITE('💬 Interactive Chat'))} - {Style.YELLOW(room.name[:30])} {' ' * (45 - len(room.name[:30]))} {Style.CYAN('║')}
    {Style.CYAN('║')} {Style.GREY('Room ID:')} {Style.CYAN(room.room_id)}{' ' * (59 - len(room.room_id))} {Style.CYAN('║')}
    {Style.CYAN('╚══════════════════════════════════════════════════════════════════════╝')}

    {Style.GREY('Commands:')} {Style.WHITE('/quit')} - Exit  {Style.WHITE('/file <path>')} - Send file  {Style.WHITE('/refresh')} - Reload messages
    """)
        print(Style.GREY('─' * 70))

        # Show recent messages
        result = self.chat_manager.get_messages(self.current_chat_room, self.current_chat_password, 20)
        message_count = 0
        if result.is_ok():
            messages = result.get()
            message_count = len(messages)
            for msg in messages[-10:]:
                self._display_message(msg)

        print(Style.GREY('─' * 70))

        # Start background listener for new messages
        def on_new_message(msg):
            # Clear current line and display new message
            print(f"\r{' ' * 80}\r", end='')  # Clear line
            self._display_message(msg)
            print(f"{Style.GREEN(f'{self.chat_manager.username}:')} ", end='', flush=True)

        listener = ChatListener(self.chat_manager, self.current_chat_room,
                                self.current_chat_password, on_new_message)
        listener.last_message_count = message_count
        listener.start()

        # Chat loop with non-blocking input
        try:
            while True:
                message = input(f"{Style.GREEN(f'{self.chat_manager.username}:')} ").strip()

                if not message:
                    continue

                if message == '/quit':
                    break

                elif message == '/refresh':
                    # Reload and show recent messages
                    result = self.chat_manager.get_messages(
                        self.current_chat_room,
                        self.current_chat_password,
                        20
                    )
                    if result.is_ok():
                        print(Style.GREY('─' * 70))
                        for msg in result.get()[-10:]:
                            self._display_message(msg)
                        print(Style.GREY('─' * 70))
                        listener.last_message_count = len(result.get())

                elif message.startswith('/file '):
                    file_path = Path(message[6:].strip())
                    self._send_file_inline(file_path)

                elif message == '/voice':
                    print(Style.YELLOW("Voice chat not yet implemented in interactive mode"))

                else:
                    result = self.chat_manager.send_message(
                        self.current_chat_room,
                        message,
                        self.current_chat_password
                    )

                    if result.is_ok():
                        # Display own message
                        self._display_message({
                            'sender': self.chat_manager.username,
                            'content': message,
                            'timestamp': datetime.now().strftime('%H:%M:%S'),
                            'message_type': 'text',
                            'is_own': True
                        })
                        # Update message count to prevent duplicate display
                        listener.last_message_count += 1
                    else:
                        print(f"{Style.RED('❌ Failed to send:')} {result.info}")

        except KeyboardInterrupt:
            pass
        finally:
            listener.stop()
            listener.join(timeout=1)

        print(f"\n{Style.YELLOW('👋 Exiting chat mode')}")
        time.sleep(1)

    def _display_message(self, msg: dict):
        """Display a chat message."""
        timestamp = Style.GREY(f"[{msg['timestamp']}]")

        if msg.get('message_type') == 'system':
            print(f"{timestamp} {Style.VIOLET2('⚙ ')} {Style.GREY(msg['content'])}")
        if msg.get('message_type') == 'file':
            sender_style = Style.GREEN if msg['is_own'] else Style.BLUE
            file_info = f"📁 {msg.get('file_name', 'Unknown')} ({msg.get('file_size', 0)} bytes)"
            sender = sender_style(f'{msg["sender"]}:')
            print(f"{timestamp} {sender} {Style.YELLOW(file_info)}")
        else:
            sender_style = Style.GREEN if msg['is_own'] else Style.BLUE
            sender = sender_style(f'{msg["sender"]}:')
            print(f"{timestamp} {sender} {Style.WHITE(msg['content'])}")

    def _send_file(self):
        """Send file in current room."""
        if not self.current_chat_room:
            print(f"{Style.RED2('❌ No active chat room')}")
            input(f"\n{Style.GREY('Press Enter to continue...')}")
            return

        print(f"\n{Style.Bold(Style.CYAN('Send File'))}")
        print(Style.GREY('─' * 70))

        file_path = input(f"{Style.WHITE('File path:')} ").strip()
        if not file_path:
            return

        self._send_file_inline(Path(file_path))
        input(f"\n{Style.GREY('Press Enter to continue...')}")

    def _send_file_inline(self, file_path: Path):
        """Send file (internal helper)."""
        if not file_path.exists():
            print(f"{Style.RED2('❌ File not found')}")
            return

        print(f"\n{Style.CYAN('📤 Sending file...')}")

        result = self.chat_manager.send_file(
            self.current_chat_room,
            file_path,
            self.current_chat_password
        )

        if result.is_ok():
            data = result.get()
            print(f"{Style.GREEN2('✅ File sent successfully!')}")
            print(f"   {Style.WHITE('File:')} {data['file_name']}")
            print(f"   {Style.WHITE('Size:')} {data['file_size']} bytes")
        else:
            print(f"{Style.RED2('❌ Failed:')} {result.info}")

    def _voice_chat(self):
        """Start live voice chat with speaker indication."""
        if not self.current_chat_room:
            print(f"{Style.RED2('❌ No active chat room')}")
            input(f"\n{Style.GREY('Press Enter to continue...')}")
            return

        room = self.chat_manager.rooms.get(self.current_chat_room)
        if not room or not room.voice_enabled:
            print(f"{Style.RED2('❌ Voice chat not enabled in this room')}")
            input(f"\n{Style.GREY('Press Enter to continue...')}")
            return

        if not VOICE_ENABLED:
            print(f"{Style.RED2('❌ pyaudio not installed')}")
            print(f"{Style.YELLOW('Install with:')} pip install pyaudio")
            input(f"\n{Style.GREY('Press Enter to continue...')}")
            return

        self.clear_screen()
        print(f"""
    {Style.CYAN('╔══════════════════════════════════════════════════════════════════════╗')}
    {Style.CYAN('║')} {Style.Bold(Style.WHITE('🎤 Live Voice Chat'))} - {Style.YELLOW(room.name[:30])} {' ' * (47 - len(room.name[:30]))} {Style.CYAN('║')}
    {Style.CYAN('║')} {Style.GREY('Press Ctrl+C to exit')} {' ' * 47} {Style.CYAN('║')}
    {Style.CYAN('╚══════════════════════════════════════════════════════════════════════╝')}
    """)

        try:
            # Initialize voice manager
            key = CryptoManager.generate_room_key(
                self.current_chat_room,
                self.current_chat_password
            )
            voice_mgr = VoiceChatManager(
                self.current_chat_room,
                key,
                self.chat_manager.username
            )

            # Check if we are the host or need to connect
            if self.current_chat_room in self.chat_manager.voice_server_info:
                # Connect to existing voice server
                host, port = self.chat_manager.voice_server_info[self.current_chat_room]
                print(f"{Style.CYAN('🔌 Connecting to voice server...')}")
                voice_mgr.connect_to_voice_server(host, port)
                print(f"{Style.GREEN2('✅ Connected to voice chat!')}\n")
            else:
                # Start as host
                print(f"{Style.CYAN('🎙️  Starting voice server...')}")
                port = voice_mgr.start_voice_server()
                self.chat_manager.voice_server_info[self.current_chat_room] = ('127.0.0.1', port)

                # Also connect to own server
                time.sleep(0.5)
                voice_mgr.connect_to_voice_server('127.0.0.1', port)
                print(f"{Style.GREEN2('✅ Voice server started on port:')} {port}")
                print(f"{Style.YELLOW('Share this info with participants:')}")
                print(f"   Host: 127.0.0.1 (or your public IP)")
                print(f"   Port: {port}\n")

            print(Style.GREY('─' * 70))
            print(f"{Style.WHITE('Voice Chat Active')} - {Style.GREEN('Speak into your microphone')}")
            print(Style.GREY('─' * 70))

            # Start recording thread
            record_thread = threading.Thread(
                target=voice_mgr.start_recording_stream,
                daemon=True
            )
            record_thread.start()

            # Display current speaker in real-time
            last_speaker = None
            print()  # Empty line for speaker display

            try:
                while True:
                    current_speaker = voice_mgr.get_current_speaker()

                    if current_speaker != last_speaker:
                        # Clear previous line and show new speaker
                        print(f"\r{' ' * 70}\r", end='')

                        if current_speaker:
                            if current_speaker == self.chat_manager.username:
                                print(f"\r{Style.GREEN('🎤 You are speaking...')}", end='', flush=True)
                            else:
                                print(f"\r{Style.CYAN(f'🎤 {current_speaker} is speaking...')}", end='', flush=True)
                        else:
                            print(f"\r{Style.GREY('🔇 Silence...')}", end='', flush=True)

                        last_speaker = current_speaker

                    time.sleep(0.1)  # Update display 10 times per second

            except KeyboardInterrupt:
                print(f"\n\n{Style.YELLOW('👋 Exiting voice chat...')}")

        except Exception as e:
            print(f"\n{Style.RED2('❌ Voice chat error:')} {e}")
            import traceback
            traceback.print_exc()

        finally:
            try:
                voice_mgr.cleanup()
            except:
                pass

        print(f"\n{Style.GREEN('Voice chat ended')}")
        input(f"\n{Style.GREY('Press Enter to continue...')}")

    def _lock_room(self):
        """Lock current room."""
        if not self.current_chat_room:
            print(f"{Style.RED2('❌ No active chat room')}")
            input(f"\n{Style.GREY('Press Enter to continue...')}")
            return

        result = self.chat_manager.lock_room(self.current_chat_room)

        if result.is_ok():
            print(f"\n{Style.GREEN2('✅ Room locked successfully!')}")
        else:
            print(f"\n{Style.RED2('❌ Failed:')} {result.info}")

        input(f"\n{Style.GREY('Press Enter to continue...')}")

    def _current_room_name(self):
        """Get name of current room."""
        if not self.current_chat_room:
            return None
        room = self.chat_manager.rooms.get(self.current_chat_room)
        return room.name if room else None

    def _leave_room(self, auto=False):
        """Leave current room."""
        if not self.current_chat_room:
            print(f"{Style.RED2('❌ No active chat room')}")
            input(f"\n{Style.GREY('Press Enter to continue...')}")
            return

        room = self.chat_manager.rooms.get(self.current_chat_room)
        room_name = room.name if room else "Unknown"

        confirm = input(f"\n{Style.YELLOW('⚠ Leave room')} '{room_name}'? (y/N): ").strip().lower() if not auto else 'y'
        if confirm != 'y':
            return

        result = self.chat_manager.leave_room(self.current_chat_room)

        if result.is_ok():
            print(f"\n{Style.GREEN2('✅ Left room successfully')}")
            self.current_chat_room = None
            self.current_chat_password = None
        else:
            print(f"\n{Style.RED2('❌ Failed:')} {result.info}")

        input(f"\n{Style.GREY('Press Enter to continue...')}")

    def p2p_menu(self):
        """P2P configuration menu."""
        while True:
            self.clear_screen()
            self.print_header()

            print(f"""
{Style.Bold(Style.WHITE('┌─ 🔧 P2P CONFIGURATION ───────────────────────────────────────────────┐'))}
{Style.WHITE('│')}                                                                      {Style.WHITE('│')}
{Style.WHITE('│')}  {Style.CYAN('1.')} {Style.WHITE('Start Relay Server')}  - Become a relay for P2P connections         {Style.WHITE('│')}
{Style.WHITE('│')}  {Style.CYAN('2.')} {Style.WHITE('Connect as Peer')}     - Connect to relay and other peers           {Style.WHITE('│')}
{Style.WHITE('│')}  {Style.CYAN('3.')} {Style.WHITE('Expose Local Service')} - Make local service accessible via P2P     {Style.WHITE('│')}
{Style.WHITE('│')}  {Style.CYAN('4.')} {Style.WHITE('Stop Instance')}       - Stop a running P2P instance                {Style.WHITE('│')}
{Style.WHITE('│')}  {Style.CYAN('0.')} {Style.WHITE('Back')}                - Return to main menu                        {Style.WHITE('│')}
{Style.WHITE('│')}                                                                      {Style.WHITE('│')}
{Style.Bold(Style.WHITE('└──────────────────────────────────────────────────────────────────────┘'))}
""")

            choice = input(f"\n{Style.CYAN('❯')} {Style.WHITE('Select option:')} ").strip()

            if choice == '0':
                break
            elif choice == '1':
                self._start_relay()
            elif choice == '2':
                self._connect_peer()
            elif choice == '3':
                self._expose_service()
            elif choice == '4':
                self._stop_instance()
            else:
                print(f"{Style.RED('Invalid option')}")
                time.sleep(1)

    def _start_relay(self):
        """Start relay server."""
        print(f"\n{Style.Bold(Style.CYAN('Start Relay Server'))}")
        print(Style.GREY('─' * 70))

        name = input(f"{Style.WHITE('Instance name (default: relay):')} ").strip() or "relay"
        bind = input(f"{Style.WHITE('Bind address (default: 0.0.0.0:9000):')} ").strip() or "0.0.0.0:9000"
        password = input(f"{Style.WHITE('Relay password:')} ").strip()

        if not password:
            print(f"{Style.RED2('❌ Password required')}")
            input(f"\n{Style.GREY('Press Enter to continue...')}")
            return

        # Get executable path
        executable = self._get_executable_path()
        if not executable:
            print(f"{Style.RED2('❌ Executable not found. Run')} {Style.WHITE('tb p2p build')} {Style.RED2('first')}")
            input(f"\n{Style.GREY('Press Enter to continue...')}")
            return

        # Create instance
        instance = EnhancedInstanceManager(name, self.app)
        config = {'bind_address': bind, 'password': password}

        success = instance.start(executable, 'relay', config)

        if success:
            self.instances[name] = instance

        input(f"\n{Style.GREY('Press Enter to continue...')}")

    def _connect_peer(self):
        """Connect as peer."""
        print(f"\n{Style.Bold(Style.CYAN('Connect as Peer'))}")
        print(Style.GREY('─' * 70))

        name = input(f"{Style.WHITE('Instance name:')} ").strip()
        if not name:
            return

        relay_addr = input(f"{Style.WHITE('Relay address (e.g., 127.0.0.1:9000):')} ").strip()
        relay_pass = input(f"{Style.WHITE('Relay password:')} ").strip()
        peer_id = input(f"{Style.WHITE('Your peer ID (default: instance name):')} ").strip() or name
        listen = input(f"{Style.WHITE('Listen address (default: 127.0.0.1:8000):')} ").strip() or "127.0.0.1:8000"
        target = input(f"{Style.WHITE('Target peer ID (optional):')} ").strip()

        if not all([relay_addr, relay_pass]):
            print(f"{Style.RED2('❌ Missing required fields')}")
            input(f"\n{Style.GREY('Press Enter to continue...')}")
            return

        # Optional: Link to chat room
        link_chat = input(f"{Style.WHITE('Link to chat room? (y/N):')} ").strip().lower() == 'y'
        chat_room = None

        if link_chat and self.current_chat_room:
            chat_room = self.current_chat_room

        # Get executable path
        executable = self._get_executable_path()
        if not executable:
            print(f"{Style.RED2('❌ Executable not found')}")
            input(f"\n{Style.GREY('Press Enter to continue...')}")
            return

        # Create instance
        instance = EnhancedInstanceManager(name, self.app)
        config = {
            'relay_address': relay_addr,
            'relay_password': relay_pass,
            'peer_id': peer_id,
            'listen_address': listen,
            'target_peer_id': target if target else None
        }

        success = instance.start(executable, 'peer', config, chat_room)

        if success:
            self.instances[name] = instance

        input(f"\n{Style.GREY('Press Enter to continue...')}")

    def _expose_service(self):
        """Expose local service via P2P."""
        print(f"\n{Style.Bold(Style.CYAN('Expose Local Service'))}")
        print(Style.GREY('─' * 70))

        name = input(f"{Style.WHITE('Instance name:')} ").strip()
        if not name:
            return

        relay_addr = input(f"{Style.WHITE('Relay address:')} ").strip()
        relay_pass = input(f"{Style.WHITE('Relay password:')} ").strip()
        peer_id = input(f"{Style.WHITE('Your peer ID:')} ").strip() or name
        listen = input(f"{Style.WHITE('Listen address (default: 127.0.0.1:8000):')} ").strip() or "127.0.0.1:8000"
        forward = input(f"{Style.WHITE('Forward to (local service, e.g., 127.0.0.1:3000):')} ").strip()

        if not all([relay_addr, relay_pass, forward]):
            print(f"{Style.RED2('❌ Missing required fields')}")
            input(f"\n{Style.GREY('Press Enter to continue...')}")
            return

        # Get executable path
        executable = self._get_executable_path()
        if not executable:
            print(f"{Style.RED2('❌ Executable not found')}")
            input(f"\n{Style.GREY('Press Enter to continue...')}")
            return

        # Create instance
        instance = EnhancedInstanceManager(name, self.app)
        config = {
            'relay_address': relay_addr,
            'relay_password': relay_pass,
            'peer_id': peer_id,
            'listen_address': listen,
            'forward_to_address': forward
        }

        success = instance.start(executable, 'peer', config)

        if success:
            self.instances[name] = instance

        input(f"\n{Style.GREY('Press Enter to continue...')}")

    def _stop_instance(self):
        """Stop running instance."""
        if not self.instances:
            print(f"\n{Style.YELLOW('No running instances')}")
            input(f"\n{Style.GREY('Press Enter to continue...')}")
            return

        print(f"\n{Style.Bold(Style.CYAN('Stop Instance'))}")
        print(Style.GREY('─' * 70))

        print(f"\n{Style.WHITE('Running instances:')}")
        running = {name: inst for name, inst in self.instances.items() if inst.is_running()}

        if not running:
            print(Style.YELLOW("  No running instances"))
            input(f"\n{Style.GREY('Press Enter to continue...')}")
            return

        for i, (name, inst) in enumerate(running.items(), 1):
            state = inst.read_state()
            mode = state.get('mode', 'Unknown')
            pid = state.get('pid', 'N/A')
            print(f"  {i}. {Style.YELLOW(name)} ({mode}, PID: {pid})")

        name = input(f"\n{Style.WHITE('Instance name to stop:')} ").strip()

        if name in running:
            running[name].stop()
        else:
            print(f"{Style.RED2('❌ Instance not found')}")

        input(f"\n{Style.GREY('Press Enter to continue...')}")

    def _get_executable_path(self) -> Optional[Path]:
        """Get executable path."""
        search_paths = [
            tb_root_dir / "bin" / EXECUTABLE_NAME,
            tb_root_dir / "tcm" / "target" / "release" / EXECUTABLE_NAME,
        ]

        for path in search_paths:
            if path.is_file():
                return path.resolve()

        return None

    def status_menu(self, do_clear=True):
        """Status and monitoring menu."""
        self.clear_screen() if do_clear else None
        self.print_header() if do_clear else None

        print(f"\n{Style.Bold(Style.CYAN('📊 System Status'))}")
        print(Style.GREY('═' * 90))

        # P2P Instances
        print(f"\n{Style.Bold(Style.WHITE('P2P Instances:'))}")
        if not self.instances:
            print(Style.YELLOW("  No instances configured"))
        else:
            print(
                f"\n{Style.Underline('NAME'):<20} {Style.Underline('MODE'):<12} {Style.Underline('STATUS'):<12} {Style.Underline('PID'):<10} {Style.Underline('CHAT ROOM')}")
            print(Style.GREY('─' * 90))

            for name, inst in self.instances.items():
                state = inst.read_state()
                mode = state.get('mode', 'Unknown')
                pid = state.get('pid', 'N/A')
                chat_room = state.get('chat_room', '-')
                status = Style.GREEN('✅ Running') if inst.is_running() else Style.RED('❌ Stopped')

                print(
                    f"{Style.YELLOW(name):<20} {mode:<12} {status:<12} {str(pid):<10} {Style.CYAN(str(chat_room)[:20])}")

        # Chat Rooms
        print(f"\n{Style.Bold(Style.WHITE('Chat Rooms:'))}")
        result = self.chat_manager.list_rooms()

        if result.is_ok():
            rooms = result.get()
            if not rooms:
                print(Style.YELLOW("  No chat rooms"))
            else:
                print(
                    f"\n{Style.Underline('NAME'):<20} {Style.Underline('PARTICIPANTS'):<15} {Style.Underline('STATUS'):<15} {Style.Underline('FEATURES')}")
                print(Style.GREY('─' * 70))

                for room in rooms:
                    name = Style.YELLOW(room['name'][:18])
                    participants = f"{room['participants_count']}/{room['max_participants']}"

                    status_parts = []
                    if room['is_locked']:
                        status_parts.append('🔒')
                    if room['is_private']:
                        status_parts.append('🔐')
                    if room['is_member']:
                        status_parts.append('✓')
                    status = ' '.join(status_parts) if status_parts else '🔓'

                    features = []
                    if room['voice_enabled']:
                        features.append('🎤 Voice')
                    if room['file_transfer_enabled']:
                        features.append('📁 Files')
                    features_str = ', '.join(features)

                    print(f"{name:<20} {participants:<15} {status:<15} {features_str}")

        input(f"\n{Style.GREY('Press Enter to continue...')}") if do_clear else None

    def settings_menu(self):
        """Settings menu."""
        while True:
            self.clear_screen()
            self.print_header()

            print(f"""
{Style.Bold(Style.WHITE('┌─ ⚙️  SETTINGS ───────────────────────────────────────────────────────┐'))}
{Style.WHITE('│')}                                                                      {Style.WHITE('│')}
{Style.WHITE('│')}  {Style.CYAN('1.')} {Style.WHITE('Change Username')}    - Set display name for chat                   {Style.WHITE('│')}
{Style.WHITE('│')}  {Style.CYAN('2.')} {Style.WHITE('Build P2P Binary')}   - Compile Rust P2P application                {Style.WHITE('│')}
{Style.WHITE('│')}  {Style.CYAN('3.')} {Style.WHITE('Clean Up')}           - Remove old instances and data               {Style.WHITE('│')}
{Style.WHITE('│')}  {Style.CYAN('0.')} {Style.WHITE('Back')}               - Return to main menu                         {Style.WHITE('│')}
{Style.WHITE('│')}                                                                      {Style.WHITE('│')}
{Style.Bold(Style.WHITE('└──────────────────────────────────────────────────────────────────────┘'))}
""")

            choice = input(f"\n{Style.CYAN('❯')} {Style.WHITE('Select option:')} ").strip()

            if choice == '0':
                break
            elif choice == '1':
                self._change_username()
            elif choice == '2':
                self._build_binary()
            elif choice == '3':
                self._cleanup()
            else:
                print(f"{Style.RED('Invalid option')}")
                time.sleep(1)

    def _change_username(self):
        """Change username."""
        print(f"\n{Style.Bold(Style.CYAN('Change Username'))}")
        print(Style.GREY('─' * 70))
        print(f"{Style.WHITE('Current:')} {Style.YELLOW(self.chat_manager.username)}")

        new_name = input(f"\n{Style.WHITE('New username:')} ").strip()
        if new_name:
            self.chat_manager.username = new_name
            print(f"\n{Style.GREEN2('✅ Username changed to:')} {Style.YELLOW(new_name)}")

        input(f"\n{Style.GREY('Press Enter to continue...')}")

    def _build_binary(self):
        """Build P2P binary."""
        print(f"\n{Style.Bold(Style.CYAN('Building P2P Binary'))}")
        print(Style.GREY('─' * 70))

        tcm_dir = tb_root_dir / "tcm"
        if not tcm_dir.exists():
            print(f"{Style.RED2('❌ TCM directory not found at:')} {tcm_dir}")
            input(f"\n{Style.GREY('Press Enter to continue...')}")
            return

        print(f"\n{Style.CYAN('⚙ Building with Cargo...')}")

        try:
            with Spinner("Compiling Rust project", symbols="t", time_in_s=120):
                process = subprocess.run(
                    ["cargo", "build", "--release"],
                    cwd=str(tcm_dir),
                    capture_output=True,
                    text=True
                )

            if process.returncode == 0:
                print(f"\n{Style.GREEN2('✅ Build successful!')}")

                # Copy to bin directory
                source = tcm_dir / "target" / "release" / EXECUTABLE_NAME
                dest_dir = tb_root_dir / "bin"
                dest_dir.mkdir(exist_ok=True)
                dest = dest_dir / EXECUTABLE_NAME

                if source.exists():
                    import shutil
                    shutil.copy2(source, dest)
                    print(f"{Style.GREEN('📦 Copied to:')} {dest}")
            else:
                print(f"\n{Style.RED2('❌ Build failed:')}")
                print(Style.GREY(process.stderr))

        except FileNotFoundError:
            print(f"\n{Style.RED2('❌ Cargo not found. Is Rust installed?')}")
        except Exception as e:
            print(f"\n{Style.RED2('❌ Build error:')} {e}")

        input(f"\n{Style.GREY('Press Enter to continue...')}")

    def _cleanup(self):
        """Cleanup old data."""
        print(f"\n{Style.Bold(Style.YELLOW('⚠ Cleanup'))}")
        print(Style.GREY('─' * 70))
        print(f"{Style.RED('This will:')}")
        print(f"  • Stop all running instances")
        print(f"  • Delete instance configurations")
        print(f"  • Keep chat rooms and messages")

        confirm = input(f"\n{Style.WHITE('Continue? (y/N):')} ").strip().lower()
        if confirm != 'y':
            return

        # Stop all instances
        for inst in self.instances.values():
            if inst.is_running():
                inst.stop()

        # Remove instance directory
        if INSTANCES_ROOT_DIR.exists():
            import shutil
            shutil.rmtree(INSTANCES_ROOT_DIR)

        self.instances = {}

        print(f"\n{Style.GREEN2('✅ Cleanup complete')}")
        input(f"\n{Style.GREY('Press Enter to continue...')}")

    def run(self):
        """Main application loop."""
        while self.running:
            self.clear_screen()
            self.print_header()
            self.print_menu()

            choice = input(f"\n{Style.CYAN('❯')} {Style.WHITE('Select option:')} ").strip()

            if choice == '0':
                print(f"\n{Style.YELLOW('👋 Goodbye!')}")
                self.running = False
            elif choice == '1':
                self.chat_menu()
            elif choice == '2':
                self.p2p_menu()
            elif choice == '3':
                self.status_menu()
            elif choice == '4':
                self.settings_menu()
            else:
                print(f"{Style.RED('Invalid option')}")
                time.sleep(1)


# =================== CLI Entry Point ===================

def cli_tcm_runner():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description=f"🚀 {Style.Bold('ToolBox P2P Manager')} - Advanced P2P with E2E Chat",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Start interactive mode (default)')
    parser.add_argument("status", nargs='?', const=True,
                        help='Check status of all instances')
    args = parser.parse_args()

    # Always start in interactive mode
    cli = InteractiveP2PCLI()

    if args.status:
        cli.status_menu(do_clear=False)
        return

    try:
        cli.run()
    except KeyboardInterrupt:
        print(f"\n\n{Style.YELLOW('👋 Interrupted by user. Goodbye!')}")
    except Exception as e:
        print(f"\n{Style.RED2('❌ Fatal error:')} {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print(f"\n{Style.GREY('Cleaning up...')}")


if __name__ == "__main__":
    cli_tcm_runner()
