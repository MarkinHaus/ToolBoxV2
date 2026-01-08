# Augment Code Agent Task: Discord Kernel Unit & E2E Tests

## Aufgabe
Erstelle umfassende Unit-Tests und E2E-Tests für den Discord Kernel und die Discord Tools des ToolBoxV2 ISAA-Moduls. Die Tests sollen **ohne echte LLM-Aufrufe** funktionieren (Mocking).

## Dateipfade

### Zu testende Dateien:
- `C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\mods\isaa\kernel\kernelin\tools\discord_tools.py`
- `C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\mods\isaa\kernel\kernelin\kernelin_discord.py`

### Test-Ausgabe:
- `C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tests\test_mods\test_discord_kernel.py`

### Referenz für Test-Struktur:
- `C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tests\test_mods\test_password_manager.py`

---

## Kontext: Discord Kernel Architektur

### discord_tools.py (~4000 Zeilen)
Die `DiscordKernelTools` Klasse bietet 59+ Discord-spezifische Tools:

**Kategorien:**
1. **Server Management**: `get_server_info()`, `get_channel_info()`, `list_channels()`
2. **User Management**: `get_user_info()`
3. **Messages**: `send_message()`, `edit_message()`, `delete_message()`
4. **Embeds**: `send_embed()`, `create_embed_template()`
5. **Templates**: `create_message_template()`, `send_template_message()`
6. **Moderation**: `kick_user()`, `ban_user()`, `timeout_user()`
7. **Roles**: `add_role()`, `remove_role()`, `create_role()`
8. **Voice**: Voice Channel Management
9. **Threads**: Thread erstellen/verwalten
10. **Invites**: `create_invite()`, `get_invites()`
11. **Reactions**: `add_reaction()`, `remove_reaction()`

**Konstruktor:**
```python
class DiscordKernelTools:
    def __init__(self, bot: 'discord.ext.commands.Bot', kernel, output_router):
        self.bot = bot
        self.kernel = kernel
        self.output_router = output_router
```

### kernelin_discord.py (~4900 Zeilen)
Der `DiscordKernel` als Interface zwischen Discord und dem ProA Kernel:

**Hauptkomponenten:**
- `WhisperAudioSink`: Voice-Input mit Groq Whisper Transkription
- `DiscordOutputRouter`: IOutputRouter Implementation für Discord
- `DiscordKernel`: Hauptklasse mit Bot-Events und Message-Handling

**Wichtige Methoden:**
```python
class DiscordKernel:
    def __init__(self, agent, app, bot_token: str, ...)
    async def start(self)
    async def stop(self)
    async def handle_message(self, message)
    async def list_all_users_with_nicknames(self)
    async def list_all_known_sessions(self)
```

---

## Test-Anforderungen

### 1. Unit Tests (ohne Discord/LLM)

#### A. DiscordKernelTools Tests
```python
class TestDiscordKernelTools(unittest.TestCase):
    """Unit tests für DiscordKernelTools - alle Discord-Calls gemockt"""

    def setUp(self):
        # Mock Bot, Kernel, OutputRouter
        self.mock_bot = Mock(spec=discord.ext.commands.Bot)
        self.mock_kernel = Mock()
        self.mock_output_router = Mock()
        self.tools = DiscordKernelTools(
            self.mock_bot,
            self.mock_kernel,
            self.mock_output_router
        )
```

**Teste folgende Methoden:**
- `get_server_info()` - mit/ohne guild_id
- `get_channel_info()` - verschiedene Channel-Typen (Text, Voice, DM)
- `list_channels()` - mit channel_type Filter
- `get_user_info()` - mit/ohne guild_id für Member-spezifische Info
- Template-System: create, get, list, delete, send templates
- Error-Handling: nicht gefundene Guilds, Channels, User

#### B. DiscordOutputRouter Tests
```python
class TestDiscordOutputRouter(unittest.TestCase):
    """Tests für den Output Router"""

    def test_route_text_output(self):
        """Test Text-Output an Discord Channel"""
        pass

    def test_route_embed_output(self):
        """Test Embed-Output"""
        pass

    def test_user_channel_mapping(self):
        """Test User -> Channel Zuordnung"""
        pass
```

#### C. WhisperAudioSink Tests (Voice/Transkription)
```python
class TestWhisperAudioSink(unittest.TestCase):
    """Tests für Audio-Sink ohne echte Groq-Calls"""

    @patch('toolboxv2.mods.isaa.kernel.kernelin.kernelin_discord.Groq')
    def test_audio_buffer_management(self, mock_groq):
        """Test Audio-Buffer pro User"""
        pass

    def test_transcription_interval(self):
        """Test dass Transkription alle 3 Sekunden triggert"""
        pass
```

### 2. Integration Tests (mit gemocktem Discord)

```python
class TestDiscordKernelIntegration(unittest.IsolatedAsyncioTestCase):
    """Async Integration Tests"""

    async def asyncSetUp(self):
        # Mock Agent, App, Bot
        self.mock_agent = AsyncMock()
        self.mock_app = Mock(spec=App)

    async def test_message_handling(self):
        """Test kompletter Message-Flow ohne LLM"""
        pass

    async def test_kernel_signal_processing(self):
        """Test Signal-Erstellung aus Discord Message"""
        pass
```

### 3. E2E Tests (Mock Discord Bot)

```python
class TestDiscordKernelE2E(unittest.IsolatedAsyncioTestCase):
    """End-to-End Tests mit gemocktem Discord Bot"""

    async def test_full_conversation_flow(self):
        """
        1. User sendet Message
        2. Kernel verarbeitet
        3. Agent wird aufgerufen (gemockt)
        4. Response wird an Channel gesendet
        """
        pass

    async def test_attachment_handling(self):
        """Test mit Bildern/Dateien"""
        pass

    async def test_voice_channel_join_leave(self):
        """Test Voice-Channel Befehle"""
        pass
```

---

## Mock-Strategien

### Discord.py Mocks

```python
import discord
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Mock Guild
def create_mock_guild(guild_id=123456789, name="Test Server"):
    guild = Mock(spec=discord.Guild)
    guild.id = guild_id
    guild.name = name
    guild.member_count = 100
    guild.owner_id = 111111111
    guild.created_at = datetime.now()
    guild.text_channels = [create_mock_text_channel()]
    guild.voice_channels = [create_mock_voice_channel()]
    guild.roles = [create_mock_role()]
    guild.emojis = []
    guild.premium_tier = 0
    guild.premium_subscription_count = 0
    return guild

# Mock TextChannel
def create_mock_text_channel(channel_id=987654321, name="general"):
    channel = Mock(spec=discord.TextChannel)
    channel.id = channel_id
    channel.name = name
    channel.type = discord.ChannelType.text
    channel.created_at = datetime.now()
    channel.topic = "Test topic"
    channel.slowmode_delay = 0
    channel.nsfw = False
    channel.send = AsyncMock(return_value=create_mock_message())
    channel.typing = MagicMock(return_value=AsyncContextManagerMock())
    return channel

# Mock Message
def create_mock_message(content="test message", author_id=222222222):
    message = Mock(spec=discord.Message)
    message.id = random.randint(100000, 999999)
    message.content = content
    message.author = create_mock_user(author_id)
    message.channel = create_mock_text_channel()
    message.guild = create_mock_guild()
    message.attachments = []
    message.mentions = []
    return message

# Mock User
def create_mock_user(user_id=222222222, name="TestUser"):
    user = Mock(spec=discord.User)
    user.id = user_id
    user.name = name
    user.display_name = name
    user.bot = False
    user.created_at = datetime.now()
    return user

# Async Context Manager Mock für channel.typing()
class AsyncContextManagerMock:
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass
```

### Kernel/Agent Mocks

```python
# Mock Kernel
def create_mock_kernel():
    kernel = Mock()
    kernel.process_signal = AsyncMock()
    kernel.memory_store = Mock()
    kernel.memory_store.user_memories = {}
    kernel.learning_engine = Mock()
    kernel.learning_engine.preferences = {}
    return kernel

# Mock Agent (ohne LLM-Calls)
def create_mock_agent():
    agent = AsyncMock()
    agent.add_tool = AsyncMock()
    agent.context_manager = Mock()
    agent.context_manager.session_managers = {}
    return agent
```

---

## Pytest Fixtures (Alternative zu unittest)

```python
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.fixture
def mock_bot():
    """Fixture für gemockten Discord Bot"""
    bot = Mock(spec=discord.ext.commands.Bot)
    bot.guilds = [create_mock_guild()]
    bot.get_guild = Mock(return_value=create_mock_guild())
    bot.get_channel = Mock(return_value=create_mock_text_channel())
    bot.get_user = Mock(return_value=create_mock_user())
    bot.user = create_mock_user(name="TestBot", bot=True)
    return bot

@pytest.fixture
def mock_kernel():
    """Fixture für gemockten Kernel"""
    return create_mock_kernel()

@pytest.fixture
def discord_tools(mock_bot, mock_kernel):
    """Fixture für DiscordKernelTools mit Mocks"""
    output_router = Mock()
    return DiscordKernelTools(mock_bot, mock_kernel, output_router)

@pytest.fixture
async def discord_kernel(mock_bot, mock_kernel):
    """Async Fixture für DiscordKernel"""
    # Patch Bot-Erstellung
    with patch('discord.ext.commands.Bot', return_value=mock_bot):
        kernel = DiscordKernel(
            agent=create_mock_agent(),
            app=Mock(spec=App),
            bot_token="fake_token"
        )
        yield kernel
```

---

## Konkrete Test-Cases

### Test 1: Server Info Abruf
```python
async def test_get_server_info_with_guild_id(discord_tools, mock_bot):
    """Test get_server_info mit spezifischer Guild ID"""
    guild_id = 123456789
    result = await discord_tools.get_server_info(guild_id)

    assert result["id"] == guild_id
    assert result["name"] == "Test Server"
    assert result["member_count"] == 100
    mock_bot.get_guild.assert_called_once_with(guild_id)

async def test_get_server_info_guild_not_found(discord_tools, mock_bot):
    """Test Error-Handling bei nicht gefundener Guild"""
    mock_bot.get_guild.return_value = None
    result = await discord_tools.get_server_info(999999)

    assert "error" in result
    assert "not found" in result["error"]
```

### Test 2: Channel Info mit verschiedenen Typen
```python
async def test_get_channel_info_text_channel(discord_tools, mock_bot):
    """Test get_channel_info für TextChannel"""
    channel = create_mock_text_channel()
    mock_bot.get_channel.return_value = channel

    result = await discord_tools.get_channel_info(channel.id)

    assert result["type"] == "text"
    assert "topic" in result
    assert "slowmode_delay" in result

async def test_get_channel_info_voice_channel(discord_tools, mock_bot):
    """Test get_channel_info für VoiceChannel"""
    channel = create_mock_voice_channel()
    mock_bot.get_channel.return_value = channel

    result = await discord_tools.get_channel_info(channel.id)

    assert result["type"] == "voice"
    assert "bitrate" in result
    assert "user_limit" in result
```

### Test 3: Message Template System
```python
async def test_create_and_use_template(discord_tools):
    """Test Template erstellen und verwenden"""
    # Template erstellen
    template_result = await discord_tools.create_message_template(
        template_name="welcome",
        content="Hello {username}!",
        embed={"title": "Welcome", "description": "Welcome to {server_name}"}
    )
    assert template_result.get("success", False) or "template_name" in template_result

    # Template abrufen
    get_result = await discord_tools.get_message_template("welcome")
    assert get_result is not None

    # Template senden
    send_result = await discord_tools.send_template_message(
        channel_id=987654321,
        template_name="welcome",
        variables={"username": "TestUser", "server_name": "Test Server"}
    )
    assert "error" not in send_result or send_result.get("success", False)
```

### Test 4: Kernel Signal Processing
```python
async def test_message_creates_correct_signal(discord_kernel, mock_bot):
    """Test dass Discord Message korrekten Kernel Signal erstellt"""
    message = create_mock_message(content="Hello Bot!")
    message.mentions = [mock_bot.user]  # Bot wurde erwähnt

    with patch.object(discord_kernel.kernel, 'process_signal') as mock_process:
        await discord_kernel.handle_message(message)

        mock_process.assert_called_once()
        signal = mock_process.call_args[0][0]

        assert signal.type == SignalType.USER_INPUT
        assert "Hello Bot!" in signal.content
        assert signal.metadata["interface"] == "discord"
```

### Test 5: Audio Sink Buffer Management
```python
def test_audio_buffer_per_user():
    """Test dass Audio-Buffer pro User getrennt sind"""
    sink = WhisperAudioSink(
        kernel=create_mock_kernel(),
        user_id="123",
        groq_client=None,
        output_router=Mock()
    )

    # Simuliere Audio-Daten von zwei Usern
    user1 = Mock(id=111, display_name="User1")
    user2 = Mock(id=222, display_name="User2")

    data1 = Mock(pcm=b"audio_data_1")
    data2 = Mock(pcm=b"audio_data_2")

    sink.write(user1, data1)
    sink.write(user2, data2)

    assert "111" in sink.audio_buffer
    assert "222" in sink.audio_buffer
    assert sink.audio_buffer["111"] != sink.audio_buffer["222"]
```

---

## Erwartete Test-Datei Struktur

```
test_discord_kernel.py
├── Imports & Fixtures
├── Mock Helper Functions
│   ├── create_mock_guild()
│   ├── create_mock_text_channel()
│   ├── create_mock_voice_channel()
│   ├── create_mock_message()
│   ├── create_mock_user()
│   └── create_mock_kernel()
├── TestDiscordKernelTools (unittest.TestCase)
│   ├── test_get_server_info_*
│   ├── test_get_channel_info_*
│   ├── test_list_channels_*
│   ├── test_get_user_info_*
│   ├── test_template_*
│   └── test_error_handling_*
├── TestDiscordOutputRouter (unittest.TestCase)
│   ├── test_route_text_output
│   ├── test_route_embed_output
│   └── test_user_channel_mapping
├── TestWhisperAudioSink (unittest.TestCase)
│   ├── test_audio_buffer_management
│   ├── test_transcription_interval
│   └── test_voice_channel_history
├── TestDiscordKernelIntegration (unittest.IsolatedAsyncioTestCase)
│   ├── test_message_handling
│   ├── test_kernel_signal_processing
│   └── test_attachment_handling
└── TestDiscordKernelE2E (unittest.IsolatedAsyncioTestCase)
    ├── test_full_conversation_flow
    ├── test_voice_channel_join_leave
    └── test_bot_mention_handling
```

---

## Ausführen der Tests

```bash
# Alle Discord Kernel Tests
pytest toolboxv2/tests/test_mods/test_discord_kernel.py -v

# Nur Unit Tests
pytest toolboxv2/tests/test_mods/test_discord_kernel.py -v -k "not Integration and not E2E"

# Nur Async Tests
pytest toolboxv2/tests/test_mods/test_discord_kernel.py -v -k "async"

# Mit Coverage
pytest toolboxv2/tests/test_mods/test_discord_kernel.py -v --cov=toolboxv2.mods.isaa.kernel.kernelin
```

---

## Wichtige Hinweise

1. **Keine echten API-Calls**: Alle Discord-API und LLM-Calls müssen gemockt sein
2. **Async-Tests**: Nutze `unittest.IsolatedAsyncioTestCase` oder `pytest-asyncio`
3. **Import-Handling**: Discord.py Import kann fehlschlagen - entsprechend behandeln
4. **Voice Support**: VOICE_SUPPORT und VOICE_RECEIVE_SUPPORT Flags beachten
5. **Error Cases**: Teste auch Fehler-Szenarien (not found, permission denied, etc.)

---

## Abhängigkeiten für Tests

```
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
```

Optional (falls discord.py installiert):
```
discord.py>=2.0.0
```

new includ feetures to support chat voce memo as input transcribe and pass to kernel.
