import asyncio, sys, types, logging
import stub
AUD = "toolboxv2.mods.isaa.base.audio_io"
NAMES = [AUD, AUD + ".audio_recorder", AUD + ".Stt", AUD + ".Tts", AUD + ".audioIo", AUD + ".omni", "sounddevice"]
from toolboxv2.mods.isaa.extras.icli2.voice import VoiceMixin, split_speakable
assert not any(n in sys.modules for n in NAMES), "Audio-Module beim Import geladen"

calls = []
def fake(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m
class Rec:
    fail = False
    def __init__(s, device=None): s.device = device
    async def start(s): calls.append("rec.start")
    async def stop(s):
        calls.append("rec.stop")
        if Rec.fail: raise RuntimeError("boom")
    async def frames(s):
        yield b"\x01\x00"; await asyncio.sleep(0.05)
        while True: await asyncio.sleep(0.05)
class Res:  text = "hallo welt"
def transcribe(audio, config=None): calls.append(("stt", audio)); return Res
class Emo:  SERIOUS = "s"; EXCITED = "e"; NEUTRAL = "n"
class Player:
    def __init__(s, player_backend=None, tts_config=None, session_id=""): s.said = []; s.dev = player_backend.device
    async def start(s): calls.append("pl.start")
    async def stop(s): raise RuntimeError("boom")
    async def queue_text(s, t, e=None): s.said.append(t)
class LP:
    def __init__(s, device=None): s.device = device
fake(AUD); fake(AUD + ".audio_recorder", LocalMicRecorder=Rec)
fake(AUD + ".Stt", STTConfig=lambda language=None: language, transcribe=transcribe)
fake(AUD + ".Tts", TTSBackend=lambda v: v, TTSConfig=lambda **k: k, TTSEmotion=Emo)
fake(AUD + ".audioIo", AudioStreamPlayer=Player, LocalPlayer=LP)
fake("sounddevice", query_devices=lambda: [
    {"name": "mic", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "spk", "max_input_channels": 0, "max_output_channels": 2}])

# split_speakable
assert split_speakable("", "Hallo.", False) == ([], "Hallo.", False)
s = "Das ist ein ausreichend langer Satz."
assert split_speakable("", s, False) == ([s], "", False)
assert split_speakable("Kurz", "x", False) == ([], "Kurzx", False)
assert split_speakable("", "```py", False) == ([], "", True)
assert split_speakable("Rest", "```", False) == (["Rest"], "", True)
assert split_speakable("", s, True) == ([], "", True)
assert split_speakable("", "```", True) == ([], "", False)
b, c = "", False
spoken = []
for ch in ["Vorher ein langer Satz, der endet.", "```", "print('x'). genug lang genug lang genug.", "```", "Nachher kurz"]:
    o, b, c = split_speakable(b, ch, c); spoken += o
assert spoken == ["Vorher ein langer Satz, der endet."] and b == "Nachher kurz" and not c, (spoken, b, c)

class V(VoiceMixin):
    def __init__(s): s.app = types.SimpleNamespace(logger=logging.getLogger("v")); s.session = "default"

async def main():
    v = V(); v._voice_init(); v._voice_init()
    assert v._v_player is None                                # ohne #audio: kein Player
    try: await v._do_voice("speak"); assert 0                 # Dauer-Schalter gibt es nicht mehr
    except ValueError: pass
    assert await v._voice_begin() is True and v._v_player is not None
    v._voice_feed(s); v._voice_feed("Rest ohne Punkt"); await asyncio.sleep(0)
    assert v._v_player.said == [s], v._v_player.said
    v._voice_flush(); await asyncio.sleep(0)
    assert v._v_player.said == [s, "Rest ohne Punkt"] and v._v_buffer == ""
    t, rows = await v._rows_voice()
    assert t == "Sprache" and all(set(x) == {"value", "label", "hint"} for x in rows)
    assert [x["value"] for x in rows] == ["omni", "in:0", "out:1"] and rows[0]["hint"] == "aus"
    await v._do_voice("in:0"); assert v._v_dev_in == 0
    await v._do_voice("out:1"); assert v._v_dev_out == 1
    # Toggle
    r = await v._do_voice_toggle(); assert r == {"text": "● Aufnahme läuft – F4 beendet"}
    await asyncio.sleep(0.1)
    r = await v._do_voice_toggle(); assert r == {"draft": "hallo welt"}, r
    assert ("stt", b"\x01\x00") in calls
    Res.text = ""
    await v._do_voice_toggle(); await asyncio.sleep(0.1)
    r = await v._do_voice_toggle(); assert r == {"text": "Nichts verstanden", "level": "warn"}, r
    # close mit werfenden Fakes
    await v._voice_begin(); v._v_recorder = Rec(); Rec.fail = True
    await v._voice_close(); assert v._v_player is None and v._v_recorder is None
    await v._voice_close()
asyncio.run(main())
print("voice ok")
