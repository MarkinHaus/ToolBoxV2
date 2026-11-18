"""Sprache fuer icli2: Aufnahme -> Text (F4), Sprachausgabe pro Anfrage (#audio), Geraete, Omni.
Es gibt bewusst keinen Dauer-Schalter fuers Vorlesen: gesprochen wird nur, wenn die Nachricht
auf "#audio" endet (wie in der alten icli) - oder in einer Omni-Session.
Alle Audio-/toolboxv2-Importe sind lazy: ohne Sprachnutzung wird kein Audio-Code geladen.
Wiederverwendet: audio_recorder.LocalMicRecorder, Stt.transcribe, audioIo.AudioStreamPlayer/LocalPlayer,
omni.OmniSession (wie flows/isaa/icli.py)."""
from __future__ import annotations

import asyncio

_MIN_LEN = 30                       # Mindestlaenge eines Satzes (wie icli _drain_agent_stream)
_ENDS = (".", "!", "?", ":", "\n\n")


def split_speakable(buffer: str, chunk: str, in_code: bool) -> tuple[list[str], str, bool]:
    """Satz-Pufferung: (sprechbare Saetze, Rest-Puffer, neuer in_code-Zustand). Rein, ohne Audio."""
    out: list[str] = []
    if "```" in chunk:                                   # Codeblock-Grenze: Chunk verwerfen
        in_code = not in_code
        if in_code and buffer.strip():
            out.append(buffer)
            buffer = ""
        return out, buffer, in_code
    if in_code:
        return out, buffer, in_code
    buffer += chunk
    if buffer.rstrip().endswith(_ENDS) and len(buffer.strip()) >= _MIN_LEN:
        out.append(buffer)
        buffer = ""
    return out, buffer, in_code


class VoiceMixin:
    def _voice_init(self) -> None:
        """Zustand anlegen (idempotent)."""
        if getattr(self, "_v_ready", False):
            return
        self._v_ready = True
        self._v_recording = False
        self._v_buffer = ""
        self._v_in_code = False
        self._v_player = None
        self._v_recorder = None
        self._v_omni = None
        self._v_frames: list[bytes] = []
        self._v_rec_task = None
        self._v_dev_in = None
        self._v_dev_out = None
        try:                                          # gespeicherte Geraetewahl
            saved_in, saved_out = self.get_voice_devices()
            if saved_in is not None:
                self._v_dev_in = int(saved_in)
            if saved_out is not None:
                self._v_dev_out = int(saved_out)
        except Exception:
            pass                                      # Standalone/ohne Backend-Settings
        self._v_tasks: set = set()

    # ---------- Aufnahme (F4) ----------
    async def _do_voice_toggle(self) -> dict:
        self._voice_init()
        if not self._v_recording:
            if self._v_omni:                             # beide wuerden das Mikrofon belegen
                raise RuntimeError("Omni-Session läuft – erst über /voice beenden")
            from toolboxv2.mods.isaa.base.audio_io.audio_recorder import LocalMicRecorder
            rec = LocalMicRecorder(device=self._v_dev_in)
            await rec.start()
            self._v_recorder, self._v_frames, self._v_recording = rec, [], True

            async def _collect():
                async for f in rec.frames():
                    self._v_frames.append(f)
            self._v_rec_task = asyncio.create_task(_collect())
            return {"text": "● Aufnahme läuft – F4 beendet"}
        self._v_recording = False
        rec, task = self._v_recorder, self._v_rec_task
        self._v_recorder = self._v_rec_task = None
        try:
            await rec.stop()
        finally:
            if task:
                task.cancel()
        audio = b"".join(self._v_frames)
        self._v_frames = []
        if not audio:
            return {"text": "Nichts verstanden", "level": "warn"}
        from toolboxv2.mods.isaa.base.audio_io.Stt import STTConfig, transcribe
        result = await asyncio.to_thread(transcribe, audio, STTConfig(language="de"))
        text = (getattr(result, "text", "") or "").strip()
        return {"draft": text} if text else {"text": "Nichts verstanden", "level": "warn"}

    # ---------- Einstellungen / Geraete ----------
    async def _devices(self) -> list[tuple[int, dict]]:
        import sounddevice as sd
        devs = await asyncio.to_thread(sd.query_devices)
        return [(i, d) for i, d in enumerate(devs)]

    async def _rows_voice(self) -> tuple[str, list[dict]]:
        self._voice_init()
        rows = [
            {"value": "omni", "label": "Omni-Session", "hint": "läuft" if self._v_omni else "aus"},
        ]
        try:
            for i, d in await self._devices():
                if d["max_input_channels"] > 0:
                    rows.append({"value": f"in:{i}", "label": f"Eingabe: {d['name']}",
                                 "hint": "aktiv" if self._v_dev_in == i else ""})
                if d["max_output_channels"] > 0:
                    rows.append({"value": f"out:{i}", "label": f"Ausgabe: {d['name']}",
                                 "hint": "aktiv" if self._v_dev_out == i else ""})
        except Exception as e:
            rows.append({"value": "", "label": "Geräte nicht lesbar", "hint": str(e)[:60]})
        return "Sprache", rows

    async def _ensure_player(self):
        if self._v_player is None:
            from toolboxv2.mods.isaa.base.audio_io.Tts import TTSBackend, TTSConfig
            from toolboxv2.mods.isaa.base.audio_io.audioIo import AudioStreamPlayer, LocalPlayer
            tts_provider, tts_voice = self.get_tts_config()
            cfg = TTSConfig(backend=TTSBackend(tts_provider), voice=tts_voice, language="de")
            self._v_player = AudioStreamPlayer(
                player_backend=LocalPlayer(device=self._v_dev_out or 0), tts_config=cfg,
                session_id=getattr(self, "session", "default"))
        await self._v_player.start()

    async def _reset_player(self) -> None:
        if self._v_player is not None:
            try:
                await self._v_player.stop()
            except Exception as e:
                self.app.logger.warning(f"voice player stop: {e}")
            self._v_player = None                        # wird bei der naechsten #audio-Anfrage neu gebaut

    async def _do_voice(self, op: str) -> dict:
        self._voice_init()
        if op == "omni":
            return await self._omni_toggle()
        kind, _, num = op.partition(":")
        if kind not in ("in", "out") or not num.lstrip("-").isdigit():
            raise ValueError(f"Unbekannte Sprach-Option: {op}")
        idx = int(num)
        if kind == "in":
            self._v_dev_in = idx
            await asyncio.to_thread(self.save_setting, "voice_input_device", idx)
        else:
            self._v_dev_out = idx
            await asyncio.to_thread(self.save_setting, "voice_output_device", idx)
            await self._reset_player()
        return {"text": f"{'Eingabe' if kind == 'in' else 'Ausgabe'}-Gerät: {idx}"}

    async def _omni_toggle(self) -> dict:
        """Minimal-Omni: Backend + Mikro + Player + Agent-Tools. Ohne delegate/VFS-Tools, VAD, State-Store."""
        if self._v_recording:
            raise RuntimeError("Aufnahme läuft – erst mit F4 beenden")
        if self._v_omni:
            sess, self._v_omni = self._v_omni, None
            sess.is_active = False
            await sess.stop()
            return {"text": "Omni-Session beendet"}
        from toolboxv2.mods.isaa.base.audio_io.omni import (
            OMNI_SYSTEM_INSTRUCTION, OmniSession, VoiceModeConfig)
        from toolboxv2.mods.isaa.base.audio_io.audio_recorder import LocalMicRecorder
        from toolboxv2.mods.isaa.base.audio_io.audioIo import LocalPlayer, StreamingLocalPlayer
        agent = await self._agent()
        cfg = VoiceModeConfig(mode="omni_cloud", device="cuda", agent=agent,
                              kwargs={"voice": "Algenib", "input_transcription": True,
                                      "output_transcription": True, "thinking_level": None})

        def _backend():
            b = cfg.build_backend()
            if b is not None and hasattr(b, "system_instruction"):
                b.system_instruction = OMNI_SYSTEM_INSTRUCTION
            return b
        backend = _backend()
        if backend is None:
            raise RuntimeError("Kein Omni-Backend verfügbar")
        stream = backend.backend_name == "GeminiLiveBackend"
        player = (StreamingLocalPlayer(device=self._v_dev_out, sample_rate=24000) if stream
                  else LocalPlayer(device=self._v_dev_out or 0))
        sess = OmniSession(backend, backend_factory=_backend, recorder=LocalMicRecorder(device=self._v_dev_in),
                           player=player, tools=getattr(agent, "tool_manager", None),
                           output_sample_rate=24000 if stream else None, buffer_audio=not stream,
                           summarizer_agent=agent)
        sess.is_active = True
        tm, specs = getattr(agent, "tool_manager", None), []
        try:
            specs = tm.get_all_litellm() if tm else []
        except Exception:
            specs = []
        try:
            await sess.start(tool_specs=specs)
        except Exception:
            sess.is_active = False
            raise
        self._v_omni = sess
        return {"text": "Omni-Session aktiv – einfach sprechen"}

    # ---------- Vorlesen ----------
    def _speak(self, text: str) -> None:
        clean = text.strip()
        try:
            from toolboxv2 import remove_styles
            clean = remove_styles(clean)
        except Exception:
            pass
        if not clean or self._v_player is None:
            return
        low = clean.lower()
        from toolboxv2.mods.isaa.base.audio_io.Tts import TTSEmotion
        emo = (TTSEmotion.SERIOUS if any(w in low for w in ("error", "fehler", "achtung", "warning", "kritisch"))
               else TTSEmotion.EXCITED if any(w in low for w in ("!", "super", "perfekt", "great"))
               else TTSEmotion.NEUTRAL)
        t = asyncio.get_running_loop().create_task(self._v_player.queue_text(clean, emo))
        self._v_tasks.add(t)
        t.add_done_callback(self._v_tasks.discard)

    async def _voice_begin(self) -> bool:
        """Vor einer #audio-Antwort: Player bereitstellen. False (statt Fehler), wenn kein Audio geht."""
        try:
            self._voice_init()
            self._v_buffer, self._v_in_code = "", False
            await self._ensure_player()
            return True
        except Exception as e:
            self.app.logger.warning(f"voice begin: {e}")
            return False

    def _voice_feed(self, text: str) -> None:
        """Content-Chunk einer #audio-Antwort. Der Aufrufer entscheidet, ob gesprochen wird."""
        try:
            self._voice_init()
            out, self._v_buffer, self._v_in_code = split_speakable(self._v_buffer, text, self._v_in_code)
            for s in out:
                self._speak(s)
        except Exception as e:
            self.app.logger.warning(f"voice feed: {e}")

    def _voice_flush(self) -> None:
        try:
            self._voice_init()
            rest, self._v_buffer, self._v_in_code = self._v_buffer, "", False
            if rest.strip():
                self._speak(rest)
        except Exception as e:
            self.app.logger.warning(f"voice flush: {e}")

    async def _voice_close(self) -> None:
        self._voice_init()
        self._v_recording = False
        steps = [(self._v_recorder, "stop"), (self._v_omni, "stop"), (self._v_player, "stop")]
        if self._v_rec_task:
            self._v_rec_task.cancel()
        if self._v_omni:
            self._v_omni.is_active = False
        for obj, meth in steps:
            if obj is None:
                continue
            try:
                await getattr(obj, meth)()
            except Exception as e:
                try:
                    self.app.logger.warning(f"voice close: {e}")
                except Exception:
                    pass
        self._v_recorder = self._v_omni = self._v_player = self._v_rec_task = None
