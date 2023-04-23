import hashlib
import json
import logging
import os
from io import BytesIO

import whisper
from playsound import playsound
import requests
import speech_recognition as sr
import pickle

from toolboxv2 import MainTool, FileHandler, Style, App

from pydub import AudioSegment
from pydub.playback import play

import pyaudio
import wave

from pyannote.audio import Pipeline

import logging
import traceback
import diart.operators as dops
import rich
import rx.operators as ops
from diart import OnlineSpeakerDiarization, PipelineConfig
from diart.sources import MicrophoneAudioSource
import os
import sys
from pyannote.core import Segment
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    # Auxiliary function to suppress Whisper logs (it is quite verbose)
    # All credit goes to: https://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

import numpy as np
from pyannote.core import Annotation, SlidingWindowFeature, SlidingWindow

def concat(chunks, collar=0.05):
    """
    Concatenate predictions and audio
    given a list of `(diarization, waveform)` pairs
    and merge contiguous single-speaker regions
    with pauses shorter than `collar` seconds.
    """
    first_annotation = chunks[0][0]
    first_waveform = chunks[0][1]
    annotation = Annotation(uri=first_annotation.uri)
    data = []
    for ann, wav in chunks:
        annotation.update(ann)
        data.append(wav.data)
    annotation = annotation.support(collar)
    window = SlidingWindow(
        first_waveform.sliding_window.duration,
        first_waveform.sliding_window.step,
        first_waveform.sliding_window.start,
    )
    data = np.concatenate(data, axis=0)
    return annotation, SlidingWindowFeature(data, window)

def colorize_transcription(transcription):
    """
    Unify a speaker-aware transcription represented as
    a list of `(speaker: int, text: str)` pairs
    into a single text colored by speakers.
    """
    colors = 2 * [
        "bright_red", "bright_blue", "bright_green", "orange3", "deep_pink1",
        "yellow2", "magenta", "cyan", "bright_magenta", "dodger_blue2"
    ]
    result = []
    for speaker, text in transcription:
        if speaker == -1:
            # No speakerfound for this text, use default terminal color
            result.append(text)
        else:
            result.append(f"[{colors[speaker]}]{text}")
    return "\n".join(result)

class WhisperTranscriber:
    def __init__(self, model="small", device=None):
        self.model = whisper.load_model(model, device=device)
        self._buffer = ""

    def transcribe(self, waveform):
        """Transcribe audio using Whisper"""
        # Pad/trim audio to fit 30 seconds as required by Whisper
        audio = waveform.data.astype("float32").reshape(-1)
        audio = whisper.pad_or_trim(audio)

        # Transcribe the given audio while suppressing logs
        with suppress_stdout():
            transcription = whisper.transcribe(
                self.model,
                audio,
                # We use past transcriptions to condition the model
                initial_prompt=self._buffer,
                verbose=True  # to avoid progress bar
            )

        return transcription

    def identify_speakers(self, transcription, diarization, time_shift):
        """Iterate over transcription segments to assign speakers"""
        speaker_captions = []
        for segment in transcription["segments"]:

            # Crop diarization to the segment timestamps
            start = time_shift + segment["words"][0]["start"]
            end = time_shift + segment["words"][-1]["end"]
            dia = diarization.crop(Segment(start, end))

            # Assign a speaker to the segment based on diarization
            speakers = dia.labels()
            num_speakers = len(speakers)
            if num_speakers == 0:
                # No speakers were detected
                caption = (-1, segment["text"])
            elif num_speakers == 1:
                # Only one speaker is active in this segment
                spk_id = int(speakers[0].split("speaker")[1])
                caption = (spk_id, segment["text"])
            else:
                # Multiple speakers, select the one that speaks the most
                max_speaker = int(np.argmax([
                    dia.label_duration(spk) for spk in speakers
                ]))
                caption = (max_speaker, segment["text"])
            speaker_captions.append(caption)

        return speaker_captions

    def __call__(self, diarization, waveform):
        # Step 1: Transcribe
        transcription = self.transcribe(waveform)
        # Update transcription buffer
        self._buffer += transcription["text"]
        # The audio may not be the beginning of the conversation
        time_shift = waveform.sliding_window.start
        # Step 2: Assign speakers
        speaker_transcriptions = self.identify_speakers(transcription, diarization, time_shift)
        return speaker_transcriptions
# Suppress whisper-timestamped warnings for a clean output

def get_live_trascription():
    config = PipelineConfig(
        duration=5,
        step=0.5,
        latency="min",
        tau_active=0.5,
        rho_update=0.1,
        delta_new=0.57
    )
    dia = OnlineSpeakerDiarization(config)
    source = MicrophoneAudioSource(config.sample_rate)

    asr = WhisperTranscriber(model="small")

    transcription_duration = 2
    batch_size = int(transcription_duration // config.step)
    source.stream.pipe(
        dops.rearrange_audio_stream(
            config.duration, config.step, config.sample_rate
        ),
        ops.buffer_with_count(count=batch_size),
        ops.map(dia),
        ops.map(concat),
        ops.filter(lambda ann_wav: ann_wav[0].get_timeline().duration() > 0),
        ops.starmap(asr),
        ops.map(colorize_transcription),
    ).subscribe(on_next=rich.print, on_error=lambda _: traceback.print_exc())

    print("Listening...")
    source.read()




def get_specar_from_file(filename):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token=os.getenv("OPENAI_API_KEY"))
    diarization = pipeline(filename)

    # 5. print the result
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(_)
        yield f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}"


voices = ["ErXwobaYiN019PkySvjV", "EXAVITQu4vr4xnSDxMaL","9Mi9dBkaxn2pCIdAAGOB"]


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.1"
        self.name = "isaa_audio"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "VIOLETBG"
        self.config = {}
        self._simpel_speech_recognizer = None
        self._simpel_speech_recognizer_mice = None
        self.get_live_trascription = get_live_trascription
        self.generate_cache_from_history = generate_cache_from_history
        self.play_from_cache = play_from_cache
        self.isaa_instance = {"Stf": {},
                              "DiA": {}}
        self.keys = {
            "KEY": "key~~~~~~~",
            "Config": "config~~~~"
        }
        self.tools = {
            "all": [["Version", "Shows current Version"],
                    ["Run", "Starts Inference"],
                    ],
            "name": "isaa",
            "Version": self.show_version,
        }

        FileHandler.__init__(self, "issa.config", app.id if app else __name__)
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=None, color=self.color, on_exit=self.on_exit)

    def show_version(self):
        self.print("Version: ", self.version)

    def on_start(self):
        self.load_file_handler()
        config = self.get_file_handler(self.keys["Config"])
        self._simpel_speech_recognizer_mice = sr.Microphone()
        self._simpel_speech_recognizer = sr.Recognizer()
        self.logger.info("simpel speech online")
        if config is not None:
            self.config = eval(config)

    def on_exit(self):
        del self._simpel_speech_recognizer
        self.add_to_save_file_handler(self.keys["Config"], str(self.config))
        self.save_file_handler()
        self.file_handler_storage.close()

    @property
    def simpel_speech_recognizer(self):
        return self._simpel_speech_recognizer

    @property
    def simpel_speech_recognizer_mice(self):
        return self._simpel_speech_recognizer_mice

    @property
    def simpel_speech(self):
        return self._simpel_speech_recognizer, self._simpel_speech_recognizer_mice

    @staticmethod
    def speech(text, voice_index=0, use_cache=True):
        chucks = []
        while len(text) > 800:
            chucks.append(text[:800])
            text = text[800:]

        if text:
            chucks.append(text)

        for chuck in chucks:
            if chuck:
                if use_cache:
                    eleven_labs_speech_(chuck, voice_index)
                else:
                    return eleven_labs_speech(chuck, voice_index)

    @staticmethod
    def speech_stream(text, voice_index=0, use_cache=True):
        chucks = []

        while len(text) > 800:
            chucks.append(text[:800])
            text = text[800:]

        if text:
            chucks.append(text)

        for chuck in chucks:
            if chuck:
                if use_cache:
                    eleven_labs_speech_s(chuck, voice_index)
                else:
                    return eleven_labs_speech_stream(chuck, voice_index)


def get_hash_key(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def load_cache_file():
    if os.path.exists("cache_file.pkl"):
        with open("cache_file.pkl", "rb") as f:
            return pickle.load(f)
    return {}


def save_cache_file(cache_data):
    with open("cache_file.pkl", "wb") as f:
        pickle.dump(cache_data, f)


def save_audio_to_cache(hash_key, audio_content):
    cache_data = load_cache_file()
    cache_data[hash_key] = audio_content
    save_cache_file(cache_data)


def play_from_cache(hash_key):
    cache_data = load_cache_file()
    if hash_key in cache_data:
        audio_content = get_audio_from_history_item(cache_data[hash_key])
        play_audio(audio_content)


def eleven_labs_speech_s(text, voice_index=0):
    hash_key = get_hash_key(text)
    cache_data = load_cache_file()

    if hash_key in cache_data:
        audio_content = get_audio_from_history_item(cache_data[hash_key])
        play_audio(audio_content)
    else:
        eleven_labs_speech_stream(text, voice_index)
        add_last_audio_to_cache()

    return True


def eleven_labs_speech_(text, voice_index=0):
    hash_key = get_hash_key(text)
    cache_data = load_cache_file()

    if hash_key in cache_data:
        audio_content = get_audio_from_history_item(cache_data[hash_key])
        play_audio(audio_content)
    else:
        eleven_labs_speech(text, voice_index)
        add_last_audio_to_cache()

    return True


def eleven_labs_speech(text, voice_index=0):
    tts_headers = {
        "Content-Type": "application/json",
        "xi-api-key": os.getenv("ELEVENLABS_API_KEY")
    }
    tts_url = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}".format(
        voice_id=voices[voice_index])
    formatted_message = {"text": text}

    hash_key = get_hash_key(text)
    cache_data = load_cache_file()

    if hash_key in cache_data:
        audio_content = get_audio_from_history_item(cache_data[hash_key])
    else:
        response = requests.post(
            tts_url, headers=tts_headers, json=formatted_message)

        if response.status_code != 200:
            print("Request failed with status code:", response.status_code)
            print("Response content:", response.content)
            return False

        audio_content = response.content
        save_audio_to_cache(text, audio_content)

    play_audio(audio_content)
    return True


def play_audio(audio_content):
    audio = AudioSegment.from_file(BytesIO(audio_content), format="mp3")
    play(audio)


def play_audio_stream(audio_stream):
    audio = AudioSegment.from_file(audio_stream, format="mp3")
    play(audio)


def eleven_labs_speech_stream(text, voice_index=0):
    tts_headers = {
        "Content-Type": "application/json",
        "xi-api-key": os.getenv("ELEVENLABS_API_KEY")
    }
    tts_url = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream".format(
        voice_id=voices[voice_index])
    formatted_message = {"text": text}

    response = requests.post(
        tts_url, headers=tts_headers, json=formatted_message, stream=True)

    if response.status_code == 200:
        play_audio_stream(response.raw)
        return True
    else:

        print("Request failed with status code:", response.status_code)
        print("Response content:", response.content)
        return False


def get_history():
    history_url = "https://api.elevenlabs.io/v1/history"
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": os.getenv("ELEVENLABS_API_KEY")
    }
    response = requests.get(history_url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print("Request failed with status code:", response.status_code)
        print("Response content:", response.content)
        return None


def get_audio_from_history_item(history_item_id):
    audio_url = f"https://api.elevenlabs.io/v1/history/{history_item_id}/audio"
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": os.getenv("ELEVENLABS_API_KEY")
    }
    response = requests.get(audio_url, headers=headers)

    if response.status_code == 200:
        return response.content
    else:
        print("Request failed with status code:", response.status_code)
        print("Response content:", response.content)
        return None


def add_last_audio_to_cache():
    item = get_history()["history"][0]

    hash_key = get_hash_key(item["text"])

    cache_data = load_cache_file()

    if hash_key not in cache_data:
        history_id = item["history_item_id"]

        if history_id is not None:
            save_audio_to_cache(hash_key, history_id)


def generate_cache_from_history():
    history = get_history()
    if history is None:
        return

    cache_data = load_cache_file()

    len_c = len(cache_data)

    for item in history["history"]:
        hash_key = get_hash_key(item["text"])
        if hash_key not in cache_data:
            history_id = item["history_item_id"]

            if history_id is None:
                continue

            print("hash key : ", hash_key)
            cache_data[hash_key] = history_id

    print(f"Adding {len(cache_data)-len_c} audio files to cache")

    save_cache_file(cache_data)
