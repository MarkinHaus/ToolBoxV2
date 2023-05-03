import hashlib
from io import BytesIO
import requests
import speech_recognition as sr
import pickle
from toolboxv2 import MainTool, FileHandler
from pydub import AudioSegment
from pydub.playback import play
from pyannote.audio import Pipeline
import logging
import traceback
import diart.operators as dops
import rich
import rx.operators as ops
from diart import OnlineSpeakerDiarization, PipelineConfig
from diart.sources import MicrophoneAudioSource
from pyannote.core import Segment
from contextlib import contextmanager
import openai
import sys
import time

try:
    import pyttsx3

    pyttsx3_init = True
except ImportError:
    pyttsx3_init = False
import speech_recognition
from gtts import gTTS
from playsound import playsound
try:
    import whisper
    whisper_init = True
except ImportError:
    whisper_init = False
import pyaudio
import wave
import os

try:
    import winsound

    winsound_init = True
except ImportError:
    winsound_init = False

import numpy as np

voices = ["ErXwobaYiN019PkySvjV", "EXAVITQu4vr4xnSDxMaL", "9Mi9dBkaxn2pCIdAAGOB"]


class Tools(MainTool, FileHandler):
    def __init__(self, app=None):
        self.version = "0.0.1"
        self.name = "isaa_audio"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "VIOLETBG"
        self.config = {}
        self._simpel_speech_recognizer = None
        self._simpel_speech_recognizer_mice = None
        self.generate_cache_from_history = generate_cache_from_history
        self.get_audio_transcribe = get_audio_transcribe
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


    def get_speech_to_text(self):
        pass


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
    try:
        item = get_history()["history"][0]

        hash_key = get_hash_key(item["text"])

        cache_data = load_cache_file()

        if hash_key not in cache_data:
            history_id = item["history_item_id"]

            if history_id is not None:
                save_audio_to_cache(hash_key, history_id)
    except TypeError:
        print("Error loading history (elevenlabs)")


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

    print(f"Adding {len(cache_data) - len_c} audio files to cache")

    save_cache_file(cache_data)


def get_audio_part(recognizer, microphone, language='de',
                   phrase_time_limit=6):  # -> Alter AutomaticSpeechRecognitionPipeline Hugg
    text = ""
    confidence = 1
    with microphone as source:
        print("listening...")
        audio = recognizer.listen(source, phrase_time_limit=phrase_time_limit)
        try:
            text, confidence = recognizer.recognize_google(audio, language=language, with_confidence=True)
        except speech_recognition.UnknownValueError:
            print("-")

    return text, confidence


def get_audio_text_c0(app, phrase_time_limit=6):
    out = app.new_ac_mod('isaa_audio')
    if isinstance(out, str):
        app.logger.critical(f'Usertalk : no isaa_audio mod {out}')
        return

    recognizer, microphone = app.AC_MOD.simpel_speech
    user_text_frac0, c = get_audio_part(recognizer=recognizer,
                                        microphone=microphone,
                                        phrase_time_limit=phrase_time_limit)
    while c < 0.6:
        c = 1
        return user_text_frac0  # Unsicher
    return user_text_frac0


def text_to_speech(text, lang='de'):
    tts = gTTS(text=text, lang=lang)
    filename = '.\\isaa_data\\speech.mp3'
    tts.save(filename)
    playsound(filename)
    os.remove(filename)


def text_to_speech3(text, engin=None):
    if engin is None:
        if pyttsx3_init:
            return text_to_speech3_(text)
        else:
            print("TTS 3 not available")
    else:
        return text_to_speech3_(text, engin)


def text_to_speech3_(text, engine=pyttsx3.init()):
    engine.say(text)
    engine.runAndWait()
    return engine


def wisper_multy_speakers(path='./isaa_data/output.wav',
                          model_size='tiny'):  # ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large']
    if not whisper_init:
        return "module whisper installed"
    model = whisper.load_model(model_size)
    result = model.transcribe(path)
    return result['text']


def get_mean_amplitude(stream, seconds=2, rate=44100):
    amplitude = []
    frames = []
    for j in range(int(rate / int(rate / 10) * seconds)):
        data = stream.read(int(rate / 10))
        frames.append(data)
        audio_np = np.frombuffer(data, dtype=np.int16)
        amplitude.append(np.abs(audio_np).mean())

    return sum(amplitude) / len(amplitude), frames


def s30sek_mean(seconds=30):
    rate = 44100
    audio = pyaudio.PyAudio()
    # Erstellen Sie einen Stream zum Aufnehmen von Audio
    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=rate, input=True,
                        frames_per_buffer=int(rate / 10))

    mean_amplitude, _ = get_mean_amplitude(stream, seconds=seconds, rate=rate)

    return mean_amplitude


def get_audio_transcribe(seconds=30,
                         filename=f"./isaa_data/output.mp3",
                         model="whisper-1",
                         rate=44100,
                         amplitude_min=82,
                         s_duration_max=1.8,
                         min_speak_duration=1.1
                         ):
    if rate <= 0:
        raise ValueError("rate must be bigger then 0 best rate: 44100")
    audio = pyaudio.PyAudio()
    # Erstellen Sie einen Stream zum Aufnehmen von Audio

    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=rate, input=True,
                        frames_per_buffer=int(rate / 10))

    frames = []
    print(f"Record : Start")

    speak_start_time = None
    speak_duration = 0
    silence_duration = 0
    if winsound_init:
        winsound.Beep(320, 125)

    for _ in range(int(rate / int(rate / 10) * seconds)):
        data = stream.read(int(rate / 10))
        frames.append(data)
        audio_np = np.frombuffer(data, dtype=np.int16)
        amplitude = np.abs(audio_np).mean()

        # Check if the amplitude has dropped below a certain threshold
        if amplitude < amplitude_min:
            # If the person has stopped speaking, update the silence duration
            if speak_start_time is not None:
                speak_duration += time.time() - speak_start_time
                speak_start_time = None
            silence_duration += int(rate / 10) / rate
        else:
            # If the person has started speaking, update the speaking duration
            if speak_start_time is None:
                speak_start_time = time.time()
                silence_duration = 0
            speak_duration += int(rate / 10) / rate

        if speak_duration != 0 and silence_duration >= s_duration_max:
            break

        if silence_duration >= seconds / 4:
            break

        print(
            f"[speak_duration] : {speak_duration:.2f} [silence_duration] : {silence_duration:.2f} [amplitude] : {amplitude:.2f}",
            end="\r")
        # print(f"[silence_duration] : {silence_duration:.2f}")
        # print(f"[amplitude]        : {amplitude:.2f}")
    if winsound_init:
        winsound.Beep(120, 175)
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print(f"")

    if speak_duration <= min_speak_duration:
        return " "

    print(f"Saving sample")

    filepath = os.path.join(os.getcwd(), filename)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(
            b''.join(frames))
        wf.close()

    print(f"transcribe sample")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    audio_file = open(filename, "rb")
    res = openai.Audio.translate("whisper-1", audio_file)["text"]
    # audio_file = open(filename, "rb")
    # print("transcribe:", openai.Audio.transcribe("whisper-1", audio_file)["text"])

    # res = wisper_multy_speakers(filename, 'small')

    return res





import pyaudio
import argparse
import asyncio
import aiohttp
import json
import os
import sys
import wave
import websockets

from datetime import datetime

startTime = datetime.now()

all_mic_data = []
all_transcripts = []

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 8000

audio_queue = asyncio.Queue()

# Mimic sending a real-time stream by sending this many seconds of audio at a time.
# Used for file "streaming" only.
REALTIME_RESOLUTION = 0.250

subtitle_line_counter = 0


def subtitle_time_formatter(seconds, separator):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


def subtitle_formatter(response, format):
    global subtitle_line_counter
    subtitle_line_counter += 1

    start = response["start"]
    end = start + response["duration"]
    transcript = (
        response.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
    )

    if format == "srt":
        separator = ","
    else:
        separator = "."

    subtitle_string = f"{subtitle_line_counter}\n"
    subtitle_string += f"{subtitle_time_formatter(start, separator)} --> "
    subtitle_string += f"{subtitle_time_formatter(end, separator)}\n"
    if format == "vtt":
        subtitle_string += "- "
    subtitle_string += f"{transcript}\n\n"

    return subtitle_string


# Used for microphone streaming only.
def mic_callback(input_data, frame_count, time_info, status_flag):
    audio_queue.put_nowait(input_data)
    return (input_data, pyaudio.paContinue)

async def run(key, method, format, **kwargs):
    deepgram_url = "wss://api.deepgram.com/v1/listen?punctuate=true"
    method = "mic"

    if method == "mic":
        deepgram_url += "&encoding=linear16&sample_rate=16000"

    # Connect to the real-time streaming endpoint, attaching our credentials.
    async with websockets.connect(
        deepgram_url, extra_headers={"Authorization": "Token {}".format(key)}
    ) as ws:
        print(f'ℹ️  Request ID: {ws.response_headers.get("dg-request-id")}')
        print("🟢 (1/5) Successfully opened Deepgram streaming connection")

        async def sender(ws):
            print(
                f'🟢 (2/5) Ready to stream {method if (method == "mic" or method == "url") else kwargs["filepath"]} audio to Deepgram{". Speak into your microphone to transcribe." if method == "mic" else ""}'
            )

            if method == "mic":
                try:
                    while True:
                        mic_data = await audio_queue.get()
                        all_mic_data.append(mic_data)
                        await ws.send(mic_data)
                except websockets.exceptions.ConnectionClosedOK:
                    await ws.send(json.dumps({"type": "CloseStream"}))
                    print(
                        "🟢 (5/5) Successfully closed Deepgram connection, waiting for final transcripts if necessary"
                    )

                except Exception as e:
                    print(f"Error while sending: {str(e)}")
                    raise

            return

        async def receiver(ws):
            """Print out the messages received from the server."""
            first_message = True
            first_transcript = True
            transcript = ""

            async for msg in ws:
                res = json.loads(msg)
                if first_message:
                    print(
                        "🟢 (3/5) Successfully receiving Deepgram messages, waiting for finalized transcription..."
                    )
                    first_message = False
                try:
                    # handle local server messages
                    if res.get("msg"):
                        print(res["msg"])
                    if res.get("is_final"):
                        transcript = (
                            res.get("channel", {})
                            .get("alternatives", [{}])[0]
                            .get("transcript", "")
                        )
                        if transcript != "":
                            if first_transcript:
                                print("🟢 (4/5) Began receiving transcription")
                                # if using webvtt, print out header
                                if format == "vtt":
                                    print("WEBVTT\n")
                                first_transcript = False
                            if format == "vtt" or format == "srt":
                                transcript = subtitle_formatter(res, format)
                            print(transcript)
                            all_transcripts.append(transcript)

                        # if using the microphone, close stream if user says "goodbye"
                        if method == "mic" and "goodbye" in transcript.lower():
                            await ws.send(json.dumps({"type": "CloseStream"}))
                            print(
                                "🟢 (5/5) Successfully closed Deepgram connection, waiting for final transcripts if necessary"
                            )

                    # handle end of stream
                    if res.get("created"):
                        # save subtitle data if specified
                        if format == "vtt" or format == "srt":
                            data_dir = os.path.abspath(
                                os.path.join(os.path.curdir, "data")
                            )
                            if not os.path.exists(data_dir):
                                os.makedirs(data_dir)

                            transcript_file_path = os.path.abspath(
                                os.path.join(
                                    data_dir,
                                    f"{startTime.strftime('%Y%m%d%H%M')}.{format}",
                                )
                            )
                            with open(transcript_file_path, "w") as f:
                                f.write("".join(all_transcripts))
                            print(f"🟢 Subtitles saved to {transcript_file_path}")

                            # also save mic data if we were live streaming audio
                            # otherwise the wav file will already be saved to disk
                            if method == "mic":
                                wave_file_path = os.path.abspath(
                                    os.path.join(
                                        data_dir,
                                        f"{startTime.strftime('%Y%m%d%H%M')}.wav",
                                    )
                                )
                                wave_file = wave.open(wave_file_path, "wb")
                                wave_file.setnchannels(CHANNELS)
                                wave_file.setsampwidth(SAMPLE_SIZE)
                                wave_file.setframerate(RATE)
                                wave_file.writeframes(b"".join(all_mic_data))
                                wave_file.close()
                                print(f"🟢 Mic audio saved to {wave_file_path}")

                        print(
                            f'🟢 Request finished with a duration of {res["duration"]} seconds. Exiting!'
                        )
                except KeyError:
                    print(f"🔴 ERROR: Received unexpected API response! {msg}")

        # Set up microphone if streaming from mic
        async def microphone():
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=mic_callback,
            )

            stream.start_stream()

            global SAMPLE_SIZE
            SAMPLE_SIZE = audio.get_sample_size(FORMAT)

            while stream.is_active():
                await asyncio.sleep(0.1)

            stream.stop_stream()
            stream.close()

        functions = [
            asyncio.ensure_future(sender(ws)),
            asyncio.ensure_future(receiver(ws)),
        ]

        if method == "mic":
            functions.append(asyncio.ensure_future(microphone()))

        await asyncio.gather(*functions)



def main():
    """Entrypoint for the example."""
    # Parse the command-line arguments.

    try:
        asyncio.run(run("", "mic", format))

    except websockets.exceptions.InvalidStatusCode as e:
        print(f'🔴 ERROR: Could not connect to Deepgram! {e.headers.get("dg-error")}')
        print(
            f'🔴 Please contact Deepgram Support (developers@deepgram.com) with request ID {e.headers.get("dg-request-id")}'
        )
        return
    except websockets.exceptions.ConnectionClosedError as e:
        error_description = f"Unknown websocket error."
        print(
            f"🔴 ERROR: Deepgram connection unexpectedly closed with code {e.code} and payload {e.reason}"
        )

        if e.reason == "DATA-0000":
            error_description = "The payload cannot be decoded as audio. It is either not audio data or is a codec unsupported by Deepgram."
        elif e.reason == "NET-0000":
            error_description = "The service has not transmitted a Text frame to the client within the timeout window. This may indicate an issue internally in Deepgram's systems or could be due to Deepgram not receiving enough audio data to transcribe a frame."
        elif e.reason == "NET-0001":
            error_description = "The service has not received a Binary frame from the client within the timeout window. This may indicate an internal issue in Deepgram's systems, the client's systems, or the network connecting them."

        print(f"🔴 {error_description}")
        # TODO: update with link to streaming troubleshooting page once available
        # print(f'🔴 Refer to our troubleshooting suggestions: ')
        print(
            f"🔴 Please contact Deepgram Support (developers@deepgram.com) with the request ID listed above."
        )
        return
    except websockets.exceptions.ConnectionClosedOK:
        return

    except Exception as e:
        print(f"🔴 ERROR: Something went wrong! {e}")
        return


if __name__ == "__main__":
    sys.exit(main() or 0)

