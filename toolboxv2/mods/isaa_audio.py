import hashlib
import os
import platform
import tempfile
import threading
import wave
from io import BytesIO
from pathlib import Path

import requests
import speech_recognition
import speech_recognition as sr
import pickle

from transformers import pipeline

from toolboxv2 import MainTool, FileHandler, get_logger, Style
from pydub import AudioSegment
from pydub.playback import play
import logging
import openai
import time
import pyaudio
import queue

try:
    import pyttsx3

    pyttsx3_init = True
except ImportError:
    pyttsx3_init = False

from gtts import gTTS
from playsound import playsound

try:
    import whisper

    whisper_init = True
except Exception:
    whisper_init = False

try:
    import winsound

    winsound_init = True
except ImportError:
    winsound_init = False

import numpy as np

voices = ["ErXwobaYiN019PkySvjV", "EXAVITQu4vr4xnSDxMaL", "9Mi9dBkaxn2pCIdAAGOB"]
_most_recent_text_spoken_to_user_via_speech_stream_isaa_audio_ = ""


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
        self.speech_stream = speech_stream
        self.isaa_instance = {"Stf": {},
                              "DiA": {}}
        self.keys = {
            "KEY": "key~~~~~~~",
            "Config": "config~~~~"
        }
        self.tools = {
            "all": [["Version", "Shows current Version"],
                    ["init", "Starts Speech for isaa"],
                    ],
            "name": "isaa",
            "Version": self.show_version,
            "init": self.init_speech,
        }

        FileHandler.__init__(self, "issaAuDi.config", app.id if app else __name__)
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

        if not os.path.exists("./data/isaa_data/"):
            Path("./data/isaa_data/").mkdir(parents=True, exist_ok=True)

        if config is not None:
            self.config = config

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

    def init_speech(self, _, app):
        isaa = app.get_mod("isaa")
        isaa.speak = self.speech_stream
        self.print("speech initialized")


def speech_stream(text, voice_index=0, use_cache=True):
    global _most_recent_text_spoken_to_user_via_speech_stream_isaa_audio_
    chucks = []
    if not text:
        return False
    if text == _most_recent_text_spoken_to_user_via_speech_stream_isaa_audio_:
        return False
    _most_recent_text_spoken_to_user_via_speech_stream_isaa_audio_ = text
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
    if os.path.exists(".data/cache_file.pkl"):
        with open(".data/cache_file.pkl", "rb") as f:
            return pickle.load(f)
    return {}


def save_cache_file(cache_data):
    with open(".data/cache_file.pkl", "wb") as f:
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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(audio_stream.read())
        temp_file.flush()
        audio = AudioSegment.from_file(temp_file.name, format="mp3")
        play(audio)
    # audio = AudioSegment.from_file(audio_stream, format="mp3")
    # play(audio)


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
    if platform.system() == "Darwin" or platform.system() == "Linux":
        filename = './data/isaa_data/speech.mp3'
    else:
        filename = '.\\data/isaa_data\\speech.mp3'
    tts.save(filename)
    playsound(filename)
    os.remove(filename)


def text_to_speech3(text, engin=None):
    if engin is None:
        if pyttsx3_init:
            def text_to_speech3_(text, engine=pyttsx3.init()):
                engine.say(text)
                engine.runAndWait()
                return engine

            return text_to_speech3_(text)
        else:
            print("TTS 3 not available")
    else:
        return text


# wisper # ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2',
# 'large']


def get_mean_amplitude(stream, seconds=2, rate=44100, p=False):
    amplitude = []
    frames = []
    for j in range(int(rate / int(rate / 10) * seconds)):
        data = stream.read(int(rate / 10))
        frames.append(data)
        audio_np = np.frombuffer(data, dtype=np.int16)
        amplitude.append(np.abs(audio_np).mean())
        if p:
            print(
                f"[last amplitude] : {amplitude[-1]:.2f} "
                f"[ac mean] : {sum(amplitude) / len(amplitude):.2f} "
                f"[min amplitude] : {min(amplitude):.2f}",
                f"[max amplitude] : {max(amplitude):.2f}",
                end="\r")

    return sum(amplitude) / len(amplitude), frames


def s30sek_mean(seconds=30, rate=44100, p=False):
    audio = pyaudio.PyAudio()
    # Erstellen Sie einen Stream zum Aufnehmen von Audio
    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=rate, input=True,
                        frames_per_buffer=int(rate / 10))

    mean_amplitude, _ = get_mean_amplitude(stream, seconds=seconds, rate=rate, p=p)

    return mean_amplitude


def get_audio_transcribe(seconds=30,
                         filename=f"./data/isaa_data/output.mp3",
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


def init_live_transcript(model="jonatasgrosman/wav2vec2-large-xlsr-53-german",
                         rate=16000, chunk_duration=10, amplitude_min=84):
    """
    models EN : 'openai/whisper-tiny'
    DE : 'jonatasgrosman/wav2vec2-large-xlsr-53-german'

    close to live chunk_duration = 1.5
    """
    # Erstelle Queues für die Kommunikation zwischen den Threads und system
    que_t0 = queue.Queue()
    que_t1 = queue.Queue()
    audio_files_que = queue.Queue()
    res_que = queue.Queue()

    def join():
        # Warte bis beide Threads fertig sind
        thread0.join()
        thread1.join()
        os.remove(temp[1]) if os.path.exists(temp[1]) else None
        os.remove(temp[-1]) if os.path.exists(temp[-1]) else None

    def put(x):
        logger.info(
            Style.ITALIC(
                Style.Bold(f"Send data to Threads: {x}")))
        que_t0.put(x)
        que_t1.put(x)

        if x == 'exit':
            join()
            if stream.is_active():
                stream.stop_stream()
            stream.close()
            audio.terminate()

    def pipe_generator():
        def helper_runner(audio_file_name):
            with open(audio_file_name, "rb") as audio_file:
                return openai.Audio.translate(model, audio_file)

        if "/" in model:
            return pipeline("automatic-speech-recognition", model)

        elif "-" in model:
            return helper_runner

        else:
            raise ValueError(f"pipe_generator : model is not suppertet : {model}")

    pipe = pipe_generator()

    if rate <= 0:
        raise ValueError("rate must be bigger then 0 best rate: 44100 | 16000")

    audio = pyaudio.PyAudio()
    # Erstellen Sie einen Stream zum Aufnehmen von Audio
    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=rate, input=True,
                        frames_per_buffer=int(rate / 10))

    logger = get_logger()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf0:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf1:

            temp = 0, tf0.name, tf1.name

            def T0():
                alive = True
                save = False
                frames = []
                logger.info(
                    Style.ITALIC(
                        Style.Bold(
                            f"Record : Start"
                        )))
                chunk_frames = 0
                index = 1
                silence_duration = 0
                speak_duration = 0
                speak_start_time = None
                while alive:

                    if not que_t0.empty():
                        data = que_t0.get()
                        logger.info(
                            Style.ITALIC(
                                Style.Bold(f"T0 Received data : {data}")))
                        if data == 'exit':
                            alive = False
                        if data == 'stop':
                            stream.stop_stream()
                            save = False
                            if winsound_init:
                                winsound.Beep(120, 175)
                        if data == 'start':
                            save = True
                            silence_duration = 0
                            speak_duration = 0
                            speak_start_time = None
                            frames = []
                            stream.start_stream()
                            if winsound_init:
                                winsound.Beep(320, 125)

                    if save:
                        data = stream.read(int(rate / 10))
                        audio_np = np.frombuffer(data, dtype=np.int16)
                        amplitude = np.abs(audio_np).mean()
                        frames.append(data)
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
                            speak_duration += int(rate / 10) / rate
                        # print( f"[speak_duration] : {speak_duration:.2f} [silence_duration] : {
                        # silence_duration:.2f} [amplitude] : {amplitude:.2f}", end="\r")
                        chunk_frames += 1
                        if chunk_frames >= int(rate / int(rate / 10) * chunk_duration) \
                            and silence_duration > 0.2 \
                            and speak_duration > chunk_duration / 5:

                            # get temp file
                            ac_temp = temp[index]
                            logger.info(
                                Style.ITALIC(
                                    Style.Bold(f"T0 Saving Sample {ac_temp}")))

                            with wave.open(ac_temp, 'wb') as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                                wf.setframerate(rate)
                                wf.writeframes(b''.join(frames))

                            frames = []
                            chunk_frames = 0
                            silence_duration = 0
                            speak_duration = 0
                            speak_start_time = None
                            audio_files_que.put(index)
                            index *= -1

                logger.info(
                    Style.ITALIC(
                        Style.Bold("T0 exiting")))

            def T1():
                alive = True
                transcribe = False
                logger.info(
                    Style.ITALIC(
                        Style.Bold("T1 started")))
                while alive:

                    if not que_t1.empty():
                        data = que_t1.get()
                        logger.info(
                            Style.ITALIC(
                                Style.Bold(f"T1 Received data : {data}")))
                        if data == 'exit':
                            alive = False
                        if data == 'stop':
                            transcribe = False
                        if data == 'start':
                            transcribe = True

                    if transcribe:
                        if not audio_files_que.empty():
                            t0 = time.time()
                            index = audio_files_que.get()
                            ac_temp = temp[index]
                            logger.info(
                                Style.ITALIC(
                                    Style.Bold(f"T1 Transcribe Sample {ac_temp}")))
                            result = pipe(ac_temp)['text']
                            res_que.put(result)
                            logger.info(
                                Style.ITALIC(
                                    Style.Bold(f"T1 Don in {time.time() - t0:.3f} chars {len(result)} :"
                                               f" {Style.GREY(result[:10])}..")))

                logger.info(
                    Style.ITALIC(
                        Style.Bold("T1 exiting")))

    thread0 = threading.Thread(target=T0)
    thread1 = threading.Thread(target=T1)

    thread0.start()
    thread1.start()

    return put, res_que
# 43 0.26
# 49 .29
# 39 0.3

