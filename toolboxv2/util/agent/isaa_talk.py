"""Console script for toolboxv2. Isaa Talk Tool"""
import datetime
import random

import openai
from colorama import Fore
from transformers import pipeline
import sys

from toolboxv2 import Style
from toolboxv2.util.agent.scripts import speak
from toolboxv2.util.agent.scripts.autoconfig import AUTOConfig
from toolboxv2.util.agent.scripts.commands import get_command
from toolboxv2.util.agent.scripts.main import construct_prompt, print_assistant_thoughts, print_to_console
from toolboxv2.util.agent.scripts.memory import PineconeMemory
from toolboxv2.util.agent.scripts.speak import eleven_labs_speech
from toolboxv2.util.agent.scripts.spinner import Spinner
from toolboxv2.util.agent.util import PromptConfig, create_chat_message
# Import default Pages
from toolboxv2.util.toolbox import App

import time

try:
    import pyttsx3
    pyttsx3_init = True
except ImportError:
    pyttsx3_init = False

from PIL import Image
import numpy as np
import speech_recognition

from gtts import gTTS
from langchain.document_loaders import UnstructuredFileLoader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from playsound import playsound

from toolboxv2.mods_dev.isaa import image_genrating_tool, AgentConfig, CollectiveMemory, \
    ShortTermMemory, append_to_file
import sounddevice as sd

import whisper
import pyaudio
import wave
import os

try:
    import winsound
    winsound_init = True
except ImportError:
    winsound_init = False

class Ability:
    answer = True

    def default(self):
        return self



def print_to_console(
    title,
    title_color,
    content,
    min_typing_speed=0.05,
    max_typing_speed=0.01):
    print(title_color +title+" "+ Style.BLUE(""), end="")
    if content:
        if isinstance(content, list):
            content = " ".join(content)
        if not isinstance(content, str):
            print(f"SYSTEM NO STR type : {type(content)}")
            print(content)
            return
        words = content.split()
        for i, word in enumerate(words):
            print(word, end="", flush=True)
            if i < len(words) - 1:
                print(" ", end="", flush=True)
            typing_speed = random.uniform(min_typing_speed, max_typing_speed)
            time.sleep(typing_speed)
            # type faster after each word
            min_typing_speed = min_typing_speed * 0.95
            max_typing_speed = max_typing_speed * 0.95
    print()




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


def get_ref_words(words: list, text: str):
    index_list = []
    text_list = text.split(' ')
    for word in words:

        if word in text_list:
            index_list.append(1)
    return sum(index_list)  # * (durchschnittliche_differenz(index_list)+0.21)


def durchschnittliche_differenz(liste):
    if not liste:
        return 0
    differenzen = [abs(liste[i + 1] - liste[i]) for i in range(len(liste) - 1)]
    if not len(differenzen):
        return 0
    if not sum(differenzen):
        return 0
    durchschnitt = sum(differenzen) / len(differenzen)
    return durchschnitt


def user_talk(app: App, functions: dict):
    get_input = False

    return_val = {
        "data": []
    }
    time_running = time.time()
    turns_running = 0

    out = app.new_ac_mod('isaa_audio')
    if isinstance(out, str):
        app.logger.critical(f'Usertalk : no isaa_audio mod {out}')
        return

    recognizer, microphone = app.AC_MOD.simpel_speech
    # session_history = [[]]
    # session_history += [c for c in app.command_history]

    user_text_frac0, c = get_audio_part(recognizer=recognizer, microphone=microphone)
    return_val['turn' + str(turns_running)] = time.time() - time_running
    return_val['confidence' + str(turns_running)] = c
    return_val['user_text' + str(turns_running)] = user_text_frac0
    time_running = time.time()

    if 'skipp' in user_text_frac0:
        get_input = False

    while get_input:
        user_text_frac1, c = get_audio_part(recognizer=recognizer, microphone=microphone)

        compleat_ = get_ref_words(
            ['verarbeite', 'berechne', 'analysiere', 'Kalkuliere', 'berechnen', 'ergebnis', 'herausfinden'],
            user_text_frac1)

        turns_running += 1
        return_val['turn' + str(turns_running)] = time.time() - time_running
        return_val['confidence' + str(turns_running)] = c
        time_running = time.time()

        if compleat_ >= 1:
            get_input = False

        # sentence_infos = combine_sentences(user_text_frac0, user_text_frac1, app)
        # sentence_infos_keys = tuple(sentence_infos.keys())
        # return_val['data'].append(sentence_infos)
        # print("[sentence_infos] :", sentence_infos)
        # if 'speed_construct' in sentence_infos_keys:
        #    if user_text_frac1 and user_text_frac0:
        #        return_val['speed-con-info' + str(turns_running)] =\
        #            functions["speed-con"](sentence_infos['speed_construct'], app, 0)
        #
        user_text_frac0 = user_text_frac1

    return return_val


def command_runner(app, todo, command):
    print("Command", command)
    print("todo", todo)

    # if command[0] == '':  # log(helper)
    #     print("Pleas enter a command or help for mor information")


#
# elif command[0] == '':
#     pass
#
# else:  # error(->)
#     print(Style.YELLOW("[-] Unknown command:") + app.pretty_print(command))


def interpret_input(app, user_input_seg, user_input):
    res = combine_sentences(user_input, user_input_seg, app)
    # prompt = f"""

    # Füge die beiden sätze so zusammen das sie einen sin ergeben fülle die lücken :

    # Satzteil 1 : {user_input}
    # Satzteil 2 : {user_input_seg}

    # """
    # res = app.run_any('isaa', 'run-sug', ["run-sug", "isaa-talk-interpret", prompt])
    # print(f"ASWERSES ::: {res}")
    return res


def combine_sentences(sentence1, sentence2, app: App):
    # add mask
    return_data = {
        "sentence1": sentence1,
        "sentence2": sentence2,
        "speed_construct": ''
    }
    sentence1 += ' [MASK] '
    sentence1 += sentence2
    combined_sentence = app.run_any('isaa', 'run-sug', ["", "fill-mask", sentence1])
    # cs = [ [ {'score': 0.15774881839752197, 'token': 2113, 'token_str': 'know', 'sequence': '[CLS] i would like to know
    # that the parameter was changed [MASK] [SEP]'}, {'score': 0.12270829826593399, 'token': 3189, 'token_str':
    # 'report', 'sequence': '[CLS] i would like to report that the parameter was changed [MASK] [SEP]'},
    # {'score': 0.10217200964689255, 'token': 2360, 'token_str': 'say', 'sequence': '[CLS] i would like to say that
    # the parameter was changed [MASK] [SEP]'}, {'score': 0.09050722420215607, 'token': 2228, 'token_str': 'think',
    # 'sequence': '[CLS] i would like to think that the parameter was changed [MASK] [SEP]'}, {'score':
    # 0.06822025030851364, 'token': 7868, 'token_str': 'assume', 'sequence': '[CLS] i would like to assume that the
    # parameter was changed [MASK] [SEP]'}],
    #
    # [ {'score': 0.9686239361763, 'token': 1012, 'token_str': '.', 'sequence': '[CLS] i would like to [MASK] that the
    # parameter was changed. [SEP]'}, {'score': 0.02599921263754368, 'token': 1025, 'token_str': ';', 'sequence': '[
    # CLS] i would like to [MASK] that the parameter was changed ; [SEP]'}, {'score': 0.0020718881860375404,
    # 'token': 1029, 'token_str': '?', 'sequence': '[CLS] i would like to [MASK] that the parameter was changed? [
    # SEP]'}, {'score': 0.001779748941771686, 'token': 999, 'token_str': '!', 'sequence': '[CLS] i would like to [
    # MASK] that the parameter was changed! [SEP]'}, {'score': 0.00031442579347640276, 'token': 1024, 'token_str':
    # ':', 'sequence': '[CLS] i would like to [MASK] that the parameter was changed : [SEP]'} ] ]

    if len(combined_sentence) == 2:

        for ob in combined_sentence[0]:
            ob['sequence'] = (ob['sequence'] if 'sequence' in ob.keys() else '').replace("[SEP]", '').replace("[CLS]",
                                                                                                              '')
        return_data["combined_sentence_word"] = combined_sentence[0][:3]
        return_data["speed_construct"] = combined_sentence[0][0]['sequence'] if 'token_str' in \
                                                                                combined_sentence[0][0].keys() else ' '

        for ob in combined_sentence[1]:
            ob['sequence'] = (ob['sequence'] if 'sequence' in ob.keys() else '').replace("[SEP]", '').replace("[CLS]",
                                                                                                              '')
        return_data["combined_sentence_punkt"] = combined_sentence[1][:3]

    if len(combined_sentence) >= 4:
        for ob in combined_sentence:
            ob['sequence'] = (ob['sequence'] if 'sequence' in ob.keys() else '').replace("[SEP]", '').replace("[CLS]",
                                                                                                              '')
        return_data["combined_sentence_word"] = combined_sentence[:3]
        return_data["speed_construct"] = combined_sentence[0]['sequence'] if 'token_str' in \
                                                                             combined_sentence[0].keys() else ' '
    # for exampel in combined_sentence:
    #    exampel = exampel.replace("[SEP]").replace("[CLS]")
    return return_data


def make_sentence(data):
    end = ""

    for step in data:
        if 'speed_construct' in step.keys():
            end += step['speed_construct']

    return end


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


def text_to_speech2(text, fm=1):
    model_name_s = ['tts_models/de/thorsten/vits', 'tts_models/de/css10/vits-neon']
    fss = [24000, 24600]
    model_name = model_name_s[fm]
    # fs = fss[fm]
    # tts = TTS(model_name, progress_bar=True, gpu=False)
    # wav = tts.tts(text)
    # sd.play(wav, fs)
    sd.wait()


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


def block_till_isaa_awake_or_kill(noice=51, wakeup_words=None, kill_words=None):
    if kill_words is None:
        kill_words = ["kill"]
    if wakeup_words is None:
        wakeup_words = ["isa", "isaa", 'wake up']

    user_called_wakeup_word = False
    ret = False

    user_text = ""

    filename = f"../../isaa_data/output_wakeup.mp3"
    filepath = os.path.join(os.getcwd(), filename)

    rate = 44100
    audio = pyaudio.PyAudio()

    # Erstellen Sie einen Stream zum Aufnehmen von Audio

    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=rate, input=True,
                        frames_per_buffer=int(rate / 10))
    print(f"Isaa - wating for words : {wakeup_words} to be detected")
    while not user_called_wakeup_word:
        silent, audio_data = get_mean_amplitude(stream, rate=rate)

        if silent >= noice:
            print(f"Isaa - Nose detected")
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(rate)
                wf.writeframes(
                    b''.join(audio_data))
                wf.close()

            user_text = wisper_multy_speakers(filename, 'base.en')

        if user_text:

            print(f"Isaa - words detected {user_text}")

            for word in wakeup_words:
                if word in user_text.lower():
                    print(f"Isaa - wakeup word detected | {word}")
                    user_called_wakeup_word = True
                    ret = True

            for word in kill_words:
                if word in user_text.lower():
                    print(f"Isaa - kill word detected | {word}")
                    user_called_wakeup_word = True
                    ret = False

            user_text = ''

            print(f"Isaa - End Nose detected ", silent)

    stream.stop_stream()
    stream.close()
    audio.terminate()
    return ret


def get_audio_transcribe(seconds=30,
                         filename=f"./isaa_data/output.mp3",
                         model='small',
                         rate=44100,
                         amplitude_min=52,
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

    res = wisper_multy_speakers(filename, model)

    return res

def doc_summary(docs):
    print(f'You have {len(docs)} document(s)')

    num_words = sum([len(doc.page_content.split(' ')) for doc in docs])

    print(f'You have roughly {num_words} words in your docs')
    print()
    if len(docs):
        print(f'Preview: \n{docs[0].page_content.split(". ")[0]}')


def summary_conversation(functions, app, n=0):
    mem_loader = UnstructuredFileLoader(f"./isaa_data/conversation{n}.txt")
    try:
        mem_doc = mem_loader.load()
    except UnicodeDecodeError:
        app.logger.error("Unable to load conversation history from file")
        return ""

    doc_summary(mem_doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0,
    )

    mem_doc = text_splitter.split_documents(mem_doc)

    doc_summary(mem_doc)

    if mem_doc:
        h0 = h1 = len(mem_doc)
        if h1 > 10:
            h1 = 5
        history = functions["speed-con"](mem_doc[h0 - h1:], app, 3)
        print(history)
        app.new_ac_mod('isaa')
        app.AC_MOD.config['speed_construct_interpret-history'] = history
        return history
    return ""


def set_up_app_for_isaa_talk(app, mem, functions=None):
    if functions is None:
        functions = {}

    app.logger.info("Setting up")
    # init setup
    # app.save_load("isaa")
    app.inplace_load("isaa", "toolboxv2.mods_dev.")
    app.logger.info("Isaa is running")

    # init interpret_input
    app.new_ac_mod('isaa')
    app.AC_MOD.add_tmk(["add_tmk", "isaa-talk-interpret", "microsoft/DialoGPT-small", "tkzca"])
    app.AC_MOD.config["isaa-talk-interpret"] = {'data': "", 't': 'talk'}
    app.AC_MOD.config["fill-mask"] = {'data': "", 't': 'fill-mask'}
    app.AC_MOD.config["de-en"] = {'data': "", 't': 'de'}
    app.AC_MOD.config["stf"] = {'data': ["ich möchte etwas mit python machen",
                                         "ich möchte informationen asu dem internet haben",
                                         "ich möchte eine matematiche frage beantwortet haben",
                                         "ich möchte mit meinem computer interagieren",
                                         "ich möchte reden"], 't': 'stf'}
    app.AC_MOD.config["stf-en"] = {'data': [
        "I want to do somethingwith python",
        "I want to have information on the internet",
        "I want to have answered a matematic question",
        "I want to interact with my computer",
        "I want to talk",
    ], 't': 'stf'}

    # api_keys(app)  # TODO use .env
    # openai.organization =
    # print(openai.Model.list())
    app.AC_MOD.lode_models()
    # app.AC_MOD.speed_construct_Interpret_agent("", app=app, memory=mem)
    functions["speed-con"] = app.AC_MOD.speed_construct_Interpret_agent
    functions["agents"] = app.AC_MOD.agent_tools
    functions["image"] = app.AC_MOD.genrate_image
    functions["ia"] = app.AC_MOD.interpret_agent

    return functions


def setup_template_g_prompt(app: App):
    app.new_ac_mod("isaa")

    app.AC_MOD.config['generator-prompt'] = {
        "input_variables": ["input"],
        "output_parser": None,
        "template": "Prompt: Erstelle eine maßgeschneiderte und ansprechende Prompt für folgende Anfrage:"
                    "\n\nAnfrage: {input}"
                    "\n\nNimm dir einen Moment Zeit, um die Anfrage sorgfältig zu analysieren"
                    " und zu verstehen. Erstelle eine Prompt basierend auf den angegebenen Informationen und dem"
                    " Fachgebiet. Achte darauf, dass die Prompt klar, präzise und für den Benutzer leicht verständlich"
                    " ist. Nutze gegebenenfalls eine freundliche und einladende Sprache,"
                    " um den Benutzer dazu zu ermutigen, mehr Informationen zu teilen oder weitere Fragen zu stellen."
                    " Wenn du Genügend informationen hast antworte mit :Final-Prompt: und gebe die finale prompt aus."
                    " \n\nVorgeschlagene Prompt:",
        "template_format": "f-string"
    }


def setup_audio(app):
    app.logger.info("Setting up audio")
    # init setup
    # app.save_load("isaa_audio")
    app.inplace_load("isaa_audio", "toolboxv2.mods_dev.")
    app.logger.info("Isaa audio is running")


def setup_alpaca(app, functions=None):
    if functions is None:
        functions = {}

    app.logger.info("Setting up: alpaca")
    # init setup
    # app.save_load("isaa")
    app.inplace_load("isaa_alpaca", "toolboxv2.mods_dev.")
    app.logger.info("Isaa alpaca is running")
    app.new_ac_mod("isaa_alpaca")
    # init interpret_input
    functions["alpaca-ez"] = app.AC_MOD.run_ez
    functions["alpaca-cov"] = app.AC_MOD.run_cov

    return functions


def run_isaa_verb_test(app: App, c=-1, awake=True):
    # c = complexity https://colab.research.google.com/drive/1V-Bt5Hm2kjaDb4P1RyMSswsDKyrzc2-3?usp=sharing#scrollTo
    # =DiE3hs3jnTlf
    from transformers import pipeline
    fi = "isaa_data/conversation0.txt"
    fm = "isaa_data/mem.txt"
    mem = False
    functions = {}
    history = ""
    compleat_conversation = []
    user_imp = ""
    ai_function_out = ""
    memory = ConversationBufferMemory(human_prefix="User:", ai_prefix="Isaa:")

    if not os.path.exists(fm):
        memory.chat_memory.add_user_message("hi!")
        memory.chat_memory.add_ai_message("whats up?")
        with open(fm, "a") as f:
            f.write(str(memory.dict()))

    with open(fm, "r") as f:
        mem = f.read()

    if mem:
        memory.load_memory_variables(eval(mem))

    functions = set_up_app_for_isaa_talk(app, memory)
    setup_audio(app)
    setup_template_g_prompt(app)

    # history = summary_conversation(functions, app)
    # user_text += history

    if not os.path.exists(fi):
        with open(fi, "a") as f:
            f.write("Nichts Hier")

    # engine = text_to_speech3("")
    # engine.setProperty('rate', 155)  # setting up new voice rate
    # engine.setProperty('volume', .6)

    app.new_ac_mod("isaa")
    fet = app.AC_MOD.bugfix()
    todolist_agent_config = app.AC_MOD.get_agent_config_class()
    step = 0

    while app.alive:
        print("", end="" + "->>\r")
        user_imp = """
Verbesssere dien code :
def insert_edit(self, command):
    "x3
    writes or edit text in a file starting at the specified line.
    "x3
    if len(command) < 4:
        return "Invalid command"
    if len(command) == 5:
        command = command[1:]
    file_, start, end, text = command
    file = self.scope + list_s_str_f(file_)
    self.print("insert_edit "+file)
    if not os.path.exists(file):
        self.create(list_s_str_f(file_))

    with open(file, 'r') as f:
        lines = f.readlines()

    lines[start:end + 1] = [text + "\\n"]

    with open(file, 'w') as f:
        f.writelines(lines)

    return f"File content updated"
und schriebe ihn in out.py"""
        ai_function_out = fet(user_imp)
        if isinstance(ai_function_out, dict):
            ai_function_out = ai_function_out['output']
        print("Isaa:", ai_function_out)
        todolist_agent_config = app.AC_MOD.get_agent_config_class("todolist")
        print(app.AC_MOD.run_agent("todolist", "", todolist_agent_config))
        user_imp = "Erzeuge mir eine Todo liste zu Ereiche Eines Haus Baus welche schritte müssen dafür " \
                            "beachtet werden?"

        print(f"{'=' * 10}STEP:{0}{'=' * 10}")
        print(f"    TODO : {app.AC_MOD.config['agent-config-todolist'].task_list}")

        ai_function_out = app.AC_MOD.run_agent("todolist", user_imp)
        ai_function_out = str(ai_function_out)
        print("#", ai_function_out, "#")
        app.alive = len(todolist_agent_config.task_list) != 0

        setup_alpaca(app, functions)

        user_imp = "Hallo alpaka das ist ein test was kannst du alles ?"
        print(user_imp)

        response2 = functions["alpaca-cov"](["", user_imp])
        text_to_speech2(response2)

        response = functions['ia'](user_text, app, 1)['output']
        print("Isaa:", response, "#2")

        if c == 4:

            print("User:", user_text)
            print("Isaa:", response)

            if not awake:
                print("Test Isaa awake call")
                block_till_isaa_awake_or_kill()
                awake = True

            input("pause;")

            user_text = get_audio_transcribe()

            print("User:", user_text)
            if " " == user_text:
                user_text = "Auf wieder sehen und bis bald ich habe dich in den ruhe modus versetzt."
                awake = False

            if "Ruhemodus" in user_text.lower():
                user_text += " Gute nacht und bis Bald "
                # awake = False

            if "exit" in user_text.lower():
                app.alive = False

            try:
                response = "Gute Nacht"
                if awake:
                    response = image_genrating_tool(user_text, app)  # functions['ia'](user_text, app, 0)
                    # response = response['output']
            except Exception as e:
                response = str(e)

            print(response)

            # if "-sleep" in response or 'Ruhemodus' in response:
            #    user_text += " Gute nacht und bis Bald "
            #    #awake = False

            # if ("Wiedersehen" in response or 'bis dann' in response) and (not 'Ruhemodus' in response):
            #    app.alive = False

            # text_to_speech3(response.replace("Isaa:", "").split("[Tool")[0])

        compleat_conversation.append("\nUser: " + user_text)
        compleat_conversation.append("\nIsaa: " + response)

        try:
            with open(fi, "a") as f:
                f.write(compleat_conversation[0])
                f.write(compleat_conversation[1])
        except UnicodeEncodeError:
            print("UnicodeEncodeError")
        compleat_conversation = []

        memory.chat_memory.add_user_message(user_text)
        memory.chat_memory.add_ai_message(response)

        if "stop" in user_text.lower():
            app.alive = False

        if "-exit" in response:
            app.alive = False

        if 'Max Tokens' in response:
            app.new_ac_mod('isaa')
            app.AC_MOD.config['speed_construct_interpret-ini'] = False
            history = summary_conversation(functions, app)
            user_text += history

        user_text = ""
        response = ""

    with open(fm, "w") as f:
        try:
            f.write(str(memory.dict()))
        except UnicodeEncodeError:
            print("Memory not encoded properly")


def test_amplitude_for_talk_mode(print=print, sek=10):
    print(f"Pleas stay silent for {sek}")
    mean_0 = s30sek_mean(10)
    print(f"Silent test done \nSpeak Test pleas Talk normal for {sek}")
    mean_1 = s30sek_mean(10)
    print(f"Speak test done \nResults silent value: {mean_0:.2f} speak value: {mean_1:.2f}")

    return mean_0, mean_1


def stop_helper(imp):
    if "Question:" in imp:
        return True
    if "User:" in imp:
        return True

    return False


def run_isaa_verb(app: App, speek_mode=True, awake=True):

    min_, max_ = 2, 20# test_amplitude_for_talk_mode(sek=2)

    if speek_mode:
        app.inplace_load("isaa_audio", "toolboxv2.mods_dev.")
        app.new_ac_mod('isaa_audio')
        # speech = app.AC_MOD.speech
        speech_stream = app.AC_MOD.speech_stream
        app.AC_MOD.generate_cache_from_history()

    ide_mode = False

    try:
        app.inplace_load("isaa_ide", "toolboxv2.mods_dev.")
        ide_mode = True
    except Exception:
        pass

    if ide_mode:
        app.new_ac_mod("isaa_ide")
        file_functions_ = app.AC_MOD.process_input
        file_functions_dis = """
witch file_functions you have access to 8 file reladead functions
format for singel input functions [create, delete, list, read] arguments is <path>
format for 2 input functions [move, copy] arguments ar <source> <destination>
format for 2 input functions [search] arguments ar <path> <keyword>
format for 2 input functions [write] arguments ar <path> <text>

syntax for function call : <function_name> <arguments>
    """

        def file_functions(x):
            try:
                if input(x + "\n\nACCEPT?\n:").lower() in ["y", "yes"]:
                    return file_functions_(x)
                return "Not authorised by user"
            except Exception as e:
                return "Error in file_functions : " + str(e)

    app.inplace_load("isaa", "toolboxv2.mods_dev.")
    app.logger.info("Isaa is running")

    sys.setrecursionlimit(1500)

    # short_term_mem_file = "isaa_data/shortMemory.txt"
    observation_term_mem_file = "isaa_data/observationMemory/"

    app.new_ac_mod('isaa')
    app.AC_MOD.observation_term_mem_file = observation_term_mem_file
    app.AC_MOD.loade_keys_from_env()

    self_agent_config: AgentConfig = app.AC_MOD.get_agent_config_class("self")

    def momory_wraper(x):
        momoey_ = CollectiveMemory().text(context=x)
        if momoey_ == "[]":
            return "No data found, Try entering other data related to your task"
        return momoey_

    # Adding Tools
    app.AC_MOD.add_tool("memory", momory_wraper, "a tool to get similar information from your memories."
                                                 " useful to get similar data. ", "memory(<related_information>)", self_agent_config.tools)

    app.AC_MOD.add_tool("save_data_to_memory", CollectiveMemory().text_add, "tool to save data to memory,"
                                                                            " write the data as specific"
                                                                            " and accurate as possible.",
                        "save_data_to_memory(<store_information>)",
                        self_agent_config.tools)

    if ide_mode:
        app.AC_MOD.add_tool("file_functions", file_functions, file_functions_dis,
                            " ",
                            self_agent_config.tools)

    def get_input():
        if speek_mode:
            input("Start listig ->")
            return get_audio_transcribe(amplitude_min=min_, s_duration_max=2)
        return input("User:")

    def print_(x, **kwargs):
        print_to_console("SYSTEM:", Fore.BLUE, x, max_typing_speed=0.04, min_typing_speed=0.08)

    def spek(x, speak_text=speek_mode, vi=0, **kwargs):
        if x.startswith("Action:"):
            return

        if not speak_text:
            return

        if ":" in x:
            x = "".join(x.split(":")[1:])

        cls_lang = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
        ln = cls_lang(x)

        if len(x) > 400:
            x = app.AC_MOD.mas_text_summaries(x, min_length=50)

        if 'de' == ln[0]["label"] and ln[0]["score"] > 0.2:
            text_to_speech3(x)

        elif 'en' == ln[0]["label"] and ln[0]["score"] > 0.5:
            speech_stream(x, voice_index=vi)
        else:
            print_(f"SPEEK SCORE TO LOW : {ln[0]['score']}")

    app.AC_MOD.speek = spek

    response = ""
    print_("Welcome")
    spek("Willkommen")
    user_text = input(":")
    print_("\n================================Starting-Agent================================")
    print_("USER0: " + user_text)
    step = 0

    while app.alive:

        if not awake:
            print("Test Isaa awake call")
            app.alive = block_till_isaa_awake_or_kill(noice=min_)
            if not app.alive:
                user_text = "Auf weider sehen"
            awake = True

        if user_text.startswith("@"):
            command = user_text.split(" ")
            user_text = str(response)
            print_(command)
            if command[1] == "mode":
                if command[2] in ["tools", "talk"]:
                    self_agent_config.mode = command[2]
                    print_("AGENT ENTER MODE: " + command[2])
                else:
                    print_.mode = "talk"
                    print("AGENT ENTER MODE: " + "talk")

        print("\nAGENT section\n")
        response = app.AC_MOD.run_agent("self", user_text)
        print("\nAGENT section END\n")

        task_done = app.AC_MOD.test_task_done(response, self_agent_config)

        print_(f"\n{'=' * 20}STEP:{step}{'=' * 20}\n")
        print_(f"\tMODE               : {self_agent_config.mode}\n")
        print_(f"\tCollectiveMemory   : {CollectiveMemory().token_in_use} | total vec num : "
               f"{CollectiveMemory().memory.get_stats()['total_vector_count']}\n")
        print_(f"\tObservationMemory  : {self_agent_config.obser_mem.tokens}\n")
        print_(f"\tShortTermMemory    : {self_agent_config.short_mem.tokens}\n\n")
        print_("AGENT: " + "\n".join(response.split(':')) + "\n")

        if stop_helper(response):

            text_to_speech3("Ich habe eine Frage")
            spek(response, vi=1)

            user_text = get_input()
            if user_text.lower() in "exit":
                app.alive = False
            if user_text.lower() in "sleep":
                awake = False

            print_(f"User: {user_text}")

        if task_done:  # TODO CLEar St mem to storage -- new task

            text_to_speech3("Ist die Aufgabe abgeschlossen?")
            print_("\tType:\n\texit\n\tstop or sleep\n\tyes or y\n\tno or n\n\tand for prompt end with -p")
            p = input("User Input: ").lower()
            if p.endswith("p"):
                print(self_agent_config.last_prompt)
                input("User Input: .. ")
            if p.startswith("e"):
                app.alive = False
            elif p.startswith("s"):
                awake = False
            elif p.startswith("y"):
                self_agent_config.short_mem.clear_to_collective()
                user_text = get_input()
                if not user_text:
                    user_text = input("Please enter ... :")
            elif p.startswith("n"):
                user_text = get_input()
                if not user_text:
                    user_text = input("Please enter ... :")
            else:
                pass

            print_(f"User: {user_text}")

        else:
            input("Agent continues - ")

        step += 1

        response = ""
