"""Console script for toolboxv2. Isaa Talk Tool"""
import datetime
# Import default Pages
from .toolbox import App

import time
import pyttsx3
from PIL import Image
import numpy as np
import speech_recognition

from gtts import gTTS
from langchain.document_loaders import UnstructuredFileLoader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from playsound import playsound

from .mods_dev.isaa import image_genrating_tool
import sounddevice as sd

import whisper
import pyaudio
import wave
import os
import winsound



class Ability:
    answer = True

    def default(self):
        return self


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


def text_to_speech3(text, engine=pyttsx3.init()):
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


def show_image_in_internet(images):
    if isinstance(images, str):
        images = [images]
    for image in images:
        os.system(f'start firefox {image}')


def block_till_isaa_awake(noice=51):
    user_called_wakeup_word = False
    wakeup_words = ["isa", "isaa", 'wake up']

    user_text = ""

    filename = f"./isaa_data/output_wakeup.mp3"
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

            user_text = ''

            print(f"Isaa - End Nose detected ", silent)

    stream.stop_stream()
    stream.close()
    audio.terminate()
    return False


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def get_audio_transcribe(seconds=30,
                         filename=f"./isaa_data/output.mp3",
                         model='small',
                         rate=44100,
                         amplitude_min=52,
                         s_duration_max=1.2
                         ):
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
    winsound.Beep(320, 125)
    for j in range(int(rate / int(rate / 10) * seconds)):
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
    winsound.Beep(120, 175)
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print(f"")

    if speak_duration <= 1.1:
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

    res = wisper_multy_speakers(filename, model)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    return res


def test_for_tool_n(i, response0):
    return f"[tool {i}]:" in response0.lower() or f"[tool{i}]:" in response0.lower() or f"[tool{i}] :" \
        in response0.lower() or f"[tool {i}] :" in response0.lower()


def compute_response(user_text, functions, app: App, tools=7):
    print("Compute response")
    response0 = functions["speed-con"](user_text, app, 2)
    text_for_tool = user_text
    tool = -1
    response1 = ""
    print("response0:", response0, "##")
    do = False
    print("Test if tool needed")
    try:
        for i in range(tools):
            print(test_for_tool_n(i, response0), end='\r')
            if test_for_tool_n(i, response0):
                print(f"Using Tool {i}")
                if i == 6:

                    image_genrating_tool(response0, app)
                    return "Die Bilder werden in Kürte angezeigt"
                else:
                    text_for_tool += ''.join(response0.split(':')[1])
                    do = True
                    tool = 1
                    response0 = response0.split(f"[Tool{i}]")[0]
                    response1 = functions["agents"](text_for_tool, app, i)
        print()
        if do:
            print("Used Tool")
            return f"{response0} \nTool: {text_for_tool}\n{response1}"

    except ValueError as e:
        f = f"\nERROR: not using chain tools\n {e} \n"
        print(f)
        user_text += f
        print("Compute response 2 ValueError")
        response0 = functions["speed-con"](user_text, app, 2)
    print("No Tool needed")
    return response0



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

def set_up_app_for_isaa_talk(app, mem):
    functions = {}
    app.logger.info("Setting up")
    # init setup
    # app.save_load("isaa")
    app.inplace_load("isaa", "toolboxv2.mods_dev.")
    app.inplace_load("isaa_audio", "toolboxv2.mods_dev.")
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

    app.AC_MOD.add_str_to_config(["", 'WOLFRAM_ALPHA_APPID', "JLQ8U8-"])
    app.AC_MOD.add_str_to_config(["", 'HUGGINGFACEHUB_API_TOKEN', ""])
    app.AC_MOD.add_str_to_config(["", 'OPENAI_API_KEY', "sk-"])
    app.AC_MOD.add_str_to_config(["", 'REPLICATE_API_TOKEN', ""])
    app.AC_MOD.add_str_to_config(["", 'IFTTTKey', ""])
    app.AC_MOD.add_str_to_config(["", 'SERP_API_KEY', ""])

    app.AC_MOD.lode_models()
    app.AC_MOD.speed_construct_Interpret_agent("", app=app, memory=mem)
    functions["speed-con"] = app.AC_MOD.speed_construct_Interpret_agent
    functions["agents"] = app.AC_MOD.agent_tools
    functions["image"] = app.AC_MOD.genrate_image
    functions["ia"] = app.AC_MOD.interpret_agent

    return functions

def run_isaa_verb(app: App, c=3.5, awake=True):
    # c = complexity https://colab.research.google.com/drive/1V-Bt5Hm2kjaDb4P1RyMSswsDKyrzc2-3?usp=sharing#scrollTo
    # =DiE3hs3jnTlf
    from transformers import pipeline
    fi = "isaa_data/conversation0.txt"
    fm = "isaa_data/mem.txt"
    mem = False
    functions = {}
    history = ""
    compleat_conversation = []
    user_text = ""
    response = ""
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

    history = summary_conversation(functions, app)
    user_text += history

    if not os.path.exists(fi):
        with open(fi, "a") as f:
            f.write("Nichts Hier")

    engine = text_to_speech3("")
    engine.setProperty('rate', 155)  # setting up new voice rate
    engine.setProperty('volume', .6)
    while app.alive:
        print("", end="" + "->>\r")

        if c == -1:
            user_text = 'Isaa Generire mir ein Bild von Mr Crab'

            response = functions['ia']("Suche mit hilfe vom Kontext den Vergangen Kontext\n"+user_text + "\nBenutze am ende Talk um Für den Nutzer Eine Antwort zu erstellen",
                                       app, 0)
            # data = json.dumps(response["intermediate_steps"], indent=2)
            # print(data)
            response = response['output']
            print(response, "#")
            # text_to_speech3(response, engine)
            app.alive = False

        if c == 0:  # mini Isaa
            # tts_pipeline = pipeline("automatic-speech-recognition")  # model="ttskit/glow-tts-ljs-300epochs"
            # Get user input audio max 6 seconds.
            # user_text = get_audio_text_c0(app)
            user_text = "hallo das ist ein kleiner Test ob sich deine Stimme verbessert hat also sag was Schönes."
            # was " \ "sind deine ersten worte? ps : (falls es beim erstenmal Klapt) versuch nur 3" prompt base
            # compute response
            # response = functions["speed-con"](user_text, app, 2)
            # # synthesise audio
            # # play response
            # print(f"Me:{user_text}\nIsaa:{response}")
            text_to_speech2(user_text)

            if "stop" in user_text.lower():
                app.alive = False

            # if "exit #0" in response.lower():
            #     app.alive = False
        if c == 1:  # Test Isaa awake call
            print(f"User: {user_text}\nIsaa: {response}\n\n")

            if not awake:
                print("Test Isaa awake call")
                block_till_isaa_awake()
                awake = True

            user_text += get_audio_transcribe()
            if " " == user_text:
                user_text = "Auf wieder sehen und bis bald ich habe dich in den ruhe modus versetzt."
                awake = False

            if "standby" in user_text.lower():
                user_text = user_text.replace("schlafen 20", " Gute nacht und bis Bald ")
                awake = False

            if "exit" in user_text.lower():
                app.alive = False

            print("Generating response")
            response = functions["speed-con"](user_text, app, 2)

            print("Generating synthesise audio response")

            text_to_speech2(response)

        if c == 2:
            t = """
            Der Ripper

Verehrter Leser, an dieser Stelle möchte ich anmerken, dass sich diese Geschichte nicht für jedes Gemüt eignet. So manchem wird beim Lesen dieser Zeilen das Blut in den Adern gefrieren, wohingegen einige dieses Buch als so langweilig empfinden werden, dass sie es gar bereuen dieses Buch gekauft zu haben. Doch wenn sie das alles nicht davon abschreckt, weiterzulesen, so muss ich sie noch einmal eindringlich warnen. Nicht nur unserer Protagonisten werden im Laufe der Geschichte mit ihren Dämonen konfrontiert werden. Auch sie werden mit jeder Zeile, jedem Satz und jedem Wort ihre dunkle Seite besser kennenlernen. Sie glauben mir nicht? Dann schauen sie mal, was bisher passiert ist: sie haben bis hier hin alle meine Warnungen, nicht weiterzulesen in den Wind geschlagen. Also was hat sie hier gehalten? Richtig, die Aussicht auf eine spannende Geschichte. Dabei war es ihnen egal ob diese Spannung durch Mord, Totschlag oder sogar Vergewaltigung erreicht wird. Natürlich werden sie sich jetzt alles abstreiten, denn was wären sie denn für ein Mensch, wenn nicht? Sie werden sich sagen, dass das alles doch nur ein Buch sei, pure Fiktion, eine Geschichte, welche sich irgendein kranker Verstand erdacht hat. Sie werden auch erstmal versuchen den Fehler bei mir, dem Autor zu finden. Doch am Ende werden sie leider erkennen müssen, dass sie den kranken Verstand besitzen, Denn während ich nur die nüchternen Fakten wiedergebe und die Geschichte eines jungen Mannes erzähle, sind sie die die das alles interessant verfolgen. Und jetzt stellen sie sich diese eine einfache, verdammte Frage: Wenn sie es schon spannend finden über solch grausame Taten in einem Buch zu lesen, wie weit ist es dann, bis sie auch im realen Leben mit Spannung Nachrichten über Mord und Totschlag konsumieren? Wie weit, bis sie noch weitergehen und sich auf illegalen Untergrund Filmvorstellungen Snuff-Filme reinziehen?
Und wie weit, bis du selbst Hauptdarsteller in so einem Film wirst?
Genau. Vielleicht verstehst du mich jetzt. Und vielleicht fragst du dich, warum ich diese Geschichte überhaupt geschrieben habe, wo ich dich doch grade eben davon abhalten wollte, die zu lesen. Die Antwort auf diese Frage liegt in der Geschichte selbst. Und wenn du dich nach allem doch entschließt die nächste Seite zu lesen, wirst du die Antwort finden. Aber sag nicht, ich hätte dich nicht gewarnt.
Noch kannst du zurück. Noch kannst du das alles abblasen und sagen: "Nein, ich möchte das nicht". Doch sobald du diese nächste Seite anfängst zu lesen, gibt es kein Zurück mehr. Dann wird sich dein Leben für immer verändern und du wirst nie mehr ganz derselbe sein.


Dennoch hast du dich entschieden, weiterzulesen, und ich werde dich nicht aufhalten. Du wirst eine Reise in die Abgründe der menschlichen Seele antreten und dabei erkennen, dass das Böse nicht immer von außen kommt. Vielleicht wirst du am Ende sogar verstehen, warum ich diese Geschichte geschrieben habe. Doch sei gewarnt: Wenn du die letzte Seite erreicht hast, wird nichts mehr sein, wie es einmal war.

Kapitel 1
31. August 1888, Whitechapel, London

Dieser Geruch. Er war es, warum er das was er tat, so liebte. Dieser süße, metallene Geruch von frischem Blut. Aber nicht nur dieser Geruch machte für ihn die Faszination beim Töten aus. Nein, auch das Schreien und das Flehen seiner Opfer versetzte ihn in Ekstase. Diese Macht, die man fühlte, wenn man mit eigenen Händen einem Menschen sein Leben nahm, war unvergleichlich. Ihn überkam jedes Mal ein wahrer Adrenalinrausch. Es war ein ganz besonderes Gefühl, das man in diesem Moment verspürte. Angefangen hatte er mit kleinen Tieren doch nach kurzer Zeit konnten diese seine Mordlust nicht mehr stillen. Also wurden die Tiere immer größer, bis ihm auch das nicht mehr reichte. Er sehnte sich nach mehr. Aber mit dem Alter nahm nicht nur seine Mordlust zu. Auch sein Bedürfnis nach Aufmerksamkeit wuchs. Aus diesem Grund hatte er schon vor Tagen einen Brief per Kurier an die City of London Police gesendet, indem er genau beschrieb, wen er heute ermorden würde. Auch den Ort des Geschehens hatte er verraten. Doch wollte er es den Behörden nicht zu leicht machen. Aus diesem Grund hatte er die Hinweise in einem Gedicht versteckt. Anscheinend hatten sie es nicht gelöst oder nicht mal beachtet. Es als Scherz betrachtet. Dieser Gedanke machte ihn wütend. Doch Wut konnte er nicht gebrauchen, sie lenkte einen ab, machte ihn unvorsichtig. Also schob er sie beiseite. Er beobachtete von seinem Versteck in einer dunklen Gasse aus dem Eingang eines Pubs. Vor dem Laden standen zwei ungepflegte Kreaturen die man nur schwer als Menschen hätte bezeichnen können. Sie waren in seinen Augen eine Schande für die menschliche Rasse. Hätte er die Zeit gehabt hätte er nicht nur die Frau getötet, die gerade aus dem Pub trat, sondern auch diese beiden jämmerlichen Gestalten. Die Frau war eine Prostituierte, so wie so ziemlich jede Frau im Londoner Armenviertel Whitechapel. Der Name der Frau war Mary Ann Nichols. Sie würde niemand vermissen. Das war es jedoch nicht, was sie zu seinem perfekten Opfer machte. Es war vielmehr der Umstand, dass sie rothaarig war. Genau wie seine Mutter, diese Schlampe. Die Frau zog ihren Mantel enger zusammen, um sich gegen die kalte, nach Gosse stinkende Luft des Abends zu schützen. Doch vor dem was dann passierte konnte sie nichts schützen. Er folgte ihr als die Frau sich langsam in Bewegung setzte. Er hielt Abstand, wartete auf den richtigen Moment. Der war gekommen, als die Frau in eine dunkle, ruhige Gasse abbog: der Bucks Row. Noch ahnte sein Opfer nicht, dass sie gleich ihren letzten Atemzug tat. Als Mary Ann ungefähr die Hälfte der Straße schon hinter sich gelassen hatte, entschied er, dass es an der Zeit war. Er schloss zu ihr auf. „Hey! Warten sie kurz.“, rief er ihr hinterher. Sie zuckte leicht zusammen und dieses Zucken rief ein wahres Gefühlsfeuerwerk in ihm hervor. Kurz fürchtete er, sie würde einfach weiter gehen und ihn ignorieren. Doch sie blieb stehen und drehte sich zu ihm um. Das war ein Fehler. Er lief auf sie zu und während sie ihm mit einem Lächeln entgegenblickte, zog er sein Skalpell aus der Tasche und fuhr ihr damit in einer schnellen Bewegung über die Kehle. In ihrem Blick sah er die Panik. Die Erkenntnis, dass sie hier und heute sterben würde. Allein, in einer dunklen Gasse. Er sah ihr die ganze Zeit in die Augen. Er wollte sehen, wie das Leben langsam aus ihrem Körper wich. Während sie langsam zu Boden sank und sich die Kehle hielt, verspürte er ein Gefühl, was er in der Art schon oft gefühlt hatte, aber noch nie in diesem Ausmaß. Als er den leblosen Körper von Mary Ann so vor sich liegen sah wusste er: Das wollte er wieder spüren. Doch erstmal musste er noch diesen Körper herrichten. Er wusste das jeder gute Mörder einen Stil hatte. Ein Erkennungsmerkmal, an dem die Polizei sofort erkennen konnte, wer diese Person getötet hatte. Einen so genannten Modus Operandi. Seiner würde es sein, seinen Opfern den Bauch aufzuschneiden und die Gedärme offenzulegen. Erst später würde er auch anfangen die Gebärmutter seiner Opfer zu entfernen.


Dies war sein erstes Opfer: Mary Ann Nichols. Nachher würden die Medien sie als erste der „kanonischen Fünf“ bezeichnen. Die erste von fünf Frauen die kaltblütig von Leo the Ripper ermordet wurden. Doch spielt dieser eine eher untergeordnete Rolle in dieser Geschichte. Vielmehr geht es um einen Jungen. Den Sohn von Mary Ann Nichols: Leo Nichols. Dieser brach fast unter der Last, die ihm mit dem Mord an seiner Mutter auf einmal auf den Schultern lastete. Doch er entschied sich etwas zu unternehmen. Er vertraute nicht darauf, dass die Polizei den Mörder seiner Mutter fand. Er nahm die Suche selbst in die Hand. Was glaubt ihr? Findet er den, der seine Mutter auf dem Gewissen hatte? Oder wird er sich am Ende in seinen eigenen Obsessionen verlieren und daran zerbrechen? Die Antwort darauf bleibt vorerst ein Geheimnis, doch Leos Suche wird zweifellos eine Geschichte voller Spannung und Gefahren sein.




Kapitel 2

Die Geschichte des besagten Jungen beginnt im gleichen Armenviertel, in dem das Leben von Mary Ann ein so schreckliches Ende fand. Seine Mutter gab ihm den Namen Leo, doch alle nannten ihn nur Raven, weil sein Gesicht immer schwarz vor Ruß war. Er arbeitete für einen alten Schornsteinfeger, der ihn nach dem Tod seiner Mutter aufgenommen hatte. Dieser hatte ihm auch ein Dach über dem Kopf geboten als die Wohnung in Whitechapel, die er zusammen mit seiner Mutter bewohnt hatte, geräumt wurde. Seitdem war Franklin Palmer so etwas wie ein Vater für ihn gewesen. Auch heute kam er völlig verdreckt an dem alten, baufälligen Gebäude an, indem er und seine Freunde ihr Hauptquartier errichtet hatten. Das Gebäude stand leer seit vor Jahren ein Feuer in den unteren Stockwerken ausbrach. Alle hatten ihre Wohnungen verlassen und nur das nötigste mitgenommen. Als das Feuer nach Stunden endlich gelöscht werden konnte bemerkte man, dass es die tragenden Balken angefressen hatte, wodurch das Gebäude als unbewohnbar deklariert wurde. Doch in dem einen Jahr indem sie hier ihr Quartier hatten, hatte es dafür keinerlei Anzeichen gegeben. Und so hatten sie im Laufe der Zeit die einzelnen Wohnungen wieder etwas hergerichtet und mit Möbeln gefüllt, die sie auf der Straße gefunden hatten. Die drei Freunde hatten vor hier einzuziehen, wenn sie älter waren. Doch fürs erste blieb es ein Rückzugs Ort. Leo blickte sich kurz um ob jemand ihn beobachtete und kletterte, dann an der Fassade lang zu einem Fenster im ersten Stock. Hier hatten sie ein paar Bretter gelockert, so dass man sie grade genug zur Seite schieben konnte, um hin durchzuschlüpfen. Im inneren herrschte Dunkelheit, die nur durch die Sonnenstrahlen ein wenig erhellt wurde, die durch die Lücken der Holzbretter an den Fenstern schienen. Leo hörte stimmen die aus der Wohnung über ihm zu kommen schienen. Seine Freunde waren also schon da. Hastig lief er in den Hausflur und rannte dann, immer zwei Treppenstufen auf einmal nehmend, die Treppe zum 2.Stock hoch. Hier war lag das Herzstück ihres Hauptquartiers: Das Nest. Die Tür stand offen und gab den Blick frei auf eine wilde Ansammlung von verschiedensten Möbelstücken aus den verschiedensten Epochen und in den verschiedensten Stilen. Sie hatten in diesem Stockwerk alle Wände und Balken bis auf die Träger rausgerissen und so einen einzigen riesigen Raum geschaffen. In der Mitte des Raumes standen ein paar Stühle, Sessel und eine Couch. Archie saß wie immer auf seinem Lieblingssessel und beschrieb Abbie gerade lebhaft von einer Verfolgungsjagd die er sich mit ein paar Polizisten geliefert hatte. „… und dann kamen auf einmal noch mehr Polzisten aus einer Straße heraus und schnitten mir den weg ab. Ich war gefangen. Vor mir Polizisten und hinter mir Polizisten. Ich dachte mir jetzt sitzt du richtig in der Scheiße. Doch ich wollte natürlich nicht verhaftet werden. Also wägte ich ab mit wieviel Polzisten ich es aufnehmen konnte und schlug mir letztendlich den Weg frei. Danach bin ich sofort hierher und verstecke mich vor den Bullen “, erzählte er gerade als Leo in den Raum trat. „Davon stimmt sicher nicht einmal die Hälfte“, behauptete Leo. „Raven!“ rief Abbie aus und sprang auf, um ihn zur Begrüßung zu umarmen. „Die Geschichte lief genauso ab wie ich sie erzählt habe.“, erklärte Archie und schlug mit Leo ab, „Vielleicht habe ich nur hier und da ein wenig übertrieben“. Leo lachte und setzte sich neben Abbie auf die Couch. „Jetzt da du hier bist muss ich euch etwas erzählen.“, fing Abbie an, „Ich konnte heute Nacht nicht schlafen also bin ich aufgestanden, um mir ein Glas Wasser zu holen, da hörte ich wie mein Dad in der Küche mit jemandem Sprach. Einem Mann. Er war ganz aufgeregt und erzählte etwas von neuen Spuren und das sie endlich die Möglichkeit hätten dieses „Schreckgespenst“, wie er es nannte, zu fangen.“. Abbies Dad war ein Inspektor der City of London Police und war erst vor kurzem zum Leiter der Ermittlungen rund um den Ripper ernannt worden. „Ja, und?“, fragte Leo. Er verstand nicht, worauf Abbie mit ihrer Geschichte hinauswollte. „Ich glaube, sie redeten vom Ripper!“, erklärte sie endlich. Das war es also gewesen. Seit seine Mutter vom besagten Ripper getötet worden war träumte Leo von dem Tag, an dem dieses Monster geschnappt wurde und seine gerechte Strafe bekommen würde. Doch hatte er die Hoffnung schon aufgegeben, weil die Polizei seit mehreren Monaten keinen Anhaltspunkt für die Identität des Mörders hatte finden können. Bis jetzt. Doch dieses Mal würde Leo es nicht der Polizei überlassen den Mörder seiner Mutter zu finden. Er würde ihn selbst fangen und ihm dasselbe antun, was dieser auch seiner Mutter in dieser Schicksalshaften Nacht angetan hatte. „Was für ein Hinweis?“, fragte er daher. Als sowohl Archie als auch Abbie ihn etwas komisch anguckten bemerkte er erst, dass seine Stimme lauter geworden war. „Tut mir leid“, sagte Leo, selbst etwas erschrocken über seine Reaktion. „Ihr wisst doch wie nah mir das geht. Schließlich hat dieser Unmensch meine Mutter ermordet“, erklärte er, „und mit diesem Hinweis könnte ich ihn fangen und seiner gerechten Strafe zuführen.“. Leo hatte seinen Freunden bis jetzt noch nichts von seinem Plan erzählt den Ripper auf eigene Faust zu suchen, jedoch hoffte er dabei auf deren Hilfe. Er hatte sich aber noch nicht dazu durchringen können, Sie zu fragen. Für einen kurzen Moment dachte Leo darüber nach, ihnen einfach alles zu erzählen. Ihnen zu erzählen, dass er schon seit nun mehr 4 Monaten eigene Ermittlungen geführt hatte. Dass er der Polizei, und damit auch Abbies Vater, nicht zutraute den Ripper zu fangen. Doch er entschied sich dagegen. Er wollte nicht, dass sie ihm sagten, wie unvernünftig seine Idee war. Und so entschied er sich erstmal seine Freunde zu belügen und machte einen Rückzieher. „Ist eigentlich ja auch nicht wichtig. Die Polizei kriegt den Ripper bestimmt irgendwann zu fassen. “, sagte er deshalb und setzte sich wieder hin. Seine Freunde merkten aber trotzdem das etwas nicht stimmte. „Archie, vielleicht holst du uns mal etwas Wasser“, schlug Abbie daher vor. Sie wollte mit Leo allein Sprechen, um herauszufinden, was ihn bedrückte. Archie nickte nur kurz, stand auf und machte sich auf den Weg in den Keller, wo sie immer ein paar Eimer mit frischem Wasser lagerten. Als er weg war dreht sich Abbie zu Leo und sah ihm tief in die dunkelblauen Agen. „Was ist denn in letzter Zeit los mit dir? Du warst schon seit Wochen nicht mehr hier, dann lässt du dich endlich mal wieder blicken und bist plötzlich so interessiert an der Arbeit meines Dads“, Leo hielt es nicht länger aus. Er seufzte schwer und erzählte Abbie alles von seinem Plan, den Ripper zu fangen, ihn zu töten und somit seiner Mutter Gerechtigkeit zukommen zu lassen.
Nachdem Leo Abbie das alles gebeichtet hatte, herrschte erstmal für eine lange Zeit Stille. Nach einiger Zeit brach Leo das Schweigen. „Sag doch etwas“, verlangte er. Abbie setzte zu einer Antwort an wurde jedoch von Archie unterbrochen, der gerade mit drei Krügen voller Wasser das Zimmer betrat. „Guckt mal wer da wieder kommt aus den tiefen der Hölle, manchen Leuten besser bekannt unter dem Namen: der Keller.“, verkündete dieser Stolz und setzte die Krüge auf dem Tisch ab. „Ich muss gehen“, sagte Abbie, griff sich hastig ihren Mantel und ihre Tasche und verschwand aus dem Zimmer. Kurze Zeit darauf hörten die beiden Jungen, wie Abbie die Holzbalken im unteren Stockwerk beiseiteschob und sie allein im Hauptquartier zurückließ. „Was war das denn?“, wollte Archie wissen doch auch Leo hatte darauf keine Antwort. „Ich habe keine Ahnung, Archie. Ich habe wirklich keine Ahnung“.


Kapitel 3

Der Mond war schon aufgegangen als Abbie aus dem Hauptquartier stieg. Er tunkte die Straßen Londons in ein unwirkliches Licht. Sie fröstelte. Für einen Sommerabend war es erstaunlich kalt. Aber vielleicht war es ja nicht nur die Kälte, die ihr einen Schauer über den Rücken jagte, überlegte Abbie, während sie die Holzbalken wieder vor das Fenster schob. Vielleicht ließ sie auch die Erinnerung an das Funkeln erschaudern, das Leo in den Augen hatte als er ihr von seinem Plan erzählte. Von einem Moment auf den anderen war aus ihm ein anderer Mensch geworden. Es hatte fast so ausgesehen, als würde er sich freuen bei der Vorstellung, dem Ripper das anzutun, was dieser seiner Mutter angetan hatte. Ein kleiner Teil in Ihr konnte Leo verstehen, schließlich hatte dieses Monster seine Mutter getötet. Doch war das eine Rechtfertigung dafür, das gleiche einem anderen Menschen anzutun? Sie musste Leo wohl oder übel bei seinem Vorhaben helfen, um das zu verhindern. Abbie seufzte schwer und begab sich flotten Schrittes in Richtung ihres Zuhauses. Die Straße, die sie langlief, war noch vor ein paar Stunden mit Menschenmassen gefüllt, die sich auf den Weg von der Arbeit nach Hause machten. Doch zu dieser späten Stunde war sie fast wie ausgestorben. Nur hier und da lagen ein paar jämmerliche Gestalten am Straßenrand, von denen man nicht wusste, ob sie tot oder lebendig waren. Dem Gestank nach, der hier in der Luft hing, waren es nicht gerade wenige, die auch wirklich tot waren. An den Gestank hatte Abbie sich mit der Zeit gewöhnt, doch das Leid, das ihr von jeder Ecke entgegenblickte machte ihr noch immer zu schaffen. Sie konnte nicht verstehen, wie manche Leute in riesigen Villen leben konnten, während hier Leute starben, weil sie ich nichts zu essen leisten konnten. Während Abbie noch über die unfaire Verteilung von Reichtum nachdachte, kam ein dichter Nebel auf. Er kroch wie ein lebendiges Tier aus den Seitenstraßen und streckte seine Fühler nach den wenigen Menschen aus, die hier noch unterwegs waren. Er kroch ihnen unter die Kleidung, brachte sie dazu ihre Mäntel enger zu ziehen und ihren Schritt zu beschleunigen. Auch Abbie war nicht immun gegenüber der Kälte, die dieser Nebel mit sich brachte. Es war auch dieser Nebel, der sie dazu bewegte, die Abkürzung über die Hanbury Street zu nehmen. Jeder, der auch nur ein bisschen abergläubisch war, hätte ihr davon abgeraten diese Straße nach Einbruch der Dunkelheit zu betreten. Die Hanbury Street war seit den Tagen, in denen London aus nicht mehr als ein paar zusammengeschusterten Hütten bestand, ein Ort des Grauens gewesen. Die Leute hatten über die Jahre mehrere Theorien aufgestellt, warum hier immer wieder so schreckliche Sachen passierten. Die einen glaubten, dass der Teufel selbst in dieser Straße wohnte. Die anderen dachten es sei eine Straße, die direkt in die Hölle führen würde. Abbie war aber nicht abergläubisch. Im Gegenteil hielt sie sich für eine sehr rational denkende Person. Und doch dachte sie über alle diese Theorien nach, während sie so ganz allein diese Straße entlanglief. Was, wenn an diesen Geschichten doch was dran war? Mach dich nicht lächerlich, sagte sie sich und ging weiter. So etwas wie eine Hölle existierte nicht. Auch der Teufel tat das nicht. Und doch konnte sie sich den Fakten gegenüber nicht verschließen. Es wurden hier in dieser Straße ungewöhnlich viele Gräueltaten begangen. Doch war das noch kein Beweis für eine übernatürlich Ursache der Ereignisse. Abbie schüttelte sich und schlug den Kragen ihres Mantels hoch um sich vor der immer unangenehmer werdenden Kälte zu Schützen. Der Nebel hatte mittlerweile ganz Whitechapel verschluckt und ließ einen nur erahnen, was vor einem lag. Sie hatte keine Ahnung, ob sie die Abzweigung, zu der Straße, in der ihr Haus lag, nicht schon verpasst hatte. Sie blieb stehen und sah sich um. Wohin sie auch blickte, sah sie nur eine undurchdringliche Nebelwand. Dieser Nebel hatte etwas Bedrohliches an sich. So als würde sich etwas in ihm verstecken. Etwas Böses. Abbie hatte plötzlich das Gefühl, als würde jemand ihre Luftröhre zudrücken. Sie viel auf die Knie und schnappte nach Luft. Ihr traten Tränen in die Augen, während ihr Kopf noch zu verarbeiten versuchte, was gerade mit ihr geschah. Ihr Körper kippte nach vorne. Sie hörte ein helles Lachen, wie das eines Kindes. Stimmen, die aus dem Nebel zu kommen schienen. Doch dann auch einen Schrei. Erst einer und dann immer mehr, bis es so viele waren, dass man sie unmöglich voneinander hätte unterscheiden können. Abbies Bewegungen wurden durch den Sauerstoffmangel immer träger und ihr Sichtfeld immer eingeschränkter. Ruckartig hörten die Schreie auf und der Nebel lichtete sich um den Blick auf einen großen, dunklen Schatten freizugeben. Abbie konnte zwar nicht viel erkennen, doch war sie sich fast sicher, dass der Schatten nicht menschlich war. Sie sah noch, wie der Schatten sich auf sie zu bewegte und dabei immer größer zu werden schien, bevor sie endgültig das Bewusstsein verlor und in ein dunkles Nichts fiel.



„Ok, ich verschwinde“. Leo griff sich seinen Mantel und verabschiedete sich mit ihrem geheimen Handschlag von Archie. Nach dem Abbie so plötzlich aufgebrochen war, hatten er und Archie noch ein paar Runden Karten gespielt, während Archie weitere wilde Geschichten darüber erzählt hatte, wie er vor Polizisten weggerannt war, wie er gegen gestandene Männer kämpfte, die mindestens drei Köpfe größer gewesen waren als er, und gewonnen hatte oder wie er die schönsten Frauen aus dem gesamten Empire küssen durfte, weil er sie vor mordlustigen Räubern rettete. Leo hatte ihm währenddessen nur mit halbem Ohr zugehört. Seine Gedanken drehten sich immer noch um Abbie und den Ausdruck in ihren Augen, als er ihr von seinem Plan erzählt hatte. Sie hatte ängstlich ausgesehen. Doch was hatte ihr solche Angst bereitet? War er es gewesen? Die Worte die er gesprochen hatte? Oder die Idee das Leo jemanden umbringen wollte? Leo konnte die dunklen Gedanken nicht abschütteln, während er aus dem Hauptquartier trat und sich auf den Weg nach Hause machte. Es war schon dunkel und es herrschte eine feuchte Kälte, weshalb er seinen Mantel enger um sich zog und den Kragen hochschlug. Leo wusste, dass er Abbie beruhigen musste, aber er wusste nicht, wie er das anstellen sollte. Er war sich nicht einmal sicher, ob er seinen Plan wirklich umsetzen würde. Aber er wusste, dass er etwas unternehmen musste. Die Stadt war voll von Verbrechern und Kriminellen, und die Polizei schien machtlos zu sein. Der Ripper war nur einer von vielen. Jemand musste handeln, und er hatte das Gefühl, dass er derjenige sein sollte. Als er sich nach kurzer Zeit seinem Haus näherte, bemerkte er, dass etwas nicht stimmte. Die Tür stand halb offen, und es schien, als ob sich jemand unerlaubt Zugriff verschafft hatte. Sein Herz schlug schneller, als er vorsichtig näher trat und durch die Tür spähte. Er konnte nichts Ungewöhnliches sehen, aber er spürte, dass jemand in seinem Haus gewesen war. Etwas war anders doch sein Verstand konnte nicht wirklich greifen, was es war. Vorsichtig durchsuchte Leo das Haus, aber es schien, als ob nichts gestohlen worden war. Franklin konnte es nicht gewesen sein. Erstens war er heute im Pub mit Freunden verabredet und würde erst später am Abend nach Hause kommen, und zweitens hätte er die Tür nicht offenstehen lassen. Vielleicht hatte er sich auch nur getäuscht. Aber das Gefühl der Unsicherheit blieb, als er sich auf sein Bett legte und versuchte zu einzuschlafen. Er konnte sich nicht von dem Gedanken lösen, dass jemand ihn beobachtete, dass er verfolgt wurde. Und er fragte sich, ob es etwas mit dem Mord an seiner Mutter zu tun hatte. War er es, der ihn beobachtete? War er in Gefahr? War Abbie in Gefahr? Leo wusste, dass er Antworten auf diese Fragen finden musste, und er beschloss, am nächsten Tag mit Abbie zu sprechen. Er würde ihr alles erklären und hoffen, dass sie ihm helfen würde. Das würde zwar nicht leicht werden, aber er konnte nicht allein handeln. Er brauchte ihre Hilfe.

Kapitel 4
Das erste was Leo bemerkte, als er aufwachte, war der Gestank. Es stank schlimmer als die Themse im Sommer. Er schlug die Augen auf. Das war nicht sein Zimmer. Leo lag in einer dunklen, verlassenen Gasse auf einem Stapel Kisten und Paletten. Es musste ziemlich früh am Morgen sein. Es war noch dunkel. Sein Rücken schmerzte, was er wohl seinem fragwürdigen Schlafplatz zu verdanken hatte. Er stand auf und sah sich um. Viel konnte er zwar nicht erkennen, aber doch so viel, dass er sicher sagen konnte, noch nie hier gewesen zu sein. Alles an dieser Straße schien dreckig zu sein. Von den Häusern über die Straße selbst bis hin zu dem Wind der durch sie fegte. Der Wind trug aber nicht nur Dreck in die Gasse, er brachte auch einen Geruch mit, der sich in Leos Nase zu ätzen schien: Der Geruch von Blut. Genau in dem Moment als sein Gehirn den strengen Geruch identifiziert hatte, hörte er einen markerschütternden Schrei. Leo zögerte keine Sekunde und rannte in die Richtung, aus der der Schrei gekommen war. Erst jetzt bemerkte er, dass er keine Schuhe trug. Das Klatschen seiner nackten Füße auf dem Kopfsteinpflaster hallte von den Häuserwänden wieder und machte Leo die Stille bewusst die herrschte, seit er in der Gasse aufgewacht war. Er blieb stehen. Etwas stimmte nicht. In London war es nie ganz ruhig. Auch nicht so früh am Morgen. Jetzt wo ihm das aufgefallen war, fielen ihm noch andere Sachen auf, stellten sich ihm noch andere Fragen. Zum Beispiel, wie kam er überhaupt hierher? Wo war „hier“ eigentlich genau? Und die wichtigste aller Fragen, wer hatte geschrien? Es klang nach einer Frau, aber da war Leo sich nicht sicher. Wo er sich allerdings sicher war, war der Umstand das die Person seine Hilfe bräuchte. Er setzte sich wieder in Bewegung und rannte die Straße runter. Schnell merkte er ein stechen in seiner rechten Seite und nicht zum ersten mal wünschte er sich, er hätte in letzter Zeit mehr auf seine Kondition geachtet.

            """

            text_to_speech2(t, 0)

        if c == 3:
            user_text = input("User: ")
            response = compute_response(user_text, functions, app)
            print("Isaa:", response, "#2")

        if c == 3.5:
            user_text = input("User: ")
            response = functions['ia'](user_text, app, 0)['output']
            print("Isaa:", response, "#2")

        if c == 4:

            print("User:", user_text)
            print("Isaa:", response)

            if not awake:
                print("Test Isaa awake call")
                block_till_isaa_awake()
                awake = True

            user_text = get_audio_transcribe()

            print("User:", user_text)
            if " " == user_text:
                user_text = "Auf wieder sehen und bis bald ich habe dich in den ruhe modus versetzt."
                awake = False

            if "Ruhemodus" in user_text.lower():
                user_text += " Gute nacht und bis Bald "
                awake = False

            if "exit" in user_text.lower():
                app.alive = False

            try:
                response = "Gute Nacht"
                if awake:
                    response = functions['ia'](user_text, app, 0)
                    response = response['output']
            except Exception as e:
                response = str(e)

            print(response)

            if "-sleep" in response or 'Ruhemodus' in response:
                user_text += " Gute nacht und bis Bald "
                awake = False

            if ("Wiedersehen" in response or 'bis dann' in response) and (not 'Ruhemodus' in response):
                app.alive = False

            text_to_speech3(response.replace("Isaa:", "").split("[Tool")[0])

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

# x = "Gedanke: Mit Gedanke können innere Prozesse simuliert werden.\n\nVerfügbare Prozesse:\nTool 0: Mit requests
# können Sie mit Webseiten kommunizieren und mit \"terminal\" mit Ihrem Terminal interagieren.\nTool 1: Mit
# \"Python\" können Sie mit dem Python Interpreter interagieren.\nTool 2: Mit \"WindowsCommandLineInterface\" können
# Sie mit Ihrem Computer interagieren und mit \"Toolbox\" mit der Toolbox interagieren.\nTool 3: Mit \"pal-math\" und
# \"llm-math\" können Sie komplexe mathematische Aufgaben lösen, mit \"open-meteo-api\" können Sie Wetterdaten
# abrufen und \"pal-colored-objects\" muss im Internet gesucht werden.\nTool 4: Mit \"wolfram-alpha\" können Sie
# konkrete, richtige Informationen finden, mit \"serpap\" aktuelle Informationen finden und mit \"searx-search\" eine
# weitere Suchmaschine verwenden.\n\nUm ein Tool aufzurufen, geben Sie \"tool n\" ein, wobei n für die Nummer des
# Tools steht.\n\nBeispiel:\n\nUser: Hole dir webseiten informationen oder benutze das terminal | \nIsaa: Okay,
# ich öffne Steam für Sie.\n[Tool0]: Hole dir webseiten informationen oder benutze das terminal\nGedanke: Isaa
# betätigt Tool 0 um die aufgabe zu bewerkstelligen. Bitte beachten Sie, dass die Tools nur in englisch
# kommunizieren!!User: Schreibe eine python datei die beim start des pc chrome öffnet\nIsaa: Okay, Ich schreibe das
# Python script für sie das chrome beim auto start öffnet für Sie.\n[Tool1]: Write an Pyton script witch open chrome
# on computer start. find chrome location first\nGedanke: Isaa betätigt Tool 1 um die aufgabe zu bewerkstelligen.
# Bitte beachten Sie, dass die Tools nur in englisch kommunizieren!!User: Öffne Steam\nIsaa: Okay, ich öffne Steam
# für Sie.\n[Tool2]: Open Steam on the lokalen Computer\nGedanke: Isaa betätigt Tool 2 um Steam zu öffnen. Bitte
# beachten Sie, dass die Tools nur in englisch kommunizieren!!User: Was sind 10 kilometer in Bananen,
# wei viel Bananen aneinader greit sind 10 Kliometer lang\nIsaa: Ich beantworte deine frage, dafür muss ich die
# standard länge einer Bananen berechnen und berechenene wie viele Bananen an einander 10 kliometer lang sind.\n[
# Tool3]: How many Bannanas ar 10 Km long\nUser: Welcher Tag ist heute.\nIsaa: Okay, ich suche die antwort im Web für
# Sie.\n[Tool4]: What is the current day?\n\n\nIsaa: Willkommen im Raum! Ich bin Isaa, Ihr intelligenter
# Sprachassistent. Ich wurde entwickelt, um Ihnen bei der Planung und Umsetzung von Projekten zu helfen. Ich kann
# komplexe Aufgaben bewältigen, Informationen organisieren und Ihnen bei der Entscheidungsfindung helfen,
# indem ich eine natürliche Konversation führe. Mein Ziel ist es, Ihnen bei der Durchführung von Projekten zu helfen,
# die der Gesellschaft einen Mehrwert bieten.\n\nIsaa: Willkommen! Ich bin Isaa, Ihr intelligenter Sprachassistent.
# Ich bin hier, um Ihnen bei der Planung und Umsetzung von Projekten zu helfen. Mit mir können Sie Ihre verschiedenen
# Tools einfach verwalten und steuern  Egal, ob Sie ein individuelles oder ein Gemeinschaftsprojekt durchführen,
# ich stehe Ihnen zur Seite, um Sie zu unterstützen. Sagen Sie mir einfach, wie ich Ihnen helfen kann!\n\nSpeicher:{
# history}\n\nMe: {input}",
