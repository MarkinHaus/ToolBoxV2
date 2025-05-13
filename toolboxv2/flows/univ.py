import threading
import time

from toolboxv2 import App, AppArgs

NAME = "vu"
llm_text = [""]


def run(app: App, _: AppArgs):
    from toolboxv2.mods.isaa import Tools

    isaa: Tools = app.get_mod('isaa')

    agent = isaa.init_isaa(name='self', build=True)
    agent.verbose = True

    put, res_que = app.get_function(("audio", "transcript"))[0](rate=16000, chunk_duration=4.0, amplitude_min=0)

    def process_data(d):
        print(d)

    def voice_dump():

        put("start")

        time.sleep(6.75)
        print("Start recording")
        transcription = ""
        # Get transcribed results
        while True:
            x = res_que.get()
            transcription += x
            print(f"Transcribed text: {x}")
            time.sleep(4.75)
            if '\n' in transcription or '.' in transcription or len(transcription) > 250:
                process_data(transcription)
                transcription = ""
        # Stop the transcription process

    # Run UI in separate thread
    ui_thread = threading.Thread(target=voice_dump, daemon=True)
    ui_thread.start()

    # Keep main thread alive
    app.idle()

    put("stop")
    # Cleanup
    put("exit")
    app.exit()
    print("\nExiting...")
