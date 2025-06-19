# stema/tts.py
# This file remains largely unchanged as it's a utility for spoken feedback.
# Ensure gTTS and simpleaudio/sounddevice are in requirements if used extensively.
# For this refactor, TTS feedback is made optional and defaults to False in config.
# If you decide to keep it, ensure it's robust.

import os
import threading

try:
    from gtts import gTTS
    import sounddevice as sd  # Using sounddevice as it's in requirements
    import soundfile as sf  # To read the MP3 gTTS saves for sounddevice

    # simpleaudio can be an alternative if preferred and installed
    # import simpleaudio as sa
    _tts_available = True
except ImportError:
    _tts_available = False
    # print("Warning: gTTS or sounddevice/soundfile not installed. TTS feedback disabled.")


def speak(text: str, lang: str = 'en', blocking: bool = False) -> None:
    """
    Speaks the given text using gTTS and sounddevice.
    Runs in a separate thread by default to not block main operations.
    """
    if not _tts_available:
        # print(f"TTS (speak): {text}") # Print to console if TTS libs not found
        return

    def _speak_thread():
        try:
            tts = gTTS(text=text, lang=lang)
            filename = "stema_feedback.mp3"  # Use a fixed name or temp file
            tts.save(filename)

            # Use soundfile to read mp3 (or convert to wav first if sounddevice needs it)
            # sounddevice typically plays numpy arrays from wav or similar.
            data, samplerate = sf.read(filename, dtype='float32')
            sd.play(data, samplerate)
            sd.wait()  # Wait for playback to finish in this thread

            os.remove(filename)
        except Exception as e:
            print(f"TTS Error: {e}")

    if blocking:
        _speak_thread()
    else:
        thread = threading.Thread(target=_speak_thread)
        thread.daemon = True  # Allows main program to exit even if thread is running
        thread.start()


if __name__ == '__main__':
    # Test TTS
    print("Testing TTS... (ensure speakers are on)")
    speak("Hello from STEMA. Text to speech is working.", blocking=True)
    speak("This is a non-blocking message.")
    print("Main thread continues while non-blocking message may be playing...")
    time.sleep(5)  # Give time for non-blocking message to play
    print("TTS test finished.")
