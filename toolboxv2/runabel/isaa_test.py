"""Console script for toolboxv2. Isaa CMD Tool"""
import sys
from datetime import datetime

from toolboxv2.mods.isaa_audio import Tools as Audio
from toolboxv2.mods.isaa import AgentChain
from toolboxv2.utils.isaa_util import sys_print, speak, run_agent_cmd, init_isaa

NAME = "isaa-test"


def run(app, args):
    speak_mode = args.speak
    audio = Audio(app)

    #isaa, self_agent_config = init_isaa(app, speak_mode=speak_mode, calendar=True, ide=False, create=False)
    #chain_instance = AgentChain()
    #response = ""
    #audio.speech_stream("Hello World")
    speak("Welcome")
