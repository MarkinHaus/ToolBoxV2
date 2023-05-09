"""Console script for toolboxv2. Isaa CMD Tool"""
import sys
from datetime import datetime

from langchain.schema import Document
from transformers import pipeline

from toolboxv2.mods.isaa_audio import get_audio_transcribe
from toolboxv2.mods.isaa import AgentConfig, CollectiveMemory, AgentChain
from toolboxv2.mods.isaa import Tools as Isaa
from toolboxv2.utils.Style import print_to_console, Style
from toolboxv2.utils.isaa_util import sys_print, speak, run_agent_cmd, init_isaa

NAME = "isaa-cmd"


def run(app, args):
    speak_mode = args.speak

    isaa, self_agent_config = init_isaa(app, speak_mode=speak_mode, calendar=False, ide=False, create=False)

    with open("E:/Markin/D/project_py/ToolBoxV2/toolboxv2/data/isaa_data/work/sum.data", 'r') as f:
        test = f.read()

    context_memory = isaa.get_context_memory()

    name = 'five-w-summarizing'

    # context_memory.split_text(name, test, 12)
    # context_memory.add_data(name)

    context_memory.init_store(name)
    context_memory.crate_live_context(name, num_clusters=5)
    summarysation = pipeline("summarization")



    user = "Diese Funktion verwendet die os-Bibliothek in Python, um alle Ordner auf derselben Ebene in einem Ordner zu listen. Die Funktion nimmt einen Parameter path, der den Pfad zum Ordner enthält, dessen Ordner auf derselben Ebene aufgelistet werden sollen. Die Funktion gibt eine Liste aller Ordner auf derselben Ebene zurück."

    while True:
        ress = context_memory.get_context_for(user, marginal=True)

        def summary(x):
            return summarysation(x, )

        last = []

        final = ''
        for res in ress:
            if last != res[1]:
                print(res[1],'--------------------')
                print_to_console("full-text:", Style.style_dic['GREEN'], res[0].page_content, max_typing_speed=0.01, min_typing_speed=0.03)
                print_to_console("Summary:", Style.style_dic['CYAN'],  summary("Related to :"+user+" "+res[0].page_content)[0]['summary_text'], max_typing_speed=0.01, min_typing_speed=0.03)
                print_to_console("Summary:", Style.style_dic['YELLOW'], isaa.mas_text_summaries("Related to :"+user+" "+res[0].page_content), max_typing_speed=0.01, min_typing_speed=0.03)
                last = res[1]
                final += res[0].page_content + '\n\n'
                print("----------------\n\n")
            else:
                print("WARNING- same")

        task = f"Act as an summary expert your specialties are writing summary. you are known to think in small and detailed steps to get the right result. Your task : write a summary reladet to {user}\n\n{final}"
        config = isaa.get_agent_config_class('think').set_model_name('gpt-3.5-turbo')
        res = isaa.run_agent('think', task)
        print_to_console("Summary:", Style.style_dic['BLACK'],
                         res,
                         max_typing_speed=0.01, min_typing_speed=0.03)

        user = input(":")

    exit(0)

    sys_print("Welcome")

    sys_print("\n================================Starting-Agent================================")

    self_agent_config.model_name = "gpt-3.5-turbo"

    self_agent_config.set_mode('execution')
    self_agent_config.completion_mode = "text"

    res = isaa.run_agent(self_agent_config, f"Rerad teh database.py file in the work dir using file_functions")
    print(res)
