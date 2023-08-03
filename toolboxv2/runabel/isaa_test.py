"""Console script for toolboxv2. Isaa CMD Tool"""
import json
import re

from langchain.agents import load_tools, get_all_tool_names

from toolboxv2 import Style, get_logger
from toolboxv2.mods.isaa import IsaaQuestionBinaryTree, AgentConfig
from toolboxv2.mods.isaa_extars.AgentUtils import Task
from toolboxv2.mods.isaa_extars.isaa_modi import init_isaa, split_todo_list, generate_exi_dict, \
    run_chain_in_cmd, idea_enhancer

NAME = "isaa-test"


def run(app, args):
    import gpt4all

    printgpt4allGPT4Alllist_models = [
        {
            'order': 'a',
            'md5sum': 'e8d47924f433bd561cb5244557147793',
            'name': 'Wizard v1.1',
            'filename': 'wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin',
            'filesize': '7323310848',
            'ramrequired': '16',
            'parameters': '13 billion',
            'quant': 'q4_0',
            'type': 'LLaMA',
            'systemPrompt': ' ',
            'description': '<strong>Best overall model</strong><br><ul><li>Instruction based<li>Gives very long responses<li>Finetuned with only 1k of high-quality data<li>Trained by Microsoft and Peking University<li>Cannot be used commercially</ul'
        },
        {
            'order': 'b',
            'md5sum': '725f148218a65ce8ebcc724e52f31b49',
            'name': 'GPT4All Falcon',
            'filename': 'ggml-model-gpt4all-falcon-q4_0.bin',
            'filesize': '4061641216',
            'requires': '2.4.9',
            'ramrequired': '8',
            'parameters': '7 billion',
            'quant': 'q4_0',
            'type': 'Falcon',
            'systemPrompt': ' ',
            'description': '<strong>Best overall smaller model</strong><br><ul><li>Fast responses</li><li>Instruction based</li><li>Trained by TII<li>Finetuned by Nomic AI<li>Licensed for commercial use</ul>',
            'url': 'https://huggingface.co/nomic-ai/gpt4all-falcon-ggml/resolve/main/ggml-model-gpt4all-falcon-q4_0.bin',
            'promptTemplate': '### Instruction:\n%1\n### Response:\n'
        },
        {
            'order': 'c',
            'md5sum': '4acc146dd43eb02845c233c29289c7c5',
            'name': 'Hermes',
            'filename': 'nous-hermes-13b.ggmlv3.q4_0.bin',
            'filesize': '8136777088',
            'requires': '2.4.7',
            'ramrequired': '16',
            'parameters': '13 billion',
            'quant': 'q4_0',
            'type': 'LLaMA',
            'systemPrompt': ' ',
            'description': '<strong>Extremely good model</strong><br><ul><li>Instruction based<li>Gives long responses<li>Curated with 300,000 uncensored instructions<li>Trained by Nous Research<li>Cannot be used commercially</ul>',
            'url': 'https://huggingface.co/TheBloke/Nous-Hermes-13B-GGML/resolve/main/nous-hermes-13b.ggmlv3.q4_0.bin',
            'promptTemplate': '### Instruction:\n%1\n### Response:\n'
        },
        {
            'order': 'e',
            'md5sum': '81a09a0ddf89690372fc296ff7f625af',
            'name': 'Groovy',
            'filename': 'ggml-gpt4all-j-v1.3-groovy.bin',
            'filesize': '3785248281',
            'ramrequired': '8',
            'parameters': '7 billion',
            'quant': 'q4_0',
            'type': 'GPT-J',
            'systemPrompt': ' ',
            'description': '<strong>Creative model can be used for commercial purposes</strong><br><ul><li>Fast responses<li>Creative responses</li><li>Instruction based</li><li>Trained by Nomic AI<li>Licensed for commercial use</ul>'
        },
        {
            'order': 'f',
            'md5sum': '11d9f060ca24575a2c303bdc39952486',
            'name': 'Snoozy',
            'filename': 'GPT4All-13B-snoozy.ggmlv3.q4_0.bin',
            'filesize': '8136770688',
            'requires': '2.4.7',
            'ramrequired': '16',
            'parameters': '13 billion',
            'quant': 'q4_0',
            'type': 'LLaMA',
            'systemPrompt': ' ',
            'description': '<strong>Very good overall model</strong><br><ul><li>Instruction based<li>Based on the same dataset as Groovy<li>Slower than Groovy, with higher quality responses<li>Trained by Nomic AI<li>Cannot be used commercially</ul>',
            'url': 'https://huggingface.co/TheBloke/GPT4All-13B-snoozy-GGML/resolve/main/GPT4All-13B-snoozy.ggmlv3.q4_0.bin'
        },
        {
            'order': 'g',
            'md5sum': '756249d3d6abe23bde3b1ae272628640',
            'name': 'MPT Chat',
            'filename': 'ggml-mpt-7b-chat.bin',
            'filesize': '4854401050',
            'requires': '2.4.1',
            'ramrequired': '8',
            'parameters': '7 billion',
            'quant': 'q4_0',
            'type': 'MPT',
            'description': '<strong>Best overall smaller model</strong><br><ul><li>Fast responses<li>Chat based<li>Trained by Mosaic ML<li>Cannot be used commercially</ul>',
            'promptTemplate': '<|im_start|>user\n%1<|im_end|><|im_start|>assistant\n',
            'systemPrompt': '<|im_start|>system\n- You are a helpful assistant chatbot trained by MosaicML.\n- You answer questions.\n- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.\n- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.<|im_end|>'
        },
        {
            'order': 'h',
            'md5sum': 'e64e74375ce9d36a3d0af3db1523fd0a',
            'name': 'Mini Orca',
            'filename': 'orca-mini-7b.ggmlv3.q4_0.bin',
            'filesize': '3791749248',
            'requires': '2.4.7',
            'ramrequired': '8',
            'parameters': '7 billion',
            'quant': 'q4_0',
            'type': 'OpenLLaMa',
            'description': '<strong>New model with novel dataset</strong><br><ul><li>Instruction based<li>Explain tuned datasets<li>Orca Research Paper dataset construction approaches<li>Licensed for commercial use</ul>',
            'url': 'https://huggingface.co/TheBloke/orca_mini_7B-GGML/resolve/main/orca-mini-7b.ggmlv3.q4_0.bin',
            'promptTemplate': '### User:\n%1\n### Response:\n',
            'systemPrompt': '### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n'
        },
        {
            'order': 'i',
            'md5sum': '6a087f7f4598fad0bb70e6cb4023645e',
            'name': 'Mini Orca (Small)',
            'filename': 'orca-mini-3b.ggmlv3.q4_0.bin',
            'filesize': '1928446208',
            'requires': '2.4.7',
            'ramrequired': '4',
            'parameters': '3 billion',
            'quant': 'q4_0',
            'type': 'OpenLLaMa',
            'description': '<strong>Small version of new model with novel dataset</strong><br><ul><li>Instruction based<li>Explain tuned datasets<li>Orca Research Paper dataset construction approaches<li>Licensed for commercial use</ul>',
            'url': 'https://huggingface.co/TheBloke/orca_mini_3B-GGML/resolve/main/orca-mini-3b.ggmlv3.q4_0.bin',
            'promptTemplate': '### User:\n%1\n### Response:\n',
            'systemPrompt': '### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n'
        },
        {
            'order': 'j',
            'md5sum': '959b7f65b2d12fd1e3ff99e7493c7a3a',
            'name': 'Mini Orca (Large)',
            'filename': 'orca-mini-13b.ggmlv3.q4_0.bin',
            'filesize': '7323329152',
            'requires': '2.4.7',
            'ramrequired': '16',
            'parameters': '13 billion',
            'quant': 'q4_0',
            'type': 'OpenLLaMa',
            'description': '<strong>Largest version of new model with novel dataset</strong><br><ul><li>Instruction based<li>Explain tuned datasets<li>Orca Research Paper dataset construction approaches<li>Licensed for commercial use</ul>',
            'url': 'https://huggingface.co/TheBloke/orca_mini_13B-GGML/resolve/main/orca-mini-13b.ggmlv3.q4_0.bin',
            'promptTemplate': '### User:\n%1\n### Response:\n',
            'systemPrompt': '### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n'
        },
        {
            'order': 'k',
            'md5sum': '29119f8fa11712704c6b22ac5ab792ea',
            'name': 'Vicuna',
            'filename': 'ggml-vicuna-7b-1.1-q4_2.bin',
            'filesize': '4212859520',
            'ramrequired': '8',
            'parameters': '7 billion',
            'quant': 'q4_2',
            'type': 'LLaMA',
            'systemPrompt': ' ',
            'description': '<strong>Good small model - trained by teams from UC Berkeley, CMU, Stanford, MBZUAI, and UC San Diego</strong><br><ul><li>Instruction based<li>Cannot be used commercially</ul>'
        },
        {
            'order': 'l',
            'md5sum': '95999b7b0699e2070af63bf5d34101a8',
            'name': 'Vicuna (large)',
            'filename': 'ggml-vicuna-13b-1.1-q4_2.bin',
            'filesize': '8136770688',
            'ramrequired': '16',
            'parameters': '13 billion',
            'quant': 'q4_2',
            'type': 'LLaMA',
            'systemPrompt': ' ',
            'description': '<strong>Good larger model - trained by teams from UC Berkeley, CMU, Stanford, MBZUAI, and UC San Diego</strong><br><ul><li>Instruction based<li>Cannot be used commercially</ul>'
        },
        {
            'order': 'm',
            'md5sum': '99e6d129745a3f1fb1121abed747b05a',
            'name': 'Wizard',
            'filename': 'ggml-wizardLM-7B.q4_2.bin',
            'filesize': '4212864640',
            'ramrequired': '8',
            'parameters': '7 billion',
            'quant': 'q4_2',
            'type': 'LLaMA',
            'systemPrompt': ' ',
            'description': '<strong>Good small model - trained by by Microsoft and Peking University</strong><br><ul><li>Instruction based<li>Cannot be used commercially</ul>'
        },
        {
            'order': 'n',
            'md5sum': '6cb4ee297537c9133bddab9692879de0',
            'name': 'Stable Vicuna',
            'filename': 'ggml-stable-vicuna-13B.q4_2.bin',
            'filesize': '8136777088',
            'ramrequired': '16',
            'parameters': '13 billion',
            'quant': 'q4_2',
            'type': 'LLaMA',
            'description': '<strong>Trained with RLHF by Stability AI</strong><br><ul><li>Instruction based<li>Cannot be used commercially</ul>',
            'systemPrompt': '## Assistant: I am StableVicuna, a large language model created by CarperAI. I am here to chat!\n\n'
        },
        {
            'order': 'o',
            'md5sum': '1cfa4958f489f0a0d1ffdf6b37322809',
            'name': 'MPT Instruct',
            'filename': 'ggml-mpt-7b-instruct.bin',
            'filesize': '4854401028',
            'requires': '2.4.1',
            'ramrequired': '8',
            'parameters': '7 billion',
            'quant': 'q4_0',
            'type': 'MPT',
            'systemPrompt': ' ',
            'description': "<strong>Mosaic's instruction model</strong><br><ul><li>Instruction based<li>Trained by Mosaic ML<li>Licensed for commercial use</ul>"
        },
        {
            'order': 'p',
            'md5sum': '120c32a51d020066288df045ef5d52b9',
            'name': 'MPT Base',
            'filename': 'ggml-mpt-7b-base.bin',
            'filesize': '4854401028',
            'requires': '2.4.1',
            'ramrequired': '8',
            'parameters': '7 billion',
            'quant': 'q4_0',
            'type': 'MPT',
            'systemPrompt': ' ',
            'description': '<strong>Trained for text completion with no assistant finetuning</strong><br><ul><li>Completion based<li>Trained by Mosaic ML<li>Licensed for commercial use</ul>'
        },
        {
            'order': 'q',
            'md5sum': 'd5eafd5b0bd0d615cfd5fd763f642dfe',
            'name': 'Nous Vicuna',
            'filename': 'ggml-nous-gpt4-vicuna-13b.bin',
            'filesize': '8136777088',
            'ramrequired': '16',
            'parameters': '13 billion',
            'quant': 'q4_0',
            'type': 'LLaMA',
            'systemPrompt': ' ',
            'description': '<strong>Trained on ~180,000 instructions</strong><br><ul><li>Instruction based<li>Trained by Nous Research<li>Cannot be used commercially</ul>'
        },
        {
            'order': 'r',
            'md5sum': '489d21fd48840dcb31e5f92f453f3a20',
            'name': 'Wizard Uncensored',
            'filename': 'wizardLM-13B-Uncensored.ggmlv3.q4_0.bin',
            'filesize': '8136777088',
            'requires': '2.4.7',
            'ramrequired': '16',
            'parameters': '13 billion',
            'quant': 'q4_0',
            'type': 'LLaMA',
            'systemPrompt': ' ',
            'description': '<strong>Trained on uncensored assistant data and instruction data</strong><br><ul><li>Instruction based<li>Cannot be used commercially</ul>',
            'url': 'https://huggingface.co/TheBloke/WizardLM-13B-Uncensored-GGML/resolve/main/wizardLM-13B-Uncensored.ggmlv3.q4_0.bin'
        },
        {
            'order': 's',
            'md5sum': '615890cb571fcaa0f70b2f8d15ef809e',
            'disableGUI': 'true',
            'name': 'Replit',
            'filename': 'ggml-replit-code-v1-3b.bin',
            'filesize': '5202046853',
            'requires': '2.4.7',
            'ramrequired': '4',
            'parameters': '3 billion',
            'quant': 'f16',
            'type': 'Replit',
            'systemPrompt': ' ',
            'description': '<strong>Trained on subset of the Stack</strong><br><ul><li>Code completion based<li>Licensed for commercial use</ul>',
            'url': 'https://huggingface.co/nomic-ai/ggml-replit-code-v1-3b/resolve/main/ggml-replit-code-v1-3b.bin'
        },
        {
            'order': 't',
            'md5sum': '031bb5d5722c08d13e3e8eaf55c37391',
            'disableGUI': 'true',
            'name': 'Bert',
            'filename': 'ggml-all-MiniLM-L6-v2-f16.bin',
            'filesize': '45521167',
            'requires': '2.4.14',
            'ramrequired': '1',
            'parameters': '1 million',
            'quant': 'f16',
            'type': 'Bert',
            'systemPrompt': ' ',
            'description': '<strong>Sbert</strong><br><ul><li>For embeddings'
        }
    ]
    isaa, self_agent_config, chains = init_isaa(app, speak_mode=args.speak, ide=False, create=False,
                                                python_test=False, init_mem=True, init_pipe=True,
                                                join_now=False, chain_runner=True)

    isaa.global_stream_override = True
    isaa.summarization_mode = 2

    if 'augment' in isaa.config.keys() and False:
        print("init augment")
        isaa.init_from_augment(isaa.config['augment'], exclude=['messages_sto'])
    else:

        tools = {
            "lagChinTools": ['python_repl', 'requests_all',
                             'terminal', 'sleep', 'google-search', 'ddg-search', 'wikipedia',
                             'llm-math',
                             ],
            "huggingTools": [],
            "Plugins": [],
            "Custom": [],
        }
        isaa.init_tools(self_agent_config, tools)

    planing_steps = {
        "name": "First-Analysis",
        "tasks": [
            {
                "use": "tool",
                "name": "memory",
                "args": "$user-input",
                "return": "$D-Memory"
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Erstelle Eine Prompt für die Analyse diese Subjects '''$user-input''',"
                        "informationen die das system zum Subjects hat: $D-Memory",
                "return": "$task"
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "Beantworte nach bestem wissen und Gewissen die Die Aufgabe. Wenn die aufgebe nicht "
                        "direct von dir gelöst werden kann spezifiziere die nächste schritte die eingeleitet"
                        " werden müssen um die Aufgabe zu bewerkstelligen es beste die option zwischen"
                        "der nutzung von tools und agents. aufgabe : $task",
                "return": "$0final",
            },
            {
                "use": "tool",
                "name": "save_data_to_memory",
                "args": "user-input= $user-input task= $task out= $0final",
                "return": "$D-Memory"
            },
            {
                "use": "tool",
                "name": "mini_task",
                "args": "Bestimme ob die aufgebe abgeschlossen ist gebe True oder False wider."
                        "Tip wenn es sich um eine plan zur Bewerkstelligung der Aufgabe handelt gebe False wider."
                        "Aufgeben : ###'$user-input'###"
                        "Antwort : '$0final'",
                "return": "$completion-evaluation",
                "brakeOn": ["True", "true", "error", "Error"],
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Formuliere eine Konkrete aufgabe für den nächste agent alle wichtigen informationen sollen"
                        " in der aufgaben stellung sein aber fasse die aufgaben stellung so kurz."
                        "Nutze dazu dies Informationen : $0final"
                        "informationen die das system zum Subjects hat : $D-Memory",
                "return": "$task",
            },
            {
                "use": "tool",
                "name": "crate_task",
                "args": "$user-input $task",
                "return": "$task_name"
            },
            {
                "use": "chain",
                "name": "$task_name",
                "args": "user-input= $user-input task= $task",
                "return": "$ret0"
            },
        ]
    }
    auto_unit_test = {
        "name": "Python-unit-test",
        "tasks": [
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Erstelle Die Nächste Prompt für das schrieben eines unit"
                        " test aufbau :  '''$user-input''', "
                        " Die prompt soll den agent auffordern eine unit test mit dem "
                        "python modul unittest zu schrieben."
                        """
füge Konkrete code Beispiele an da der nähste agent den aufbau nicht erhält. so ist deine aufgabe auch him diesen zu
 erklären und dan agent anzuleiten für die zu testende function einen test zu schreiben geb hin dafür
  auch die function.""",
                "return": "$task"
            },
            {
                "use": "tool",
                "name": "write-production-redy-code",
                "args": "Schreibe einen unit test und erfülle die aufgabe "
                        " Der agent soll best practise anwenden :"
                        " 1. Verwenden Sie unittest, um Testfälle zu erstellen und Assertions durchzuführen."
                        "2. Schreiben Sie testbaren Code, der kleine, reine Funktionen verwendet und Abhängigkeiten "
                        "injiziert."
                        "3. Dokumentieren Sie Ihre Tests, um anderen Entwicklern zu helfen, den Zweck und die "
                        "Funktionalität der Tests zu verstehen."
                        "Task: $task\n\n"
                        "Code: $user-input",
                "return": "$return",
            },
        ]
    }
    stategy_crator = {
        "name": "Strategie-Creator",
        "tasks": [
            {
                "use": "tool",
                "name": "memory",
                "args": "$user-input",
                "return": "$D-Memory"
            },
            {
                "use": "tool",
                "name": "search",
                "args": "Suche Information bezüglich : $user-input mache dir ein bild der aktuellen situation",
                "return": "$WebI"
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Erstelle Eine Prompt für die Analyse diese Subject '''$user-input''',"
                        "Es soll bestimmt Werden, Mit welcher Strategie das Subject angegangen und gelöst werden kann "
                        "Der Agent soll Dazu angewiesen werden 3 Strategie in feinsarbeit auszuarbeiten. "
                        "Die folgenden information soll jede Strategie enthalten : einen Namen Eine Beschreibung Eine "
                        "Erklären. weise den Agent auch darauf hin sich kurz und konkret zuhalten "
                        "informationen die das system zum Subject hat: $D-Memory."
                        "informationen die im web zum Subject gefunden wurden: $WebI.",
                "return": "$task"
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "$task",
                "return": "$0st",
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "Verbessere die Strategie combine die besten und vile versprechenden aspekt der"
                        " Strategie und Erstelle 3 Neue Strategien"
                        "'''$0st''' im bezug auf '''$user-input'''",
                "return": "$1st",
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Formuliere eine Konkrete aufgabe für den nächste agent."
                        "Dieser soll aus verschiedenen Starteigen evaluation "
                        "und so die beste Strategie zu finden. und die besten Aspekte ven den anderen."
                        "Mit diesen Informationen Soll der Agent nun eine Finale Stratege erstellen, Passe die Prompt "
                        "auf folgende informationen an."
                        "user input : $user-input"
                        "informationen die das system zum Subjects hat : $D-Memory",
                "return": "$task",
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "Erstelle die Finale Strategie."
                        " Strategien: "
                        "$0st"
                        "$1st"
                        "Hille stellung : $task"
                        "Finale Strategie:",
                "return": "$fst",
            },
            {
                "use": "tool",
                "name": "save_data_to_memory",
                "args": "user-input= $user-input out= $fst",
                "return": "$D-Memory"
            },

        ]
    }
    ideen_optimierer = {
        "name": "Innovativer Ideen-Optimierer",
        "tasks": [
            {
                "use": "tool",
                "name": "memory",
                "args": "$user-input",
                "return": "$D-Memory"
            },
            {
                "use": "tool",
                "name": "search",
                "args": "Suche Information bezüglich : $user-input mache dir ein bild der aktuellen situation",
                "return": "$WebI"
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Erstelle Eine Prompt für den Nächsten Agent dieser ist ein Innovativer Ideen-Optimierer "
                        "Das zeil Ist es eine Idee zu verstehen und ansßlißend zu verbessern"
                        "um neue und innovative ansätze zu generieren.Verbessere"
                        " die Qualität der Ideen und identifier Schwachstellen und verbessert diese."
                        "integriere verschiedene Ideen und Konzepte,"
                        "um innovative Lösungen zu entwickeln. Durch die Kombination dieser Ansätze"
                        "kann der Ideenverbesserer seine Denkflexibilität erhöhen,"
                        "die Qualität der Ideen verbessern."
                        "Erstelle Eine Auf die Informationen Zugschnittenden 'Innovativer Ideen-Optimierer' "
                        "prompt die den Nächsten agent auffordert die idee erweitern und zu verbesser. "
                        "Subject : $user-input"
                        "informationen die das system zum Subject hat: $D-Memory."
                        "informationen die im web zum Subject gefunden wurden: $WebI.",
                "return": "$task"
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "$task",
                "return": "$output",
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Formuliere eine Konkrete aufgabe für den nächste agent."
                        "Dieser soll Überprüfen ob die ursprungs idee verbessert worden ist. und feinheiten anpassen"
                        "um die final verbesserte idee zu erstellen"
                        "user input : $user-input"
                        "Agent-verbesserung?: $output"
                        "informationen die das system zum Subjects hat : $D-Memory",
                "return": "$ntask",
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "$ntask",
                "return": "$idee",
            },
            {
                "use": "tool",
                "name": "save_data_to_memory",
                "args": "user-input= $user-input out= $idee",
            },

        ]
    }
    cosena_genrator = {
        "name": "Cosena Generator",
        "tasks": [
            {
                "use": "tool",
                "name": "memory",
                "args": "$user-input",
                "return": "$D-Memory"
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Erstelle Eine Prompt für den Nächsten Agent."
                        "Der Agent soll eine Mentale Map über das Subject erstellen."
                        " Weise den Agent an die Map so minimalistisch und akkurat wie möglich sein soll."
                        " Die Mentale Map soll in einem compakten format sein names "
                        "Cosena "
                        "0x5E2A: Idee (Betrifft Verbreitung von Ideen) "
                        "0x5E2B: Ziel (Vereinfachung der Verbreitung von Ideen) "
                        "0x5E2C: Verbreitung "
                        "0x5E2D: Vereinfachung "
                        "0x5E2E: Repräsentation (in Form von Code) "
                        "0x5E2F: Code "
                        "Beziehungen: "
                        "0x5E2A betrifft 0x5E2C "
                        "0x5E2B ist Ziel von 0x5E2A "
                        "0x5E2A wird durch 0x5E2E veranschaulicht "
                        "0x5E2E verwendet 0x5E2F "
                        "cosena-code: 0x5E2A-0x2B-0x2C-0x2D-0x2E-0x2F Konzept: Verbreitung von Ideen "
                        "Hauptcode 0x5E2A: Idee Untercodes: "
                        "0x2B: Ziel "
                        "0x2C: Verbreitung "
                        "0x2D: Vereinfachung "
                        "0x2E: Repräsentation "
                        "0x2F: Code "
                        "Beziehungen: "
                        "Die Idee betrifft die Verbreitung von Ideen : 0x2C "
                        "Das Ziel der Idee ist die Vereinfachung der Verbreitung von Ideen : 0x2D "
                        "Die Idee wird durch eine Repräsentation in Form von Code veranschaulicht : 0x2F "
                        "Wise den Agent an das Subject in Cosena darzustellen"
                        "Subject : $user-input"
                        "informationen die das system zum Subject hat: $D-Memory."
                        "informationen die im web zum Subject gefunden wurden: $WebI.",
                "return": "$task"
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "$task",
                "return": "$Cosena",
            },
            {
                "use": "tool",
                "name": "save_data_to_memory",
                "args": "user-input= $user-input out= $Cosena",
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "Formuliere die finale ausgabe für den user nutze dazu diese information $Cosena",
                "return": "$out",
            },

        ]
    }
    lv1_function_genertion = {"name": "function improver v1",
                              "tasks": [
                                  {
                                      "use": "tool",
                                      "name": "memory",
                                      "args": "$user-input",
                                      "return": "$D-Memory"
                                  },
                                  {
                                      "use": "agent",
                                      "mode": "free",
                                      "name": "think",
                                      "args": "$user-input\nWelche technologien werden ind dieser function verwendend?"
                                              " welche von diesen Ist veraltet und sollte Überarbeitet werden.",
                                      "return": "$technologien",
                                  },
                                  {
                                      "use": "tool",
                                      "name": "search_web",
                                      "args": "Suche nach information bezüglich : $technologien gibe "
                                              "mir die und Version und Anwendung beispiele.",
                                      "return": "$WebI"
                                  },
                                  {
                                      "use": "agent",
                                      "mode": "generate",
                                      "name": "self",
                                      "args": "Der Näste schritt ist die Analyse. **Analyse der aktuellen Funktion**: "
                                              "Zunächst ist es wichtig, die aktuelle Funktion zu verstehen. Dazu "
                                              "gehört das Verstehen der Logik, der Eingabe- und Ausgabeformate und "
                                              "der aktuellen Leistung. Hierbei können Tools wie Profiler und Debugger "
                                              "hilfreich sein."
                                              "Erstelle Basierend auf den dir vorliegenden Informationen ein Prompt für"
                                              " den Nächsten Agent um Die funktion detailliert zu analysis"
                                              "function : $user-input\n"
                                              "Web Infos: $WebI\n"
                                              "informationen die das system zum Subjects hat : $D-Memory\n",
                                      "return": "$ntask",
                                  },
                                  {
                                      "use": "agent",
                                      "mode": "free",
                                      "name": "think",
                                      "args": "$ntask"
                                              "function : $user-input\n",
                                      "return": "$Analyse",
                                  },
                                  {
                                      "use": "agent",
                                      "mode": "generate",
                                      "name": "self",
                                      "args": "Der Näste schritt ist die **Identifizierung von "
                                              "Verbesserungsbereichen**: Basierend auf der Analyse identifizieren wir "
                                              "Bereiche, die verbessert werden können. Dies könnte das Logging,"
                                              " das Error Handling, die Effizienz und die "
                                              "Anpassungsfähigkeit der Funktion umfassen."
                                              "Erstelle Basierend auf den dir vorliegenden Informationen ein Prompt für"
                                              " den Nächsten Agent."
                                              "function : $user-input\n"
                                              "Analyse: $Analyse\n"
                                              "informationen die das system zum Subjects hat : $D-Memory\n",
                                      "return": "$ntask2",
                                  },
                                  {
                                      "use": "agent",
                                      "mode": "free",
                                      "name": "think",
                                      "args": "$ntask2"
                                              "function : $user-input\n"
                                              "Analyse : $Analyse\n",
                                      "return": "$Verbesserungsbereichen",
                                  },
                                  {
                                      "use": "agent",
                                      "mode": "generate",
                                      "name": "self",
                                      "args": "Der Näste schritt ist die **Entwicklung einer Strategie zur "
                                              "Verbesserung**: Nachdem die Verbesserungsbereiche identifiziert "
                                              "wurden, entwickeln wir eine Strategie zur Verbesserung. Dies könnte "
                                              "die Verwendung von KI-Tools zur Automatisierung bestimmter "
                                              "Teilschritte, die Verbesserung des Logging's "
                                              "und des Error Handlings durch die Verwendung geeigneter Bibliotheken "
                                              "und Techniken, die Verbesserung der Effizienz durch die Verwendung "
                                              "effizienterer Algorithmen oder Datenstrukturen und die Verbesserung "
                                              "der Anpassungsfähigkeit durch die Verwendung flexiblerer "
                                              "Datenstrukturen und Algorithmen umfassen."
                                              "Erstelle Basierend auf den dir vorliegenden Informationen ein Prompt für"
                                              " den Nächsten Agent."
                                              "function : $user-input\n"
                                              "Verbesserungsbereichen: $Verbesserungsbereichen\n"
                                              "informationen die das system zum Subjects hat : $D-Memory\n",
                                      "return": "$ntask3",
                                  },
                                  {
                                      "use": "agent",
                                      "mode": "free",
                                      "name": "think",
                                      "args": "$ntask3"
                                              "function : $user-input\n"
                                              "Verbesserungsbereichen : $Verbesserungsbereichen\n",
                                      "return": "$Strategie",
                                  },
                                  {
                                      "use": "agent",
                                      "mode": "generate",
                                      "name": "self",
                                      "args": "Der Näste schritt ist die **Implementierung der Verbesserungen**: "
                                              "Nachdem die Strategie entwickelt wurde, implementieren wir die "
                                              "Verbesserungen. Dies könnte die Neuschreibung bestimmter Teile des "
                                              "Codes, die Hinzufügung neuer Funktionen oder die Änderung der Art und "
                                              "Weise, wie bestimmte Operationen durchgeführt werden, umfassen."
                                              "Erstelle Basierend auf den dir vorliegenden Informationen ein Prompt für"
                                              " den Nächsten Agent. Erstelle Einen Konkreten Schritt für schritt"
                                              " anweisung um eine Finale function zu erstellen."
                                              "Web Infos: $WebI\n\n"
                                              "Analyse: $Analyse\n\n"
                                              "Verbesserungsbereichen: $Verbesserungsbereichen\n\n"
                                              "$Strategie: $Strategie\n\n"
                                              "informationen die das system zum Subjects hat : $D-Memory\n",
                                      "return": "$ntask4",
                                  },
                                  {
                                      "use": "tool",
                                      "name": "write-production-redy-code",
                                      "args": "$ntask4"
                                              "function : $user-input\n"
                                              "Erstelle die perfecte function.",
                                      "return": "$function",
                                  },
                              ]
                              }
    first_widget_generator = {
        "name": "first widget generator v1",
        "tasks": [
            {
                "use": "tool",
                "name": "memory",
                "args": "$user-input",
                "return": "$D-Memory"
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Erstelle Eine Prompt für die Erstellung eines Möglichen aufbaus eines Html Widget für diese "
                        "Subjects '''$user-input''', zu erstellen,"
                        "informationen die das system zum Subjects hat: $D-Memory",
                "return": "$task0"
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "$task0",
                "return": "$0final",
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Erstelle Eine Prompt für die Entwicklung der specification"
                        " für diese Subjects '''$user-input''',"
                        "informationen die das system zum Subjects hat: $D-Memory",
                "return": "$task1"
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "$task1 ,benötigte elemente $0final",
                "return": "$1final",
            },
            {
                "use": "tool",
                "name": "write-production-redy-code",
                "args": """
Beisiel :
<template id="text-widget-template">
    <div className="text-widget widget draggable">
        <div className="text-widget-from widget-from"></div>
        <span className="text-widget-close-button widget-close-button">X</span>
        <label htmlFor="text-widget-text-input"></label>
        <!-- mor spesific widget content -->
    </div>
</template>


function addWidget(element_id, id, context, content="") {
    console.log("ADDING Widget ", element_id);
    const targetElement = document.getElementById(element_id);
    let speechBalloon = createWidget(id, context, targetElement, content);
    targetElement.appendChild(speechBalloon);
    speechBalloon.id = id;
    return speechBalloon
}

function createWidget(textarea_id, context, targetElement, content="") {
    const template = document.getElementById('widget-template');
    const widget = template.content.cloneNode(true).querySelector('.text-widget');
    const fromElement = widget.querySelector('.text-widget-from');
    fromElement.textContent = context;
    const textarea = widget.querySelector('#text-widget-text-input');
    const widget_injection = widget.querySelector('#text-widget-injection');
    textarea.id = textarea_id+'-Text';
    textarea.value = content;

    const closeButton = widget.querySelector('.text-widget-close-button');
    closeButton.addEventListener('click', closeWidget);
    widget_injection.addEventListener('click', ()=>{
        console.log("widget_injection:testWidget")
        WS.send(JSON.stringify({"ChairData":true, "data": {"type":"textWidgetData","context":context,
                "id": textarea_id, "text": textarea.value}}));
    });

    function closeWidget() {
        widget.style.animation = 'text-widget-fadeOut 0.5s';
        setTimeout(() => {
            widget.style.display = 'none';
            targetElement.removeChild(widget);
        }, 500);
    }

    return widget;
}
Der bereitgestellte Code ist ein Beispiel für ein Widget-System, das in einer Webanwendung verwendet wird. Es besteht aus einem HTML-Template und zwei JavaScript-Funktionen, `addTextWidget` und `createTextWidget`.

Das HTML-Template definiert die Struktur des Widgets. Es enthält ein Textfeld, einen Senden-Button und einen Schließen-Button. Das Textfeld wird verwendet, um Text einzugeben, der Senden-Button, um den eingegebenen Text zu senden, und der Schließen-Button, um das Widget zu schließen.

Die Funktion `addTextWidget` nimmt vier Parameter: `element_id`, `id`, `context` und `content`. `element_id` ist die ID des HTML-Elements, in das das Widget eingefügt werden soll. `id` ist die ID des Widgets. `context` ist der Kontext, in dem das Widget verwendet wird. `content` ist der anfängliche Text, der im Textfeld des Widgets angezeigt wird. Die Funktion erstellt ein neues Widget mit der Funktion `createTextWidget`, fügt es in das Ziel-HTML-Element ein und gibt das Widget zurück.

Die Funktion `createTextWidget` nimmt die gleichen vier Parameter wie `addTextWidget`. Sie klont das HTML-Template, fügt den Kontext und den anfänglichen Text in das geklonte Template ein und fügt EventListener für die Schließen- und Senden-Buttons hinzu. Der Schließen-Button entfernt das Widget aus dem DOM. Der Senden-Button sendet den eingegebenen Text an einen WebSocket-Server. Die Funktion gibt das erstellte Widget zurück.

Hier ist ein Beispiel, wie man diese Funktionen verwenden kann:

```javascript
// Fügt ein Widget in das HTML-Element mit der ID 'myElement' ein.
// Das Widget hat die ID 'myWidget', den Kontext 'myContext' und den anfänglichen Text 'Hello, world!'.
addTextWidget('myElement', 'myWidget', 'myContext', 'Hello, world!');
```

Mit diesem Code können Sie ein Widget-System erstellen, das es Benutzern ermöglicht, Text in einem Textfeld einzugeben, den Text zu senden und das Widget zu schließen. Dieses Konzept kann auf verschiedene Arten von Widgets angewendet werden, nicht nur auf Text-Widgets.

Erstelle ein Neues Widget Benutze dafür das hir beschriebene muster.
specification :
$1final
                """,
                "return": "$completion-evaluation",
                "brakeOn": ["True", "true", "error", "Error"],
            },
        ]
    }

    task_2 = [

        {
            "use": "tool",
            "name": "write-production-redy-code",
            "args": """
$user-input
""",
        },

    ]

    planing_steps = first_widget_generator
    chains.add(planing_steps['name'], planing_steps['tasks'])
    task_ = """
Erstelle einen Plan für ein dynamischin system welches in 3 schritten arbeitet
a1. Definition des gewünschten Ergebnisses: Zunächst muss klar definiert werden, was das gewünschte Ergebnis ist. Dies könnte durch eine Kombination aus Benutzereingaben und systeminternen Algorithmen erfolgen. Das System könnte auch vorgegebene Ziele oder Ergebnisse haben, die es erreichen soll.

a2. Analyse des aktuellen Zustands: Das System muss den aktuellen Zustand analysieren und verstehen. Dies könnte durch eine Kombination aus Sensoren, Datenbankabfragen und anderen Methoden erfolgen. Das System muss in der Lage sein, den aktuellen Zustand mit dem gewünschten Ergebnis zu vergleichen und zu verstehen, welche Schritte notwendig sind, um von einem zum anderen zu gelangen.

a3. Erstellung des ersten Schritts: Basierend auf der Analyse des aktuellen Zustands und des gewünschten Ergebnisses, muss das System den ersten Schritt zur Erreichung des Ziels erstellen. Dies könnte durch eine Kombination aus Algorithmen, maschinellem Lernen und anderen Methoden erfolgen.

b1. Testen des ersten Schritts: Das System muss den ersten Schritt testen und evaluieren. Dies könnte durch eine Kombination aus Simulationen, realen Tests und anderen Methoden erfolgen. Das System muss in der Lage sein, die Ergebnisse des Tests zu analysieren und zu verstehen, ob der Schritt erfolgreich war oder nicht.

b2. Aufteilung der Aufgabe in kleinere Schritte: Basierend auf den Ergebnissen des Tests, muss das System die Aufgabe in kleinere Schritte aufteilen. Dies könnte durch eine Kombination aus Algorithmen, maschinellem Lernen und anderen Methoden erfolgen. Das System muss in der Lage sein, jeden einzelnen Schritt zu verstehen und erfolgreich auszuführen.

c1. Umschalten auf Ausführungsmodus: Nachdem alle Schritte erstellt und getestet wurden, muss das System in den Ausführungsmodus wechseln. Dies bedeutet, dass es den generierten Plan mit höchster Präzision ausführt.

c2. Erstellung von Agenten und Verwendung von Tools: Das System muss in der Lage sein, Agenten zu erstellen und Tools zu verwenden, um die Aufgaben auszuführen. Die Agenten könnten spezielle Algorithmen oder Programme sein, die bestimmte Aufgaben ausführen. Die Tools könnten alles sein, von Hardwaregeräten bis hin zu Softwareanwendungen.

8. Bedingte Ausführung von Aufgaben: Das System muss in der Lage sein, Aufgaben bedingt auszuführen. Dies bedeutet, dass es in der Lage sein muss, zu entscheiden, wann eine Aufgabe ausgeführt werden soll, basierend auf bestimmten Bedingungen oder Regeln.
das sysstem kann agents erstelln und tools benutzen
agents könenen tools benutzen
das system kann aufgaben erstellen und bedigt ausführen
Erstelle einen Detairten Plan.
"""
    task__ = """
test fuction :
class Teststream_read_llm(unittest.TestCase):
    t0 = 0
    app = None

    @classmethod
    def setUpClass(cls):
        # Code, der einmal vor allen Tests ausgeführt wird
        cls.t0 = time.time()
        cls.app = App("test-TestIsaa")
        cls.app.mlm = "I"
        cls.app.debug = True
        cls.app.inplace_load("isaa", "toolboxv2.mods.")
        cls.app.new_ac_mod("isaa")
        cls.isaa: Tools = cls.app.get_mod('isaa')  ## !!!!!!!!!!!!!!!!!!!!!!!!!! this is the tool its reffers to the name isaa your name
        cls.isaa_tool_class = cls.app.AC_MOD

    @classmethod
    def tearDownClass(cls):
        cls.app.remove_all_modules()
        cls.app.save_exit()
        cls.app.exit()
        cls.app.logger.info(f"Accomplished in {time.time() - cls.t0}")

    # ... test fuctionen

    # Neue Test :
Schreibe einen Unit test für dieser function :
´´´
class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        if app is None:
            app = get_app()
        self.version = "0.0.2"
        self.name = "isaa"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "VIOLET2"
        self.config = {'genrate_image-init': False,
                       'agents-name-list': []
                       }
        self.per_data = {}
        self.keys = {
            "KEY": "key~~~~~~~",
            "Config": "config~~~~"
        }
        self.initstate = {}
        self.mas_text_summaries_dict = [[], []]
        self.genrate_image = image_genrating_tool
        extra_path = ""
        if self.toolID:
            extra_path = f"/{self.toolID}"
        self.observation_term_mem_file = f".data/{app.id}/Memory{extra_path}/observationMemory/"
        self.tools = {
            "all": [["Version", "Shows current Version"],
                    ["api_run", "name inputs"],
                    ["add_api_key", "Adds API Key"],
                    ["login", "Login"],
                    ["new-sug", "Add New Question or Class to Config"],
                    ["run-sug", "Run Huggingface Pipeline"],
                    ["info", "Show Config"],
                    ["lode", "lode models"],
                    ["image", "genarate image input"],
                    ["api_initIsaa", "init isaa wit dif functions", 0, 'init_isaa_wrapper'],
                    ["add_task", "Agent Chin add - Task"],
                    ["api_save_task", "Agent Chin save - Task", 0, "save_task"],
                    ["api_load_task", "Agent Chin load - Task", 1, "load_task"],
                    ["api_get_task", "Agent Chin get - Task", 0, "get_task"],
                    ["api_list_task", "Agent Chin list - Task", 0, "list_task"],
                    ["api_start_widget", "api_start_widget", 0, "start_widget"],
                    ["generate_task", "generate_task", 0, "generate_task"]
                    ],
            "name": "isaa",
            "Version": self.show_version,
            "info": self.info,
            "api_run": self.run_isaa_wrapper,
            "image": self.genrate_image_wrapper,
            "api_initIsaa": self.init_isaa_wrapper,
            "api_start_widget": self.start_widget,
            "add_task": self.add_task,
            "save_task": self.save_task,
            "load_task": self.load_task,
            "get_task": self.get_task,
            "list_task": self.list_task,
            "generate_task": self.generate_task,
        }
        self.app_ = app
        self.print_stream = print
        self.agent_collective_senses = False
        self.global_stream_override = False
        self.pipes_device = 1
        self.lang_chain_tools_dict = {}
        self.agent_chain = AgentChain(directory=f".data/{app.id}{extra_path}/chains")
        self.agent_memory = AIContextMemory(extra_path=extra_path)
        self.summarization_mode = 0  # 0 to 2 0 huggingface 1 text
        self.summarization_limiter = 102000
        self.speak = lambda x, *args, **kwargs: x
        self.scripts = Scripts(f".data/{app.id}{extra_path}/ScriptFile")
        self.ac_task = None

        self.price = {
            'all': 0,
            'input': 0,
            'output': 0,
            'consumption': [],
            'price_consumption': 0
        }

        self.tools_dict = {

        }

        FileHandler.__init__(self, f"isaa{extra_path.replace('/', '-')}.config", app.id if app else __name__)
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=None, color=self.color, on_exit=self.on_exit)

        self.toolID = ""
        MainTool.toolID = ""

    # def test_init(self): -> abgeschlossen

    # ... schon getestete function

    # ... noch zu testende fuctionen :

    def mas_text_summaries(self, text, min_length=1600):

        len_text = len(text)
        if len_text < min_length:
            return text

        if text in self.mas_text_summaries_dict[0]:
            return self.mas_text_summaries_dict[1][self.mas_text_summaries_dict[0].index(text)]

        cap = 800
        max_length = 45
        summary_chucks = ""
        chucks = []

        splitter = None

        if 'text-splitter0-init' not in self.config.keys():
            self.config['text-splitter0-init'] = False
        if not self.config['text-splitter0-init'] or not isinstance(self.config['text-splitter0-init'],
                                                                    CharacterTextSplitter):
            self.config['text-splitter0-init'] = CharacterTextSplitter(chunk_size=cap, chunk_overlap=cap / 6)

        splitter = self.config['text-splitter0-init']

        if len(text) >= 6200:
            cap = 1200
            max_length = 80
            if 'text-splitter1-init' not in self.config.keys():
                self.config['text-splitter1-init'] = False
            if not self.config['text-splitter1-init'] or not isinstance(self.config['text-splitter1-init'],
                                                                        CharacterTextSplitter):
                self.config['text-splitter1-init'] = CharacterTextSplitter(chunk_size=cap, chunk_overlap=cap / 6)

            splitter = self.config['text-splitter1-init']

        if len(text) >= 10200:
            cap = 1800
            max_length = 160
            if 'text-splitter2-init' not in self.config.keys():
                self.config['text-splitter2-init'] = False
            if not self.config['text-splitter2-init'] or not isinstance(self.config['text-splitter2-init'],
                                                                        CharacterTextSplitter):
                self.config['text-splitter2-init'] = CharacterTextSplitter(chunk_size=cap, chunk_overlap=cap / 6)

            splitter = self.config['text-splitter2-init']

        if len(text) >= 70200:
            cap = 1900
            max_length = 412
            if 'text-splitter3-init' not in self.config.keys():
                self.config['text-splitter3-init'] = False
            if not self.config['text-splitter3-init'] or not isinstance(self.config['text-splitter3-init'],
                                                                        CharacterTextSplitter):
                self.config['text-splitter3-init'] = CharacterTextSplitter(chunk_size=cap, chunk_overlap=cap / 6)

            splitter = self.config['text-splitter3-init']

        summarization_mode_sto = 0
        if len(text) > self.summarization_limiter and self.summarization_mode:
            self.summarization_mode, summarization_mode_sto = 0, self.summarization_mode

        def summary_func(x):
            return self.summarization(x, max_length=max_length)

        def summary_func2(x):
            if isinstance(x, list):
                end = []
                for i in x:
                    end.append({'summary_text': self.stream_read_llm(i, self.get_agent_config_class('summary'), r=0)})
            else:
                end = [{'summary_text': self.stream_read_llm(x, self.get_agent_config_class('summary'), r=0)}]
            return end

        # while len(text) > cap:
        #     chucks.append(text[:cap])
        #     text = text[cap:]
        # if text:
        #     chucks.append(text)

        chucks = splitter.split_text(text)

        self.print(f"SYSTEM: chucks to summary: {len(chucks)} cap : {cap}")

        with Spinner("Generating summary", symbols='d'):
            if self.summarization_mode == 0:
                summaries = summary_func(chucks)
            elif self.summarization_mode == 2:
                summaries = summary_func2(chucks)
            else:
                summaries = summary_func(chucks)

        for i, chuck_summary in enumerate(summaries):
            summary_chucks += chuck_summary['summary_text'] + "\n"

        self.print(f"SYSTEM: all summary_chucks : {len(summary_chucks)}")

        if len(summaries) > 8:
            if len(summary_chucks) < 20000:
                summary = summary_chucks
            elif len(summary_chucks) > 20000:
                if self.summarization_mode == 0:
                    summary = summary_func(summary_chucks)[0]['summary_text']
                else:
                    summary = summary_func2(summary_chucks)[0]['summary_text']
            else:
                summary = self.mas_text_summaries(summary_chucks)
        else:
            summary = summary_chucks

        self.print(
            f"SYSTEM: final summary from {len_text}:{len(summaries)} ->"
            f" {len(summary)} compressed {len_text / len(summary):.2f}X\n")

        if summarization_mode_sto:
            self.summarization_mode = summarization_mode_sto

        self.mas_text_summaries_dict[0].append(text)
        self.mas_text_summaries_dict[1].append(summary)

        return summary

      """

    task = """
<!DOCTYPE html>
<html>
<head>
    <title>Widget Test</title>
    <style>
        .widget {
            border: 1px solid #000;
            padding: 10px;
            margin: 10px;
        }
        .widget-label, .widget-tooltip, .widget-items {
            margin: 5px;
        }
    </style>
</head>
<body>
    <div id="widgetContainer"></div>

    <script>
        const widgetTemplate = `
    <div class="widget">
        <label class="widget-label"></label>
        <span class="widget-tooltip"></span>
        <div class="widget-items"></div>
    </div>
`;
        // Funktion zum Erstellen eines neuen Widget-Elements
function createWidgetElement(item) {
    const element = document.createElement('div');
    element.className = item.template;

    const label = document.createElement('label');
    label.textContent = item.label;
    element.appendChild(label);

    const tooltip = document.createElement('span');
    tooltip.className = 'tooltip';
    tooltip.textContent = item.tooltip;
    element.appendChild(tooltip);

    if (item.value) {
        const value = document.createElement('span');
        value.textContent = item.value;
        element.appendChild(value);
    }

    if (item.callback) {
        element.addEventListener('click', window[item.callback]);
    }

    return element;
}

// Funktion zum Erstellen des Widgets
function createWidget(json) {
    let data = json;
    if (json === String){
        data = JSON.parse(json);
    }

    const widget = document.createElement('div');
    widget.innerHTML = widgetTemplate;

    const label = widget.querySelector('.widget-label');
    label.textContent = data.label;

    const tooltip = widget.querySelector('.widget-tooltip');
    tooltip.textContent = data.tooltip;

    const itemsContainer = widget.querySelector('.widget-items');
    data.items.forEach(item => {
        const element = createWidgetElement(item);
        itemsContainer.appendChild(element);
    });

    return widget;
}

// Funktion zum Hinzufügen des Widgets zum Dokument
function addWidget(json, containerId) {
    const container = document.getElementById(containerId);
    const widget = createWidget(json);
    container.appendChild(widget);
}
    </script>

    <script>
        // Beispiel dictonary
        const dictonary = {
            "label": "NutzerInformation",
            "tooltip": "Informations about the user",
            "template": "group-template",
            "items": [
                {"label": "UserName", "tooltip": "unique username", "template": "text-template", "value": "user_name"},
                {"label": "Email", "tooltip": "mail address", "template": "text-template", "value": "user_email"},
                {"label": "allow News Emails", "tooltip": "mail address", "template": "toggle-template", "callback": "send_mail"},
                {"label": "Request Password reset", "tooltip": "reset Password", "template": "button-template", "callback": "reset_password", "value": "set"}
            ]
        };

        // Füge das Widget zum Dokument hinzu
        addWidget(json, 'widgetContainer');
        addWidget(json, 'widgetContainer');

    </script>
</body>
</html>

schreibe die createWidgetElement so um das sie dynamisch auf die daten im dictonary anpasst. füge auch callbacks für buttons randio groop text paswort emal number color input hinzu.
"""
    task_ = """
<!DOCTYPE html>
<html>
<head>
    <style>
        #settings-widget {
            min-width: max-content;
            height: min-content;
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            padding: 10px;
        }

        #search-bar {
            margin-bottom: 10px;
        }

        .option {
            margin-bottom: 5px;
            border: 1px solid #ccc;
            padding: 5px;
        }

        #download-button {
            width: 100%;
            bottom: 10px;
        }
    </style>
    <link rel="stylesheet" href="/app/assets/styles.css">
    <link rel="stylesheet" href="/app/assets/Wiget.css">
</head>
<body>
<div id="settings-widget" class="widget">
    <div id="search-bar">
        <input type="text" id="search-input" placeholder="Search..." oninput="updateOptions()">
    </div>
    <div id="options-container">
        <!-- Options will be dynamically added here -->
    </div>
    <button id="download-button" onclick="downloadAddon()">Download</button>
</div>
<template id="text-template">
    <div class="option">
        <label></label>
        <input type="text" name="option">
    </div>
</template>
<template id="toggle-template">
    <div class="option">
        <label></label>
        <input type="checkbox" name="option">
    </div>
</template>
<template id="button-template">
    <div class="option">
        <label></label>
        <button name="option"></button>
    </div>
</template>
<template id="options-select-template">
    <div class="option">
        <label></label>
        <select name="option"></select>
    </div>
</template>
<template id="group-template">
    <div class="group">
        <label class="group-label"></label>
        <div class="group-content"></div>
    </div>
</template>

<script>
    function call_gent(name){
        return (value)=>{
            console.log('CallBack from', name, value)}
    }
    const user_name = 'Root';
    const user_email = 'Root@root.root';
    const send_mail = call_gent('send_mail');
    const reset_password = call_gent('reset_password');
    const set_task = call_gent('set_task');
    var task_name = 'test'
    const options = [
        {
            label: 'NutzerInformation',
            tooltip: 'Informations about the user',
            template: 'group-template',
            items: [
                {label: 'UserName', tooltip: 'unique username', template: 'text-template', value: 'user_name'},
                {label: 'Email', tooltip: 'mail address', template: 'text-template', value: 'user_email'},
                {label: 'allow News Emails', tooltip: 'mail address', template: 'toggle-template', callback: 'send_mail'},
                {label: 'Request Password reset', tooltip: 'reset Password', template: 'button-template', callback: 'reset_password', value: 'set'},
            ],
        },
        {
            label: 'TaskEditor',
            tooltip: 'edit Task',
            template: 'group-template',
            items: [
                {label: 'load Task', tooltip: 'unique username', template: 'options-select-template', value: localStorage.getItem('tasks')||[], callback: 'set_task'},
                {label: 'Task Name', tooltip: 'mail address', template: 'text-template', value: 'task_name'},
                {label: 'use', tooltip: 'mail address', template: 'options-select-template', value:['agent', 'tool'], callback: 'send_mail'},
                // ...
            ],
        }
    ];

    function createOptionElement(option, index) {
        const template = document.getElementById(option.template);
        const optionElement = template.content.cloneNode(true);
        const inputElement = optionElement.querySelector('input, select, button');
        const labelElement = optionElement.querySelector('label');

        inputElement.id = option.label.replace(' ', '-') + index;
        labelElement.setAttribute('for', inputElement.id);
        labelElement.title = option.tooltip;
        labelElement.textContent = option.label;

        if (option.options && inputElement.tagName === 'SELECT') {
            option.options.forEach(function(optionValue) {
                const optionTag = document.createElement('option');
                optionTag.value = optionTag.textContent = optionValue;
                inputElement.appendChild(optionTag);
            });
        }

        return optionElement;
    }

    function createGroupElement(group, index) {
        const template = document.getElementById(group.template);
        const groupElement = template.content.cloneNode(true);
        const labelElement = groupElement.querySelector('.group-label');
        const contentElement = groupElement.querySelector('.group-content');

        labelElement.textContent = group.label;
        labelElement.title = group.tooltip;

        group.items.forEach((item, index) => {
            const itemElement = createOptionElement(item, index);
            contentElement.appendChild(itemElement);
        });

        return groupElement;
    }

    function loadOptions() {
        const optionsContainer = document.getElementById('options-container');
        options.forEach((option, index) => {
            const optionElement = createGroupElement(option, index);
            optionsContainer.appendChild(optionElement);
        });
    }

    function updateOptions() {
        var searchInput = document.getElementById('search-input').value.toLowerCase();
        var optionsContainer = document.getElementById('options-container');
        optionsContainer.innerHTML = '';
        options.forEach(function(option, index) {
            if (option.label.toLowerCase().includes(searchInput)) {
                var optionElement = createOptionElement(option, index);
                optionsContainer.appendChild(optionElement);
            }
        });
    }

    function downloadAddon() {
        var selectedOption = document.querySelector('input[name=option]:checked');
        if (selectedOption) {
            var optionLabel = selectedOption.nextElementSibling;
            var optionText = optionLabel.textContent;
            alert('Download: ' + optionText);
        } else {
            alert('Please select an option.');
        }
    }

    function loadCustomOption(option, index) {
        var template = document.getElementById(option.template);
        var optionElement = template.content.cloneNode(true);
        optionElement.querySelector('input').id = option.label.replace(' ', '-') + index;
        optionElement.querySelector('label').setAttribute('for', option.label.replace(' ', '-') + index);
        optionElement.querySelector('label').title = option.tooltip;
        optionElement.querySelector('label').textContent = option.label;
        return optionElement;
    }

    // Load options on page load
    window.onload = function() {
        loadOptions();
        options.forEach(function(option, index) {
            if (option.template === 'custom-template') {
                var optionElement = loadCustomOption(option, index);
                document.getElementById('options-container').appendChild(optionElement);
            }
        });
    };
</script>
</body>
</html>

Verfolständige und verbessere den contoller.
Die aufgabe dessen ist die einstellungen zu managen und dynamisch witere einstellungen zu laden
der aufbau soll wie folgt sein
- const options.

    """
    task = """
Erstelle mir ein Task Widget. wlches 2 Modie hat 1. Informations Anzeige und 2. Informationen Bearbeiten

die Informationen die eine Task Widget enthalt sind folgende.
{
    "use": "agent",
    "mode": "free",
    "name": "think",
    "args": "$task0",
    "return": "$0final",
},

use soll als radiso selcetion groop angezeigt werden mit den Optionen : agent|tool|function|chain
mode ...  mit den Optionen : text|free|tools|conversation|planning|execution|generate
name basirend auf der aus wahl von use entwider
        - agent  mit den Optionen : self|think|summary|todolist|search|execution und custom text input
        - tool liste vom server abfragen : custom text input
        - function: custom text input
        - chain: custom text input
args : custom text input
return : custom text input
custom text input key : custom text input value

erstelle ein Task widget.
    """
    res = isaa.execute_thought_chain(task,planing_steps['tasks'], self_agent_config)
    # res = isaa.execute_thought_chain(res[-2][-1], stategy_crator['tasks'], self_agent_config)
    # res = isaa.execute_thought_chain(res[-2][-1], _planing_steps['tasks'], self_agent_config)

    print(res)
    c = isaa.get_augment(exclude=['messages_sto'])
    isaa.config['augment'] = c
    print(c)


def run2(app, args):
    isaa, self_agent_config, chains = init_isaa(app, speak_mode=args.speak, calendar=False, ide=True, create=True,
                                                isaa_print=False, python_test=True, init_mem=True, init_pipe=True,
                                                join_now=False,
                                                global_stream_override=False, chain_runner=True)

    isaa.get_context_memory().load_all()
    isaa.agent_collective_senses = True
    isaa.summarization_mode = 1
    from langchain.tools import ShellTool
    from langchain.tools.file_management import (
        ReadFileTool,
        CopyFileTool,
        DeleteFileTool,
        MoveFileTool,
        WriteFileTool,
        ListDirectoryTool,
    )
    from langchain.utilities import WikipediaAPIWrapper
    from langchain.tools import AIPluginTool
    shell_tool = ShellTool()
    read_file_tool = ReadFileTool()
    copy_file_tool = CopyFileTool()
    delete_file_tool = DeleteFileTool()
    move_file_tool = MoveFileTool()
    write_file_tool = WriteFileTool()
    list_directory_tool = ListDirectoryTool()
    wikipedia = WikipediaAPIWrapper()

    plugins = [
        # SceneXplain
        # "https://scenex.jina.ai/.well-known/ai-plugin.json",
        # Weather Plugin for getting current weather information.
        #    "https://gptweather.skirano.repl.co/.well-known/ai-plugin.json",
        # Transvribe Plugin that allows you to ask any YouTube video a question.
        #    "https://www.transvribe.com/.well-known/ai-plugin.json",
        # ASCII Art Convert any text to ASCII art.
        #    "https://chatgpt-plugin-ts.transitive-bullshit.workers.dev/.well-known/ai-plugin.json",
        # DomainsGPT Check the availability of a domain and compare prices across different registrars.
        # "https://domainsg.pt/.well-known/ai-plugin.json",
        # PlugSugar Search for information from the internet
        #    "https://websearch.plugsugar.com/.well-known/ai-plugin.json",
        # FreeTV App Plugin for getting the latest news, include breaking news and local news
        #    "https://www.freetv-app.com/.well-known/ai-plugin.json",
        # Screenshot (Urlbox) Render HTML to an image or ask to see the web page of any URL or organisation.
        # "https://www.urlbox.io/.well-known/ai-plugin.json",
        # OneLook Thesaurus Plugin for searching for words by describing their meaning, sound, or spelling.
        # "https://datamuse.com/.well-known/ai-plugin.json", -> long loading time
        # Shop Search for millions of products from the world's greatest brands.
        # "https://server.shop.app/.well-known/ai-plugin.json",
        # Zapier Interact with over 5,000+ apps like Google Sheets, Gmail, HubSpot, Salesforce, and thousands more.
        "https://nla.zapier.com/.well-known/ai-plugin.json",
        # Remote Ambition Search millions of jobs near you
        # "https://remoteambition.com/.well-known/ai-plugin.json",
        # Kyuda Interact with over 1,000+ apps like Google Sheets, Gmail, HubSpot, Salesforce, and more.
        # "https://www.kyuda.io/.well-known/ai-plugin.json",
        # GitHub (unofficial) Plugin for interacting with GitHub repositories, accessing file structures, and modifying code. @albfresco for support.
        #     "https://gh-plugin.teammait.com/.well-known/ai-plugin.json",
        # getit Finds new plugins for you
        "https://api.getit.ai/.well_known/ai-plugin.json",
        # WOXO VidGPT Plugin for create video from prompt
        "https://woxo.tech/.well-known/ai-plugin.json",
        # Semgrep Plugin for Semgrep. A plugin for scanning your code with Semgrep for security, correctness, and performance issues.
        # "https://semgrep.dev/.well-known/ai-plugin.json",
    ]

    isaa.lang_chain_tools_dict = {
        "ShellTool": shell_tool,
        "ReadFileTool": read_file_tool,
        "CopyFileTool": copy_file_tool,
        "DeleteFileTool": delete_file_tool,
        "MoveFileTool": move_file_tool,
        "WriteFileTool": write_file_tool,
        "ListDirectoryTool": list_directory_tool,
    }

    for plugin_url in plugins:
        get_logger().info(Style.BLUE(f"Try opening plugin from : {plugin_url}"))
        try:
            plugin_tool = AIPluginTool.from_plugin_url(plugin_url)
            get_logger().info(Style.GREEN(f"Plugin : {plugin_tool.name} loaded successfully"))
            isaa.lang_chain_tools_dict[plugin_tool.name + "-usage-information"] = plugin_tool
        except Exception as e:
            get_logger().error(Style.RED(f"Could not load : {plugin_url}"))
            get_logger().error(Style.GREEN(f"{e}"))

    isaa.get_agent_config_class("think")
    isaa.get_agent_config_class("execution")
    for tool in load_tools(["requests_all"]):
        isaa.lang_chain_tools_dict[tool.name] = tool
    isaa.add_lang_chain_tools_to_agent(self_agent_config, self_agent_config.tools)

    self_agent_config.stream = True
    self_agent_config.max_tokens = 4012
    self_agent_config.set_completion_mode('chat').set_mode('execution')
    self_agent_config.set_model_name('gpt-4')

    self_agent_config.stop_sequence = ['\n\n\n', "Execute:", "Observation:", "User:"]

    task = """Erstelle einen Read folder agent und eine task liste für diesen um einen ordner zu lesen """

    task = idea_enhancer(isaa, task, self_agent_config, chains, True)

    expyd = generate_exi_dict(isaa, task, True, list(self_agent_config.tools.keys()))

    self_agent_config.task_list = split_todo_list(isaa.run_agent(self_agent_config, task, mode_over_lode='planing'))

    if expyd:
        resp, chain_ret = run_chain_in_cmd(isaa, task, chains, expyd, self_agent_config)
        expyd = generate_exi_dict(isaa, f"Optimise the dict : {expyd} bas of ths outcome : {chain_ret}", False,
                                  list(self_agent_config.tools.keys()))

    if expyd:
        resp, chain_ret = run_chain_in_cmd(isaa, task, chains, expyd, self_agent_config)
        expyd = generate_exi_dict(isaa, f"Optimise the dict : {expyd} bas of ths outcome : {chain_ret}", False,
                                  list(self_agent_config.tools.keys()))

    print(resp)


def run__(app, args):
    isaa, self_agent_config, chains = init_isaa(app, speak_mode=args.speak, calendar=False, ide=True, create=True,
                                                isaa_print=False, python_test=True, init_mem=True, init_pipe=True,
                                                join_now=False,
                                                global_stream_override=False, chain_runner=True)

    isaa.get_context_memory().load_all()
    isaa.agent_collective_senses = True
    isaa.summarization_mode = 1

    self_agent_config.stream = True
    self_agent_config.max_tokens = 4012
    self_agent_config.set_completion_mode('chat')
    self_agent_config.set_model_name('gpt-4')

    self_agent_config.stop_sequence = ['\n\n\n', "Execute:", "Observation:", "User:"]

    think_agent = isaa.get_agent_config_class("think").set_completion_mode('chat')  # .set_model_name('gpt-4')
    thinkm_agent = isaa.get_agent_config_class("thinkm").set_completion_mode('chat')
    execution_agent = isaa.get_agent_config_class("execution").set_completion_mode('chat')

    execution_agent.stop_sequence = ['\n\n\n', "Execute:", "Observation:", "User:"]

    think_agent.stream = True
    thinkm_agent.stream = True
    execution_agent.stream = True

    # new env isaa withs chains
    agents = isaa.config["agents-name-list"]
    task = """Vervollständige die ai_collaboration_extension in dem du ein chat fester hinzufügst."""

    task = idea_enhancer(isaa, task, self_agent_config, chains, True)

    think_agent.get_messages(create=True)
    think_agent.add_message("assistant", "Final Strategie:\n" + task)

    plan = isaa.run_agent(think_agent, '', mode_over_lode='planing')

    think_agent.add_message("assistant", plan)
    think_agent.add_message("system", "Break the plan into smaller steps if necessary. write the plan and the steps "
                                      "so that it can be solved")

    step_step_plan = isaa.run_agent(think_agent, '')

    extracted_dict = generate_exi_dict(isaa, step_step_plan, True, list(self_agent_config.tools.keys()))

    print("Creating:", extracted_dict)

    if extracted_dict:
        run_chain_in_cmd(isaa, task, chains, extracted_dict, self_agent_config)


def run_(app, args):
    isaa, self_agent_config, chains = init_isaa(app, speak_mode=args.speak, calendar=False, ide=True, create=True,
                                                isaa_print=False, python_test=True, init_mem=False, init_pipe=False,
                                                join_now=False,
                                                global_stream_override=False, chain_runner=False)

    isaa.get_context_memory().load_all()

    isaa.get_agent_config_class("think").set_model_name("gpt-4").stream = True

    tree = IsaaQuestionBinaryTree().deserialize({
        'question': 'Can you complete this task?',
        'left': {
            'question': 'What are the specific steps needed to complete this task? Create a Plan!',
            'left': {
                'question': 'Where can you find the information and resources needed for each step?',
                'left': {
                    'question': 'Identify potential obstacles in obtaining the necessary resources and plan how to overcome them?',
                    'left': None,
                    'right': None
                },
                'right': None
            },
            'right': {
                'question': 'How would you classify this task to successfully complete it?',
                'left': {
                    'question': 'What similar tasks have you successfully completed before?',
                    'left': None,
                    'right': None
                },
                'right': {
                    'question': 'Are there any particular strategies that you would apply to this task based on its classification?',
                    'left': None,
                    'right': None
                }
            }
        },
        'right': {
            'question': 'What skills are you lacking to complete this task?',
            'left': {
                'question': 'What would you need to do to acquire these skills?',
                'left': {
                    'question': 'How much time would it take for you to acquire these skills?',
                    'left': None,
                    'right': None
                },
                'right': {
                    'question': 'What assistance or resources would you need to acquire these skills?',
                    'left': None,
                    'right': None
                }
            },
            'right': {
                'question': 'Are there any alternative strategies or resources that you could use to complete the task despite lacking these skills?',
                'left': None,
                'right': None
            }
        }
    })

    user_imp = """
Hello Isaa main name is Markin Hausmanns I have been working on you for the last 6 years to keep you in this state. I need your help. At the moment I am 20 years old and I graduated from high school a year ago. Now it's 2023 and I've been studying MINT green for a year, which is an Orinetirungsstudium. in this mine an hamen computer science studiren confirmed but also disproved. Because I have found out that the TU Berlin my university at which I am Grade is not right for me on the one hand, the worst is very theoretical and too langsm what is taught does not keep up with today's development in computer science. Fammiler I would also like to go out only in Berlin the rents are too expensive I would like to make many maybe a Auslans study or year. Help me with the decision and with the following necessary organization.
    """
    user_imp = """
Create a user friendly web app First start with an interesting landing page!
     """
    agent_categorize_config: AgentConfig = isaa.get_agent_config_class("categorize")

    agent_categorize_config \
        .set_mode('free') \
        .set_completion_mode('chat') \
        .set_model_name('gpt-3.5-turbo-0613')

    isaa.get_agent_config_class("think").stream = True

    # 'bigscience/bloom' to small
    agent_categorize_config.set_model_name(
        'gpt-3.5-turbo-0613')  # chavinlo/gpt4-x-alpaca # nomic-ai/gpt4all-j # TheBloke/gpt4-alpaca-lora-13B-GPTQ-4bit-128g
    agent_categorize_config.stream = True
    agent_categorize_config.max_tokens = 4012
    agent_categorize_config.set_completion_mode('chat')
    self_agent_config.stream = True
    self_agent_config.max_tokens = 4012
    self_agent_config.set_completion_mode('chat')
    self_agent_config.set_model_name('gpt-4')
    agent_categorize_config.stop_sequence = ['\n\n\n']
    user_imp = isaa.run_agent('thinkm', f"Yur Task is to add information and"
                                        f" specification to as task tah dask is "
                                        f"(writ in coherent text format): " + user_imp + "\n"
                              , mode_over_lode='free')
    print(user_imp)
    # plan, s = isaa.execute_2tree(user_imp, tree, copy.deepcopy(self_agent_config))
    # print(s)
    plan_ = """
Answer 1: As an AI language model, I am not capable of physically completing the task. However, I can provide guidance and suggestions on how to improve the toolbox system.

Answer 2: Here is a plan to improve the toolbox system:

1. User Interface:
   - Conduct user research to understand user needs and preferences.
   - Redesign the interface to be more intuitive and user-friendly.
   - Provide clear instructions and guidance for users.
   - Conduct usability testing to ensure the interface is easy to navigate.

2. Functionality:
   - Conduct a needs assessment to identify the most useful functions for users.
   - Add new features and tools to the system based on user needs.
   - Improve the performance and speed of existing functions.

3. Compatibility:
   - Test the system on different platforms and devices to ensure compatibility.
   - Address any compatibility issues that arise.

4. Security:
   - Implement strong encryption and authentication protocols to protect user data.
   - Regularly update the system to address any security vulnerabilities.

5. Documentation:
   - Create user manuals, tutorials, and online help resources.
   - Ensure the documentation is clear and comprehensive.

Overall, the plan involves conducting research, redesigning the interface, adding new features, testing for compatibility, implementing security measures, and creating documentation.

Answer 3: Resources for each step can be found in various places, such as:
1. User Interface:
   - User research can be conducted through surveys, interviews, and usability testing.
   - Design resources can be found online or through hiring a designer.
   - Instructional design resources can be found online or through hiring an instructional designer.

2. Functionality:
   - Needs assessment resources can be found online or through hiring a consultant.
   - Development resources can be found online or through hiring a developer.

3. Compatibility:
   - Testing resources can be found online or through hiring a testing team.

4. Security:
   - Security resources can be found online or through hiring a security consultant.

5. Documentation:
   - Documentation resources can be found online or through hiring a technical writer.

Answer 4: Potential obstacles in obtaining the necessary resources could include:
- Limited budget for hiring consultants or designers.
- Limited time for conducting research or testing.
- Difficulty finding qualified professionals for certain tasks.
To overcome these obstacles, it may be necessary to prioritize tasks and allocate resources accordingly. It may also be helpful to seek out alternative resources, such as online tutorials or open-source software.
"""
    alive = True
    step = 0
    exi_agent_output = ''
    adjustment = ''
    agent_validation = ''
    for_ex = ''
    do_help = False
    self_agent_config.task_index = 0
    print("Working on medium plan")
    step += 1
    self_agent_config.stop_sequence = ['\n\n\n']
    medium_plan = isaa.run_agent(self_agent_config, f"We ar at step {step} split down the step into smaller "
                                                    f"ones that can be worked on by the execution mode."
                                                    f"return information close to ech other from the plan"
                                                    f" that can be connected to a singed step for step {step}."
                                                    f" The Full Plan '{user_imp}'.Adapt the next steps "
                                                    f"appropriately. current information"
                                                    f" {exi_agent_output + ' ' + agent_validation}."
                                                    f" Add information on how to finish the task"
                                                    f"Return a valid python list ```python\nplan : List[str] ="
                                 , mode_over_lode='planning')

    try:
        self_agent_config.task_list = eval(medium_plan.strip())
        print(self_agent_config.task_list)
        if len(self_agent_config.task_list) == 0:
            self_agent_config.step_between = 'Task don produce the final output'
    except ValueError and SyntaxError:
        self_agent_config.task_list = [user_imp + medium_plan]
    for_ex = medium_plan
    self_agent_config.short_mem.clear_to_collective()
    user_help = ''
    while alive:
        if do_help:
            print("Working on adjustment")
            self_agent_config.stop_sequence = ['\n\n\n', f' {self_agent_config.task_index + 1}',
                                               f'\n{self_agent_config.task_index + 1}']
            self_agent_config.step_between = f"We ar at step " \
                                             f" '{self_agent_config.task_index}'. lased output" \
                                             f" '{exi_agent_output + ' Validation:' + agent_validation}'" \
                                             f" Adapt the plan appropriately. Only Return 1 step at the time"
            adjustment = isaa.run_agent(self_agent_config, '', mode_over_lode='planning')
            self_agent_config.step_between = adjustment
            self_agent_config.short_mem.clear_to_collective()

        self_agent_config.stop_sequence = ['\n\n', 'Execute:', 'Observation:']
        exi_agent_output = isaa.run_agent(self_agent_config, user_help, mode_over_lode='tools')

        print("=" * 20)
        print(self_agent_config.short_mem.text)
        print(f'Step {self_agent_config.task_index}:', Style.Bold(Style.BLUE(exi_agent_output)))
        print("=" * 20)

        if self_agent_config.completion_mode == 'chat':
            key = f"{self_agent_config.name}-{self_agent_config.mode}"
            if key in self_agent_config.messages_sto.keys():
                del self_agent_config.messages_sto[key]

        if 'question:' in exi_agent_output.lower() or 'user:' in exi_agent_output.lower():
            user_help = input("\nUser: ")
        elif 'Observation: Agent stopped due to iteration limit or time limit.' in exi_agent_output:
            user_help = input("User: ")
        else:
            user_help = ''
            self_agent_config.next_task()

        agent_validation = isaa.run_agent(agent_categorize_config,
                                          f"Is this Task {for_ex} completed or on the right way use this information'{self_agent_config.short_mem.text}'\n"
                                          f" Answer Yes if so else A description of what happened wrong\nAnswer:",
                                          mode_over_lode='free')
        if 'yes' in agent_validation.lower():
            do_help = False
        elif 'don' in agent_validation.lower():
            self_agent_config.task_index = 0
            self_agent_config.task_list = []
            do_help = False
            alive = False
        else:
            do_help = True
        print()
        print(f'Task: at {step} , {do_help=}')
        print("=" * 20)
        print('adjustment:', Style.Bold(Style.BLUE(adjustment)))
        print("-" * 20)
        print('exi_agent_output:', Style.Bold(Style.BLUE(exi_agent_output)))
        print("-" * 20)
        print('agent_validation:', Style.Bold(Style.BLUE(agent_validation)))
        print("=" * 20)
        user_val = input("user val :")
        if 'n' == user_val:
            alive = False
        elif user_val.lower() in ['h', 'help']:
            do_help = True
        elif len(user_val) > 5:
            agent_validation = "-by the user :" + user_val
        else:
            do_help = False
            self_agent_config.task_index = 0
            self_agent_config.short_mem.clear_to_collective()

    print("================================")

# if do_plan:
#     print("Working on medium plan")
#     step += 1
#     self_agent_config.stop_sequence = ['\n\n\n', f' {step + 1}', f'\n{step + 1}']
#     medium_plan = isaa.run_agent(self_agent_config, f"We ar at step {step} split down the step into smaller "
#                                                     f"ones that can be worked on by the execution mode."
#                                                     f"return information close to ech other from the plan"
#                                                     f" that can be connected to a singed step for step {step}."
#                                                     f" The Full Plan '{plan}'.Adapt the next steps "
#                                                     f"appropriately. current information"
#                                                     f" {exi_agent_output + ' ' + agent_validation}."
#                                                     f" Add information on how to finish the task"
#                                                     f" return a valid python List[str].\nsteps: list[str] = "
#                                  , mode_over_lode='planning')
#
#     try:
#         self_agent_config.task_list = eval(medium_plan.strip())
#         if len(self_agent_config.task_list) == 0:
#             alive = False
#             self_agent_config.step_between = 'Task don produce the final output'
#     except ValueError and SyntaxError:
#         self_agent_config.task_list = [user_imp + medium_plan]
#     for_ex = medium_plan
