"""Console script for toolboxv2. Isaa CMD Tool"""
import random
import time

import tiktoken

from toolboxv2 import Style
from toolboxv2.mods.isaa_extars.isaa_modi import init_isaa, get_code_files

NAME = "isaa-gitDocs"
# tools.text_classification.TextClassificationTool()
# tools.document_question_answering.DocumentQuestionAnsweringTool()
# tools.image_captioning.ImageCaptioningTool()
# tools.speech_to_text.SpeechToTextTool()
# tools.text_to_speech.TextToSpeechTool()
# tools.image_segmentation.ImageSegmentationTool()
# tools.text_summarization.TextSummarizationTool()

def run(app, args):
    isaa, self_agent_config, chains = init_isaa(app, speak_mode=args.speak, init_pipe=False, ide=True, create=False,
                                                join_now=True,
                                                global_stream_override=True)

    # slauw87/bart_summarisation
    # facebook/bart-large-cnn
    # "ashwinR/CodeExplainer"
    # SEBIS/code_trans_t5_large_source_code_summarization_python_multitask_finetune
    # SEBIS/code_trans_t5_base_code_documentation_generation_java_multitask

    isaa.init_pipeline('summarization', "SEBIS/code_trans_t5_large_source_code_summarization_python_multitask_finetune")

    isaa.get_context_memory().load_all()
    print("Starring")
    isaa_memory = isaa.get_context_memory()
    isaa.get_chain().load_from_file()

    isaa.get_agent_config_class("think").set_model_name("gpt-3.5-turbo-0613").stream = True
    # get project in isaa_work dir
    repo_url = "https://github.com/MarkinHaus/ToolBoxV2.git"
    branch = "init-isaa"
    destination_folder = "isaa_work/ai_collaboration_extension/"
    project_name = "ai_collaboration_extension"

    # download_github_project(repo_url, branch, destination_folder)

    code_and_md_files = get_code_files(destination_folder)

    tokens = 0
    prices = {
        "gpt-3.5-turbo-0613": 0.002,
        "Davinci": 0.02,
        "Babbage": 0.0005,
        "GPT-4-p": 0.03,
        "GPT-4-c": 0.06,
    }
    for file in code_and_md_files:
       with open("isaa_work/"+file, 'r', encoding='utf-8') as f:
           tokens += len(tiktoken.encoding_for_model("gpt-3.5-turbo-0613").encode(f.read()))

    out = ((tokens*3.7)/1000)*prices['gpt-3.5-turbo-0613']
    inp = (tokens/1000)*prices['gpt-3.5-turbo-0613']
    print(f"ALL of tokens : {tokens} text input price : ${inp}\n"
         f"estimated output price :${out} \nfull price {inp+out}")

    if input(" AKZEPTIREN :") not in ['y']:
       return

    self_agent_config.stream = True

    def do_on_file(filename):
        time.sleep(random.randint(1, 100) / 100)
        description, file_doc = isaa.execute_thought_chain(filename,
                                                           chains.get("Generate_docs")
                                                           , self_agent_config)
        print("=" * 20)
        print(file_doc)
        print("Description:\n", Style.Bold(Style.BLUE(description)))
        print("=" * 20)

        isaa_memory.add_data(project_name, file_doc[-1][1])
        return file_doc[-1][1]

    print(code_and_md_files)

   #code_and_md_files = [
   #    'toolbox/toolboxv2\\main_tool.py',
   #    'toolbox/toolboxv2\\__init__.py', 'toolbox/toolboxv2\\api\\fast_api.py',
   #    'toolbox/toolboxv2\\api\\fast_api_install.py', 'toolbox/toolboxv2\\api\\fast_api_main.py',
   #    'toolbox/toolboxv2\\api\\fast_app.py', 'toolbox/toolboxv2\\api\\util.py',
   #    'toolbox/toolboxv2\\mods\\api_manager.py',
   #    'toolbox/toolboxv2\\mods\\cloudM.py', 'toolbox/toolboxv2\\mods\\DB.py', 'toolbox/toolboxv2\\mods\\isaa.py',
   #    'toolbox/toolboxv2\\mods\\isaa_audio.py',
   #    'toolbox/toolboxv2\\mods\\isaa_ide.py', 'toolbox/toolboxv2\\runabel\\isaa_conversation.py',
   #    'toolbox/toolboxv2\\runabel\\isaa_init_chains.py',
   #    'toolbox/toolboxv2\\utils\\file_handler.py', 'toolbox/toolboxv2\\utils\\isaa_util.py',
   #    'toolbox/toolboxv2\\utils\\Style.py', 'toolbox/toolboxv2\\utils\\TBConfig.py',
   #    'toolbox/toolboxv2\\utils\\tb_logger.py', 'toolbox/toolboxv2\\utils\\toolbox.py']

    for file in code_and_md_files:
        do_on_file(file)

    isaa_memory.crate_live_context(project_name, len(code_and_md_files))
    isaa_memory.init_store(project_name)
    qa = isaa.init_db_questions(project_name, self_agent_config)
    if qa is None:
        return
    chat_history = []
    while True:
        question = input('Question:')
        if question == 'quit':
            break
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result['answer']))
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")

        print("================================")
        infos = isaa_memory.get_context_for(question, name=project_name)
        if infos:
            add = f"Additional Information's : {infos}"
            question += '\n' + add
            result = qa({"question": question, "chat_history": chat_history})
            chat_history.append((question, result['answer']))
            print(f"-> **Question**: {question} \n")
            print(f"**Answer**: {result['answer']} \n")

    print("================================")
    # isaa_memory.add_data(project_name, )

    # reade project split functions add informations to functions save to vector DB
    # reade Documentation online for Task relavent informations save to vector DB
    # Get Task and Informations form DB get code sample to edit and start developing
    # Prompt Isaa first step
    # Isaa use relavent informations and code to generate code
    # Prompt verbesserungen >>-<<

