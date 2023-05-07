"""Console script for toolboxv2. Isaa CMD Tool"""
import sys
from datetime import datetime

from toolboxv2.mods.isaa_audio import get_audio_transcribe
from toolboxv2.mods.isaa import AgentConfig, CollectiveMemory, AgentChain
from toolboxv2.mods.isaa import Tools as Isaa
from toolboxv2.utils.isaa_util import sys_print, speak, run_agent_cmd, init_isaa

NAME = "isaa-cmd"


def run(app, args):
    speak_mode = args.speak

    isaa, self_agent_config = init_isaa(app, speak_mode=speak_mode, calendar=True, ide=False, create=False)
    chain_instance = AgentChain()
    response = ""
    sys_print("Welcome")
    # app.AC_MOD.genrate_image("Blue 4k Sky", app)
    # user_text = get_input()
    sys_print("\n================================Starting-Agent================================")
    # sys_print("USER0: " + user_text)
    step = 0

    task_done = False
    # while app.alive:

    self_agent_config.model_name = "gpt-3.5-turbo"

    self_agent_config.mode = "free"
    self_agent_config.completion_mode = "chat"
    json_input = [{"use-tool": True, "tool-name": "search",
                   "args": "Suche nach arumenten und infomationen die für $user-text stehen"},
                  {"use-tool": True, "tool-name": "search",
                   "args": "Suche nach arumenten und infomationen die gegen $user-text stehen"},
                  {"use-agent": True, "agent-name": "Think",
                   "args": "fasse die ergebnis aus $step1 und $2 zusammen berichte diferentziert"}
                  ]
    # json_input = '{"name":"diff-seche", "tasks": [{"use-tool": true, "tool-name":"search", "args": "Suche ' \
    #                  'nach arumenten und infomationen die für $user-text stehen"},{"use-tool": true, ' \
    #                  '"tool-name":"search", "args": "Suche nach arumenten und infomationen die gegen $user-text stehen"},' \
    #                  '{"use-agent": true, "agent-name":"Think", "args": "fasse die ergebnis aus $step1 und $2 zusammen ' \
    #                  'berichte diferentziert"}]}'

    agent_categorize_config: AgentConfig = isaa.get_agent_config_class("categorize")

    agent_categorize_config\
        .set_mode('free')\
        .set_completion_mode('text')\
        .set_model_name('gpt-3.5-turbo')\
        .stream = True

    chain_instance.add("get_a_differentiated_point_of_view", [
        {
            "use": 'tool',
            "name": "search",
            "args": "Suche Information zu $user-input",
            "return": "$infos-0",
        },
        {
            "use": 'tool',
            "name": "search",
            "args": "Suche nach argument und information die für $user-input sprechen bezier $infos-0 mit ein",
            "return": "$infos-1",
        },
        {
            "use": 'tool',
            "name": "search",
            "args": "Suche nach argument und information die gegen $user-input sprechen bezier $infos-0 mit ein",
            "return": "$infos-2",
        },
        {
            "use": 'agent',
            "name": "think",
            "args": "fasse die information zu Thema $infos-0 \nPro seite $infos-1 \n\nCon seite $infos-2 "
                    "\n\ndiffernzirte zusammen und berichte",
        }
    ])

    chain_instance.add("calendar_entry", [
        {
            "use": 'agent',
            "name": "categorize",
            "args": "Bestimme den Typ des Kalendereintrags basierend auf $user-input",
            "return": "$entry-type",
        },
        {
            "use": 'tool',
            "name": "Calender",
            "args": "Speichere den Eintrag Typ: $entry-type \nUser: $user-input",
            "infos": f"$Date"
        }
    ])

    chain_instance.add("next_three_days", [
        {
            "use": "tool",
            "name": "Calender",
            "args": "Rufe die Ereignisse der nächsten 3 Tage ab",
            "return": "$events"
        },
        {
            "use": "agent",
            "name": "summary",
            "args": "Fasse die Ereignisse $events der nächsten 3 Tage übersichtlich zusammen",
            "return": "$summary"
        }
    ])
    # chain_instance.save_to_file()
    #user_text = "Was ist das Problem mit detonations antrieben"
    #res = isaa.run_agent(agent_categorize_config, f"What chain '{str(chain_instance)}'"
    #                                              f" is fitting for this input '{user_text}'")
    #chain = ''
    #for chain_key in chain_instance.chains.keys():
    #    if res in chain_key:
    #        chain = chain_key
    #        break
    #print("chain: ", chain)
    #user_vlidation = input(f"if its the chain is wrong type the corresponding number {list(zip(chain_instance.chains.keys(),range(len(chain_instance.chains.keys()))))} :")
    #if user_vlidation:
    #    user_vlidation = int(user_vlidation)-1
    #    chain = list(chain_instance.chains.keys())[user_vlidation]
    #sumry, res = isaa.execute_thought_chain(user_text, chain_instance.get(chain), self_agent_config.set_model_name("gpt-4"), speak=speak)
    #print("sumry: ", sumry)


    # user_text = """Create a coda creation mod for the tool box. withe 2 modes 1 get information by reading github
    # repos or url docs, and 2 mode wirte code providet with data from mode 1 reladed to th task"""
    # sys_print("USER0: " + user_text)
    # CollectiveMemory.memory.clear()
    # user_text = 'Define a function to scrape data from Github repositories or URL documents. This function should take a URL as input and return the relevant data.'
    #
    # self_agent_config.stop_sequence, st = ["Execute:"], self_agent_config.stop_sequence
    # response, task_done = run_agent_cmd(isaa, user_text, self_agent_config, step, speak)
    # self_agent_config.stop_sequence = st
    #
    ##response = """spawn_agent(Name: LifePlan, Mode: talk, Task: Planning user next 2 jears, Personal: Technically-oriented, Interests: Programming, Bouldering)"""
    ## memory(Interest: Programming, Age: 20, Trial study status: In progress, Location preference: Outside Berlin)
    ## todolist(Tech related job search, Job applications, Internship search, Network building, Learning new programming languages)
    ## todolist(Bouldering schedule, Bouldering locations: Outside Berlin)
    ## search(Programming Job Opportunities, Internships outside Berlin, Bouldering locations)
    ## """
    ## TODO test if valid and exiqute steps via system
    ##self_agent_config.task_list = response.split('\n')
    ##for task in self_agent_config.task_list:
    ##    print('Task: ', task)
    ##    valid_task, function_name, function_args = isaa.test_use_tools("Action: " + task, self_agent_config)
    ##    if valid_task:
    ##        ret = isaa.run_tool(function_args, function_name, self_agent_config)
    ##    else:
    ##        # task ist not valid
    ##        ret = f"Invalid function call {task}"
    ##    sys_print(f"RET: {ret}")
    ##    # check if the function is run successfully and adjust further execution
    #
    # self_agent_config.mode = 'execution'
    # user_text = 'Define a function to scrape data from Github repositories or URL documents. This function should take a URL as input and return the relevant data.'
    # task_done = True
    # while not task_done:
    #    response, task_done = run_agent_cmd(isaa, user_text, self_agent_config, step, speak)
    #    print(self_agent_config.last_prompt)
    #    print(response)
    #    print(task_done)
    #    input("-")

    a = """
    if stop_helper(response):
        text_to_speech3("Ich habe eine Frage")
        spek(response, vi=1)
        user_text = get_input()
        if user_text.lower() in "exit":
            app.alive = False
        if user_text.lower() in "sleep":
            awake = False
        sys_print(f"User: {user_text}")
    if task_done:  # TODO CLEar St mem to storage -- new task
        if speek_mode:
            text_to_speech3("Ist die Aufgabe abgeschlossen?")
        else:
            print("Ist die Aufgabe abgeschlossen?")
        print("Type:\n\texit\n\tstop or sleep\n\tyes or y\n\tno or n\n\tand for prompt end with -p")
        p = input("User Input: ").lower()
        if p.endswith("p"):
            print(self_agent_config.last_prompt)
            input("User Input: .. ")
        if p.startswith("e") or p.startswith("y"):
            app.alive = False
        elif p.startswith("n"):
            user_text = get_input()
            if not user_text:
                user_text = input("Sorry got no input ... :")
        else:
            pass
        sys_print(f"User: {user_text}")
    else:
        if input("Agent continues - ") in ['e', 'exit']:
            app.alive = False
"""
    step += 1
