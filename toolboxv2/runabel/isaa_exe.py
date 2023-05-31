"""Console script for toolboxv2. Isaa CMD Tool"""

from toolboxv2 import Style, Spinner
from toolboxv2.mods.isaa import AgentChain, IsaaQuestionBinaryTree, AgentConfig
from toolboxv2.utils.isaa_util import init_isaa, extract_dict_from_string, split_todo_list

NAME = "isaa-exe"


def run(app, args):
    isaa, self_agent_config, chains = init_isaa(app, speak_mode=args.speak, calendar=False, ide=True, create=True,
                                                isaa_print=False, python_test=True, init_mem=True, init_pipe=True,
                                                join_now=False,
                                                global_stream_override=False, chain_runner=True)

    isaa.get_context_memory().load_all()
    isaa.summarization_mode = 2

    self_agent_config.stream = True
    self_agent_config.max_tokens = 4012
    self_agent_config.set_completion_mode('chat')
    self_agent_config.set_model_name('gpt-4')

    self_agent_config.stop_sequence = ['\n\n\n', "Execute:", "Observation:", "User:"]

    isaa.get_agent_config_class("think").stream = True
    isaa.get_agent_config_class("thinkm").stream = True

    # new env isaa withs chains
    print(isaa.config["agents-name-lsit"])
    skills = []
    agents = isaa.config["agents-name-lsit"]
    task = """
Wird an einen Kondensator eine Spannung angelegt, springt die Spannung nicht, sondern steigt langsam auf ihren Endwert an. Genauso sinkt die Spannung langsam auf 0V, sobald keine Spannung mehr anliegt. Dieser Umstand beruht auf dem Aufbau und der Funktionsweise des Kondensators.
Baut eine Schaltung, welche einen Kondensator immer wieder lädt und entlädt (Schaltplan im Anhang), und zeichnet dabei den zeitlichen Spannungsverlauf des Kondensators mit einem Arduino Programm auf dem seriellen Plotter auf. Die Baudrate bei der seriellen Kommunikation soll auf den Wert 9600 eingestellt sein. (10 Punkte)
TIPPS:
Die Aufgabe ist so gedacht, dass ihr Arrays verwendet.
Es ist in Ordnung, wenn die Aufzeichnung des Ladens und des Entladens getrennt geschieht. Ihr könnt also 2 Arduino Programme hochladen.
Der Schaltplan befindet sich im Anhang.
Um eine Idee zu bekommen, wie das Ergebnis auszusehen hat, könnt ihr einfach "Kondensator Ladekurve" googeln.
"""
    alive = True
    new_agent = isaa.config["agents-name-lsit"][-1]
    if len(agents) != 3:
        for agent_name in agents[3:]:
            isaa.get_agent_config_class(agent_name)
    while alive:
        env_text = f"""Welcome, you are in a hostile environment, your name is isaa.
you have several basic skills 1. creating agents 2. creating some agents 3. using skills, agents and tools

you have created {len(agents)}agents so far these are : {agents}.
you have learned {len(skills)}skills so far these are : {skills}.

use your basic functions with the agent and skills to complete a task.

for your further support you have a python environment at your disposal. write python code to access it.
if you have no ather wy then to ask for help write Question: 'your question'\nUser:

Task : {task}"""

        sim = isaa.run_agent(self_agent_config, env_text, mode_over_lode='execution')
        task += str(sim)

        if "user:" in sim.lower():
            print("USER QUESTION")
            task += input()

        print(isaa.config["agents-name-lsit"])
        agents = isaa.config["agents-name-lsit"]

        if new_agent != isaa.config["agents-name-lsit"][-1]:
            new_agent = isaa.config["agents-name-lsit"][-1]
            isaa.get_agent_config_class(new_agent).save_to_file()

        print(split_todo_list(sim))
        user_val = input("")
        if user_val == "n":
            alive = False
        elif len(user_val) > 3:
            task += "User: " + user_val
