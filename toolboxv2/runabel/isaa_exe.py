"""Console script for toolboxv2. Isaa CMD Tool"""

from toolboxv2.mods.isaa_extars.isaa_modi import init_isaa, split_todo_list

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
    self_agent_config.short_mem.max_length = 3000
    self_agent_config.set_model_name('gpt-4')

    self_agent_config.stop_sequence = ['\n\n\n', "Execute:", "Observation:", "User:"]

    isaa.get_agent_config_class("think").stream = True
    isaa.get_agent_config_class("thinkm").stream = True

    # new env isaa withs chains
    print(isaa.config["agents-name-list"])
    skills = []
    agents = isaa.config["agents-name-list"]
    task = """
"""

    alive = True
    new_agent = isaa.config["agents-name-list"][-1]
    if len(agents) != 3:
        for agent_name in agents[3:]:
            isaa.get_agent_config_class(agent_name)
    while alive:
        env_text = f"""Welcome, you are in a hostile environment, your name is isaa.
you have several basic skills 1. creating agents 2. using skills, agents and tools

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

        print(isaa.config["agents-name-list"])
        agents = isaa.config["agents-name-list"]

        if new_agent != isaa.config["agents-name-list"][-1]:
            new_agent = isaa.config["agents-name-list"][-1]
            isaa.get_agent_config_class(new_agent).save_to_file()

        print(split_todo_list(sim))
        user_val = input("")
        if user_val == "n":
            alive = False
        elif len(user_val) > 3:
            task += "User: " + user_val
