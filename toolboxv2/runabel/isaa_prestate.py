"""Console script for toolboxv2. Isaa CMD Tool"""

from toolboxv2.mods.isaa_extars.isaa_modi import init_isaa, split_todo_list, generate_exi_dict

NAME = "isaa-s-auto"


def run(app, args):
    isaa, self_agent_config, chains = init_isaa(app, speak_mode=args.speak, calendar=False, ide=True, create=True,
                                                isaa_print=False, python_test=True, init_mem=True, init_pipe=True,
                                                join_now=False,
                                                global_stream_override=False, chain_runner=True)

    isaa.get_context_memory().load_all()
    isaa.agent_collective_senses = True
    isaa.summarization_mode = 2

    self_agent_config.stream = True
    self_agent_config.max_tokens = 4012
    self_agent_config.set_completion_mode('chat')
    self_agent_config.set_model_name('gpt-4')

    self_agent_config.stop_sequence = ['\n\n\n', "Execute:", "Observation:", "User:"]

    think_agent = isaa.get_agent_config_class("think").set_completion_mode('chat')  # .set_model_name('gpt-4')
    thinkm_agent = isaa.get_agent_config_class("thinkm").set_completion_mode('chat')

    clarification_agent = isaa.get_agent_config_class("user_input_helper") \
        .set_completion_mode('chat') \
        .set_model_name('gpt-4').set_mode('free')

    create_agent = isaa.get_agent_config_class("create_or") \
        .set_completion_mode('chat') \
        .set_model_name('gpt-4').set_mode('execution')

    create_agent.stop_sequence = ['\n\n', "Execute:", "Observation:", "User:"]
    clarification_agent.stop_sequence = ['\n\n\n']

    think_agent.stream = True
    thinkm_agent.stream = True
    clarification_agent.stream = True
    create_agent.stream = True

    # new env isaa withs chains
    agents = isaa.config["agents-name-list"]
    task = """Work on the fierfox plung-in in the ai_collaboration_extension file."""
    infos_c = f"List of Avalabel Agents : {agents}\n Tools : {list(self_agent_config.tools.keys())}\n" \
              f" Functions : {isaa.scripts.get_scripts_list()}\n Chains : {str(chains)} " \
              f"chains can be run withe the execute-chain tool "

    print(infos_c)

    clarification_agent.get_messages(create=True)
    clarification_agent.add_message("user", task)
    clarification_agent.add_message("system", infos_c)
    clarification_agent.add_message("system", "Reproduce the task four times in your own words and"
                                              " think about Possible Solution approaches ."
                                              " with which you can understand this problem better."
                                              " For each variant you should specify a Understanding from 0% to 100%."
                                              " For each variant you should specify a Complexity"
                                              "  approximate the numbers of step taken to compleat"
                                              "For each variant you should specify a deviation from the task probability "
                                              "from 100% to 0%. -> 0 = no deviation is in perfect alignment to the task."
                                              " 100 = full deviation not related to the task ")
    clarification_agent.add_message("assistant", "After long and detailed step by step thinking and evaluating,"
                                                 " brainstormen of the task I have created the following variant."
                                                 "variant :")
    perfact = False
    new_task = ""
    while not perfact:

        new_task = isaa.run_agent(clarification_agent, '')

        clarification_agent.add_message("assistant", new_task)

        u = input(":")

        if u == 'x':
            exit(0)
        if u == 'y':
            clarification_agent.add_message("system", "Return an Elaborate task for the next agent"
                                                      " consider what the user ask and the best variant.")
            new_task = isaa.run_agent(clarification_agent, '')
            perfact = True

        clarification_agent.add_message("user", u)

    create_agent.get_messages(create=True)
    create_agent.add_message("user", new_task)
    create_agent.add_message("system", infos_c)
    create_agent.add_message("system", "Generate new agent if needed to extend your skills")
    create_agent.add_message("assistant", "After long and detailed step by step thinking and evaluating,"
                                          " brainstormen of the task I comt to the conclusion")
    task_don = False

    create_agent_prompt = isaa.run_agent('self', f"Generate a prompt for an agent its"
                                                 f" task is to crate new agents fitting to ths task '''{new_task}'''"
                                                 f"to extend the skills of issa.",
                                         mode_over_lode='generate')
    u = ''
    while not task_don:
        agents = isaa.run_agent(create_agent, create_agent_prompt,
                                mode_over_lode='tools')
        task_don = isaa.test_task_done(agents)

        create_agent.add_message("assistant", agents)

        u = input(":")

        if u == 'x':
            exit(0)
        if u == 'y':
            clarification_agent.add_message("system", "return a summary of what happened during execution")
            agent_speak = isaa.run_agent(clarification_agent, '')
            task_don = True

        clarification_agent.add_message("user", u)

    task = new_task

    infos_c = f"List of Avalabel Agents : {agents}\n Tools : {list(self_agent_config.tools.keys())}\n" \
              f" Functions : {isaa.scripts.get_scripts_list()}\n Chains : {str(chains)} " \
              f"chains can be run withe the execute-chain tool "

    think_agent.get_messages(create=True)
    think_agent.add_message("user", task)
    think_agent.add_message("system", infos_c)
    think_agent.add_message("system", "Create 4 strategies with which you can solve this problem."
                                      "Specify the required agent tools and scripts in each strategie."
                                      " For each stratagem you should specify a success probability from 0% to 100%."
                                      "For each stratagem you should specify a deviation from the task probability "
                                      "from 100% to 0%. -> 0 = no deviation is in perfect alignment to the task."
                                      " 100 = full deviation not related to the task ")
    think_agent.add_message("assistant", "After long and detailed step by step thinking and evaluating,"
                                         " brainstormen of the task I have created the following strategies."
                                         "Strategies :")

    strategies = isaa.run_agent(think_agent, '')

    think_agent.add_message("assistant", strategies)

    think_agent.add_message("system", "Think about 3 further strategies with an lower Deviation then the best strategy."
                                      "Brainstorm new ideas and add old knowledge by extracted and or combined with "
                                      "new ideas."
                                      "Consider your stills,"
                                      " Reflect the successes "
                                      "as well as the deviation from the task at hand. Give both numbers.")

    strategies2 = isaa.run_agent(think_agent, '')

    think_agent.add_message("assistant", strategies2)

    think_agent.add_message("system", "Extract the best Strategie, and Elaborate them.")

    strategies_final = isaa.run_agent(think_agent, '')

    think_agent.add_message("assistant", "Final Strategie:\n" + strategies_final)

    plan = isaa.run_agent(think_agent, '', mode_over_lode='planing')

    think_agent.add_message("assistant", plan)
    think_agent.add_message("system", "Break the plan into smaller steps if necessary. write the plan and the steps "
                                      "so that it can be solved")

    step_step_plan = isaa.run_agent(think_agent, '')

    extracted_dict = generate_exi_dict(isaa, step_step_plan, True, list(self_agent_config.tools.keys()))

    print("Creating:", extracted_dict)

    if extracted_dict:
        task_done = False
        free_run = False
    else:
        task_done = True
        free_run = True

    while not task_done:
        # try:
        evaluation, chain_ret = isaa.execute_thought_chain(task, chains.get(extracted_dict['name']), self_agent_config)
        # except Exception as e:
        #    print(e, '🔴')
        #    return "ERROR"
        evaluation = evaluation[::-1][:300][::-1]
        pipe_res = isaa.text_classification(evaluation)
        # aweeeewedspeak(f"The evaluation of the chain is {evaluation} i am {int(pipe_res[0]['score'])*10} Peasant sure")
        print(chain_ret)
        print(pipe_res)
        if pipe_res[0]['label'] == "NEGATIVE":
            print('🟡')
            task_done = True
            if "y" in input("retry ? : "):
                task_done = False
            response = chain_ret[-1][1]
        else:
            print(pipe_res[0]['score'])
            print(f'🟢')
            task_done = True
            response = chain_ret[-1][1]

    new_agent = isaa.config["agents-name-list"][-1]
    if len(agents) != 3:
        for agent_name in agents[3:]:
            isaa.get_agent_config_class(agent_name)
    while free_run:
        env_text = f"""Welcome, you are in a hostile environment, your name is isaa.
    you have several basic skills 1. creating agents 2. creating some agents 3. using skills, agents and tools

    you have created {len(agents)}agents so far these are : {agents}.

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
            free_run = False
        elif len(user_val) > 3:
            task += "User: " + user_val


"""
It seems like you're looking for information on creating a virtual agent development lab and developing tools for agent testing. Here's a summary of the process and the features needed for a virtual agent development lab:

1. Integrated Development Environment (IDE): The lab should have an IDE that allows developers to write, test, and debug code for their virtual agents.
2. Agent Creation Tools: The lab should have tools that allow developers to create and customize virtual agents, including their appearance, behavior, and communication capabilities.
3. Agent Testing and Debugging Tools: The lab should have tools that allow developers to test and debug their virtual agents, including simulating different scenarios and environments.
4. Agent Deployment and Management Tools: The lab should have tools that allow developers to deploy and manage their virtual agents, including loading and saving agents, and monitoring their performance.
5. Collaboration and Sharing Tools: The lab should have tools that allow developers to collaborate and share their virtual agents with others, including version control and sharing features.
6. User-Friendly Interface: The lab should have a user-friendly interface that is easy to navigate and use, even for non-technical users.
7. Documentation and Support: The lab should have comprehensive documentation and support resources, including tutorials, FAQs, and user forums.

To create a virtual agent development lab, follow these steps:

1. Define the purpose and scope of the lab.
2. Choose the development platform.
3. Develop the IDE.
4. Create agent creation tools.
5. Develop agent testing and debugging tools.
6. Create agent deployment and management tools.
7. Develop collaboration and sharing tools.
8. Design a user-friendly interface.
9. Develop documentation and support resources.
10. Test and refine the lab.

Developing tools for agent testing involves creating a system that can simulate different scenarios and environments, identify and fix errors and bugs, and provide feedback to developers. This will help ensure that the virtual agents are functioning correctly and efficiently.
"""
