import pyperclip

from toolboxv2 import Spinner
from toolboxv2.utils.toolbox import get_app

"""Console script for toolboxv2. Isaa Conversation Tool"""
from toolboxv2.mods.isaa_audio import init_live_transcript, speech_stream
from toolboxv2.mods.isaa_extars.isaa_modi import sys_print, init_isaa

NAME = "isaa-talk"


def run(app, args):
    isaa, self_agent_config, chains = init_isaa(app, speak_mode=True, calendar=True, ide=True, create=True,
                                                isaa_print=False, python_test=True, init_mem=True, init_pipe=True,
                                                join_now=True, global_stream_override=True, chain_runner=True
                                                )

    mean = 82#s30sek_mean(seconds=.5, p=True)
    comm, que = init_live_transcript(chunk_duration=1.55, amplitude_min=mean)

    mode = 'free'
    alive = True

    self_agent_config.stream = True
    self_agent_config.max_tokens = 4012
    self_agent_config.set_completion_mode('chat')
    self_agent_config.set_model_name('gpt-4')

    user_text = """
    Based on the information provided, Markin is currently studying MINT green, an orientation program for computer science studies at TU Berlin. However, he is not satisfied with the quality of education and is considering studying abroad or taking a gap year. To create a life plan for the next 2 years, we need more information about Markin's goals, interests, and priorities.

To help Markin, the following agents and tools can be utilized:

1. Career counselor: To identify career goals, interests, and suggest suitable paths and educational programs.
2. Education consultant: To evaluate the current educational program and suggest alternatives or universities that align with Markin's goals.
3. Financial advisor: To evaluate financial situations and suggest ways to finance education and living expenses.
4. Language learning tools: To learn a new language if studying abroad, such as Duolingo or Rosetta Stone.
5. Travel booking tools: To make travel arrangements, such as Expedia or Kayak.
6. Time management tools: To balance studies and other activities, such as calendars, to-do lists, and productivity apps.
7. Communication tools: To stay in touch with family and friends, such as Skype, WhatsApp, or Zoom.

The following skills would be helpful for this task:

- Get a differentiated point of view: To understand Markin's perspective and priorities.
- Search for information: To gather information about potential universities or programs.
- Calendar entry: To organize and schedule important dates and deadlines.
- Generate documents: To create and organize documents related to education and travel plans.
- Utilize tools: To identify and use tools and resources for organizing and planning.
- Read, analyze, and write summaries: To summarize and organize information about potential universities or programs.

It is essential to gather more information about Markin's situation and preferences before making any decisions or recommendations.

Ask questions to help to find a decisions or recommendations.
    """

    while alive:
        while not que.empty():
            data = que.get()
            print(f'\n{data}\n')
            user_text += data + ' '
            if 'stop' in data:
                comm('stop')
                user_text = ''
            if 'ende' in data:
                comm('stop')
                comm('exit')
                alive = False

        buffer = input(f"\ncurrent input : {user_text}\ncurrent mode : {mode}\ncommands ar : exit\n\tstart #s"
                       "\n\treset input #r"
                       "\n\treset and stop #rs"
                       "\n\trun isaa #"
                       "\n\tclear memory #cl"
                       "\n"
                       "\nswitch mode commands:"
                       "\n\texecution #e"
                       "\n\tfree #f"
                       "\n\tconversation #c \n\n:")

        if buffer.endswith("exit"):
            alive = False
            que.put('stop')
        if buffer.endswith("#e"):
            mode = 'execution'
            print(f"Switch to {mode=}")
        if buffer.endswith("#f"):
            mode = 'free'
            print(f"Switch to {mode=}")
        if buffer.endswith("#c"):
            mode = 'conversation'
            print(f"Switch to {mode=}")
        if buffer.endswith("#s"):
            comm('start')
            print(f"Start")
        if buffer.endswith("#rs"):
            comm('stop')
            print(f"stop")
            user_text = ''
        if buffer.endswith("#r"):
            user_text = ''
        if buffer.endswith("#cl"):
            self_agent_config.short_mem.memory_data = []
            self_agent_config.observe_mem.memory_data = []
            self_agent_config.edit_text.memory_data = []
            isaa.get_chain().load_from_file()
            print("Memory cleared")
        if buffer.endswith("#"):
            if buffer.endswith("##"):
                user_text = buffer.replace("##", "")
            print("You pressed enter!")
            comm('stop')
            # self_agent_config.set_completion_mode('chat').set_model_name('gpt-3.5-turbo-0613')
            while not que.empty():
                user_text += que.get() + ' '
            sys_print("\n================================Starting-Agent================================")
            print("User text :", user_text)
            context = pyperclip.paste()
            if context:
                self_agent_config.edit_text.text = context

            res = isaa.run_agent(self_agent_config, user_text, mode_over_lode='free')
            print(res)
            with Spinner("Generating audio..."):
                speech_stream(isaa.mas_text_summaries(res, min_length=50), voice_index=0)

            user_text = ''
            print("\n")

    comm('stop')
    comm('exit')
    print("Auf wieder sehen")

    qa = isaa.init_db_questions('main', self_agent_config)
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


if __name__ == '__main__':
    isaa, self_agent_config, chains = init_isaa(get_app('debug'), speak_mode=True, calendar=True, ide=True, create=True,
                                                isaa_print=False, python_test=True, init_mem=True, init_pipe=True,
                                                join_now=True, global_stream_override=True, chain_runner=True
                                                )
    res = """
    Based on the information provided, Markin is currently studying MINT green, an orientation program for computer science studies at TU Berlin. However, he is not satisfied with the quality of education and is considering studying abroad or taking a gap year. To create a life plan for the next 2 years, we need more information about Markin's goals, interests, and priorities.

To help Markin, the following agents and tools can be utilized:

1. Career counselor: To identify career goals, interests, and suggest suitable paths and educational programs.
2. Education consultant: To evaluate the current educational program and suggest alternatives or universities that align with Markin's goals.
3. Financial advisor: To evaluate financial situations and suggest ways to finance education and living expenses.
4. Language learning tools: To learn a new language if studying abroad, such as Duolingo or Rosetta Stone.
5. Travel booking tools: To make travel arrangements, such as Expedia or Kayak.
6. Time management tools: To balance studies and other activities, such as calendars, to-do lists, and productivity apps.
7. Communication tools: To stay in touch with family and friends, such as Skype, WhatsApp, or Zoom.

The following skills would be helpful for this task:

- Get a differentiated point of view: To understand Markin's perspective and priorities.
- Search for information: To gather information about potential universities or programs.
- Calendar entry: To organize and schedule important dates and deadlines.
- Generate documents: To create and organize documents related to education and travel plans.
- Utilize tools: To identify and use tools and resources for organizing and planning.
- Read, analyze, and write summaries: To summarize and organize information about potential universities or programs.

It is essential to gather more information about Markin's situation and preferences before making any decisions or recommendations.
"""

    next = """
Removed ~ 630 tokens from ShortTerm tokens in use: 375 max : 448
None
isaa: Running agent self Coordination
isaa: TOKENS: 1287 | left = 448
isaa: SYSTEM: chucks to summary: 7 cap : 800
isaa: SYSTEM: final summary from 5483:7 -> 862 compressed 6.36X

Generating response (/) stream (\) self gpt-4 Coordination chat
To create a comprehensive life plan for the next two years, including a gap year in Berlin, we need to consider several factors such as academic goals, personal development, and financial planning. Here is a suggested life plan:

1. Academic Goals:
- Research and identify the specific computer science courses and programs offered at TU Berlin or other universities in Berlin.
- Determine the prerequisites and admission requirements for these programs.
- Plan to complete any necessary prerequisite courses or exams during the first year.
- Apply for the chosen computer science program at TU Berlin or another university in Berlin during the second year.

2. Personal Development:
- Utilize the gap year to gain practical experience in the field of computer science through internships, part-time jobs, or volunteering.
- Attend workshops, conferences, and networking events related to computer science to expand your knowledge and connections in the industry.
- Develop soft skills such as communication, teamwork, and problem-solving through participation in clubs, organizations, or online courses.
- Learn German to enhance your cultural experience in Berlin and improve your chances of securing internships or job opportunities.

3. Financial Planning:
- Research and apply for scholarships, grants, or financial aid programs available for international students studying in Berlin.
- Create a budget for living expenses, tuition fees, and other costs associated with studying and living in Berlin.
- Explore part-time job opportunities or freelance work to supplement your income during your gap year and while studying.
- Save money during the first year to cover expenses during the second year of the life plan.

4. Travel Plans:
- Research and plan trips to explore Germany and other European countries during your gap year and study breaks.
- Consider joining a travel group or connecting with other students to share travel experiences and reduce costs.

5. Health and Well-being:
- Maintain a balanced lifestyle by incorporating regular exercise, a healthy diet, and sufficient sleep.
- Seek out social opportunities to make friends and build a support network in Berlin.
- Utilize university resources such as counseling services or mental health support if needed.

By following this life plan, you can make the most of your gap year in Berlin and set yourself up for success in your computer science studies at TU Berlin or another university in the city.
------stream-end------
    """

    text = isaa.mas_text_summaries(res, min_length=50)

    with Spinner("Generating audio..."):
        speech_stream(text, voice_index=0)
