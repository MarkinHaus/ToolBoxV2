from toolboxv2.util.agent.isaa_talk import get_audio_transcribe




# get user Input.
user_txt = get_audio_transcribe()


# get ai response
# get relevant Context for ai
def genrate_memory_prompt(log_memory, short_memory, objective, compleat_text):
    big_context = log_memory.get_relevant(compleat_text[-10:])
    pin_context = short_memory.get_relevant(objective)

    return f"""
    Memory-Section:
    ###
    Overall Context : {big_context}
    ---
    Specific Context : {pin_context}
    ###
    """
