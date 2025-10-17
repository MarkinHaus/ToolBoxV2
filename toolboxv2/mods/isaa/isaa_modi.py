import os

from toolboxv2.mods import BROWSER

PIPLINE = None

try:
    from toolboxv2.mods.isaa_audio import (
        get_audio_transcribe,
        s30sek_mean,
        speech_stream,
        text_to_speech3,
    )

    SPEAK = True
except ImportError:
    SPEAK = False

try:
    import inquirer

    INQUIRER = True
except ImportError:
    INQUIRER = False

from toolboxv2.utils.extras.Style import Style, print_to_console


def sys_print(x, **kwargs):
    print_to_console("SYSTEM:", Style.style_dic['BLUE'], x, max_typing_speed=0.04, min_typing_speed=0.08)



def get_multiline_input(init_text="", line_starter=""):
    lines = []
    if init_text:
        print(init_text, end='')
    while True:
        line = input(line_starter)
        if line:
            lines.append(line)
        else:
            break
    return "\n".join(lines)


try:
    import inquirer

    INQUIRER = True
except ImportError:
    INQUIRER = False


def choiceList(all_chains, print_=print, input_=input, do_INQUIRER=True):
    all_chains += ['None']
    if INQUIRER and do_INQUIRER:

        questions = [
            inquirer.List('chain',
                          message="Choose a chain?",
                          choices=all_chains,
                          ),
        ]
        choice = inquirer.prompt(questions)['chain']

    else:
        choice = input_(f"{all_chains} select one (q) to quit:")
        while choice not in all_chains:
            if choice.lower() == 'q':
                return "None"
            print_("Invalid Chain name")
            choice = input_(f"{all_chains} select one (q) to quit:")
    return choice


def show_image_in_internet(images_url, browser=BROWSER):
    if isinstance(images_url, str):
        images_url = [images_url]
    for image_url in images_url:
        os.system(f'start {browser} {image_url}')


