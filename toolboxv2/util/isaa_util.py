import random
import time

from toolboxv2 import Style


def print_to_console(
    title,
    title_color,
    content,
    min_typing_speed=0.05,
    max_typing_speed=0.01):
    print(title_color + title + " " + Style.BLUE(""), end="")
    if content:
        if isinstance(content, list):
            content = " ".join(content)
        if not isinstance(content, str):
            print(f"SYSTEM NO STR type : {type(content)}")
            print(content)
            return
        words = content.split()
        for i, word in enumerate(words):
            print(word, end="", flush=True)
            if i < len(words) - 1:
                print(" ", end="", flush=True)
            typing_speed = random.uniform(min_typing_speed, max_typing_speed)
            time.sleep(typing_speed)
            # type faster after each word
            min_typing_speed = min_typing_speed * 0.95
            max_typing_speed = max_typing_speed * 0.95
    print()
