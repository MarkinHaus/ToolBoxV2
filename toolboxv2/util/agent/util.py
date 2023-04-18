from toolboxv2.util.TBConfig import Config


class PromptConfig:
    FORMAT: str = """{
    "thoughts":
    {
        "text": "thought",
        "reasoning": "reasoning",
        "plan": "- short bulleted\n- list that conveys\n- long-term plan",
        "criticism": "constructive self-criticism",
        "speak": "thoughts summary to say to user"
    },
    "command": {
        "name": "command name",
        "args":{
            "arg name": "value"
        }
    },
}"""
    EVALUATION: str = """1. ~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.
2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.
3. Exclusively use the commands listed in double quotes e.g. "command name"
"""
    COMMANDS: str = """1. Talk to User: "talk", args: "speak": "<text for user>"
2. Use Tools: "use_tools", args: "goal": "<what_you_want_to_accomplish>"
3. Brake down Task: "divide_et_impera",  args: "task": "<on_task_form_task_list>", "task": "<short_task_desc>"
"""
    RESOURCES: str = """1. Internet access for searches and information gathering.
2. Long Term memory management.
3. GPT-3.5 powered Agents for delegation of simple tasks.
4. File output."""
    CONSTRAINTS: str = """1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
2. Constructively self-criticize your big-picture behavior constantly.
3. Reflect on past decisions and strategies to refine your approach.
4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.

You should only respond in JSON format as described below"""

    def get_default_prompt(self):
        return f"""
CONSTRAINTS:\n{self.CONSTRAINTS}
COMMANDS:\n{self.COMMANDS}
RESOURCES:\n{self.RESOURCES}
PERFORMANCE EVALUATION:\n{self.EVALUATION}
You should only respond the format as described below.
RESPONSE FORMAT:\n{self.FORMAT}
Ensure the response can be parsed.
"""


def get_prompt(config):
    return f"""
CONSTRAINTS:\n{config.CONSTRAINTS}
COMMANDS:\n{config.COMMANDS}
RESOURCES:\n{config.RESOURCES}
PERFORMANCE EVALUATION:\n{config.EVALUATION}
You should only respond the format as described below.
RESPONSE FORMAT:\n{config.FORMAT}
Ensure the response can be parsed.
"""


def question(prompt: str, user_input: str, role: str):
    return f"""
    {prompt}
    generate the next
    {role}: {user_input}
    """

def create_chat_message(role, content):
    """
    Create a chat message with the given role and content.

    Args:
    role (str): The role of the message sender, e.g., "system", "user", or "assistant".
    content (str): The content of the message.

    Returns:
    dict: A dictionary containing the role and content of the message.
    """
    return {"role": role, "content": content}
def welcome_sequence():
    start_prompt = """

    """
    # Isaa ask user if they want to continue on last project or start a new on.
    # Yes
    # 1. Isaa gets relevante information about resent project to name ()
    # 2. Do project specific stuff
    # No
    # 1. Design new Project wirth Isaa By Imagen a Goal.
    # 2.
    pass
