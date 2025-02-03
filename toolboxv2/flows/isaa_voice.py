from toolboxv2 import App, AppArgs, TBEF

import threading
import time
import os
import signal
import pyperclip
import queue
from typing import Optional, Dict, Callable
from PIL import Image
from dataclasses import dataclass
import curses

# For Windows
import pystray
from pystray import MenuItem as Item
from pystray import Menu

# Timing adjustment for text responses
TRANSCRIPTION_WAIT_TIME = 8.75
TEXT_CHUNK_WAIT_TIME = 6.0


@dataclass
class Command:
    name: str
    callback: Callable
    description: str


class SystemTrayApp:
    def __init__(self, commands: Dict[str, Command], app):
        self.commands = commands
        self.app = app
        self.running = True

        # Create base icon
        icon_path = os.path.join(app.start_dir, 'web', 'icons', 'icon.png')
        if not os.path.exists(icon_path):
            # Create a patterned icon using the documented method
            self.icon = self.create_default_icon()
        else:
            self.icon = Image.open(icon_path)

    def create_default_icon(self):
        """Create a patterned icon using the pystray documented method"""
        width, height = 64, 64
        image = Image.new('RGB', (width, height), 'blue')
        return image

    def create_menu(self) -> Menu:
        """Create the menu structure using the new Menu API"""
        return Menu(
            # Voice Controls submenu
            *[Item(
                'Voice Controls',
                Menu(
                    Item(
                        'Start Voice Assistant',
                        lambda icon, item: self.execute_command(self.commands['voice_dispatch'])
                    ),
                    Item(
                        'Voice Recording',
                        lambda icon, item: self.execute_command(self.commands['voice_dump'])
                    ),
                    Item(
                        'Voice Conversation',
                        lambda icon, item: self.execute_command(self.commands['conversation'])
                    )
                )
            ),
                # Analysis Tools submenu
                Item(
                    'Analysis Tools',
                    Menu(
                        Item(
                            'Analyze Context',
                            lambda icon, item: self.execute_command(self.commands['analyze'])
                        ),
                        Item(
                            'Get References',
                            lambda icon, item: self.execute_command(self.commands['references'])
                        ),
                        Item(
                            'Mini Task',
                            lambda icon, item: self.execute_command(self.commands['mini_task'])
                        )
                    )
                ),
                # File Processing submenu
                Item(
                    'File Processing',
                    Menu(
                        Item(
                            'Process Web URL',
                            lambda icon, item: self.execute_command(self.commands['web_url'])
                        ),
                        Item(
                            'Process Local File',
                            lambda icon, item: self.execute_command(self.commands['local_file'])
                        )
                    )
                ),
                Menu.SEPARATOR,
                Item(
                    'Exit',
                    self.exit_app
                )]
        )

    def execute_command(self, command: Command):
        """Execute a command and handle the clipboard operations"""
        try:
            context = pyperclip.paste()
            result = command.callback(context)
            if result:
                pyperclip.copy(result)
            print(f"Executed: {command.name}")
        except Exception as e:
            print(f"Error executing {command.name}: {e}")

    def exit_app(self, icon: pystray.Icon, item):
        """Properly handle application exit"""
        self.running = False
        icon.stop()
        self.app.alive = False

    def setup(self, icon):
        """Setup function to run after the icon is ready"""
        icon.visible = True

    def run(self):
        """Run the system tray application with proper setup"""
        icon = pystray.Icon(
            name="SystemTray",
            icon=self.icon,
            title="AI Assistant Controls",
            menu=self.create_menu()
        )

        # Use the recommended setup pattern
        print("Tray icon Online")
        icon.run(setup=self.setup)


class TerminalUI:
    def __init__(self, commands: Dict[str, Command], app):
        self.commands = commands
        self.app = app
        self.running = True
        self.current_selection = 0
        signal.signal(signal.SIGINT, self.handle_sigint)

    def handle_sigint(self, signum, frame):
        self.running = False
        self.app.alive = False

    def draw_menu(self, stdscr):
        while self.running:
            stdscr.clear()
            height, width = stdscr.getmaxyx()

            # Draw header
            header = "AI Assistant Controls"
            stdscr.addstr(0, (width - len(header)) // 2, header, curses.A_BOLD)

            # Draw categories and commands
            current_line = 2
            categories = {
                "Voice Controls": ["Start Voice Assistant", "Voice Recording", "Voice Conversation"],
                "Analysis Tools": ["Analyze Context", "Get References", "Mini Task"],
                "File Processing": ["Process Web URL", "Process Local File"]
            }

            command_map = {}  # Map menu items to commands
            item_counter = 0

            for category, items in categories.items():
                if current_line < height - 1:
                    stdscr.addstr(current_line, 2, category, curses.A_BOLD)
                    current_line += 1

                    for item in items:
                        if current_line < height - 1:
                            # Highlight selected item
                            if item_counter == self.current_selection:
                                stdscr.addstr(current_line, 4, f"> {item}", curses.A_REVERSE)
                            else:
                                stdscr.addstr(current_line, 4, f"  {item}")

                            # Map menu item to command
                            command_map[item_counter] = next(
                                (cmd for cmd in self.commands.values() if cmd.name == item),
                                None
                            )

                            item_counter += 1
                            current_line += 1

                    current_line += 1

            # Draw footer
            footer = "↑/↓: Navigate | Enter: Select | q: Quit"
            if height - 1 > current_line:
                stdscr.addstr(height - 2, (width - len(footer)) // 2, footer)

            stdscr.refresh()

            # Handle input
            try:
                key = stdscr.getch()
                if key == ord('q'):
                    self.running = False
                    self.app.alive = False
                    break
                elif key == curses.KEY_UP and self.current_selection > 0:
                    self.current_selection -= 1
                elif key == curses.KEY_DOWN and self.current_selection < item_counter - 1:
                    self.current_selection += 1
                elif key == ord('\n'):  # Enter key
                    if command_map.get(self.current_selection):
                        context = pyperclip.paste()
                        result = command_map[self.current_selection].callback(context)
                        if result:
                            pyperclip.copy(result)

                        # Show execution message
                        msg = f"Executed: {command_map[self.current_selection].name}"
                        stdscr.addstr(height - 1, 0, msg + " " * (width - len(msg) - 1))
                        stdscr.refresh()
                        time.sleep(1)
            except curses.error:
                continue

    def run(self):
        curses.wrapper(self.draw_menu)


def analyze_speech_queue(text_queue: queue.Queue,
                         punctuation_pause: float = 1.5,
                         max_pause: float = 3.0) -> str:
    """
    Analyzes a queue of spoken text and determines when the speaker has finished
    based on punctuation and pause duration.

    Args:
        text_queue: Queue containing incoming text segments
        punctuation_pause: Time to wait after punctuation (default 1.5s)
        max_pause: Maximum time to wait for new text (default 5.0s)

    Returns:
        Complete spoken text as a string
    """
    full_text = []
    last_text_time = time.time()

    def get_next_text(timeout__: float) -> Optional[str]:
        try:
            return text_queue.get(timeout=timeout__)
        except queue.Empty:
            return None

    while True:
        # Get next text segment with timeout
        current_time = time.time()
        elapsed_time = current_time - last_text_time

        # If max pause time exceeded, return accumulated text
        if elapsed_time >= max_pause and full_text:
            return ' '.join(full_text)

        # Calculate remaining timeout
        timeout_ = min(
            max_pause - elapsed_time if full_text else max_pause,
            punctuation_pause if full_text and full_text[-1][-1] in '.!?' else max_pause
        )

        text_segment = get_next_text(timeout_)

        # If no new text received
        if text_segment is None:
            if full_text:
                # Return if we hit punctuation pause or max pause
                if (full_text[-1][-1] in '.!?' and elapsed_time >= punctuation_pause) or elapsed_time >= max_pause:
                    return ' '.join(full_text)
            continue
        print(text_segment, end=' ')
        # Add new text and update timestamp
        full_text.append(text_segment.strip())
        last_text_time = time.time()


NAME = "v-isaa"
llm_text = [""]


def run(app: App, _: AppArgs, voice=True, timeout=8):
    from toolboxv2.mods.isaa import Tools

    from toolboxv2.mods.isaa.extras.modes import ConversationMode, PreciseResponder
    from toolboxv2.mods.isaa.isaa_modi import browse_website
    from toolboxv2.mods.isaa.subtools.file_loder import route_local_file_to_function
    from toolboxv2.mods.isaa.extras.session import isaa_mode_dispatch, intelligent_isaa_dispatcher, ChatSession

    if isinstance(timeout, str):
        timeout = int(timeout)

    isaa: Tools = app.get_mod('isaa')

    if not app.mod_online("audio"):
        app.print(f"TalK is starting AUDIO")
        audio_mod = app.load_mod("audio")
        if audio_mod:
            audio_mod.on_start()
        isaa.speak = audio_mod.speech_stream

    def stream_text_audio(text, *args, **kwargs):
        llm_text[0] += text
        if len(text.split('\n')) > 1:
            if llm_text[0] == "":
                return text
            if not voice:
                print(llm_text[0])
                llm_text[0] = ""
                return text
            app.run_any(TBEF.AUDIO.SPEECH, text=llm_text[0], voice_index=0,
                        use_cache=False,
                        provider="piper",
                        config={'play_local': True, 'model_name': 'ryan'},
                        local=True,
                        save=False)
            llm_text[0] = ""
        return text

    # Fastest vector db Qdrant Chroma milvus ...
    # isaa.register_agents_setter(lambda x: x.set_verbose(True))
    agent = isaa.init_isaa(name='self', build=True, v_name=app.id.split('-')[0])

    # Analysiere den Titel

    put, res_que = app.get_function(("audio", "transcript"))[0](rate=16000, chunk_duration=4.0, amplitude_min=30, microphone_index=-1)
    isaa.agent_memory.ai_content = True
    cs = ChatSession(isaa.get_context_memory())

    def complexity_callback(user_input, complexity):

        if complexity < 4:
            return

        def helper():
            out = isaa.mini_task_completion(
                "Generate an short intermediate answer for that request outlining main steps",
                user_input, mode=isaa.controller.rget(PreciseResponder))
            stream_text_audio(out)

        threading.Thread(target=helper, daemon=True).start()

    def get_user_input_text():
        put("start")
        time.sleep(3)
        try:
            result = res_que.get(timeout=timeout)
        except queue.Empty:
            put("stop")
            return ""
        result += analyze_speech_queue(res_que)
        put("stop")
        return result

    def voice_intelligent_isaa_dispatcher():
        called, text_start = app.get_function(("audio", "wake_word"))[0](word="Computer", variants=["system",
                                                                                                    "computer", "pc"],
                                                                         microphone_index=-1,
                                                                         amplitude_min=-1.,
                                                                         model="openai/whisper-small",
                                                                         do_exit=False,
                                                                         do_stop=False, ques=(put, res_que))
        if not called:
            print("No Wake word")
            return ""
        text_start += analyze_speech_queue(res_que)
        put("stop")

        if not text_start:
            print("No text input")
            return

        agent.stream = True
        agent.stream_function = stream_text_audio
        llm_message = agent.get_llm_message(text_start, persist=True, fetch_memory=False,
                                            isaa=isaa,
                                            task_from="user", all_meme=False)

        out = agent.run_model(llm_message=llm_message, persist_local=True,
                                      persist_mem=False)
        agent.stream_function = False
        if agent.if_for_fuction_use(out):
            out += agent.execute_fuction(persist=True, persist_mem=False)

        notes = isaa.agent_memory.process_data(out)
        content = text_start + '\n\n' + notes[0].content
        if not content:
            print("No content")
            return ""

        evaluation = isaa.mini_task_completion(
            f"evaluate in the content is a task only reply withe None or the task it self ok?",
            user_task=f"content: '{content}'",
            examples=[
                f"content: 'what is the wetter in Berlin?'",
                'wetter in Berlin?',
                f"content: 'Hey whats going on'",
                'None',
            ], task_from="system", max_tokens=10)

        if agent.fuzzy_string_match(evaluation, ['None', 'none', 'task']) != 'task':
            print("Fuzzy", evaluation)
            return ""

        agent_task = evaluation

        mode = isaa_mode_dispatch(agent_task, isaa)
        input_infos, result = intelligent_isaa_dispatcher(isaa, agent_task, mode=isaa.controller.get(mode),
                                                          complexity_rating_callback=complexity_callback)

        if input_infos.get('complexity_rating') > 2:
            result = isaa.mini_task_completion("""### Task: Generate a User-Friendly, Conversational Summary of System Actions

        **Objective:**
        Create an agent that processes system actions and generates a natural, user-friendly, and conversational response based on the provided user request and preferences.

        ---

        #### Inputs:
        1. **System Actions**: A list of actions executed by the system, containing:
           - **Action Type**: e.g., "file deleted," "database query," "API call."
           - **Timestamp**: When each action occurred.
           - **Status**: e.g., "successful," "failed."
           - **Details**: Any relevant information about the action, such as results or data modified.

        2. **User Request**: The original user request or question to provide context for the agent's response. For example, the user might ask, "Did the database update go through?"

        3. **User Preferences**: Preferences for response style, including:
           - **Conciseness**: e.g., "concise" for a brief summary, or "detailed" for more in-depth explanations.

        ---

        #### Output:
        **A User-Friendly, Conversational Response:**
        The response should directly address the user's request, summarizing the actions in a natural manner, similar to a helpful AI assistant (e.g., Jarvis). Depending on user preferences, the response should adjust its level of detail.

        """, user_task=text_start + "\nSYSTEM RESULT\n " + result, mode=isaa.controller.rget(PreciseResponder),
                                               message=cs.get_past_x(3))

        app.run_any(TBEF.AUDIO.SPEECH, text=result, voice_index=0,
                    use_cache=False,
                    provider="piper",
                    config={'play_local': True, 'model_name': 'ryan'},
                    local=True,
                    save=False)

    def mian_voice_intelligent_isaa_dispatcher(users_context):

        text_start = get_user_input_text()

        if not text_start:
            print("No text input")
            return

        mode = isaa_mode_dispatch(text_start, isaa)
        input_infos, result = intelligent_isaa_dispatcher(isaa, text_start, mode=isaa.controller.get(mode),
                                                          complexity_rating_callback=complexity_callback)

        if input_infos.get('complexity_rating') > 2:
            result = isaa.mini_task_completion("""### Task: Generate a User-Friendly, Conversational Summary of System Actions

        **Objective:**
        Create an agent that processes system actions and generates a natural, user-friendly, and conversational response based on the provided user request and preferences.

        ---

        #### Inputs:
        1. **System Actions**: A list of actions executed by the system, containing:
           - **Action Type**: e.g., "file deleted," "database query," "API call."
           - **Timestamp**: When each action occurred.
           - **Status**: e.g., "successful," "failed."
           - **Details**: Any relevant information about the action, such as results or data modified.

        2. **User Request**: The original user request or question to provide context for the agent's response. For example, the user might ask, "Did the database update go through?"

        3. **User Preferences**: Preferences for response style, including:
           - **Conciseness**: e.g., "concise" for a brief summary, or "detailed" for more in-depth explanations.

        ---

        #### Output:
        **A User-Friendly, Conversational Response:**
        The response should directly address the user's request, summarizing the actions in a natural manner, similar to a helpful AI assistant (e.g., Jarvis). Depending on user preferences, the response should adjust its level of detail.

        """, user_task=text_start + "\nSYSTEM RESULT\n " + result, mode=isaa.controller.rget(PreciseResponder),
                                               message=cs.get_past_x(3))

        app.run_any(TBEF.AUDIO.SPEECH, text=result, voice_index=0,
                    use_cache=False,
                    provider="piper",
                    config={'play_local': True, 'model_name': 'ryan'},
                    local=True,
                    save=False)

    def mini_task_run(users_context):

        text_start = get_user_input_text()
        if not text_start:
            print("No text input")
            return
        result = isaa.run(text_start + ("\nUsers chipboard context: " + users_context), reference=False,
                          persist=True)

        app.run_any(TBEF.AUDIO.SPEECH, text=result, voice_index=0,
                    use_cache=False,
                    provider="piper",
                    config={'play_local': True, 'model_name': 'ryan'},
                    local=True,
                    save=False)

    def voice_dump(_):

        put("start")

        time.sleep(6.75)
        print("Start recording")
        transcription = ""
        # Get transcribed results
        while not res_que.empty():
            x = res_que.get()
            transcription += x
            print(f"Transcribed text: {x}")
            time.sleep(4.75)
            if '\n' in transcription or '.' in transcription or len(transcription) > 250:
                isaa.agent_memory.process_data(transcription)
                transcription = ""
        # Stop the transcription process
        put("stop")

    def conversation_voice_dump(_):

        while user_input := get_user_input_text():
            if 'stop conversation' in user_input:
                return

            isaa.agent_memory.process_data(user_input)

            agent.mode = isaa.controller.rget(ConversationMode)
            agent.stream = True
            agent.stream_function = stream_text_audio
            res = isaa.run_agent("self", user_input, running_mode="once", persist=True, fetch_memory=True)
            agent.mode = None
            isaa.agent_memory.process_data(res)

    def helper_c_to_r(user_context, sys_context):
        out = isaa.mini_task_completion(
            f"Generate an Contextual relevant response for the user based on the user given context and the system "
            f"context : {sys_context}",
            user_context, mode=isaa.controller.rget(PreciseResponder))
        stream_text_audio(out)
        return sys_context

    def get_ref(user_context):
        sys_context = isaa.agent_memory.search_notes(user_context)
        print(sys_context)
        return helper_c_to_r(user_context, sys_context)

    def main_analyses(user_context):
        sys_graph = isaa.agent_memory.analyze_and_process_vault(process=True, all_files=False)
        return sys_graph

    def inject_web_url(url):
        question = get_user_input_text()
        if not question:
            print("No question")
            return
        content = browse_website(url, question, isaa.mas_text_summaries)
        isaa.agent_memory.process_data(content)
        return helper_c_to_r(question, content)

    def exit_helper(*_):
        app.alive = False

    def path_getter(path):
        loder, docs_loder = route_local_file_to_function(path)
        docs = docs_loder()
        notes = isaa.agent_memory.process_data([doc.page_content for doc in docs])
        return '\n'.join([n.filename for n in notes])

    # Function to handle triggering the registered function

    commands = {}

    # Function to register keyboard shortcuts
    commands["voice_dispatch"] = Command(
        "Start Voice Assistant",
        mian_voice_intelligent_isaa_dispatcher,
        "Start voice command processing"
    )

    commands["voice_dump"] = Command(
        "Voice Recording",
        voice_dump,
        "Record and process voice input"
    )

    commands["conversation"] = Command(
        "Voice Conversation",
        conversation_voice_dump,
        "Start interactive voice conversation"
    )

    commands["analyze"] = Command(
        "Analyze Context",
        main_analyses,
        "Analyze and process system context"
    )

    commands["references"] = Command(
        "Get References",
        get_ref,
        "Retrieve contextual references"
    )

    commands["mini_task"] = Command(
        "Mini Task",
        mini_task_run,
        "Execute a small task"
    )

    commands["web_url"] = Command(
        "Process Web URL",
        inject_web_url,
        "Process content from a web URL"
    )

    commands["local_file"] = Command(
        "Process Local File",
        path_getter,
        "Process a local file"
    )

    # Create and run appropriate UI based on platform
    #if platform.system() == "Windows":
    #    ui = SystemTrayApp(commands, app)
    #else:
    ui = TerminalUI(commands, app)

    # Run UI in separate thread
    ui_thread = threading.Thread(target=ui.run, daemon=True)
    ui_thread.start()

    # Keep main thread alive
    app.idle()

    # Cleanup
    put("exit")
    app.exit()
    print("\nExiting...")
