import json
import threading
import time
from dataclasses import dataclass
from typing import Callable

from whatsapp import WhatsApp

from toolboxv2.mods.WhatsAppTb.utils import ProgressMessenger, emoji_set_thermometer, emoji_set_work_phases
from toolboxv2.mods.WhatsAppTb.server import AppManager
from toolboxv2.mods.isaa import Tools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from datetime import datetime
import threading
from google.oauth2.credentials import Credentials
import logging
import json

@dataclass
class WhClient:
    messenger: WhatsApp
    disconnect: Callable
    s_callbacks: Callable
    progress_messenger0: ProgressMessenger
    progress_messenger1: ProgressMessenger
    progress_messenger2: ProgressMessenger



class AssistantState(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    MAINTENANCE = "maintenance"


@dataclass
class WhatsAppAssistant:
    whc: 'WhClient'
    isaa: 'Tools'
    agent: Optional['Agent'] = None
    credentials: Optional[Credentials] = None
    state: AssistantState = AssistantState.OFFLINE

    # Service clients
    gmail_service: Any = None
    calendar_service: Any = None

    # Progress messengers
    progress_messengers: Dict[str, 'ProgressMessenger'] = field(default_factory=dict)

    def __post_init__(self):
        self.setup_google_services()
        self.setup_progress_messengers()
        self.setup_interaction_buttons()
        self.state = AssistantState.ONLINE

    def setup_google_services(self):
        """Initialize Gmail and Calendar API services"""
        SCOPES = ['https://www.googleapis.com/auth/gmail.modify',
                  'https://www.googleapis.com/auth/calendar']
        # from google_auth_oauthlib.flow import InstalledAppFlow
        # from googleapiclient.discovery import build
        # flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        # self.credentials = flow.run_local_server(port=0)

        # self.gmail_service = build('gmail', 'v1', credentials=self.credentials)
        # self.calendar_service = build('calendar', 'v3', credentials=self.credentials)

    def setup_progress_messengers(self):
        """Initialize progress messengers for different types of tasks"""
        self.progress_messengers = {
            'task': self.whc.progress_messenger0,
            'email': self.whc.progress_messenger1,
            'calendar': self.whc.progress_messenger2
        }

    def setup_interaction_buttons(self):
        """Define WhatsApp interaction buttons for different functionalities"""
        self.buttons = {
            'main_menu': {
                'header': 'Digital Assistant',
                'body': 'Please select an option:',
                'footer': 'Choose wisely!',
                'action': {
                    'button': 'Menu',
                    'sections': [
                        {
                            'title': 'Main Functions',
                            'rows': [
                                {'id': 'agent', 'title': 'Agent Controls', 'description': 'Manage your AI assistant'},
                                {'id': 'email', 'title': 'Email Management', 'description': 'Handle your emails'},
                                {'id': 'calendar', 'title': 'Calendar', 'description': 'Manage your schedule'},
                                {'id': 'docs', 'title': 'Documents', 'description': 'Handle documents'},
                                {'id': 'system', 'title': 'System', 'description': 'System controls and metrics'}
                            ]
                        }
                    ]
                }
            },
            'agent_controls': self._create_agent_controls_buttons(),
            'email_controls': self._create_email_controls_buttons(),
            'calendar_controls': self._create_calendar_controls_buttons(),
            'docs_controls': self._create_docs_controls_buttons(),
            'system_controls': self._create_system_controls_buttons()
        }

    def _create_agent_controls_buttons(self):
        return {
            'header': 'Agent Controls',
            'body': 'Manage your AI assistant:',
            'action': {
                'button': 'Select',
                'sections': [{
                    'title': 'Actions',
                    'rows': [
                        {'id': 'start', 'title': 'Start Agent', 'description': 'Run in background'},
                        {'id': 'stop', 'title': 'Stop Agent', 'description': 'Stop current tasks'},
                        {'id': 'tasks', 'title': 'Task Stack', 'description': 'View and manage tasks'},
                        {'id': 'memory', 'title': 'Clear Memory', 'description': 'Reset agent memory'}
                    ]
                }]
            }
        }

    def _create_email_controls_buttons(self):
        return {
            'header': 'Email Management',
            'body': 'Handle your emails:',
            'action': {
                'button': 'Select',
                'sections': [{
                    'title': 'Actions',
                    'rows': [
                        {'id': 'check', 'title': 'Check Emails', 'description': 'View recent emails'},
                        {'id': 'send', 'title': 'Send Email', 'description': 'Compose new email'},
                        {'id': 'summary', 'title': 'Get Summary', 'description': 'Summarize emails'},
                        {'id': 'search', 'title': 'Search', 'description': 'Search emails'}
                    ]
                }]
            }
        }

    def _create_calendar_controls_buttons(self):
        return {
            'header': 'Calendar Management',
            'body': 'Manage your schedule:',
            'action': {
                'button': 'Select',
                'sections': [{
                    'title': 'Actions',
                    'rows': [
                        {'id': 'today', 'title': 'Today\'s Events', 'description': 'View today\'s schedule'},
                        {'id': 'add', 'title': 'Add Event', 'description': 'Create new event'},
                        {'id': 'upcoming', 'title': 'Upcoming', 'description': 'View upcoming events'},
                        {'id': 'find_slot', 'title': 'Find Time Slot', 'description': 'Find available time'}
                    ]
                }]
            }
        }

    def _create_docs_controls_buttons(self):
        return {
            'header': 'Document Management',
            'body': 'Handle your documents:',
            'action': {
                'button': 'Select',
                'sections': [{
                    'title': 'Actions',
                    'rows': [
                        {'id': 'upload', 'title': 'Upload', 'description': 'Add new document'},
                        {'id': 'list', 'title': 'List Documents', 'description': 'View all documents'},
                        {'id': 'search', 'title': 'Search', 'description': 'Search documents'},
                        {'id': 'delete', 'title': 'Delete', 'description': 'Remove document'}
                    ]
                }]
            }
        }

    def _create_system_controls_buttons(self):
        return {
            'header': 'System Controls',
            'body': 'System management:',
            'action': {
                'button': 'Select',
                'sections': [{
                    'title': 'Actions',
                    'rows': [
                        {'id': 'status', 'title': 'System Status', 'description': 'View current status'},
                        {'id': 'metrics', 'title': 'Metrics', 'description': 'System metrics'},
                        {'id': 'restart', 'title': 'Restart', 'description': 'Restart system'},
                        {'id': 'logs', 'title': 'Logs', 'description': 'View system logs'}
                    ]
                }]
            }
        }

    async def handle_message(self, message: 'Message'):
        """Main message handler for incoming WhatsApp messages"""
        # Mark message as read
        await message.mark_as_read()

        # Extract content and type
        content_type = message.get_type()
        content = message.get_content()

        if content_type == 'button_response':
            await self.handle_button_interaction(content, message)
        elif content_type == 'text':
            if content == "main_menu":
                await message.send_button(
                    recipient_id=message.sender,
                    button=self.buttons[content]
                )
            else:
                await self.handle_text_message(content, message)
        elif content_type in ['audio', 'document', 'image']:
            await message.reply("Sorry, I don't understand this type of message yet.")# self.handle_media_message(content, content_type, message)
        else:
            await message.reply("Sorry, I don't understand this type of message.")

    async def handle_button_interaction(self, content: str, message: 'Message'):
        """Handle button click interactions"""
        if content in self.buttons:
            await message.send_button(
                recipient_id=message.sender,
                button=self.buttons[content]
            )
        else:
            # Handle specific button actions
            actions = {
                'start': self.start_agent,
                'stop': self.stop_agent,
                'tasks': self.show_task_stack,
                # ... add more action handlers
            }
            if content in actions:
                await actions[content](message)

    async def handle_text_message(self, content: str, message: 'Message'):
        """Handle text messages"""
        if content.startswith('/'):
            await message.reply("Sorry, I don't understand this type of message yet.") #self.handle_command(content[1:], message)
        else:
            # Pass to agent for processing
            if self.agent:
                response = await self.agent.run(content)
                await message.reply(response)

    def start_agent(self):
        """Start the agent in background mode"""
        if self.agent:
            self.agent.run_in_background()
            return True
        return False

    def stop_agent(self):
        """Stop the currently running agent"""
        if self.agent:
            self.agent.stop()
            return True
        return False

    def show_task_stack(self):
        """Display current task stack"""
        if self.agent and self.agent.taskstack:
            tasks = self.agent.taskstack.tasks
            return "\n".join([f"Task {t.id}: {t.description}" for t in tasks])
        return "No tasks in stack"

    def run(self):
        """Start the WhatsApp assistant"""
        try:
            self.state = AssistantState.ONLINE
            # Send welcome message

            mas = self.whc.messenger.create_message(
                content="Digital Assistant is online! Send /help for available commands.",to=self.whc.progress_messenger0.recipient_phone,
            ).send(sender=0)
            mas_id = mas.get("messages", [{}])[0].get("id")
            print(mas_id)

        except Exception as e:
            logging.error(f"Assistant error: {str(e)}")
            self.state = AssistantState.OFFLINE
            raise


def connect(app, phone_number_id):
    key = app.config_fh.one_way_hash(phone_number_id, "WhatsappAppManager",
                                     AppManager.pepper)

    messenger, s_callbacks = AppManager().online(key)

    emoji_set_loading = ["🔄", "🌀", "⏳", "⌛", "🔃"]  # Custom Loading Emoji Set
    progress_messenger0 = ProgressMessenger(messenger, "", emoji_set=emoji_set_loading)
    progress_messenger1 = ProgressMessenger(messenger, "", emoji_set=emoji_set_thermometer)
    progress_messenger2 = ProgressMessenger(messenger, "", emoji_set=emoji_set_work_phases)
    whc = WhClient(messenger=messenger,
                   s_callbacks=s_callbacks,
                   disconnect=AppManager().offline(key),
                   progress_messenger0=progress_messenger0,
                   progress_messenger1=progress_messenger1,
                   progress_messenger2=progress_messenger2)

    #step_flag = threading.Event()
    #message_id = progress_messenger0.send_initial_message(mode="progress")
    #print(progress_messenger0.max_steps)
    #progress_messenger0.start_progress_in_background(step_flag=step_flag)
    #for i in range(progress_messenger0.max_steps):
    #    time.sleep(2)
    #    step_flag.set()
    # Simulate work, then stop loading
    # time.sleep(10)  # Simulate work duration
    # stop_flag.set()

    # stop_flag = threading.Event()
    # message_id = progress_messenger0.send_initial_message(mode="loading")
    # progress_messenger0.start_loading_in_background(stop_flag)

    # Simulate work, then stop loading
    # time.sleep(10)  # Simulate work duration
    # stop_flag.set()

    return whc

def runner(app, phone_number_id):

    whc = connect(app, phone_number_id)
    # setup
    isaa = app.get_mod("isaa")

    self_agent = isaa.get_agent_class("self")

    waa = WhatsAppAssistant(whc=whc, isaa=isaa, agent=self_agent, credentials=None)
    whc.s_callbacks(waa.handle_message, print)
    waa.run()

    return waa

# @dataclass is full inplent do not tuch only for help !!!
    # class WhClient:
    #     messenger: WhatsApp
    #     disconnect: Callable
    #     s_callbacks: Callable
    #     progress_messenger0: ProgressMessenger
    #     progress_messenger1: ProgressMessenger
    #     progress_messenger2: ProgressMessenger


    # @dataclass
    # class Task:
    #     id: str
    #     description: str
    #     priority: int
    #     estimated_complexity: float  # Range 0.0 to 1.0
    #     time_sensitivity: float  # Range 0.0 to 1.0
    #     created_at: datetime

    #   class TaskStack:
    #     def __init__(self):
    #         self.tasks: List[Task] = []
    #         self.current_task: Optional[Task] = None
    #         ...
    #
    #     def add_task(self, task: Task):
    #         ...
    #
    #     def _sort_tasks(self):
    #         ....
    #
    #     def get_next_task(self) -> Optional[Task]:
    #         ...
    #
    #     def remove_task(self, task_id: str):
    #         ...
    #
    #     def emtpy(self):
    #         return len(self.tasks) == 0


    #  class AgentState(Enum):
    #    IDLE = "idle"
    #    RUNNING = "running"
    #    STOPPED = "stopped"

    # @dataclass
    # class TaskStatus:
    #     task_id: str
    #     status: str  # queued, running, completed, error
    #     progress: float  # Range 0.0 to 1.0
    #     result: Optional[Any] = None
    #     error: Optional[str] = None

    # set up base client


"""


    @dataclass
    class Agent:
        amd: AgentModelData = field(default_factory=AgentModelData)

        stream: bool = field(default=False) # set Flase
        messages: List[Dict[str, str]] = field(default_factory=list)

        max_history_length: int = field(default=10)
        similarity_threshold: int = field(default=75)

        verbose: bool = field(default=logger.level == logging.DEBUG) # must be Tro for Clabbacks ( print_verbose )

        stream_function: Callable[[str], bool or None] = field(default_factory=print) # (live strem callback do not use ..

        taskstack: Optional[TaskStack] = field(default_factory=TaskStack)
        status_dict: Dict[str, TaskStatus] = field(default_factory=dict)
        state: AgentState = AgentState.IDLE
        world_model: Dict[str, str] = field(default_factory=dict)

        post_callback: Optional[Callable] = field(default=None) # gets calls d wit the final result str
        progress_callback: Optional[Callable] = field(default=None) # gets onlled wit an status object

        mode: Optional[LLMMode or ModeController] = field(default=None) # add an inteface to sent the modes ( isaa controller for modes :  @dataclass
                                            class ControllerManager:
                                                controllers: Dict[str, ModeController] = field(default_factory=dict)

                                                def rget(self, llm_mode: LLMMode, name: str = None):
                                                    if name is None:
                                                        name = llm_mode.name
                                                    if not self.registered(name):
                                                        self.add(name, llm_mode)
                                                    return self.get(name)

                                                def list_names(self):
                                                    return list(self.controllers.keys())
            # avalabel with isaa.controller

        last_result: Optional[Dict[str, Any]] = field(default=None) # opionl

        def show_word_model(self):
            if not self.world_model:
                return "balnk"
            return "Key <> Value\n" + "\n".join([f"{k} <> {v}" for k, v in self.world_model.items()])

        def flow_world_model(self, query):
                #.... add dircet information to the agents word modell
                pass

        def run_in_background(self):
            ""Start a task in background mode""
           # ... start fuction of the agent non blocking !

        def stop(self):
            self._stop_event.set()
            # dos not sop instante


        def run(self, user_input_or_task: str or Task, with_memory=None, with_functions=None, max_iterations=3, **kwargs):
            # ... entry pont for the Agent, uses callbacks  self.progress_callback with an status object
            # dos not send the final completet stae only inf run in backrund not a dirct task from the user -> returns agent autup as string / noot good for dircte viw for the user ..
          # ...
            #update_progress()

            #self.status_dict[task.id].status = "completed"
            #self.status_dict[task.id].result = out
            #self.status_dict[task.id].progress = 1.0
            #return out

        def _to_task(self, query:str) -> Task:
            ## creat a task from a str input
            # return task

        def invoke(self, user_input, with_mem=False, max_iterations=3, **kwargs):
            # run in decret agentcy mode + sepace mode for the user to utilyse
            #except Exception as e:
            #    return str(e).split("Result:")[-1]

        def mini_task(self, user_task, task_from="user", mini_task=None, message=None):
            ## mini task fro the agent sutch as refactoring the autu to whastapp syle syntax and make it perises and optimest for minimlistk conrret relavent chat Asisstatnt

        def format_class(self, format_class, task, re_try=4, gen=None):
           # tasks a BasModel Cass as input and a str string and retus a  full version for the model example :
            prompt = f"Determen if to change the current world model ##{self.show_word_model()}## basd on the new informaiton :" + query

            class WorldModelAdaption(BaseModel):
                ""world model adaption action['remove' or 'add' or ' change' or None] ;
                key from the existing word model or new one ;
                informations changed or added""
                action: Optional[str] = field(default=None)
                key: Optional[str] = field(default=None)
                informations: Optional[str] = field(default=None)

            model_action = self.format_class(WorldModelAdaption, prompt)

        def reset_context(self):
            self.messages = []
            self.world_model = {}


        def reset_memory(self):
            self.content_memory.text = ""
            self.messages = []

"""

    # long runing tasks
    # config stop task
    #   list lask stak , stask list
    #   reoderd / cancel / edit tasks
    #   stop agent | run in background
    #  toggels spesific insights from task execution

    # Emails
    #   Summary of N Keywort
    #   Last 5 Emals /
    #   Send email with attachment

    # Docs
    #   input .txt .pdf. png .jpg .csv .mp3 .mp4
    #   save and add or remove from agent knolage base
    #   list in nive form
    #   agent space for interaction

    # Kalender
    #   list day scope
    #   show nex event
    #   add event
    #   remove events
    #   agent get-, add-, list-events

    # System
    #   Metrics
    #   Online since

    # stt speech to text ->
        #     talk_generate = app.run_any(TBEF.AUDIO.STT_GENERATE,
#                                 model="openai/whisper-small",
#                                 row=True, device=1)
# audio_data: bytes =
#             text: str = talk_generate(audio_data)['text']

    # tts ->
    # filepaths: bytes = app.run_any(TBEF.AUDIO.SPEECH, text="hallo das ist ein test", voice_index=0,
#                                             use_cache=False,
#                                             provider='piper',
#                                             config={'play_local': False, 'model_name': 'kathleen'},
#                                             local=False,
#                                             save=True) # simply convert  to str ic neddet witch .decode()

    # give tasks
    #   inputs text audio docs
    #
    #   commands /config /emails /Kalender /docs /System
    #   flows -> config Interaction Button
    #               - Edit Agent | Edit Tasks
    #               - start stop agent clear temp memory | show Taskstakc (edit, roder, remove singel tasks) , show status_dict (show one task as progress par ( toogel on / off ) show result in a nice format mardown auto to -> whatsapp
    #               - crate new agent / switch bewenn agent and (dircht call isaa.run) for norma tasks
    #   flows -> emails Interaction Button
    #               - send | recive
    #               - send with all detais and atachemts | filter for (Time, context, emaladress) -> convert to summary send as audio / gent spesific insigt and send location / get sepsific insind and save event to calender + user aproval !

    #   flows -> Kalender Interaction Button
    #               - Get latest evnt and send -> location -> summary , audio summary
    #               - gent Day / nDays overwiy and send -> summary , audio summary
    #               - find optimual spoot for x event using perrafances and exsiting events
    #               - adding an event
    #   flows -> docs Interaction Button
    #               - add remove view
    #   flows -> System Interaction Button
    #               - show Nice Texte view , restart
    #   flows -> Default text or audio recived
    #               - tirgger active agent or (issa.run)


    # return results
    # message = Message(instance=yourclient, id="MESSAGEID")
    #   - make as red
    #       - to see if isystem is online
    #            message.mark_as_read()
    #   - react to user massage
    #       message.react("👍")
    #   - show inital welcome and explantion
    #       message.send_button
    #   - reply to user massage
    #       message.reply("Hello world!")
    #   - reply to agent massage
    #       message.reply("Hello world!")
    #   - show progrss bar
    #       step_flag = threading.Event()
    #     message_id = progress_messenger0.send_initial_message(mode="progress")
    #     print(progress_messenger0.max_steps)
    #     progress_messenger0.start_progress_in_background(step_flag=step_flag)
    #     for i in range(progress_messenger0.max_steps):
    #         time.sleep(2)
    #         step_flag.set()
    #   - shwo loding state
    #        # stop_flag = threading.Event()
    #     message_id = progress_messenger0.send_initial_message(mode="loading")
    #     # progress_messenger0.start_loading_in_background(stop_flag)
    #
    #     # Simulate work, then stop loading
    #     # time.sleep(10)  # Simulate work duration
    #     # stop_flag.set()
    #   - send audio
    #       messenger.send_audio(
    #         audio="https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
    #         recipient_id="255757xxxxxx",
    #         sender=0,
    #     )
    #   - send image
    #       media_id = messenger.upload_media(
    #         media='path/to/media',
    #     )['id']
    # >>> messenger.send_image(
    #         image=media_id,
    #         recipient_id="255757xxxxxx",
    #         link=False
    #         sender=0,
    #     )
    #   - send location
    #       messenger.send_location(
    #         lat=1.29,
    #         long=103.85,
    #         name="Singapore",
    #         address="Singapore",
    #         recipient_id="255757xxxxxx",
    #         sender=0,
    #     )

    # exaple button .send_button(
    #         recipient_id="255757xxxxxx",
    #         button={
    #             "header": "Header Testing",
    #             "body": "Body Testing",
    #             "footer": "Footer Testing",
    #             "action": {
    #                 "button": "Button Testing",
    #                 "sections": [
    #                     {
    #                         "title": "iBank",
    #                         "rows": [
    #                             {"id": "row 1", "title": "Send Money", "description": ""},
    #                             {
    #                                 "id": "row 2",
    #                                 "title": "Withdraw money",
    #                                 "description": "",
    #                             },
    #                         ],
    #                     }
    #                 ],
    #             },
    #         },
    #         sender=0,
    #     )


    # tools for the agent :        tools = {**tools, **{
    #
    #             "saveDataToMemory": {"func": ad_data, "description": "tool to save data to memory,"
    #                                                                  " write the data as specific"
    #                                                                  " and accurate as possible."},

