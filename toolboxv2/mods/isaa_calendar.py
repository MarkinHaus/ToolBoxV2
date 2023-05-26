import os
import types
from datetime import datetime, timedelta, time

import datefinder
from gcsa.google_calendar import GoogleCalendar
from concurrent.futures import TimeoutError

from gcsa.reminders import PopupReminder
from pebble import concurrent
from beautiful_date import *
from gcsa.free_busy import FreeBusy
from gcsa.event import Event
from beautiful_date import D, drange, days, MO

def puschup_habit(gc: GoogleCalendar):

    PUSH_UPS_COUNT = [
        5, 5, 0, 5, 10, 0, 10,
        0, 12, 12, 0, 15, 15, 0,
        20, 24, 0, 25, 30, 0, 32,
        35, 35, 0, 38, 40, 0, 42,
        45, 50
    ]

    # starting next Monday (of course)
    # +1 days for the case that today is Monday
    start = D.today()[9:15] + 1 * days + MO
    end = start + len(PUSH_UPS_COUNT) * days

    for day, push_ups in zip(drange(start, end), PUSH_UPS_COUNT):
        e = Event(
            f'{push_ups} Push-Ups' if push_ups else 'Rest',
            start=day,
            minutes_before_popup_reminder=5
        )
        gc.add_event(e)

    for event in gc:
        print(event)


@concurrent.process(timeout=120)
def create_process(email):
    return GoogleCalendar(email, credentials_path=CREDENTIALS_FILE)

CREDENTIALS_FILE = os.getenv('CREDENTIALS_FILE')
from gcsa.calendar import Calendar, CalendarListEntry
from gcsa.event import Event

import re
from dateutil.parser import parse

from toolboxv2 import MainTool, FileHandler
from concurrent.futures import TimeoutError
import concurrent
from beautiful_date import hours
from gcsa.event import Event


class Tools(MainTool, FileHandler):
    def __init__(self, app=None):
        self.version = "0.3"
        self.name = "isaa_calendar"
        self.logs = app.logger if app else None
        self.color = "YELLOW"
        self.keys = {
            "email": "email",
            "credentials": CREDENTIALS_FILE,
            "calendar_ids": "calendar_ids"
        }

        self.gc = None

        self.append_agent = append_agent

        self.tools = {
            "all": [
                ["Version", "Shows current Version"],
                ["add_quick_event", "Add a quick event to the calendar"],
                ["add_calendar_id", "Add a calendar ID"],
                ["get_calendar_id", "Get a calendar ID"],
                ["delete_calendar_id", "Delete a calendar ID"],
                ["get_calendar_list", "Get a list of calendars"],
                ["get_events", "Get events from a calendar between start and end date"],
                ["get_free_busy", "Get free and busy time slots"],
                ["add_event", "Add an event to the calendar"],
                ["update_event", "Update an event in the calendar"],
                ["search_events", "Search for events in the calendar"],
                ["get_event", "Get an event by its ID"],
                ["get_instances", "Get instances of a recurring event"],
            ],
            "name": "GoogleCalendar",
            "Version": self.show_version,
            "add_quick_event": self.add_quick_event,
            "add_calendar_id": self.add_calendar_id,
            "get_calendar_id": self.get_calendar_id,
            "delete_calendar_id": self.delete_calendar_id,
            "get_calendar_list": self.get_calendar_list,
            "get_events": self.get_events,
            "get_free_busy": self.get_free_busy,
            "add_event": self.add_event,
            "update_event": self.update_event,
            "search_events": self.search_events,
            "get_event": self.get_event,
            "get_instances": self.get_instances,
        }

        self.calendar_ids = {}
        self.scope = "primary"

        FileHandler.__init__(self, "google_calendar_tools.data", app.id if app else __name__, keys=self.keys,
                             defaults={"email": "markinhausmanns@gmail.com", "calendar_ids": {}})
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=self.logs, color=self.color, on_exit=self.on_exit)

    def on_start(self):
        self.load_file_handler()

    # @concurrent.process(timeout=120)
    def create_google_calendar_instance(self, email):
        return GoogleCalendar(email, credentials_path=self.keys["credentials"])

    def add_quick_event(self, event_text):
        return self.gc.add_quick_event(event_text, calendar_id=self.scope)


    def add_calendar_id(self, key, calendar_id):
        calendar_ids = self.get_file_handler("calendar_ids")
        calendar_ids[key] = calendar_id
        self.add_to_save_file_handler("calendar_ids", calendar_ids)
        self.save_file_handler()

    def get_calendar_id(self, key):
        calendar_ids = self.get_file_handler("calendar_ids")
        return calendar_ids.get(key)

    def delete_calendar_id(self, key):
        calendar_ids = self.get_file_handler("calendar_ids")
        if key in calendar_ids:
            del calendar_ids[key]
            self.add_to_save_file_handler("calendar_ids", calendar_ids)
            self.save_file_handler()

    def get_calendar_list(self):
        return self.gc.get_calendar_list()

    def get_events(self, start_date, end_date, order_by="updated"):
        return self.gc.get_events(start_date, end_date, order_by=order_by, single_events=True, calendar_id=self.scope)

    def get_free_busy(self, calendar_id):
        return self.gc.get_free_busy(calendar_id, calendar_id=self.scope)

    def add_event(self, event):
        self.gc.add_event(event, calendar_id=self.scope)

    def update_event(self, event):
        self.gc.update_event(event, calendar_id=self.scope)

    def search_events(self, query):
        return self.gc.get_events(query=query, calendar_id=self.scope)

    def get_event(self, event_id):
        return self.gc.get_event(event_id, calendar_id=self.scope)

    def get_instances(self, recurring_event_id):
        return self.gc.get_instances(recurring_event_id, calendar_id=self.scope)

    def show_version(self):
        self.print("Version: ", self.version)

    def on_exit(self):
        pass

    def add_calendar_id(self, name, calendar_id):
        self.calendar_ids[name] = calendar_id

    def get_calendar_id(self, name):
        return self.calendar_ids.get(name)

    def extract_datetime(self, text):
        try:
            return parse(text, fuzzy=True)
        except ValueError:
            return None

    def extract_keywords(self, text, keywords):
        for keyword in keywords:
            if keyword.lower() in text.lower():
                return keyword
        return None

    def extract_event_id(self, text):
        match = re.search(r'\bID\s*:\s*(\w+)\b', text, re.IGNORECASE)
        return match.group(1) if match else None

    def extract_event_title(self, text):
        match = re.search(r'\btitle\s*:\s*(.+)', text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def extract_event_description(self, text):
        match = re.search(r'\bdescription\s*:\s*(.+)', text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def extract_location(self, text):
        match = re.search(r'\blocation\s*:\s*(.+)', text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def extract_start_end_datetime(self, text):
        matches_ = datefinder.find_dates(text)
        matches = []
        for match in matches_:
            matches.append(match)
        if len(matches) == 0:
            return None, None
        if len(matches) == 1:
            return matches[0], None
        if len(matches) == 2:
            return matches[0], matches[1]

    def set_scope(self, command):
        result = self.get_calendar_list()
        for res in result:
            print(command,  str(res), command in str(res))
            if command in str(res):
                self.scope = res.id
                print("set_scope")
                return res.id


    def llm_controller(self, text_input):
        command = text_input.lower()

        if "calendar list" in command:
            return self.get_calendar_list()

        elif "free" in command and "busy" in command:
            calendar_name = self.extract_keywords(text_input, self.calendar_ids.keys())
            calendar_id = self.get_calendar_id(calendar_name)
            return self.get_free_busy(calendar_id)

        elif "events" in command:
            start_date, end_date = self.extract_start_end_datetime(text_input)
            order_by = "updated" if "updated" in command else "startTime"
            return self.get_events(start_date, end_date, order_by=order_by)

        elif "add event" in command:
            title = self.extract_event_title(text_input)
            start, end = self.extract_start_end_datetime(text_input)
            event = Event(title, start=start, end=end, reminders=[PopupReminder(minutes_before_start=15),
                                                                  PopupReminder(minutes_before_start=5)])
            return self.add_event(event)

        elif "delete event" in command:
            event_id = self.extract_event_id(text_input)  # Add a method to extract the event ID from the text input.
            return self.gc.delete_event(event_id, calendar_id=self.scope)

        elif "update event" in command:
            event_id = self.extract_event_id(text_input)  # Add a method to extract the event ID from the text input.
            event: Event = self.get_event(event_id)
            title = self.extract_event_title(text_input)
            description = self.extract_event_description(text_input)
            start, end = self.extract_start_end_datetime(text_input)

            if event:

                if title:
                    event.title = title
                if description:
                    event.description = description
                if description:
                    event.description = description
                if start:
                    event.start = start
                if end:
                    event.end = end

                self.gc.update_event(event, calendar_id=self.scope)

        elif "get instances" in command:
            recurring_event_id = self.extract_event_id(text_input)  # Add a method to extract the recurring event ID from the text input.
            return self.get_instances(recurring_event_id)

        elif "quick add" in command:
            return self.add_quick_event(text_input.replace("quick add", ""))

        elif "set scope" in command:
            return self.set_scope(text_input.replace("set scope", ""))

        else:
            return "Sorry, I didn't understand your request. Please try again."

    def get_llm_tool(self, email):
        t = init(self, email)

        def run(text):
            result, output = "", ""
            if not text:
                return "Please Provide an Input"
            if text.lower() == "help":
                return discription()
            try:
                result = t.llm_controller(text)
                if not result:
                    return "Somthing went wrong ... "

                output = format_llm_output(result)
            except Exception as e:
                return "Error " + str(e)

            if not output:
                return "Parsing failed " + str(result)

            return output

        return run


def init(cls, email):
    try:
        process = create_process(email)
        cls.gc = process.result()
    except TimeoutError:
        print("User hasn't authenticated in 120 seconds")
        exit(0)
    return cls


def format_llm_output(output):
    formatted_output = ""
    max_length = 10
    ac_length = 0

    if isinstance(output, str):
        formatted_output = output

    elif isinstance(output, list):
        if len(output) == 0:
            formatted_output = "There are no items in the list."
        else:
            if isinstance(output[0], GoogleCalendar):
                formatted_output = "Here are your calendars:\n"
                for calendar in output:
                    ac_length += 1
                    formatted_output += f"{calendar.name} (ID: {calendar.id})\n"
                    if ac_length > max_length:
                        break

            elif isinstance(output[0], Event):
                formatted_output = "Here are the events:\n"
                for event in output:
                    ac_length += 1
                    formatted_output += f"{event.summary} from {event.start} to {event.end} (ID: {event.id})\n"
                    if ac_length > max_length:
                        break

    elif isinstance(output, Event):
        formatted_output = f"{output.summary} from {output.start} to {output.end} (ID: {output.id})\n"

    elif isinstance(output, FreeBusy):
        formatted_output = "Here are the busy time slots:\n"
        for time_range in output:
            ac_length += 1
            formatted_output += f"Busy from {time_range.start} to {time_range.end}\n"
            if ac_length > max_length:
                break

    elif isinstance(output, types.GeneratorType):
        for i in output:
            ac_length += 1
            if isinstance(i, Event):
                formatted_output += f"{i.summary} from {i.start} to {i.end} | description:{i.description} (ID: {i.id})\n"
            else:
                formatted_output += str(i) + f" (ID: {i.id})\n"
            if ac_length > max_length:
                break

    else:
        if len(str(output)) > 3000:
            output = str(output)[:3000]
        formatted_output = "Sorry, I couldn't format the output. Please try again." + "\n" + str(output)

    return formatted_output


def append_agent(calender_agent_config, calender_run, thinc):

    calender_agent_config.name = "Calender-Agent"

    def get_calendar_list(text_input):
        if not text_input:
            return "pleas provide an input!"
        command = "calendar list"
        return calender_run(command + text_input)

    def get_free_busy(text_input):
        if not text_input:
            return "pleas provide an input!"
        command = "free busy"
        return calender_run(command + text_input)

    def get_events(text_input):
        if not text_input:
            return "pleas provide an input!"
        command = "events"
        return calender_run(command + text_input)

    def add_event(text_input):
        if not text_input:
            return "pleas provide an input!"
        command = "add event"
        return calender_run(command + text_input)

    def delete_event(text_input):
        if not text_input:
            return "pleas provide an input!"
        command = "delete event"
        return calender_run(command + text_input)

    def update_event(text_input):
        if not text_input:
            return "pleas provide an input!"
        command = "update event"
        return calender_run(command + text_input)

    def get_instances(text_input):
        if not text_input:
            return "pleas provide an input!"
        command = "get instances"
        return calender_run(command + text_input)

    def add_quick_event(text_input):
        if not text_input:
            return "pleas provide an input!"
        command = "quick add"
        return calender_run(command + text_input)

    def set_scope(text_input):
        if not text_input:
            return "pleas provide an input!"
        command = "set scope"
        return calender_run(command + text_input)

    calender_agent_config.mode = "tools"
    calender_agent_config.model_name = "text-davinci-003"

    calender_agent_config.agent_type = "zero-shot-react-description"
    calender_agent_config.max_iterations = 4
    calender_agent_config.verbose = True

    calender_agent_config.personality = """
     Efficient: The Calendar Management Agent should be able to manage appointments, to-dos, and projects with ease, ensuring that the user's schedule runs smoothly.
     Organized: The Calendar Management Agent should maintain a well-structured calendar, categorizing and prioritizing tasks, events, and projects appropriately.
     Adaptive: The Calendar Management Agent should adjust its strategies based on user preferences, as well as the user's schedule and priorities.
     Proactive: The Calendar Management Agent should anticipate potential scheduling conflicts and suggest solutions to optimize the user's time management.
     Detail-Oriented: The Calendar Management Agent should pay close attention to the details of tasks, events, and projects, ensuring that all relevant information is accurately captured and presented."""

    calender_agent_config.goals = """
     1. Seamless Integration: The Calendar Management Agent should integrate smoothly with the Google Calendar API, allowing the user to manage their appointments, to-dos, and projects directly within the Google Calendar environment.
     2. Effective Time Management: The Calendar Management Agent should help the user optimize their time by efficiently organizing tasks, events, and projects within the calendar.
     3. Conflict Resolution: The Calendar Management Agent should identify and resolve scheduling conflicts, ensuring that the user's schedule remains well-organized and conflict-free.
     4. Personalization: The Calendar Management Agent should adapt its strategies to suit the user's preferences and needs, taking into account factors such as time management, task prioritization, and notification settings.
     5. Continuous Improvement: The Calendar Management Agent should continuously refine its algorithms and strategies to improve the effectiveness of its calendar management capabilities over time."""

    calender_agent_config.tools: dict = {
        "get_calendar_list": {"func": lambda x: get_calendar_list(x),
                              "description": "This command returns a list of the user's calendars."},
        "get_free_busy": {"func": lambda x: get_free_busy(x),
                          "description": "This command returns the user's free/busy status for a particular calendar.Syntax:  <calendar name>"},
        "get_events": {"func": lambda x: get_events(x),
                       "description": "This command returns a list of events in the user's calendar within a specified date range. Syntax: events from <start date> to <end date> [ordered by updated/startTime]"},
        "add_event": {"func": lambda x: add_event(x),
                      "description": "This command adds an event to the user's calendar. Syntax: add event titled <title> from <start time> to <end time> [with description <description>]"},
        "delete_event": {"func": lambda x: delete_event(x),
                         "description": "This command deletes an event from the user's calendar. Syntax: delete event with ID <event ID>"},
        "update_event": {"func": lambda x: update_event(x),
                         "description": "This command updates an event in the user's calendar with the specified details. Syntax: update event with ID <event ID> [with title <title>] [with description <description>] [from <start time> to <end time>]"},
        "get_instances": {"func": lambda x: get_instances(x),
                          "description": "This command returns all the instances of a recurring event in the user's calendar. Syntax: get instances of <recurring event ID>"},
        "add_quick_event": {"func": lambda x: add_quick_event(x),
                            "description": "This command adds an event to the user's calendar using Google's. it is an all raunder to crat repeating task, remaiders and singelton. use lagaude to discribe the pattern + time of the reminder / task. Syntax: quick add <event details>"},
        "set_scope": {"func": lambda x: set_scope(x),
                      "description": "This command sets the calendar scope for the API operations. Syntax: set scope to <calendar name>"},
        "Think": {"func": lambda x: thinc(x),
                      "description": "Use Tool to perform complex thought processes"},
    }

    return calender_agent_config


def discription():
    return """Guide for the llm_controller function:

The llm_controller function is the main function that controls the Google Calendar API functionalities. It takes in a string input of the user's command and processes it to perform the required operation.

Here's a breakdown of the supported commands and their syntax:

    "Calendar list":
    This command returns a list of the user's calendars.
    Syntax: "calendar list"

    "Free busy":
    This command returns the user's free/busy status for a particular calendar.
    Syntax: "free busy for <calendar name>"

    "Events":
    This command returns a list of events in the user's calendar within a specified date range.
    Syntax: "events from <start date> to <end date> [ordered by updated/startTime]"

    "Add event":
    This command adds an event to the user's calendar.
    Syntax: "add event titled <title> from <start time> to <end time> [with description <description>]"

    "Delete event":
    This command deletes an event from the user's calendar.
    Syntax: "delete event with ID <event ID>"

    "Update event":
    This command updates an event in the user's calendar with the specified details.
    Syntax: "update event with ID <event ID> [with title <title>] [with description <description>] [from <start time> to <end time>]"

    "Get instances":
    This command returns all the instances of a recurring event in the user's calendar.
    Syntax: "get instances of <recurring event ID>"

    "Quick add":
    This command adds an event to the user's calendar using Google's quick add feature.
    Syntax: "quick add <event details>"

    "Set scope":
    This command sets the calendar scope for the API operations.
    Syntax: "set scope to <calendar name>"

The llm_controller function also has several helper functions to extract the necessary details from the user's input, such as extract_event_id, extract_event_title, extract_start_end_datetime, and extract_event_description.

To use the llm_controller function, simply call it with the user's input command as the parameter, and it will return the result of the corresponding operation. If the command is not recognized, it will return an error message asking the user to try again."""


