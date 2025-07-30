import asyncio
import json
from pathlib import Path
from typing import Any, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.history import FileHistory

from toolboxv2 import get_app
from toolboxv2.mods.isaa.module import Tools as Isaatools
from toolboxv2.mods.isaa.CodingAgent.live import EnhancedVerboseOutput
from toolboxv2.utils.extras.Style import Spinner

NAME = "isaa_cli"

class CLIFormatter(EnhancedVerboseOutput):
    def __init__(self, app_instance):
        super().__init__(verbose=True, print_f=app_instance.print)

    async def print_agent_response(self, response: str):
        await self.log_message("assistant", response)

    async def print_thought(self, thought: str):
        await self.log_message("assistant", f"Thought: {thought}")


class IsaasCli:
    def __init__(self, app_instance: Any):
        self.app = app_instance
        self.isaa_tools: Isaatools = app_instance.get_mod("isaa")
        self.formatter = CLIFormatter(app_instance)
        self.active_agent_name = "cli_agent"
        self.session_id = "cli_session"
        self.history = FileHistory(Path(self.app.data_dir) / "isaa_cli_history.txt")
        self.completer = self.build_completer()
        self.prompt_session = PromptSession(
            history=self.history,
            completer=self.completer,
            complete_while_typing=True,
        )
        self.background_tasks = {}
        self.interrupt_count = 0

    def build_completer(self):
        return NestedCompleter.from_nested_dict({
            "/agent": {
                "list": None,
                "switch": None,
                "create": None,
                "run_bg": None,
            },
            "/world": {
                "show": None,
                "set": None,
                "remove": None,
            },
            "/task": {
                "list": None,
                "run": None,
                "create": None,
                "improve": None,
            },
            "/tasks": {
                "list": None,
                "attach": None,
                "detach": None,
            },
            "/context": {
                "save": None,
                "load": None,
                "clear": None,
            },
            "/help": None,
            "/exit": None,
        })

    async def init(self):
        with Spinner("Initializing ISAA CLI...", symbols="c"):
            await self.isaa_tools.init_isaa()
        if self.active_agent_name not in self.isaa_tools.config.get("agents-name-list", []):
            builder = self.isaa_tools.get_agent_builder(self.active_agent_name)
            builder.with_system_message(
                "You are a supervisor agent in a CLI. Your primary role is to understand user requests and then create, configure, and manage other specialized agents to accomplish those tasks. You can run tasks in the background, monitor their progress, and report back to the user. Use your tools to create new agents and delegate tasks to them."
            )
            await self.add_cli_tools_to_agent(builder)
            await self.isaa_tools.register_agent(builder)
        self.formatter.print_header("ISAA CLI Initialized. Type `/help` for commands.")

    async def add_cli_tools_to_agent(self, builder):
        async def create_agent_tool(agent_name: str, system_prompt: str, model: str = None, mode: str = None):
            """Creates and registers a new agent."""
            new_builder = self.isaa_tools.get_agent_builder(agent_name)
            new_builder.with_system_message(system_prompt)
            if model:
                new_builder.with_model(model)
            if mode:
                llm_mode = self.isaa_tools.controller.rget(mode)
                if llm_mode:
                    new_builder.with_system_message(llm_mode.system_msg)
            await self.isaa_tools.register_agent(new_builder)
            return f"Agent '{agent_name}' created successfully."

        async def run_background_task_tool(agent_name: str, prompt: str):
            """Runs a task in the background for a specified agent."""
            task = asyncio.create_task(self.isaa_tools.run_agent(agent_name, prompt, session_id=f"bg_{len(self.background_tasks)}", progress_callback=self.progress_callback))
            task_id = len(self.background_tasks)
            self.background_tasks[task_id] = task
            return f"Started background task {task_id} for agent '{agent_name}'."

        async def list_background_tasks_tool():
            """Lists all running background tasks."""
            if not self.background_tasks:
                return "No background tasks running."
            return "\n".join([f"Task {task_id}: {'Running' if not task.done() else 'Done'}" for task_id, task in self.background_tasks.items()])

        builder.with_adk_tool_function(create_agent_tool, name="create_agent")
        builder.with_adk_tool_function(run_background_task_tool, name="run_background_task")
        builder.with_adk_tool_function(list_background_tasks_tool, name="list_background_tasks")

    async def run(self):
        await self.init()
        while True:
            try:
                self.interrupt_count = 0
                prompt_text = f"({self.active_agent_name}) >>> "
                user_input = await self.prompt_session.prompt_async(prompt_text)

                if user_input.startswith("/"):
                    await self.handle_command(user_input)
                else:
                    await self.handle_prompt(user_input)

            except (EOFError, KeyboardInterrupt) as e:
                if self.interrupt_count == 0 and not isinstance(e, EOFError):
                    self.interrupt_count += 1
                    self.formatter.print_info("Press Ctrl+C again to exit.")
                    continue
                break
            except Exception as e:
                self.formatter.print_error(f"An unexpected error occurred: {e}")

    async def handle_prompt(self, prompt: str):
        agent = await self.isaa_tools.get_agent(self.active_agent_name)
        agent.progress_callback = self.formatter.print_event
        response = await self.isaa_tools.run_agent(
            name=agent,
            text=prompt,
            session_id=self.session_id,
            progress_callback=self.progress_callback
        )
        await self.formatter.print_agent_response(response)

    async def progress_callback(self, data: dict):
        print("TYYPE:",data.get("type"))
        self.formatter.print_event(data)

    async def handle_command(self, user_input: str):
        parts = user_input.split()
        command = parts[0]
        args = parts[1:]

        command_map = {
            "/agent": self.handle_agent_command,
            "/world": self.handle_world_command,
            "/task": self.handle_task_command,
            "/tasks": self.handle_background_tasks_command,
            "/context": self.handle_context_command,
            "/help": self.handle_help_command,
            "/exit": self.handle_exit_command,
        }

        handler = command_map.get(command)
        if handler:
            await handler(args)
        else:
            self.formatter.print_error(f"Unknown command: {command}")

    async def handle_agent_command(self, args: list[str]):
        if not args:
            self.formatter.print_error("Usage: /agent <list|switch|create|run_bg> [name] [params]")
            return

        sub_command = args[0]
        if sub_command == "list":
            agents = self.isaa_tools.config.get("agents-name-list", [])
            self.formatter.print_info(f"Available agents: {', '.join(agents)}")
        elif sub_command == "switch":
            if len(args) < 2:
                self.formatter.print_error("Usage: /agent switch <name>")
                return
            agent_name = args[1]
            if agent_name not in self.isaa_tools.config.get("agents-name-list", []):
                self.formatter.print_error(f"Agent '{agent_name}' not found.")
                return
            self.active_agent_name = agent_name
            self.formatter.print_success(f"Switched to agent: {self.active_agent_name}")
        elif sub_command == "create":
            if len(args) < 2:
                self.formatter.print_error("Usage: /agent create <name> [key=value...]")
                return
            agent_name = args[1]
            builder = self.isaa_tools.get_agent_builder(agent_name)
            for param in args[2:]:
                key, value = param.split("=")
                if key == "model":
                    builder.with_model(value)
                elif key == "system_prompt":
                    builder.with_system_message(value)
                elif key == "mode":
                    mode = self.isaa_tools.controller.rget(value)
                    if mode:
                        builder.with_system_message(mode.system_msg)
                elif key == "mcp":
                    builder.enable_mcp_server()
            with Spinner(f"Creating agent '{agent_name}'...", symbols="c"):
                await self.isaa_tools.register_agent(builder)
            self.formatter.print_success(f"Agent '{agent_name}' created.")
        elif sub_command == "run_bg":
            if len(args) < 3:
                self.formatter.print_error("Usage: /agent run_bg <name> <prompt>")
                return
            agent_name = args[1]
            prompt = " ".join(args[2:])
            task = asyncio.create_task(self.isaa_tools.run_agent(agent_name, prompt, session_id=f"bg_{len(self.background_tasks)}", progress_callback=self.progress_callback))
            self.background_tasks[len(self.background_tasks)] = task
            self.formatter.print_success(f"Started background task {len(self.background_tasks) - 1} for agent '{agent_name}'.")
        else:
            self.formatter.print_error(f"Unknown agent command: {sub_command}")

    async def handle_world_command(self, args: list[str]):
        if not args:
            self.formatter.print_error("Usage: /world <show|set|remove> [key] [value]")
            return

        sub_command = args[0]
        agent = await self.isaa_tools.get_agent(self.active_agent_name)

        if sub_command == "show":
            world_model = agent.world_model.show()
            self.formatter.print_info(f"World Model for '{self.active_agent_name}':\n{world_model}")
        elif sub_command == "set":
            if len(args) < 3:
                self.formatter.print_error("Usage: /world set <key> <value>")
                return
            key = args[1]
            value = " ".join(args[2:])
            agent.world_model.set(key, value)
            self.formatter.print_success(f"Set '{key}' in world model.")
        elif sub_command == "remove":
            if len(args) < 2:
                self.formatter.print_error("Usage: /world remove <key>")
                return
            key = args[1]
            agent.world_model.remove(key)
            self.formatter.print_success(f"Removed '{key}' from world model.")
        else:
            self.formatter.print_error(f"Unknown world command: {sub_command}")

    async def handle_task_command(self, args: list[str]):
        if not args:
            self.formatter.print_error("Usage: /task <list|run|create|improve> [name] [prompt]")
            return

        sub_command = args[0]
        if sub_command == "list":
            tasks = self.isaa_tools.list_task()
            self.formatter.print_info(f"Available tasks:\n{tasks}")
        elif sub_command == "run":
            if len(args) < 3:
                self.formatter.print_error("Usage: /task run <name> <input>")
                return
            task_name = args[1]
            user_input = " ".join(args[2:])
            with Spinner(f"Running task '{task_name}'...", symbols="c"):
                result = await self.isaa_tools.run_task(user_input, task_name)
            await self.formatter.print_agent_response(str(result))
        elif sub_command == "create":
            if len(args) < 2:
                self.formatter.print_error("Usage: /task create <prompt>")
                return
            prompt = " ".join(args[1:])
            with Spinner("Creating task chain...", symbols="c"):
                chain_name = await self.isaa_tools.crate_task_chain(prompt)
            if chain_name:
                self.isaa_tools.save_task(chain_name)
                self.formatter.print_success(f"Task chain '{chain_name}' created and saved.")
            else:
                self.formatter.print_error("Failed to create task chain.")
        elif sub_command == "improve":
            if len(args) < 3:
                self.formatter.print_error("Usage: /task improve <name> <instructions>")
                return
            task_name = args[1]
            instructions = " ".join(args[2:])
            self.formatter.print_info(f"Improving task '{task_name}'...")

            task_chain = self.isaa_tools.get_task(task_name)
            if not task_chain:
                self.formatter.print_error(f"Task '{task_name}' not found.")
                return

            prompt = f"""
            Here is a task chain named '{task_name}':
            ```json
            {json.dumps(task_chain, indent=2)}
            ```
            Please improve this task chain based on the following instructions:
            "{instructions}"

            Respond with the improved task chain in the same JSON format.
            """

            with Spinner(f"Improving task '{task_name}'...", symbols="c"):
                response = await self.isaa_tools.run_agent(
                    name=self.active_agent_name,
                    text=prompt,
                    session_id=self.session_id
                )

            try:
                improved_task_chain_str = response[response.find("```json")+7:response.rfind("```")]
                improved_task_chain = json.loads(improved_task_chain_str)

                self.isaa_tools.add_task(task_name, improved_task_chain)
                self.isaa_tools.save_task(task_name)
                self.formatter.print_success(f"Task chain '{task_name}' improved and saved.")
            except (json.JSONDecodeError, KeyError) as e:
                self.formatter.print_error(f"Failed to parse the improved task chain from the agent's response: {e}")
                self.formatter.print_info("Agent's raw response:")
                await self.formatter.print_agent_response(response)

        else:
            self.formatter.print_error(f"Unknown task command: {sub_command}")

    async def handle_background_tasks_command(self, args: list[str]):
        if not args:
            self.formatter.print_error("Usage: /tasks <list|attach|detach> [task_id]")
            return

        sub_command = args[0]
        if sub_command == "list":
            if not self.background_tasks:
                self.formatter.print_info("No background tasks running.")
                return
            for task_id, task in self.background_tasks.items():
                status = "Running" if not task.done() else "Done"
                self.formatter.print_info(f"Task {task_id}: {status}")
        elif sub_command == "attach":
            if len(args) < 2:
                self.formatter.print_error("Usage: /tasks attach <task_id>")
                return
            task_id = int(args[1])
            if task_id not in self.background_tasks:
                self.formatter.print_error(f"Task {task_id} not found.")
                return
            task = self.background_tasks[task_id]
            with Spinner(f"Attaching to task {task_id}...", symbols="c"):
                await task
            await self.formatter.print_agent_response(task.result())
        elif sub_command == "detach":
            self.formatter.print_info("Detaching from all tasks.")
        else:
            self.formatter.print_error(f"Unknown tasks command: {sub_command}")

    async def handle_context_command(self, args: list[str]):
        if not args:
            self.formatter.print_error("Usage: /context <save|load|clear> [session_name]")
            return

        sub_command = args[0]
        agent = await self.isaa_tools.get_agent(self.active_agent_name)

        if sub_command == "save":
            session_name = args[1] if len(args) > 1 else self.session_id
            history = agent.message_history.get(self.session_id, [])
            with open(Path(self.app.data_dir) / f"{session_name}.json", "w") as f:
                json.dump(history, f)
            self.formatter.print_success(f"Context '{session_name}' saved.")
        elif sub_command == "load":
            session_name = args[1] if len(args) > 1 else self.session_id
            try:
                with open(Path(self.app.data_dir) / f"{session_name}.json", "r") as f:
                    history = json.load(f)
                agent.message_history[self.session_id] = history
                self.formatter.print_success(f"Context '{session_name}' loaded.")
            except FileNotFoundError:
                self.formatter.print_error(f"Context '{session_name}' not found.")
        elif sub_command == "clear":
            agent.message_history[self.session_id] = []
            self.formatter.print_success("Context cleared.")
        else:
            self.formatter.print_error(f"Unknown context command: {sub_command}")

    async def handle_help_command(self, args: list[str]):
        self.formatter.print_header("ISAA CLI Help")
        self.formatter.print_info("""
        Available commands:
        /agent <list|switch|create|run_bg> [name] [params] - Manage agents
        /world <show|set|remove> [key] [value] - Manage the active agent's world model
        /task <list|run|create|improve> [name] [prompt] - Manage and run task chains
        /tasks <list|attach|detach> [task_id] - Manage background tasks
        /context <save|load|clear> [session_name] - Manage conversation history
        /help - Show this help message
        /exit - Exit the CLI
        """)

    async def handle_exit_command(self, args: list[str]):
        self.formatter.print_info("Exiting ISAA CLI...")
        raise EOFError


async def run(app, *args, **kwargs):
    app = get_app("isaa_cli_instance")
    cli = IsaasCli(app)
    await cli.run()

if __name__ == "__main__":
    asyncio.run(run(None))
