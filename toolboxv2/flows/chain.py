"""
Chain CLI - Modern minimalistic chain management console
A powerful, intuitive interface for creating, testing, and deploying AI agent chains.
"""

import asyncio
import json
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import FuzzyCompleter, NestedCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import confirm
from pydantic import Field as PydanticField
from pydantic import create_model

from toolboxv2 import get_app

# Import the complete Chain implementation
from toolboxv2.mods.isaa.base.Agent.chain import (
    CF,
    IS,
    Chain,
    ConditionalChain,
    ErrorHandlingChain,
    ParallelChain,
)
from toolboxv2.mods.isaa.base.Agent.types import NodeStatus, ProgressEvent, ChainMetadata
from toolboxv2.mods.isaa.extras.cahin_printer import ChainProgressTracker, ChainPrinter

NAME = "chain"



class ChainStorage:
    """Handles chain persistence and metadata management"""

    def __init__(self, app_instance):
        self.app = app_instance
        self.chains_dir = Path(app_instance.data_dir) / "chains"
        self.chains_dir.mkdir(exist_ok=True)
        self.metadata_file = self.chains_dir / "metadata.json"
        self._metadata_cache = {}
        self._load_metadata()

    def _load_metadata(self):
        """Load chain metadata from storage"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    data = json.load(f)
                    self._metadata_cache = {
                        name: ChainMetadata(**meta) for name, meta in data.items()
                    }
            except Exception as e:
                print(f"Warning: Could not load chain metadata: {e}")

    def _save_metadata(self):
        """Save chain metadata to storage"""
        try:
            data = {
                name: {
                    "name": meta.name,
                    "description": meta.description,
                    "created_at": meta.created_at.isoformat(),
                    "modified_at": meta.modified_at.isoformat(),
                    "version": meta.version,
                    "tags": meta.tags,
                    "author": meta.author,
                    "complexity": meta.complexity,
                    "agent_count": meta.agent_count,
                    "has_conditionals": meta.has_conditionals,
                    "has_parallels": meta.has_parallels,
                    "has_error_handling": meta.has_error_handling
                }
                for name, meta in self._metadata_cache.items()
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save chain metadata: {e}")

    def save_chain(self, name: str, chain_config: dict, metadata: ChainMetadata):
        """Save chain configuration and metadata"""
        chain_file = self.chains_dir / f"{name}.json"

        try:
            with open(chain_file, 'w') as f:
                json.dump(chain_config, f, indent=2)

            metadata.modified_at = datetime.now()
            self._metadata_cache[name] = metadata
            self._save_metadata()
            return True
        except Exception as e:
            print(f"Error saving chain '{name}': {e}")
            return False

    def load_chain(self, name: str) -> dict | None:
        """Load chain configuration"""
        chain_file = self.chains_dir / f"{name}.json"

        if not chain_file.exists():
            return None

        try:
            with open(chain_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading chain '{name}': {e}")
            return None

    def delete_chain(self, name: str) -> bool:
        """Delete chain and metadata"""
        chain_file = self.chains_dir / f"{name}.json"

        try:
            if chain_file.exists():
                chain_file.unlink()
            if name in self._metadata_cache:
                del self._metadata_cache[name]
                self._save_metadata()
            return True
        except Exception as e:
            print(f"Error deleting chain '{name}': {e}")
            return False

    def list_chains(self) -> list[tuple[str, ChainMetadata]]:
        """List all available chains with metadata"""
        return [(name, meta) for name, meta in self._metadata_cache.items()]

    def get_metadata(self, name: str) -> ChainMetadata | None:
        """Get metadata for a specific chain"""
        return self._metadata_cache.get(name)


class ChainBuilder:
    """Interactive chain builder with guided creation"""

    def __init__(self, isaa_tools, printer: ChainPrinter):
        self.isaa = isaa_tools
        self.printer = printer

    def _format_chain_display(self, chain_agents):
        """Format chain for display"""
        if not chain_agents:
            return "(empty)"

        display_parts = []
        for agent in chain_agents:
            if hasattr(agent, 'amd') and agent.amd:
                display_parts.append(agent.amd.name)
            elif hasattr(agent, 'format_class'):
                display_parts.append(f"CF({agent.format_class.__name__})")
            elif hasattr(agent, 'agents'):  # ParallelChain
                parallel_names = [a.amd.name if hasattr(a, 'amd') else str(a) for a in agent.agents]
                display_parts.append(f"({' + '.join(parallel_names)})")
            else:
                display_parts.append(str(type(agent).__name__))

        return " >> ".join(display_parts)

    async def interactive_create(self, name: str) -> Chain | None:
        """Create a chain through interactive prompts"""
        self.printer.print_header(f"Creating Chain: {name}")

        # Get available agents
        agents = self.isaa.config.get("agents-name-list", [])
        if not agents:
            self.printer.print_error("No agents available. Please create agents first using ISAA.")
            return None

        print(f"\n{self.printer._colorize('Chain Metadata:', 'bold')}")
        description = input("Description: ").strip()
        author = input("Author (optional): ").strip()
        tags = input("Tags (comma-separated, optional): ").strip()
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        self.printer.print_info(f"Available agents: {', '.join(agents)}")

        chain_agents = []

        while True:

            # Dann in der while-Schleife:
            print(f"\n{self.printer._colorize('Current chain:', 'bold')} {self._format_chain_display(chain_agents)}")
            print("\nOptions:")
            print("  [agent_name] - Add agent to chain")
            print("  parallel [agent1,agent2] - Add parallel execution")
            print("  condition [field:value] - Add conditional logic")
            print("  format [ModelName] - Add data formatting")
            print("  done - Finish chain creation")
            print("  cancel - Cancel creation")

            choice = input("\n> ").strip()

            if choice == "done":
                if chain_agents:
                    chain = self._build_chain(chain_agents)
                    chain.name = name  # Set the correct chain name

                    # Create metadata with user input
                    metadata = ChainMetadata(
                        name=name,
                        description=description,
                        author=author,
                        tags=tag_list,
                        complexity="simple",  # Will be updated by _analyze_chain
                        agent_count=len([a for a in chain_agents if hasattr(a, 'amd')])
                    )

                    return chain, metadata  # Return both chain and metadata
                else:
                    self.printer.print_warning("Chain is empty. Add at least one agent.")
                    continue

            elif choice == "cancel":
                return None

            elif choice.startswith("parallel"):
                # Parse parallel agents
                try:
                    agent_list = choice.split("[")[1].split("]")[0].split(",")
                    parallel_agents = []
                    for agent_name in agent_list:
                        agent_name = agent_name.strip()
                        if agent_name in agents:
                            agent = await self.isaa.get_agent(agent_name)
                            parallel_agents.append(agent)
                        else:
                            self.printer.print_error(f"Agent '{agent_name}' not found")
                            break
                    else:
                        chain_agents.append(ParallelChain(parallel_agents))
                        self.printer.print_success(f"Added parallel execution: {agent_list}")
                except:
                    self.printer.print_error("Invalid parallel syntax. Use: parallel [agent1,agent2,...]")

            elif choice.startswith("condition"):
                # Parse condition
                try:
                    condition_part = choice.split("[")[1].split("]")[0]
                    field, value = condition_part.split(":")
                    condition = IS(field.strip(), value.strip())

                    print("\nTrue branch agent:")
                    true_agent = input("> ").strip()
                    if true_agent in agents:
                        true_agent_obj = await self.isaa.get_agent(true_agent)

                        print("False branch agent (optional):")
                        false_agent = input("> ").strip()
                        false_agent_obj = None
                        if false_agent and false_agent in agents:
                            false_agent_obj = await self.isaa.get_agent(false_agent)

                        chain_agents.append(ConditionalChain(condition, true_agent_obj, false_agent_obj))
                        self.printer.print_success(f"Added condition: {field}={value}")
                    else:
                        self.printer.print_error(f"Agent '{true_agent}' not found")
                except:
                    self.printer.print_error("Invalid condition syntax. Use: condition [field:value]")

            elif choice.startswith("format"):
                # Parse format model name and key extraction
                try:
                    if "[" in choice and "]" in choice:
                        # Extract model name and key: format [ModelName] key
                        parts = choice.split("[")[1].split("]")
                        model_name = parts[0].strip()
                        key_part = choice.split("]")[1].strip() if len(choice.split("]")) > 1 else ""
                    else:
                        # Just format [ModelName]
                        model_name = choice.split("[")[1].split("]")[0].strip()
                        key_part = ""

                    # Create dynamic Pydantic model
                    print(f"\nDefining fields for {model_name} model:")
                    print("Enter field definitions (name:type:description), 'done' when finished:")

                    fields = {}
                    while True:
                        field_input = input("Field > ").strip()
                        if field_input.lower() == "done":
                            break
                        if field_input.lower() == "cancel":
                            break

                        try:
                            # Parse field: name:type:description
                            parts = field_input.split(":", 2)
                            if len(parts) >= 2:
                                field_name = parts[0].strip()
                                field_type = parts[1].strip()
                                field_desc = parts[2].strip() if len(parts) > 2 else ""

                                # Map string types to Python types
                                type_map = {
                                    "str": str, "string": str,
                                    "int": int, "integer": int,
                                    "float": float,
                                    "bool": bool, "boolean": bool,
                                    "list": list[str], "list[str]": list[str],
                                    "list[int]": list[int]
                                }

                                python_type = type_map.get(field_type.lower(), str)
                                if field_desc:
                                    fields[field_name] = (python_type, PydanticField(description=field_desc))
                                else:
                                    fields[field_name] = (python_type, ...)
                            else:
                                self.printer.print_error("Invalid format. Use: name:type:description")
                        except Exception as e:
                            self.printer.print_error(f"Invalid field definition: {e}")

                    if not fields:
                        self.printer.print_error("No fields defined for model")
                        continue

                    # Create dynamic Pydantic model
                    DynamicModel = create_model(model_name, **fields)

                    # Create CF with optional key extraction
                    cf = CF(DynamicModel)

                    # Handle key extraction
                    if key_part:
                        if key_part.startswith("*"):
                            cf = cf - key_part[1:] if len(key_part) > 1 else cf - "*"
                        elif "[n]" in key_part:
                            cf = cf - key_part
                        elif "," in key_part:
                            keys = tuple(k.strip() for k in key_part.split(","))
                            cf = cf - keys
                        else:
                            cf = cf - key_part

                    chain_agents.append(cf)

                    extraction_info = f" -> {key_part}" if key_part else ""
                    self.printer.print_success(f"Added format: {model_name}{extraction_info}")

                except Exception as e:
                    self.printer.print_error(f"Invalid format syntax: {e}")
                    self.printer.print_info("Usage: format [ModelName] or format [ModelName] key_name")

            elif choice in agents:
                # Add single agent
                agent = await self.isaa.get_agent(choice)
                chain_agents.append(agent)
                self.printer.print_success(f"Added agent: {choice}")

            else:
                self.printer.print_error(f"Unknown option or agent: {choice}")

    def _build_chain(self, components: list) -> Chain:
        """Build chain from components"""
        if len(components) == 1:
            if isinstance(components[0], Chain | ParallelChain | ConditionalChain):
                return components[0]
            else:
                return Chain(components[0])

        # Build sequential chain
        chain = components[0]
        for component in components[1:]:
            if isinstance(chain, Chain):
                chain = chain >> component
            else:
                chain = Chain(chain) >> component

        return chain


class ChainCLI:
    """
    Modern minimalistic chain management console.
    Create, test, manage, and deploy AI agent chains with intuitive commands.
    """

    def __init__(self, app_instance):
        self.app = app_instance
        self.isaa = app_instance.get_mod("isaa")
        self.printer = ChainPrinter(verbose=True)
        self.storage = ChainStorage(app_instance)
        self.builder = ChainBuilder(self.isaa, self.printer)

        # Session management
        self.session_id = f"chain_cli_{int(time.time())}"
        self.current_chain: Chain | None = None
        self.current_chain_name: str | None = None

        # History and completion
        self.history = FileHistory(Path(app_instance.data_dir) / "chain_cli_history.txt")
        self.completion_dict = self._build_completions()

        # Prompt session
        self.prompt_session = PromptSession(
            history=self.history,
            completer=FuzzyCompleter(NestedCompleter.from_nested_dict(self.completion_dict)),
            complete_while_typing=True,
        )

        # Command mapping
        self.commands = {
            'help': self.cmd_help,
            'list': self.cmd_list,
            'create': self.cmd_create,
            'load': self.cmd_load,
            'save': self.cmd_save,
            'delete': self.cmd_delete,
            'show': self.cmd_show,
            'test': self.cmd_test,
            'run': self.cmd_run,
            'deploy': self.cmd_deploy,
            'info': self.cmd_info,
            'clear': self.cmd_clear,
            'status': self.cmd_status,
            'agents': self.cmd_agents,
            'export': self.cmd_export,
            'import': self.cmd_import,
            'exit': self.cmd_exit,
            'quit': self.cmd_exit,
        }

    def _build_completions(self) -> dict:
        """Build command completions"""
        base_commands = {
            'help': None,
            'list': None,
            'create': None,
            'load': {name: None for name, _ in self.storage.list_chains()},
            'save': None,
            'delete': {name: None for name, _ in self.storage.list_chains()},
            'show': {name: None for name, _ in self.storage.list_chains()},
            'test': {name: None for name, _ in self.storage.list_chains()},
            'run': {name: None for name, _ in self.storage.list_chains()},
            'deploy': {name: None for name, _ in self.storage.list_chains()},
            'info': {name: None for name, _ in self.storage.list_chains()},
            'clear': None,
            'status': None,
            'agents': None,
            'export': {name: None for name, _ in self.storage.list_chains()},
            'import': None,
            'exit': None,
            'quit': None,
        }

        return base_commands

    async def run(self):
        """Main CLI event loop"""
        await self.isaa.init_isaa()

        self.printer.print_header(
            "Chain CLI",
            "Modern minimalistic chain management • Type 'help' for commands"
        )

        try:
            while True:
                # Update completions
                self.completion_dict = self._build_completions()
                self.prompt_session.completer = FuzzyCompleter(
                    NestedCompleter.from_nested_dict(self.completion_dict)
                )

                # Create dynamic prompt
                prompt_text = self._get_prompt()

                try:
                    user_input = await self.prompt_session.prompt_async(
                        HTML(prompt_text),
                        complete_while_typing=True
                    )

                    if not user_input.strip():
                        continue

                    await self._process_command(user_input.strip())

                except KeyboardInterrupt:
                    self.printer.print_info("Use 'exit' to quit")
                    continue
                except EOFError:
                    break

        except Exception as e:
            self.printer.print_error(f"Unexpected error: {e}")
            if self.app.debug:
                traceback.print_exc()
        finally:
            await self._cleanup()

    def _get_prompt(self) -> str:
        """Generate dynamic prompt based on current state"""
        chain_info = ""
        if self.current_chain_name:
            chain_info = f"<ansired>[{self.current_chain_name}]</ansired> "

        return f"{chain_info}<ansiblue>chain</ansiblue><ansiwhite>></ansiwhite> "

    async def _process_command(self, command_line: str):
        """Process and execute commands"""
        parts = command_line.split()
        if not parts:
            return

        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        if command in self.commands:
            try:
                await self.commands[command](args)
            except EOFError:
                raise
            except Exception:
                self.printer.print_error(f"Command failed: {traceback.print_exc()}")
                if self.app.debug:
                    traceback.print_exc()
        else:
            self.printer.print_error(f"Unknown command: {command}. Type 'help' for available commands.")

    # Command implementations
    async def cmd_help(self, args: list[str]):
        """Show help information"""
        if args:
            # Specific command help
            cmd = args[0].lower()
            if cmd in self.commands:
                help_text = getattr(self.commands[cmd], '__doc__', 'No help available')
                self.printer.print_info(f"{cmd}: {help_text}")
            else:
                self.printer.print_error(f"Unknown command: {cmd}")
        else:
            # General help
            self.printer.print_header("Available Commands")

            categories = {
                "Chain Management": [
                    ("create <name>", "Create new chain interactively"),
                    ("load <name>", "Load existing chain"),
                    ("save [name]", "Save current chain"),
                    ("delete <name>", "Delete chain"),
                    ("list", "List all chains"),
                ],
                "Chain Operations": [
                    ("show [name]", "Visualize chain structure"),
                    ("test [name] [input]", "Test chain execution"),
                    ("run <name> <input>", "Run chain with input"),
                    ("deploy <name>", "Deploy chain as service"),
                ],
                "Information": [
                    ("info [name]", "Show chain information"),
                    ("status", "Show CLI status"),
                    ("agents", "List available agents"),
                ],
                "Utilities": [
                    ("export <name> [file]", "Export chain to file"),
                    ("import <file>", "Import chain from file"),
                    ("clear", "Clear screen"),
                    ("exit/quit", "Exit CLI"),
                ]
            }

            for category, commands in categories.items():
                print(f"\n{self.printer._colorize(category + ':', 'bold')}")
                for cmd, desc in commands:
                    print(f"  {self.printer._colorize(cmd, 'highlight'):20} {desc}")

    async def cmd_list(self, args: list[str]):
        """List all available chains"""
        chains = self.storage.list_chains()
        self.printer.print_chain_list(chains)

    async def cmd_create(self, args: list[str]):
        """Create new chain interactively"""
        if not args:
            self.printer.print_error("Usage: create <chain_name>")
            return

        name = args[0]

        # Check if chain already exists
        if self.storage.load_chain(name):
            if not confirm(f"Chain '{name}' already exists. Overwrite?"):
                return

        # Interactive creation
        result = await self.builder.interactive_create(name)

        if result:
            chain, user_metadata = result
            # Analyze chain for technical metadata and merge with user metadata
            tech_metadata = self._analyze_chain(chain, name)
            tech_metadata.description = user_metadata.description
            tech_metadata.author = user_metadata.author
            tech_metadata.tags = user_metadata.tags

            # Convert chain to serializable format
            chain_config = self._chain_to_config(chain)

            if self.storage.save_chain(name, chain_config, tech_metadata):
                self.current_chain = chain
                self.current_chain_name = name
                self.printer.print_success(f"Chain '{name}' created and loaded")

                # Show chain visualization
                await self.cmd_show([])
            else:
                self.printer.print_error(f"Failed to save chain '{name}'")
        else:
            self.printer.print_info("Chain creation cancelled")

    async def cmd_load(self, args: list[str]):
        """Load existing chain"""
        if not args:
            self.printer.print_error("Usage: load <chain_name>")
            return

        name = args[0]
        chain_config = self.storage.load_chain(name)

        if not chain_config:
            self.printer.print_error(f"Chain '{name}' not found")
            return

        try:
            chain = await self._config_to_chain(chain_config)
            self.current_chain = chain
            self.current_chain_name = name
            self.printer.print_success(f"Chain '{name}' loaded")

        except Exception as e:
            self.printer.print_error(f"Failed to load chain '{name}': {e}")

    async def cmd_save(self, args: list[str]):
        """Save current chain"""
        if not self.current_chain:
            self.printer.print_error("No chain loaded")
            return

        name = args[0] if args else self.current_chain_name
        if not name:
            self.printer.print_error("Usage: save <chain_name>")
            return

        metadata = self._analyze_chain(self.current_chain, name)
        chain_config = self._chain_to_config(self.current_chain)

        if self.storage.save_chain(name, chain_config, metadata):
            self.current_chain_name = name
            self.printer.print_success(f"Chain saved as '{name}'")
        else:
            self.printer.print_error(f"Failed to save chain '{name}'")

    async def cmd_delete(self, args: list[str]):
        """Delete chain"""
        if not args:
            self.printer.print_error("Usage: delete <chain_name>")
            return

        name = args[0]

        if not self.storage.load_chain(name):
            self.printer.print_error(f"Chain '{name}' not found")
            return

        if confirm(f"Delete chain '{name}'? This cannot be undone."):
            if self.storage.delete_chain(name):
                if self.current_chain_name == name:
                    self.current_chain = None
                    self.current_chain_name = None
                self.printer.print_success(f"Chain '{name}' deleted")
            else:
                self.printer.print_error(f"Failed to delete chain '{name}'")

    async def cmd_show(self, args: list[str]):
        """Visualize chain structure"""
        chain = self.current_chain
        name = args[0] if args else self.current_chain_name

        if args and args[0] != self.current_chain_name:
            # Load different chain for visualization
            chain_config = self.storage.load_chain(args[0])
            if not chain_config:
                self.printer.print_error(f"Chain '{args[0]}' not found")
                return
            try:
                chain = await self._config_to_chain(chain_config)
                name = args[0]
            except Exception as e:
                self.printer.print_error(f"Failed to load chain '{args[0]}': {e}")
                return

        if not chain:
            self.printer.print_error("No chain to show. Use 'load <name>' or 'create <name>'")
            return

        self.printer.print_header(f"Chain Structure: {name or 'Unnamed'}")

        try:
            # Use the chain's built-in visualization
            chain.print_graph()
        except Exception as e:
            self.printer.print_error(f"Failed to visualize chain: {e}")

    async def cmd_test(self, args: list[str]):
        """Test chain execution"""
        chain = self.current_chain
        chain_name = self.current_chain_name
        test_input = "Hello, this is a test input."

        if args:
            if len(args) == 1:
                # If only one arg, could be name or input
                if self.storage.load_chain(args[0]):
                    # It's a chain name
                    chain_config = self.storage.load_chain(args[0])
                    try:
                        chain = await self._config_to_chain(chain_config)
                        chain_name = args[0]
                    except Exception as e:
                        self.printer.print_error(f"Failed to load chain '{args[0]}': {e}")
                        return
                else:
                    # It's test input
                    test_input = args[0]
            elif len(args) >= 2:
                # First arg is chain name, rest is input
                chain_config = self.storage.load_chain(args[0])
                if not chain_config:
                    self.printer.print_error(f"Chain '{args[0]}' not found")
                    return
                try:
                    chain = await self._config_to_chain(chain_config)
                    chain_name = args[0]
                    test_input = " ".join(args[1:])
                except Exception as e:
                    self.printer.print_error(f"Failed to load chain '{args[0]}': {e}")
                    return

        if not chain:
            self.printer.print_error("No chain to test. Use 'load <name>' or 'create <name>'")
            return

        self.printer.print_header(f"Testing Chain: {chain_name or 'Unnamed'}")
        self.printer.print_info(f"Input: {test_input}")

        # Set up progress tracking
        tracker = ChainProgressTracker(self.printer)
        chain.set_progress_callback(tracker.emit_event)

        try:
            start_time = time.time()
            result = await chain.a_run(test_input)
            end_time = time.time()

            self.printer.print_success(f"Test completed in {end_time - start_time:.2f}s")
            print(f"\n{self.printer._colorize('Result:', 'bold')}")
            print(f"{result}")

            # Show execution statistics
            self._print_execution_stats(tracker.events)

        except Exception as e:
            self.printer.print_error(f"Test failed: {e}")
            if self.app.debug:
                traceback.print_exc()

    async def cmd_run(self, args: list[str]):
        """Run chain with input"""
        if len(args) < 2:
            self.printer.print_error("Usage: run <chain_name> <input>")
            return

        name = args[0]
        input_text = " ".join(args[1:])

        chain_config = self.storage.load_chain(name)
        if not chain_config:
            self.printer.print_error(f"Chain '{name}' not found")
            return

        try:
            chain = await self._config_to_chain(chain_config)
        except Exception as e:
            self.printer.print_error(f"Failed to load chain '{name}': {e}")
            return

        self.printer.print_header(f"Running Chain: {name}")

        # Set up progress tracking
        tracker = ChainProgressTracker(self.printer)
        chain.set_progress_callback(tracker)

        try:
            result = await chain.a_run(input_text)

            print(f"\n{self.printer._colorize('Output:', 'bold')}")
            print(f"{result}")

        except Exception as e:
            self.printer.print_error(f"Execution failed: {e}")
            if self.app.debug:
                traceback.print_exc()

    async def cmd_deploy(self, args: list[str]):
        """Deploy chain as service"""
        if not args:
            self.printer.print_error("Usage: deploy <chain_name>")
            return

        name = args[0]
        chain_config = self.storage.load_chain(name)

        if not chain_config:
            self.printer.print_error(f"Chain '{name}' not found")
            return

        self.printer.print_header(f"Deploying Chain: {name}")

        try:
            # Load chain
            chain = await self._config_to_chain(chain_config)
            metadata = self.storage.get_metadata(name)

            # Deploy using ISAA's publish_and_host_agent
            result = await self.isaa.publish_and_host_agent(
                agent=chain,
                public_name=f"Chain Service: {name}",
                description=f"Deployed chain service - {metadata.description if metadata else 'No description'}",
                access_level="public",
                registry_server= "wss://simplecore.app/ws/registry/connect" if 'remote' in args else "ws://localhost:8080/ws/registry/connect"
            )

            if result.get('success'):
                self.printer.print_success(f"Chain '{name}' deployed successfully!")
                print(f"\n{self.printer._colorize('Deployment Details:', 'bold')}")
                print(f"🌐 Public URL: {result['public_url']}")
                print(f"🔑 API Key: {result['public_api_key']}")
                if result.get('ui_url'):
                    print(f"🖥️  UI: {result['ui_url']}")
                print(f"🔌 WebSocket: {result['websocket_url']}")
            else:
                self.printer.print_error(f"Deployment failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            self.printer.print_error(f"Deployment failed: {e}")
            if self.app.debug:
                traceback.print_exc()

    async def cmd_info(self, args: list[str]):
        """Show chain information"""
        name = args[0] if args else self.current_chain_name

        if not name:
            self.printer.print_error("Usage: info <chain_name> or load a chain first")
            return

        metadata = self.storage.get_metadata(name)
        if not metadata:
            self.printer.print_error(f"Chain '{name}' not found")
            return

        self.printer.print_header(f"Chain Information: {name}")

        info_items = [
            ("Name", metadata.name),
            ("Description", metadata.description or "No description"),
            ("Version", metadata.version),
            ("Author", metadata.author or "Unknown"),
            ("Created", metadata.created_at.strftime("%Y-%m-%d %H:%M:%S")),
            ("Modified", metadata.modified_at.strftime("%Y-%m-%d %H:%M:%S")),
            ("Complexity", metadata.complexity),
            ("Agent Count", metadata.agent_count),
            ("Has Conditionals", "Yes" if metadata.has_conditionals else "No"),
            ("Has Parallels", "Yes" if metadata.has_parallels else "No"),
            ("Has Error Handling", "Yes" if metadata.has_error_handling else "No"),
        ]

        for label, value in info_items:
            print(f"  {self.printer._colorize(label + ':', 'bold'):20} {value}")

        if metadata.tags:
            tags_str = ", ".join(metadata.tags)
            print(f"  {self.printer._colorize('Tags:', 'bold'):20} {tags_str}")

    async def cmd_clear(self, args: list[str]):
        """Clear screen"""
        os.system('clear' if os.name == 'posix' else 'cls')

    async def cmd_status(self, args: list[str]):
        """Show CLI status"""
        self.printer.print_header("Chain CLI Status")

        # Current state
        current_chain = self.current_chain_name or "None"
        total_chains = len(self.storage.list_chains())
        available_agents = len(self.isaa.config.get("agents-name-list", []))

        status_items = [
            ("Current Chain", current_chain),
            ("Total Chains", total_chains),
            ("Available Agents", available_agents),
            ("Session ID", self.session_id),
            ("Data Directory", str(self.storage.chains_dir)),
        ]

        for label, value in status_items:
            print(f"  {self.printer._colorize(label + ':', 'bold'):20} {value}")

    async def cmd_agents(self, args: list[str]):
        """List available agents"""
        agents = self.isaa.config.get("agents-name-list", [])

        if not agents:
            self.printer.print_info("No agents available. Create agents using ISAA first.")
            return

        self.printer.print_header(f"Available Agents ({len(agents)})")

        for i, agent_name in enumerate(agents, 1):
            print(f"  {i:2d}. {self.printer._colorize(agent_name, 'highlight')}")

    async def cmd_export(self, args: list[str]):
        """Export chain to file"""
        if not args:
            self.printer.print_error("Usage: export <chain_name> [output_file]")
            return

        name = args[0]
        output_file = args[1] if len(args) > 1 else f"{name}_export.json"

        chain_config = self.storage.load_chain(name)
        metadata = self.storage.get_metadata(name)

        if not chain_config:
            self.printer.print_error(f"Chain '{name}' not found")
            return

        try:
            export_data = {
                "chain_name": name,
                "chain_config": chain_config,
                "metadata": {
                    "name": metadata.name,
                    "description": metadata.description,
                    "created_at": metadata.created_at.isoformat(),
                    "modified_at": metadata.modified_at.isoformat(),
                    "version": metadata.version,
                    "tags": metadata.tags,
                    "author": metadata.author,
                    "complexity": metadata.complexity,
                    "agent_count": metadata.agent_count,
                    "has_conditionals": metadata.has_conditionals,
                    "has_parallels": metadata.has_parallels,
                    "has_error_handling": metadata.has_error_handling
                } if metadata else {},
                "export_timestamp": datetime.now().isoformat(),
                "cli_version": "1.0.0"
            }

            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)

            self.printer.print_success(f"Chain '{name}' exported to '{output_file}'")

        except Exception as e:
            self.printer.print_error(f"Export failed: {e}")

    async def cmd_import(self, args: list[str]):
        """Import chain from file"""
        if not args:
            self.printer.print_error("Usage: import <file_path>")
            return

        file_path = args[0]

        if not os.path.exists(file_path):
            self.printer.print_error(f"File '{file_path}' not found")
            return

        try:
            with open(file_path) as f:
                import_data = json.load(f)

            chain_name = import_data["chain_name"]
            chain_config = import_data["chain_config"]
            metadata_data = import_data["metadata"]

            # Check if chain exists
            if self.storage.load_chain(chain_name):
                if not confirm(f"Chain '{chain_name}' already exists. Overwrite?"):
                    return

            # Create metadata object
            metadata = ChainMetadata(
                name=metadata_data.get("name", chain_name),
                description=metadata_data.get("description", ""),
                created_at=datetime.fromisoformat(metadata_data.get("created_at", datetime.now().isoformat())),
                modified_at=datetime.now(),  # Update to current time
                version=metadata_data.get("version", "1.0.0"),
                tags=metadata_data.get("tags", []),
                author=metadata_data.get("author", ""),
                complexity=metadata_data.get("complexity", "simple"),
                agent_count=metadata_data.get("agent_count", 0),
                has_conditionals=metadata_data.get("has_conditionals", False),
                has_parallels=metadata_data.get("has_parallels", False),
                has_error_handling=metadata_data.get("has_error_handling", False)
            )

            if self.storage.save_chain(chain_name, chain_config, metadata):
                self.printer.print_success(f"Chain '{chain_name}' imported successfully")
            else:
                self.printer.print_error(f"Failed to save imported chain '{chain_name}'")

        except Exception as e:
            self.printer.print_error(f"Import failed: {e}")
            if self.app.debug:
                traceback.print_exc()

    async def cmd_exit(self, args: list[str]):
        """Exit CLI"""
        self.printer.print_info("Goodbye! 👋")
        raise EOFError()

    # Helper methods
    def _analyze_chain(self, chain: Chain, name: str) -> ChainMetadata:
        """Analyze chain and generate metadata"""
        try:
            graph_data = chain.chain_to_graph()
            structure = graph_data.get("structure", {})

            return ChainMetadata(
                name=name,
                description="",  # Will be filled by user
                complexity=structure.get("chain_type", "simple").lower(),
                agent_count=self._count_agents(chain),
                has_conditionals=structure.get("has_conditionals", False),
                has_parallels=structure.get("has_parallels", False),
                has_error_handling=structure.get("has_error_handling", False)
            )
        except:
            return ChainMetadata(name=name)

    def _count_agents(self, chain) -> int:
        """Count total agents in chain"""
        try:
            if hasattr(chain, 'tasks'):
                return len([t for t in chain.tasks if hasattr(t, 'amd')])
            return 0
        except:
            return 0

    def _chain_to_config(self, chain: Chain) -> dict:
        """Convert chain to serializable configuration"""

        def serialize_component(comp, comp_id=None):
            """Recursively serialize chain components"""
            if comp is None:
                return {"type": "null"}

            comp_id = comp_id or str(uuid.uuid4())

            # Handle FlowAgent
            if hasattr(comp, 'amd') and comp.amd:
                return {
                    "type": "agent",
                    "id": comp_id,
                    "name": comp.amd.name,
                    "agent_type": "flow_agent"
                }

            # Handle CF (Chain Format)
            if hasattr(comp, 'format_class'):
                config = {
                    "type": "format",
                    "id": comp_id,
                    "format_class": comp.format_class.__name__,
                    "module": comp.format_class.__module__,
                }

                # Serialize model schema for reconstruction
                if hasattr(comp.format_class, 'model_json_schema'):
                    config["schema"] = comp.format_class.model_json_schema()

                # Handle extraction parameters
                if hasattr(comp, 'extract_key') and comp.extract_key:
                    config["extract_key"] = comp.extract_key
                if hasattr(comp, 'extract_multiple') and comp.extract_multiple:
                    config["extract_multiple"] = comp.extract_multiple
                if hasattr(comp, 'parallel_count') and comp.parallel_count:
                    config["parallel_count"] = comp.parallel_count

                return config

            # Handle IS (Conditional)
            if hasattr(comp, 'key') and hasattr(comp, 'expected_value'):
                return {
                    "type": "condition",
                    "id": comp_id,
                    "key": comp.key,
                    "expected_value": comp.expected_value
                }

            # Handle ParallelChain
            if hasattr(comp, 'agents') and isinstance(comp.agents, list | tuple):
                return {
                    "type": "parallel",
                    "id": comp_id,
                    "agents": [serialize_component(agent, f"{comp_id}_agent_{i}")
                               for i, agent in enumerate(comp.agents)]
                }

            # Handle ConditionalChain
            if hasattr(comp, 'condition') and hasattr(comp, 'true_branch'):
                config = {
                    "type": "conditional",
                    "id": comp_id,
                    "condition": serialize_component(comp.condition, f"{comp_id}_condition"),
                    "true_branch": serialize_component(comp.true_branch, f"{comp_id}_true")
                }
                if hasattr(comp, 'false_branch') and comp.false_branch:
                    config["false_branch"] = serialize_component(comp.false_branch, f"{comp_id}_false")
                return config

            # Handle ErrorHandlingChain
            if hasattr(comp, 'primary') and hasattr(comp, 'fallback'):
                return {
                    "type": "error_handling",
                    "id": comp_id,
                    "primary": serialize_component(comp.primary, f"{comp_id}_primary"),
                    "fallback": serialize_component(comp.fallback, f"{comp_id}_fallback")
                }

            # Handle Chain
            if hasattr(comp, 'tasks') and isinstance(comp.tasks, list | tuple):
                return {
                    "type": "chain",
                    "id": comp_id,
                    "name": getattr(comp, 'name', 'unnamed_chain'),
                    "tasks": [serialize_component(task, f"{comp_id}_task_{i}")
                              for i, task in enumerate(comp.tasks)]
                }

            # Fallback for unknown components
            return {
                "type": "unknown",
                "id": comp_id,
                "class_name": type(comp).__name__,
                "str_repr": str(comp)
            }

        # Main serialization
        main_config = serialize_component(chain, "main_chain")

        return {
            "version": "1.0.0",
            "created_by": "chain_cli",
            "created_at": datetime.now().isoformat(),
            "chain_type": main_config.get("type", "unknown"),
            "structure": main_config,
            "metadata": {
                "total_components": self._count_components(chain),
                "agent_names": self._extract_agent_names(chain),
                "has_formatting": self._has_formatting(chain),
                "has_conditions": self._has_conditions(chain),
                "complexity_score": self._calculate_complexity(chain)
            }
        }

    def _count_components(self, chain) -> int:
        """Count total components in chain"""

        def count_recursive(comp):
            if comp is None:
                return 0
            count = 1
            if hasattr(comp, 'tasks'):
                count += sum(count_recursive(task) for task in comp.tasks)
            elif hasattr(comp, 'agents'):
                count += sum(count_recursive(agent) for agent in comp.agents)
            elif hasattr(comp, 'true_branch'):
                count += count_recursive(comp.true_branch)
                if hasattr(comp, 'false_branch'):
                    count += count_recursive(comp.false_branch)
            elif hasattr(comp, 'primary'):
                count += count_recursive(comp.primary)
                count += count_recursive(comp.fallback)
            return count

        return count_recursive(chain)

    def _extract_agent_names(self, chain) -> list[str]:
        """Extract all agent names from chain"""
        names = []

        def extract_recursive(comp):
            if comp is None:
                return
            if hasattr(comp, 'amd') and comp.amd and hasattr(comp.amd, 'name'):
                names.append(comp.amd.name)
            elif hasattr(comp, 'tasks'):
                for task in comp.tasks:
                    extract_recursive(task)
            elif hasattr(comp, 'agents'):
                for agent in comp.agents:
                    extract_recursive(agent)
            elif hasattr(comp, 'true_branch'):
                extract_recursive(comp.true_branch)
                if hasattr(comp, 'false_branch'):
                    extract_recursive(comp.false_branch)
            elif hasattr(comp, 'primary'):
                extract_recursive(comp.primary)
                extract_recursive(comp.fallback)

        extract_recursive(chain)
        return list(set(names))  # Remove duplicates

    def _has_formatting(self, chain) -> bool:
        """Check if chain has formatting components"""

        def check_recursive(comp):
            if comp is None:
                return False
            if hasattr(comp, 'format_class'):
                return True
            if hasattr(comp, 'tasks'):
                return any(check_recursive(task) for task in comp.tasks)
            elif hasattr(comp, 'agents'):
                return any(check_recursive(agent) for agent in comp.agents)
            elif hasattr(comp, 'true_branch'):
                return (check_recursive(comp.true_branch) or
                        (hasattr(comp, 'false_branch') and check_recursive(comp.false_branch)))
            elif hasattr(comp, 'primary'):
                return check_recursive(comp.primary) or check_recursive(comp.fallback)
            return False

        return check_recursive(chain)

    def _has_conditions(self, chain) -> bool:
        """Check if chain has conditional components"""

        def check_recursive(comp):
            if comp is None:
                return False
            if hasattr(comp, 'key') and hasattr(comp, 'expected_value'):
                return True
            if hasattr(comp, 'condition'):
                return True
            if hasattr(comp, 'tasks'):
                return any(check_recursive(task) for task in comp.tasks)
            elif hasattr(comp, 'agents'):
                return any(check_recursive(agent) for agent in comp.agents)
            elif hasattr(comp, 'true_branch'):
                return (check_recursive(comp.true_branch) or
                        (hasattr(comp, 'false_branch') and check_recursive(comp.false_branch)))
            elif hasattr(comp, 'primary'):
                return check_recursive(comp.primary) or check_recursive(comp.fallback)
            return False

        return check_recursive(chain)

    def _calculate_complexity(self, chain) -> int:
        """Calculate complexity score for chain"""
        score = 0

        def score_recursive(comp):
            nonlocal score
            if comp is None:
                return

            # Basic component = 1 point
            score += 1

            # Parallel execution = +2 points
            if hasattr(comp, 'agents'):
                score += 2
                for agent in comp.agents:
                    score_recursive(agent)

            # Conditional logic = +3 points
            elif hasattr(comp, 'condition'):
                score += 3
                score_recursive(comp.true_branch)
                if hasattr(comp, 'false_branch'):
                    score_recursive(comp.false_branch)

            # Error handling = +2 points
            elif hasattr(comp, 'primary'):
                score += 2
                score_recursive(comp.primary)
                score_recursive(comp.fallback)

            # Format with extraction = +1 point
            elif hasattr(comp, 'format_class'):
                if hasattr(comp, 'extract_key') and comp.extract_key:
                    score += 1

            # Sequential tasks
            elif hasattr(comp, 'tasks'):
                for task in comp.tasks:
                    score_recursive(task)

        score_recursive(chain)
        return score

    async def _config_to_chain(self, config: dict) -> Chain:
        """Convert configuration back to chain"""

        async def deserialize_component(comp_config):
            """Recursively deserialize chain components"""
            if not comp_config or comp_config.get("type") == "null":
                return None

            comp_type = comp_config.get("type")

            # Handle FlowAgent
            if comp_type == "agent":
                agent_name = comp_config["name"]
                try:
                    return await self.isaa.get_agent(agent_name)
                except Exception as e:
                    raise Exception(f"Failed to load agent '{agent_name}': {e}")

            # Handle CF (Chain Format)
            elif comp_type == "format":
                try:
                    # Reconstruct Pydantic model from schema
                    schema = comp_config.get("schema", {})
                    model_name = comp_config["format_class"]

                    if schema and "properties" in schema:
                        # Rebuild model from schema
                        fields = {}
                        for field_name, field_info in schema["properties"].items():
                            field_type = str  # Default type
                            field_desc = field_info.get("description", "")

                            # Map JSON schema types to Python types
                            json_type = field_info.get("type")
                            if json_type == "integer":
                                field_type = int
                            elif json_type == "number":
                                field_type = float
                            elif json_type == "boolean":
                                field_type = bool
                            elif json_type == "array":
                                field_type = list[str]  # Simplified

                            if field_desc:
                                fields[field_name] = (field_type, PydanticField(description=field_desc))
                            else:
                                fields[field_name] = (field_type, ...)

                        DynamicModel = create_model(model_name, **fields)
                    else:
                        # Fallback: create simple model
                        DynamicModel = create_model(model_name, value=(str, PydanticField(description="Dynamic field")))

                    # Create CF instance
                    cf = CF(DynamicModel)

                    # Apply extraction parameters
                    if "extract_key" in comp_config:
                        cf = cf - comp_config["extract_key"]
                    if comp_config.get("extract_multiple"):
                        cf.extract_multiple = True
                    if comp_config.get("parallel_count"):
                        cf.parallel_count = comp_config["parallel_count"]

                    return cf

                except Exception as e:
                    raise Exception(f"Failed to reconstruct format model '{model_name}': {e}")

            # Handle IS (Conditional)
            elif comp_type == "condition":
                return IS(comp_config["key"], comp_config["expected_value"])

            # Handle ParallelChain
            elif comp_type == "parallel":
                agents = []
                for agent_config in comp_config.get("agents", []):
                    agent = await deserialize_component(agent_config)
                    if agent:
                        agents.append(agent)
                return ParallelChain(agents)

            # Handle ConditionalChain
            elif comp_type == "conditional":
                condition = await deserialize_component(comp_config["condition"])
                true_branch = await deserialize_component(comp_config["true_branch"])
                false_branch = None
                if "false_branch" in comp_config:
                    false_branch = await deserialize_component(comp_config["false_branch"])

                return ConditionalChain(condition, true_branch, false_branch)

            # Handle ErrorHandlingChain
            elif comp_type == "error_handling":
                primary = await deserialize_component(comp_config["primary"])
                fallback = await deserialize_component(comp_config["fallback"])
                return ErrorHandlingChain(primary, fallback)

            # Handle Chain
            elif comp_type == "chain":
                tasks = []
                for task_config in comp_config.get("tasks", []):
                    task = await deserialize_component(task_config)
                    if task:
                        tasks.append(task)

                chain = Chain._create_chain(tasks)
                if "name" in comp_config:
                    chain.name = comp_config["name"]
                return chain

            else:
                raise Exception(f"Unknown component type: {comp_type}")

        # Validate config version
        if config.get("version", "1.0.0") != "1.0.0":
            self.printer.print_warning(f"Config version {config.get('version')} may not be fully compatible")

        # Deserialize main structure
        structure = config.get("structure", {})
        if not structure:
            raise Exception("Invalid config: missing structure")

        # Check if required agents exist
        required_agents = config.get("metadata", {}).get("agent_names", [])
        available_agents = self.isaa.config.get("agents-name-list", [])
        missing_agents = [name for name in required_agents if name not in available_agents]

        if missing_agents:
            available_str = ", ".join(available_agents) if available_agents else "None"
            raise Exception(f"Missing required agents: {', '.join(missing_agents)}. Available agents: {available_str}")

        # Build the chain
        try:
            main_chain = await deserialize_component(structure)

            if not isinstance(main_chain, Chain | ParallelChain | ConditionalChain | ErrorHandlingChain):
                # Wrap single components in a Chain
                if main_chain:
                    main_chain = Chain(main_chain)
                else:
                    raise Exception("Failed to reconstruct chain: no valid components")

            return main_chain

        except Exception as e:
            raise Exception(f"Failed to reconstruct chain: {e}")

    def _print_execution_stats(self, events: list[ProgressEvent]):
        """Print execution statistics"""
        if not events:
            return

        print(f"\n{self.printer._colorize('Execution Statistics:', 'bold')}")

        total_events = len(events)
        completed_tasks = len([e for e in events if e.success])
        failed_tasks = len([e for e in events if e.event_type == "task_error"])

        stats = [
            ("Total Events", total_events),
            ("Completed Tasks", completed_tasks),
            ("Failed Tasks", failed_tasks),
        ]

        for label, value in stats:
            color = "success" if "Completed" in label else "error" if "Failed" in label else "info"
            print(f"  {label}: {self.printer._colorize(str(value), color)}")

    async def _cleanup(self):
        """Cleanup resources"""
        try:
            if self.current_chain and hasattr(self.current_chain, 'close'):
                await self.current_chain.close()
        except:
            pass


# Main entry point
async def run(app_instance, *args):
    """Entry point for Chain CLI"""
    if not app_instance:
        app_instance = get_app("chain_cli_instance")

    cli = ChainCLI(app_instance)

    try:
        await cli.run()
    except KeyboardInterrupt:
        print("\n🛑 Chain CLI interrupted.")
    except Exception as e:
        print(f"💥 Fatal error in Chain CLI: {e}")
        if app_instance.debug:
            traceback.print_exc()


if __name__ == "__main__":
    # For direct execution
    app = get_app("chain_cli_standalone")
    asyncio.run(run(app))
