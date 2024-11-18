import os
try:
    import customtkinter as ctk
except ImportError:
    os.system("pip install customtkinter")
    import customtkinter as ctk

import inspect
from typing import Callable
import asyncio
from toolboxv2.runabel import runnable_dict as runnable_dict_func


class DynamicFunctionApp:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Function Runner")
        self.window.geometry("1000x800")

        # State variables
        self.is_logged_in = False
        self.username = ""
        self.tb_app = None
        self.runnable_dict = {}
        self.edit_mode = False

        # Configure grid layout
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_rowconfigure(1, weight=1)

        self._create_header()
        self._create_main_content()

        # Start async login process
        self.window.after(100, self._initialize_login)

    def _create_header(self):
        """Create the header with login/logout functionality"""
        header_frame = ctk.CTkFrame(self.window)
        header_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        header_frame.grid_columnconfigure(2, weight=1)

        # App title
        title_label = ctk.CTkLabel(
            header_frame,
            text="Function Runner",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.grid(row=0, column=0, padx=10, pady=5)

        # Username display
        self.username_label = ctk.CTkLabel(
            header_frame,
            text="Logging in...",
            font=ctk.CTkFont(size=14)
        )
        self.username_label.grid(row=0, column=1, padx=10)

        # Edit mode toggle
        self.edit_toggle = ctk.CTkSwitch(
            header_frame,
            text="Edit Mode",
            command=self._toggle_edit_mode
        )
        self.edit_toggle.grid(row=0, column=2, padx=10)

    def _create_main_content(self):
        """Create the grid layout for function cards"""
        self.main_frame = ctk.CTkScrollableFrame(self.window)
        self.main_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        # Configure grid columns
        for i in range(3):  # 3 cards per row
            self.main_frame.grid_columnconfigure(i, weight=1, uniform="column")

    def _create_function_cards(self):
        """Create and layout function cards in a grid"""
        # Clear existing cards
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Create cards in a grid layout (3 columns)
        for idx, (func_name, func) in enumerate(self.runnable_dict.items()):
            row = idx // 3
            col = idx % 3

            self._create_function_card(func_name, func, row, col)

    def _create_function_card(self, func_name: str, func: Callable, row: int, col: int):
        """Create a card for each function with its parameters"""
        # Card frame
        card = ctk.CTkFrame(self.main_frame)
        card.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)

        # Function name
        name_label = ctk.CTkLabel(
            card,
            text=func_name,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        name_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        # Get function parameters
        params = inspect.signature(func).parameters
        param_widgets = {}
        current_row = 1

        for param_name, param in params.items():
            # Skip 'self' and 'app' parameters
            if param_name in ['self', 'app', 'args', '_', '__']:
                continue

            if self.edit_mode:
                # Parameter label
                param_label = ctk.CTkLabel(card, text=param_name)
                param_label.grid(row=current_row, column=0, padx=10, pady=2, sticky="w")

                # Parameter input
                param_input = ctk.CTkEntry(card)
                param_input.grid(row=current_row, column=1, padx=10, pady=2, sticky="ew")

                # Set default value if exists
                if param.default is not param.empty:
                    param_input.insert(0, str(param.default))

                param_widgets[param_name] = param_input
                current_row += 1

        # Run button
        def run_function():
            kwargs = []
            if self.edit_mode:
                kwargs = [
                    f"--kwargs {name}={widget.get()}"
                    for name, widget in param_widgets.items()
                ]
            import subprocess
            import threading

            command = ' '.join(['tb', '-m', func_name] + kwargs)

            # Windows Terminal öffnen und den PowerShell-Befehl ausführen
            g_command = os.getenv("GUI_COMMAND", "wt new-tab powershell -NoExit -Command").strip()
            if "${command}" in g_command:
                g_command = g_command.replace("${command}", command)
            else:
                g_command += ' '+command
            wt_command = g_command

            # Ausführen des Befehls in einem separaten Thread
            threading.Thread(
                target=subprocess.run,
                args=(wt_command,),
                kwargs={"shell": True},
                daemon=True
            ).start()

        run_button = ctk.CTkButton(
            card,
            text="Run",
            command=run_function
        )
        run_button.grid(row=current_row, column=0, pady=10, sticky="ew")

    async def _perform_login(self):
        """Handle the async login process"""
        try:
            from toolboxv2 import get_app
            self.tb_app = get_app()
            login_success = await self.tb_app.session.login()

            if not login_success:
                # Show magic link dialog
                self._show_magic_link_dialog()
            else:
                self._complete_login()
        except Exception as e:
            self._show_error(f"Login failed: {str(e)}")

    def _initialize_login(self):
        """Initialize the async login process"""
        asyncio.run(self._perform_login())

    def _show_magic_link_dialog(self):
        """Show dialog for magic link input"""
        dialog = ctk.CTkInputDialog(
            text="Please enter the magic link:",
            title="Magic Link Login"
        )
        magic_link = dialog.get_input()

        if magic_link:
            asyncio.run(self._magic_link_login(magic_link))

    async def _magic_link_login(self, magic_link: str):
        """Handle magic link login"""
        try:
            success = await self.tb_app.session.init_log_in_mk_link(magic_link)
            if success:
                self._complete_login()
            else:
                self._show_error("Invalid magic link")
        except Exception as e:
            self._show_error(f"Magic link login failed: {str(e)}")

    def _complete_login(self):
        """Complete the login process"""
        self.is_logged_in = True
        self.username = self.tb_app.get_username()
        self.username_label.configure(text=f"Logged in as: {self.username}")

        # Set runnable dict
        self.runnable_dict = runnable_dict_func(remote=False)
        self.runnable_dict.update(runnable_dict_func(s='', remote=True))

        # Create function cards
        self._create_function_cards()

    def _toggle_edit_mode(self):
        """Toggle edit mode for function parameters"""
        self.edit_mode = self.edit_toggle.get()
        self._create_function_cards()  # Recreate cards with new mode

    def _show_error(self, message: str):
        """Display error message in a popup"""
        error_window = ctk.CTkToplevel(self.window)
        error_window.title("Error")
        error_window.geometry("400x150")

        label = ctk.CTkLabel(
            error_window,
            text=message,
            wraplength=350
        )
        label.pack(padx=20, pady=20)

        button = ctk.CTkButton(
            error_window,
            text="OK",
            command=error_window.destroy
        )
        button.pack(pady=10)

    def run(self):
        """Start the application"""
        self.window.mainloop()


def start():
    app = DynamicFunctionApp()
    app.run()


if __name__ == "__main__":
    app = DynamicFunctionApp()
    app.run()
