import os
import platform
import sys

from toolboxv2 import Style, show_console

try:
    import customtkinter as ctk
except ImportError:
    os.system("pip install customtkinter")
    import customtkinter as ctk

import asyncio
import inspect
import subprocess
import threading
from collections.abc import Callable

from toolboxv2.flows import flows_dict as flows_dict_func


class ContextualInputDialog(ctk.CTkToplevel):
    def __init__(self, master, title, text, default_value='',
                 validator=None, context_info=None):
        super().__init__(master)
        self.title(title)
        self.geometry("400x250")
        self.result = None

        # Context Information
        if context_info:
            context_label = ctk.CTkLabel(
                self,
                text=context_info,
                wraplength=350,
                font=ctk.CTkFont(size=12)
            )
            context_label.pack(padx=20, pady=(20, 10))

        # Main Input Label
        label = ctk.CTkLabel(
            self,
            text=text,
            wraplength=350
        )
        label.pack(padx=20, pady=(10, 5))

        # Input Entry
        self.entry = ctk.CTkEntry(
            self,
            width=300,
            height=35
        )
        self.entry.insert(0, default_value)
        self.entry.pack(padx=20, pady=10)
        self.entry.focus()

        # Validation Indicator
        self.validation_label = ctk.CTkLabel(
            self,
            text="",
            text_color="red"
        )
        self.validation_label.pack(padx=20, pady=(0, 10))

        # Button Frame
        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=10)

        confirm_btn = ctk.CTkButton(
            btn_frame,
            text="Confirm",
            command=self._on_confirm
        )
        confirm_btn.pack(side="left", padx=5)

        cancel_btn = ctk.CTkButton(
            btn_frame,
            text="Cancel",
            fg_color="gray",
            command=self.destroy
        )
        cancel_btn.pack(side="left", padx=5)

        self.validator = validator

        # Bind Enter key
        self.entry.bind("<Return>", lambda e: self._on_confirm())
        self.entry.bind("<Escape>", lambda e: self.destroy())

    def _on_confirm(self):
        value = self.entry.get()

        if self.validator:
            validation_result = self.validator(value)
            if not validation_result[0]:
                self.validation_label.configure(text=validation_result[1])
                return

        self.result = value
        self.destroy()

    @classmethod
    def show(cls, master, **kwargs):
        dialog = cls(master, **kwargs)
        dialog.wait_window()
        return dialog.result


class DynamicFunctionApp:

    def __init__(self):
        self.run_command_function_holder = False
        self.window = ctk.CTk()
        self.window.title("Function Runner")
        self.window.geometry("1200x800")

        # Theme und Farben
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        # State variables
        self.is_logged_in = False
        self.username = ""
        self.tb_app = None
        self.flows_dict = {}
        self.card_edit_states = {}
        self.search_var = ctk.StringVar()
        self.search_var.trace_add("write", self._on_search_change)

        # Configure grid layout
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_rowconfigure(1, weight=1)  # Hauptinhalt bekommt mehr Platz

        self._create_top_section()
        self._create_main_content()
        self._initialize_flowss()

        # Asynchrone Login-Initialisierung
        self.window.after(100, self._initialize_login)

    def _create_top_section(self):
        """Erstellt den oberen Bereich mit Header, Suche und Übersicht"""
        top_frame = ctk.CTkFrame(self.window)
        top_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        top_frame.grid_columnconfigure(1, weight=1)

        # Header
        header_frame = ctk.CTkFrame(top_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        header_frame.grid_columnconfigure(1, weight=1)

        title_label = ctk.CTkLabel(
            header_frame,
            text="Function Runner",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.grid(row=0, column=0, padx=10, pady=5)

        self.username_label = ctk.CTkLabel(
            header_frame,
            text="Not logged in",
            font=ctk.CTkFont(size=14)
        )
        self.username_label.grid(row=0, column=1, padx=10, sticky="e")

        # Suchleiste
        search_frame = ctk.CTkFrame(top_frame)
        search_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        search_frame.grid_columnconfigure(0, weight=1)

        search_entry = ctk.CTkEntry(
            search_frame,
            placeholder_text="Search functions or type command...",
            textvariable=self.search_var,
            height=35
        )
        search_entry.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        run_command_btn = ctk.CTkButton(
            search_frame,
            text="Run Command",
            command=self._run_search_command,
            height=35,
        )

        run_command_btn.grid(row=0, column=1, padx=10, pady=5)

        # Kategorien-Übersicht
        self.overview_frame = ctk.CTkFrame(top_frame)
        self.overview_frame.grid(row=2, column=1, columnspan=2, sticky="ew", pady=5)

    def _create_main_content(self):
        """Erstellt den Hauptbereich mit den Funktionskarten"""
        self.main_frame = ctk.CTkScrollableFrame(self.window)
        self.main_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        # 3-Spalten-Layout für Karten
        for i in range(3):
            self.main_frame.grid_columnconfigure(i, weight=1)

    def _create_function_cards(self, filter_text=""):
        """Erstellt die Funktionskarten im Raster-Layout"""
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        filtered_funcs = {
            name: func for name, func in self.flows_dict.items()
            if filter_text.lower() in name.lower()
        }

        for idx, (func_name, func) in enumerate(filtered_funcs.items()):
            row = idx // 3
            col = idx % 3
            self._create_function_card(func_name, func, row, col, solo=len(filtered_funcs.items()) == 1)

    def _create_function_card(self, func_name: str, func: Callable, row: int, col: int, solo=False):
        """Erstellt eine einzelne Funktionskarte"""
        # Hauptkarte
        card = ctk.CTkFrame(self.main_frame, fg_color=self._get_card_color(func_name, solo))
        card.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)

        # Karten-Header mit Funktionsname und Edit-Toggle
        header_frame = ctk.CTkFrame(card)
        header_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        header_frame.grid_columnconfigure(0, weight=1)

        # Erster Buchstabe groß im Hintergrund
        first_letter = ctk.CTkLabel(
            header_frame,
            text=func_name[0].upper(),
            font=ctk.CTkFont(size=48, weight="bold"),
            text_color=("lightblue", "darkblue") if solo else ("gray70", "gray30")
        )
        first_letter.grid(row=1, column=0, padx=10, pady=5)

        name_label = ctk.CTkLabel(
            header_frame,
            text=func_name,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        name_label.grid(row=0, column=0, padx=10, pady=5)

        # Parameter-Sektion
        params = inspect.signature(func).parameters
        param_widgets = {}
        current_row = 0

        if len(params.items()) > 2:

            edit_toggle = ctk.CTkSwitch(
                header_frame,
                text="Edit",
                command=lambda: self._toggle_card_edit(func_name)
            )
            edit_toggle.grid(row=2, column=0, padx=10, pady=5)
            self.card_edit_states[func_name] = self.card_edit_states.get(func_name, False)
            if self.card_edit_states[func_name]:

                params_frame = ctk.CTkFrame(card)
                params_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
                params_frame.grid_columnconfigure(1, weight=1)

                for param_name, param in params.items():
                    # Ignoriere spezielle Parameter
                    if param_name in ['self', 'app', 'args', '_', '__', 'app_args', '*', '*_']:
                        continue

                    # Parameter-Label immer sichtbar
                    param_label = ctk.CTkLabel(params_frame, text=param_name)
                    param_label.grid(row=current_row, column=0, padx=5, pady=2, sticky="w")

                    # Wert-Anzeige oder Edit-Feld
                    if self.card_edit_states.get(func_name, False):
                        param_input = ctk.CTkEntry(params_frame)
                        param_input.grid(row=current_row, column=1, padx=5, pady=2, sticky="ew")
                        if param.default is not param.empty:
                            param_input.insert(0, str(param.default))
                        param_widgets[param_name] = param_input
                    else:
                        value_label = ctk.CTkLabel(
                            params_frame,
                            text=str(param.default) if param.default is not param.empty else "N/A"
                        )
                        value_label.grid(row=current_row, column=1, padx=5, pady=2, sticky="w")

                    current_row += 1

                # Run-Button
                run_button = ctk.CTkButton(
                    card,
                    text="Run",
                    command=lambda: self._run_function(func_name, param_widgets)
                )
                run_button.grid(row=2, column=0, pady=0, padx=10, sticky="ew")
        else:
            # Run-Button
            run_button = ctk.CTkButton(
                card,
                text="Run",
                command=lambda: self._run_function(func_name, param_widgets)
            )
            run_button.grid(row=1, column=0, pady=0, padx=10, sticky="ew")

    def _get_card_color(self, func_name, solo=False):
        """Dynamic color based on function characteristics"""
        func_name.split('_')[0]
        if solo:
            return ("magenta", "darkmagenta")[ctk.get_appearance_mode() == "Dark"]
        return ("gray90", "gray20")[ctk.get_appearance_mode() == "Dark"]

    def _run_function(self, func_name: str, param_widgets: dict):
        """Führt die ausgewählte Funktion aus"""
        kwargs = []
        if self.card_edit_states.get(func_name, False):
            kwargs = [
                f"--kwargs {name}={widget.get()}"
                for name, widget in param_widgets.items()
            ]

        command = ' '.join([sys.executable, '-m', 'toolboxv2', '-m', func_name] + kwargs)
        g_command = os.getenv("GUI_COMMAND", "wt new-tab powershell -NoExit -Command").strip()

        if "${command}" in g_command:
            g_command = g_command.replace("${command}", command)
        else:
            g_command += ' ' + command

        import subprocess
        import threading
        threading.Thread(
            target=subprocess.run,
            args=(g_command,),
            kwargs={"shell": True},
            daemon=True
        ).start()

    # [Rest der Klasse bleibt gleich...]

    def _update_quick_overview(self):
        """Aktualisiert die Kategorien-Übersicht"""
        for widget in self.overview_frame.winfo_children():
            widget.destroy()

        # Create scrollable frame for functions
        func_scroll_frame = ctk.CTkFrame(
            self.overview_frame,
            height=25,
            width=800
        )
        func_scroll_frame.pack(fill="x", padx=10, pady=5)

        # Dynamic function placement
        max_columns = 36  # Adjust based on screen width
        current_row = 0
        current_column = 0

        for _category, funcs in self._categorize_functions().items():
            current_column += 1

            for func_name in funcs:
                func_btn = ctk.CTkButton(
                    func_scroll_frame,
                    text=func_name,
                    width=50,
                    command=lambda f=func_name: self._filter_by_function(f)
                )
                func_btn.grid(row=current_row, column=current_column, padx=5, pady=2)

                current_column += 1
                if current_column >= max_columns:
                    current_row += 1
                    current_column = 0

    def _categorize_functions(self):
        """Group functions by category"""
        categories = {}
        for func_name in self.flows_dict:
            category = func_name.split('_')[0]
            categories.setdefault(category, []).append(func_name)
        return categories

    def _filter_by_function(self, func_name):
        """Filter functions by specific function name"""
        self.search_var.set(func_name)

    def _create_header(self):
        header_frame = ctk.CTkFrame(self.window)
        header_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        header_frame.grid_columnconfigure(1, weight=1)

        title_label = ctk.CTkLabel(
            header_frame,
            text="Function Runner",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.grid(row=0, column=0, padx=10, pady=5)

        self.username_label = ctk.CTkLabel(
            header_frame,
            text="Not logged in",
            font=ctk.CTkFont(size=14)
        )
        self.username_label.grid(row=0, column=1, padx=10, sticky="e")

    def _toggle_card_edit(self, func_name: str):
        """Toggle edit mode for a specific card"""
        self.card_edit_states[func_name] = not self.card_edit_states.get(func_name, False)
        self._create_function_cards(self.search_var.get())

    def _on_search_change(self, *args):
        """Handle search input changes"""
        self._create_function_cards(self.search_var.get())

    def _create_tab_with_close_button(self, tab_name):
        """Erstellt einen neuen Tab mit Schließen-Button"""
        self.tab_view.add(tab_name)
        tab = self.tab_view.tab(tab_name)

        # Container für Tab-Inhalt und Button
        header_frame = ctk.CTkFrame(tab)
        header_frame.pack(fill="x", pady=(0, 5))

        # Schließen-Button
        close_button = ctk.CTkButton(
            header_frame,
            text="×",
            width=20,
            height=20,
            command=lambda: self._close_tab(tab_name),
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30")
        )
        close_button.pack(side="right", padx=5, pady=5)

        # Output-Textbox unterhalb des Headers
        output_text = ctk.CTkTextbox(tab, height=200)
        output_text.pack(fill="both", expand=True)

        return output_text

    def _close_tab(self, tab_name):
        """Schließt den angegebenen Tab"""
        if self.tab_view.get() == tab_name:  # Wenn der aktuelle Tab geschlossen wird
            # Wechsle zu einem anderen Tab, falls vorhanden
            tabs = self.tab_view._tab_dict.keys()  # Alle verfügbaren Tabs
            current_index = list(tabs).index(tab_name)
            if len(tabs) > 1:
                # Wähle den nächsten oder vorherigen Tab
                next_tab = list(tabs)[current_index - 1 if current_index > 0 else 1]
                self.tab_view.set(next_tab)

        self.tab_view.delete(tab_name)

        if len(self.tab_view._tab_dict.keys()) == 0:
            # Entferne Output-Frame und Tab-View komplett
            if hasattr(self, 'output_frame'):
                self.output_frame.destroy()
                del self.output_frame

            # Entferne die zweite Zeile (Tab-Bereich) komplett
            self.window.grid_rowconfigure(2, weight=0)

            # Stelle ursprüngliche Grid-Konfiguration wieder her
            self.window.grid_columnconfigure(0, weight=1)
            self.window.grid_rowconfigure(1, weight=1)

            # Rekonstruiere Hauptansicht
            self._create_top_section()
            self._create_main_content()
            self._create_function_cards()
            self._update_quick_overview()
            self._initialize_flowss()

    def _run_search_command(self):
        """Execute the command from search bar and display output in tabbed GUI using PowerShell with colors"""
        command = self.search_var.get()
        if not command:
            return

        # Initialisiere das Tab-System falls noch nicht vorhanden
        if not hasattr(self, 'output_frame'):
            self.output_frame = ctk.CTkFrame(self.window)
            self.output_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
            self.window.grid_rowconfigure(2, weight=1)

            self.tab_view = ctk.CTkTabview(self.output_frame)
            self.tab_view.pack(fill="both", expand=True)
            self.tab_count = 0

        # Erstelle neuen Tab für diesen Befehl
        tab_name = f"T{command[:5]} {self.tab_count + 1}"

        # Erstelle Textbox im neuen Tab
        output_text = self._create_tab_with_close_button(tab_name)
        self.tab_count += 1

        # Konfiguriere Text-Tags für Farben
        for color_name, color_code in Style.style_dic.items():
            output_text.tag_config(color_name, foreground=self._ansi_to_rgb(color_code))

        # Zeige initialen Befehl
        output_text.insert("end", f"Ausführung von Befehl: {command}\n\n", "CYAN")
        self.tab_view.set(tab_name)  # Aktiviere den neuen Tab

        def stream_output(process):
            while True:
                try:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        formatted_output = self._parse_ansi_colors(line)
                        self.window.after(0, lambda out=formatted_output: self._update_output(output_text, out))
                except UnicodeDecodeError:
                    # Bei Encoding-Fehlern zeige eine Warnung
                    self.window.after(0,
                                      lambda: output_text.insert("end", "[Encoding-Fehler bei der Ausgabe]\n", "RED"))

        # Shell-Konfiguration mit PowerShell-Präferenz
        if platform.system() == "Windows":
            powershell_path = os.getenv("GUI_PW", r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
            if os.path.exists(powershell_path):
                shell_cmd = powershell_path
                # PowerShell mit UTF-8 Encoding und ANSI-Farben
                cmd = [
                    shell_cmd,
                    "-NoProfile",
                    "-NoLogo",
                    "-InputFormat", "Text",
                    "-OutputFormat", "Text",
                    "-Command",
                    command
                ]
                encoding = 'utf-8'
            else:
                shell_cmd = os.environ.get("COMSPEC", "cmd.exe")
                cmd = [shell_cmd, "/c", command]
                encoding = 'cp850'  # Alternatives Encoding für CMD
        else:
            shell_cmd = os.environ.get("SHELL", "/bin/sh")
            cmd = [shell_cmd, "-c", command]
            encoding = 'utf-8'

        try:
            # Erstelle den Prozess mit spezifischem Encoding
            startupinfo = None
            if platform.system() == "Windows":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding=encoding,
                errors='replace',  # Ersetze nicht-decodierbare Zeichen
                env=os.environ,
                startupinfo=startupinfo,
                universal_newlines=True
            )

            threading.Thread(
                target=stream_output,
                args=(process,),
                daemon=True
            ).start()

        except Exception as e:
            output_text.insert("end", Style.RED(f"Fehler: {str(e)}\n"))

    def _ansi_to_rgb(self, ansi_code):
        """Konvertiert ANSI-Farbcodes in RGB-Werte"""
        color_map = {
            '\33[30m': '#000000',  # BLACK
            '\33[31m': '#FF0000',  # RED
            '\33[32m': '#00FF00',  # GREEN
            '\33[33m': '#FFFF00',  # YELLOW
            '\33[34m': '#0000FF',  # BLUE
            '\33[35m': '#FF00FF',  # MAGENTA
            '\33[36m': '#00FFFF',  # CYAN
            '\33[37m': '#FFFFFF',  # WHITE
        }
        return color_map.get(ansi_code, '#FFFFFF')

    def _parse_ansi_colors(self, text):
        """Parst ANSI-Farbcodes und erstellt eine Liste von (Text, Farbe) Tupeln"""
        result = []
        current_text = ""
        current_style = None

        i = 0
        while i < len(text):
            if text[i:i + 2] == '\33[':
                if current_text and current_style:
                    result.append((current_text, current_style))
                current_text = ""
                end = text.find('m', i)
                if end != -1:
                    color_code = text[i:end + 1]
                    for style_name, style_code in Style.style_dic.items():
                        if style_code == color_code:
                            current_style = style_name
                            break
                    i = end + 1
                    continue
            current_text += text[i]
            i += 1

        if current_text and current_style:
            result.append((current_text, current_style))
        elif current_text:  # Falls kein Farbstil gefunden wurde
            result.append((current_text, None))
        return result

    def _update_output(self, output_text, formatted_output):
        """Aktualisiert das Output-Fenster mit formatierten Text"""
        for text, style in formatted_output:
            output_text.insert("end", text, style if style else None)
        output_text.see("end")

    def _initialize_flowss(self):
        """Initialize flows dictionary without waiting for login"""
        self.flows_dict = flows_dict_func(remote=False)
        self.flows_dict.update(flows_dict_func(s='', remote=True))
        self._create_function_cards()
        self._update_quick_overview()

    async def _perform_login(self):
        #try:
            from toolboxv2 import get_app
            self.tb_app = get_app()
            show_console(False)
            login_success = await self.tb_app.session.login()

            if not login_success:
                await self._show_magic_link_dialog()
            else:
                self._complete_login()
        #except Exception as e:
        #    self._show_error(f"Login failed: {str(e)}")
        #    self.username_label.configure(text="Not logged in")

    def _initialize_login(self):
        asyncio.run(self._perform_login())

    async def _show_magic_link_dialog(self):
        dialog = ctk.CTkInputDialog(
            text="Please enter the magic link:",
            title="Magic Link Login"
        )
        magic_link = dialog.get_input()
        if magic_link:
            await self._magic_link_login(magic_link)

    async def _magic_link_login(self, magic_link: str):
        #try:
            success = await self.tb_app.session.init_log_in_mk_link(magic_link)
            if success:
                self._complete_login()
            else:
                self._show_error("Invalid magic link")
        #except Exception as e:
        #    self._show_error(f"Magic link login failed: {str(e)}")

    def _complete_login(self):
        self.is_logged_in = True
        self.username = self.tb_app.get_username()
        self.username_label.configure(text=f"Logged in as: {self.username}")

    def _show_error(self, message: str):
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
        self.window.mainloop()


def start():
    app = DynamicFunctionApp()
    app.run()


if __name__ == "__main__":
    start()
