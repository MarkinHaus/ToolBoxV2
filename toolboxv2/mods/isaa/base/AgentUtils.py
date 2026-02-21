import json
import locale
import os
import pickle
import platform
import re
import subprocess
import sys
import threading

import requests
import tiktoken

from toolboxv2 import Singleton, get_logger, remove_styles
from toolboxv2.mods.isaa.base.KnowledgeBase import KnowledgeBase


def get_ip():
    response = requests.get('https://api64.ipify.org?format=json').json()
    return response["ip"]


def get_location():
    ip_address = get_ip()
    response = requests.get(f'https://ipapi.co/{ip_address}/json/').json()
    location_data = f"city: {response.get('city')},region: {response.get('region')},country: {response.get('country_name')},"

    return location_data, ip_address


import shutil


def detect_shell() -> tuple[str, str]:
    """
    Detects the best available shell and the argument to execute a command.
    Returns:
        A tuple of (shell_executable, command_argument).
        e.g., ('/bin/bash', '-c') or ('powershell.exe', '-Command')
    """
    if platform.system() == "Windows":
        if shell_path := shutil.which("pwsh"):
            return shell_path, "-Command"
        if shell_path := shutil.which("powershell"):
            return shell_path, "-Command"
        return "cmd.exe", "/c"

    shell_env = os.environ.get("SHELL")
    if shell_env and shutil.which(shell_env):
        return shell_env, "-c"

    for shell in ["bash", "zsh", "sh"]:
        if shell_path := shutil.which(shell):
            return shell_path, "-c"

    return "/bin/sh", "-c"


def safe_decode(data: bytes) -> str:
    encodings = [sys.stdout.encoding, locale.getpreferredencoding(), 'utf-8', 'latin-1', 'iso-8859-1']
    for enc in encodings:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode('utf-8', errors='replace')


import io
import xml.etree.ElementTree as ET
import zipfile

# Import der PyPDF2-Bibliothek
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None


def extract_text_natively(data: bytes, filename: str = "") -> str:
    """
    Extrahiert Text aus verschiedenen Dateitypen mit nativen Python-Methoden
    oder reinen Python-Bibliotheken (speziell PyPDF2 für PDFs).

    Args:
        data (bytes): Der Inhalt der Datei als Bytes.
        filename (str, optional): Der Originaldateiname, um den Typ zu bestimmen.

    Returns:
        str: Der extrahierte Text.

    Raises:
        ValueError: Wenn der Dateityp nicht unterstützt wird oder die Verarbeitung fehlschlägt.
        ImportError: Wenn PyPDF2 für die Verarbeitung von PDF-Dateien benötigt, aber nicht installiert ist.
    """
    file_ext = filename.lower().split('.')[-1] if '.' in filename else ''

    # 1. DOCX-Verarbeitung (nativ mit zipfile und xml)
    if data.startswith(b'PK\x03\x04'):
        try:
            docx_file = io.BytesIO(data)
            text_parts = []
            with zipfile.ZipFile(docx_file) as zf:
                namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
                body_path = "word/document.xml"
                if body_path in zf.namelist():
                    xml_content = zf.read(body_path)
                    tree = ET.fromstring(xml_content)
                    for para in tree.iter(f"{namespace}p"):
                        texts_in_para = [node.text for node in para.iter(f"{namespace}t") if node.text]
                        if texts_in_para:
                            text_parts.append("".join(texts_in_para))
                return "\n".join(text_parts)
        except (zipfile.BadZipFile, ET.ParseError):
            pass  # Fährt fort, falls es eine ZIP-Datei, aber kein gültiges DOCX ist

    # 2. PDF-Verarbeitung (mit PyPDF2)
    if data.startswith(b'%PDF-'):
        if PyPDF2 is None:
            raise ImportError(
                "Die Bibliothek 'PyPDF2' wird benötigt, um PDF-Dateien zu verarbeiten. Bitte installieren Sie sie mit 'pip install PyPDF2'.")

        try:
            # Erstelle ein In-Memory-Dateiobjekt für PyPDF2
            pdf_file = io.BytesIO(data)
            # Verwende PdfFileReader aus PyPDF2
            pdf_reader = PyPDF2.PdfFileReader(pdf_file)

            text_parts = []
            # Iteriere durch die Seiten
            for page_num in range(pdf_reader.numPages):
                page = pdf_reader.getPage(page_num)
                # Extrahiere Text mit extractText()
                page_text = page.extractText()
                if page_text:
                    text_parts.append(page_text)

            return "\n".join(text_parts)
        except Exception as e:
            raise ValueError(f"PDF-Verarbeitung mit PyPDF2 fehlgeschlagen: {e}")

    # 3. Fallback auf reinen Text (TXT)

    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        try:
            return data.decode('latin-1')
        except Exception as e:
            raise ValueError(f"Text-Dekodierung fehlgeschlagen: {e}")


def get_json_from_json_str(json_str: str or list or dict, repeat: int = 1) -> dict or None:
    """Versucht, einen JSON-String in ein Python-Objekt umzuwandeln.

    Wenn beim Parsen ein Fehler auftritt, versucht die Funktion, das Problem zu beheben,
    indem sie das Zeichen an der Position des Fehlers durch ein Escape-Zeichen ersetzt.
    Dieser Vorgang wird bis zu `repeat`-mal wiederholt.

    Args:
        json_str: Der JSON-String, der geparst werden soll.
        repeat: Die Anzahl der Versuche, das Parsen durchzuführen.

    Returns:
        Das resultierende Python-Objekt.
    """
    for _ in range(repeat):
        try:
            return parse_json_with_auto_detection(json_str)
        except json.JSONDecodeError as e:
            unexp = int(re.findall(r'\(char (\d+)\)', str(e))[0])
            unesc = json_str.rfind(r'"', 0, unexp)
            json_str = json_str[:unesc] + r'\"' + json_str[unesc + 1:]
            closg = json_str.find(r'"', unesc + 2)
            json_str = json_str[:closg] + r'\"' + json_str[closg + 1:]
        new = fix_json_object(json_str)
        if new is not None:
            json_str = new
    get_logger().info(f"Unable to parse JSON string after {json_str}")
    return None


def parse_json_with_auto_detection(json_data):
    """
    Parses JSON data, automatically detecting if a value is a JSON string and parsing it accordingly.
    If a value cannot be parsed as JSON, it is returned as is.
    """

    def try_parse_json(value):
        """
        Tries to parse a value as JSON. If the parsing fails, the original value is returned.
        """
        try:
            # print("parse_json_with_auto_detection:", type(value), value)
            parsed_value = json.loads(value)
            # print("parsed_value:", type(parsed_value), parsed_value)
            # If the parsed value is a string, it might be a JSON string, so we try to parse it again
            if isinstance(parsed_value, str):
                return eval(parsed_value)
            else:
                return parsed_value
        except Exception:
            # logging.warning(f"Failed to parse value as JSON: {value}. Exception: {e}")
            return value

    get_logger()

    if isinstance(json_data, dict):
        return {key: parse_json_with_auto_detection(value) for key, value in json_data.items()}
    elif isinstance(json_data, list):
        return [parse_json_with_auto_detection(item) for item in json_data]
    else:
        return try_parse_json(json_data)


def extract_json_objects(text: str, matches_only=False):
    pattern = r'\{.*?\}'
    matches = re.findall(pattern,
                         text
                         .replace("'{", '{')
                         .replace("}'", '}')
                         .replace('"', "'")
                         .replace("':'", '":"')
                         .replace("': '", '": "')
                         .replace("','", '","')
                         .replace("', '", '", "')
                         .replace("{'", '{"')
                         .replace("'}", '"}')
                         .replace("':{", '":{')
                         .replace("' :{", '" :{')
                         .replace("': {", '": {')
                         ,
                         flags=re.DOTALL)
    json_objects = []
    if matches_only:
        return matches

    for match in matches:
        try:
            x = json.loads(match)
            print("Found", x)
            json_objects.append(x)
        except json.JSONDecodeError:
            # Wenn die JSON-Dekodierung fehlschlägt, versuchen Sie, das JSON-Objekt zu reparieren
            fixed_match = fix_json_object(match)
            if fixed_match:
                try:
                    y = json.loads(fixed_match)
                    json_objects.append(y)
                except json.JSONDecodeError as e:
                    print(e)
                    try:
                        y = json.loads(fixed_match.replace("\n", "#New-Line#"))
                        for k in y:
                            if isinstance(y[k], str):
                                y[k] = y[k].replace("#New-Line#", "\n")
                            if isinstance(y[k], dict):
                                for k1 in y[k]:
                                    if isinstance(y[k][k1], str):
                                        y[k][k1] = y[k][k1].replace("#New-Line#", "\n")
                        json_objects.append(y)
                    except json.JSONDecodeError as e:
                        print(e)
                        pass
    return json_objects


def fix_json_object(match: str):
    # Überprüfen Sie, wie viele mehr "}" als "{" vorhanden sind
    extra_opening_braces = match.count("}") - match.count("{")
    if extra_opening_braces > 0:
        # Fügen Sie die fehlenden öffnenden Klammern am Anfang des Matches hinzu
        opening_braces_to_add = "{" * extra_opening_braces
        fixed_match = opening_braces_to_add + match
        return fixed_match
    extra_closing_braces = match.count("{") - match.count("}")
    if extra_closing_braces > 0:
        # Fügen Sie die fehlenden öffnenden Klammern am Anfang des Matches hinzu
        closing_braces_to_add = "}" * extra_closing_braces
        fixed_match = match + closing_braces_to_add
        return fixed_match
    return None


def find_json_objects_in_str(data: str):
    """
    Sucht nach JSON-Objekten innerhalb eines Strings.
    Gibt eine Liste von JSON-Objekten zurück, die im String gefunden wurden.
    """
    json_objects = extract_json_objects(data)
    if not isinstance(json_objects, list):
        json_objects = [json_objects]
    return [get_json_from_json_str(ob, 10) for ob in json_objects if get_json_from_json_str(ob, 10) is not None]


def complete_json_object(data: str, mini_task):
    """
    Ruft eine Funktion auf, um einen String in das richtige Format zu bringen.
    Gibt das resultierende JSON-Objekt zurück, wenn die Funktion erfolgreich ist, sonst None.
    """
    ret = mini_task(
        f"Vervollständige das Json Object. Und bringe den string in das Richtige format. data={data}\nJson=")
    if ret:
        return anything_from_str_to_dict(ret)
    return None


def fix_json(json_str, current_index=0, max_index=10):
    if current_index > max_index:
        return json_str
    try:
        return json.loads(json_str)  # Wenn der JSON-String bereits gültig ist, gib ihn unverändert zurück
    except json.JSONDecodeError as e:
        error_message = str(e)
        # print("Error message:", error_message)

        # Handle specific error cases
        if "Expecting property name enclosed in double quotes" in error_message:
            # Korrigiere einfache Anführungszeichen in doppelte Anführungszeichen
            json_str = json_str.replace("'", '"')

        elif "Expecting ':' delimiter" in error_message:
            # Setze fehlende Werte auf null
            json_str = json_str.replace(':,', ':null,')

        elif "Expecting '" in error_message and "' delimiter:" in error_message:
            # Setze fehlende Werte auf null
            line_i = int(error_message[error_message.rfind('line') + 4:error_message.rfind('column')].strip())
            colom_i = int(error_message[error_message.rfind('char') + 4:-1].strip())
            sp = error_message.split("'")[1]

            json_lines = json_str.split('\n')
            corrected_json_lines = json_lines[:line_i - 1]  # Bis zur Zeile des Fehlers
            faulty_line = json_lines[line_i - 1]  # Die Zeile, in der der Fehler aufgetreten ist
            corrected_line = faulty_line[:colom_i] + sp + faulty_line[colom_i:]
            corrected_json_lines.append(corrected_line)
            remaining_lines = json_lines[line_i:]  # Nach der Zeile des Fehlers
            corrected_json_lines.extend(remaining_lines)

            json_str = '\n'.join(corrected_json_lines)

        elif "Extra data" in error_message:
            # Entferne Daten vor dem JSON-String
            start_index = json_str.find('{')
            if start_index != -1:
                json_str = json_str[start_index:]

        elif "Unterminated string starting at" in error_message:
            # Entferne Daten nach dem JSON-String
            line_i = int(error_message[error_message.rfind('line') + 4:error_message.rfind('column')].strip())
            colom_i = int(error_message[error_message.rfind('char') + 4:-1].strip())
            # print(line_i, colom_i)
            index = 1
            new_json_str = ""
            for line in json_str.split('\n'):
                if index == line_i:
                    line = line[:colom_i - 1] + line[colom_i + 1:]
                new_json_str += line
                index += 1
            json_str = new_json_str
        # Versuche erneut, den reparierten JSON-String zu laden
        # {"name": "John", "age": 30, "city": "New York", }

        start_index = json_str.find('{')
        if start_index != -1:
            json_str = json_str[start_index:]

        # Füge fehlende schließende Klammern ein
        count_open = json_str.count('{')
        count_close = json_str.count('}')
        for _i in range(count_open - count_close):
            json_str += '}'

        count_open = json_str.count('[')
        count_close = json_str.count(']')
        for _i in range(count_open - count_close):
            json_str += ']'

        return fix_json(json_str, current_index + 1)


def anything_from_str_to_dict(data: str, expected_keys: dict = None, mini_task=lambda x: ''):
    """
    Versucht, einen String in ein oder mehrere Dictionaries umzuwandeln.
    Berücksichtigt dabei die erwarteten Schlüssel und ihre Standardwerte.
    """
    if len(data) < 4:
        return []

    if expected_keys is None:
        expected_keys = {}

    result = []
    json_objects = find_json_objects_in_str(data)
    if not json_objects and data.startswith('[') and data.endswith(']'):
        json_objects = eval(data)
    if json_objects and len(json_objects) > 0 and isinstance(json_objects[0], dict):
        result.extend([{**expected_keys, **ob} for ob in json_objects])
    if not result:
        completed_object = complete_json_object(data, mini_task)
        if completed_object is not None:
            result.append(completed_object)
    if len(result) == 0 and expected_keys:
        result = [{list(expected_keys.keys())[0]: data}]
    for res in result:
        if isinstance(res, list) and len(res) > 0:
            res = res[0]
        for key, value in expected_keys.items():
            if key not in res:
                res[key] = value

    if len(result) == 0:
        fixed = fix_json(data)
        if fixed:
            result.append(fixed)

    return result


import os
from dataclasses import asdict, dataclass, field


@dataclass
class LLMMode:
    name: str
    description: str
    system_msg: str
    post_msg: str | None = None
    examples: list[str] | None = None

    def __str__(self):
        return f"LLMMode: {self.name} (description) {self.description}"

@dataclass
class ModeController(LLMMode):
    shots: list = field(default_factory=list)

    def add_shot(self, user_input, agent_output):
        self.shots.append([user_input, agent_output])

    def add_user_feedback(self):

        add_list = []

        for index, shot in enumerate(self.shots):
            print(f"Input : {shot[0]} -> llm output : {shot[1]}")
            user_evalution = input("Rank from 0 to 10: -1 to exit\n:")
            if user_evalution == '-1':
                break
            else:
                add_list.append([index, user_evalution])

        for index, evaluation in add_list:
            self.shots[index].append(evaluation)

    def auto_grade(self):
        pass

    @classmethod
    def from_llm_mode(cls, llm_mode: LLMMode, shots: list | None = None):
        if shots is None:
            shots = []

        return cls(
            name=llm_mode.name,
            description=llm_mode.description,
            system_msg=llm_mode.system_msg,
            post_msg=llm_mode.post_msg,
            examples=llm_mode.examples,
            shots=shots
        )


@dataclass
class ControllerManager:
    controllers: dict[str, ModeController] = field(default_factory=dict)

    def rget(self, llm_mode: LLMMode, name: str = None):
        if name is None:
            name = llm_mode.name
        if not self.registered(name):
            self.add(name, llm_mode)
        return self.get(name)

    def registered(self, name):
        return name in self.controllers

    def get(self, name):
        if name is None:
            return None
        if name in self.controllers:
            return self.controllers[name]
        return None

    def add(self, name, llm_mode, shots=None):
        if name in self.controllers:
            return "Name already defined"

        if shots is None:
            shots = []

        self.controllers[name] = ModeController.from_llm_mode(llm_mode=llm_mode, shots=shots)

    def list_names(self):
        return list(self.controllers.keys())

    def list_description(self):
        return [d.description for d in self.controllers.values()]

    def __str__(self):
        return "LLMModes \n" + "\n\t".join([str(m).replace('LLMMode: ', '') for m in self.controllers.values()])

    def save(self, filename: str | None, get_data=False):

        data = asdict(self)

        if filename is not None:
            with open(filename, 'w') as f:
                json.dump(data, f)

        if get_data:
            return json.dumps(data)

    @classmethod
    def init(cls, filename: str | None, json_data: str | None = None):

        controllers = {}

        if filename is None and json_data is None:
            print("No data provided for ControllerManager")
            return cls(controllers=controllers)

        if filename is not None and json_data is not None:
            raise ValueError("filename and json_data are provided only one accepted filename or json_data")

        if filename is not None:
            if os.path.exists(filename) and os.path.isfile(filename):
                with open(filename) as f:
                    controllers = json.load(f)
            else:
                print("file not found")

        if json_data is not None:
            controllers = json.loads(json_data)

        return cls(controllers=controllers)
