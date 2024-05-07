import os
import sys
import subprocess
from pathlib import Path


def create_virtual_env(path):
    """ Erstellt eine virtuelle Umgebung im angegebenen Pfad. """
    subprocess.check_call([sys.executable, '-m', 'venv', path])


def install_package(env_path, package):
    """ Installiert ein Paket in der virtuellen Umgebung. """
    # subprocess.check_call([os.path.join(env_path, 'Scripts', 'python'), '-m', 'pip', 'install', '--upgrade', package])
    subprocess.check_call(['pip', 'install', '--upgrade', package])


def run_command(env_path, command):
    """Führt einen Befehl in der virtuellen Umgebung aus und zeigt stdout/stderr an."""
    if not env_path:
        env_path = Path.home() / ".local/share/virtualenvs/toolboxv2_env"
    env_path = Path(env_path)
    process = subprocess.Popen(
        # [os.path.join(env_path, 'Scripts', 'python'), '-m'] +
        command.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    out = ''
    # Drucke die Ausgabe live
    while True:
        try:
            output = process.stdout.readline()
        except UnicodeDecodeError:
            continue
        if output == '' and process.poll() is not None:
            break
        if output:
            out += '\n' + output.strip()

    exit_code = process.wait()
    if exit_code != 0:
        s = subprocess.CalledProcessError(returncode=exit_code, cmd=command)
        print("Process ERROR:", s)
    return out, exit_code


def main(env_path, retries=5):
    if not env_path:
        env_path = Path.home() / ".local/share/virtualenvs/toolboxv2_env"
    env_path = Path(env_path)

    print(env_path)
    # Erstelle die virtuelle Umgebung, falls nicht vorhanden
    if not env_path.exists():
        print("Erstelle virtuelle Umgebung...")
        create_virtual_env(str(env_path))

    print("Installiere Toolboxv2...")
    install_package(str(env_path), 'ToolBoxV2')

    command = 'toolboxv2 -fg -v'
    for attempt in range(retries):
        print(f"Versuch {attempt + 1} von {retries}...")
        error, exit_code = run_command(str(env_path), command)

        if exit_code == 0:
            print(error)
            print("Erfolgreich ausgeführt!")
            break
        else:
            error = str(error)
            info = error.split(':')[-1].strip()
            if info.startswith('No module named'):
                module = info.replace("No module named '", "").replace("'", "")
                print(module)
                print("Versuche ", module, "Zu installer")
                install_package(str(env_path), module)

            print("Fehler beim Ausführen des Befehls:", error)


if __name__ == "__main__":
    #_env_path = sys.argv[1] if len(sys.argv) > 1 else None
    #main(_env_path)

    command = 'toolboxv2 --test'
    while not input("Enter "):
        error, exit_code = run_command(sys.executable.replace('\python.exe', ''), command)

        if exit_code == 0:
            print(error)
            print("Erfolgreich ausgeführt!")
            break
        else:
            error = str(error)
            info = error.split(':')[-1].strip()
            if info.startswith('No module named'):
                module = info.replace("No module named '", "").replace("'", "").split('\n')[0]
                print(module)
                print("Versuche ", module, "Zu installer")
                install_package(sys.executable, module)

            print("Fehler beim Ausführen des Befehls:", error)

"""rich>=13.5.2
coverage>=7.2.7
setuptools>=68.0.0
fastapi>=0.99.1
pyjwt>=2.8.0
redis>=5.0.0
requests>=2.31.0
beautifulsoup4>=4.12.2
tqdm>=4.65.0
python-dotenv>=1.0.0
readchar>=4.0.5
mock>=5.1.0
docker>=6.1.0
dill~=0.3.7
cryptography>=41.0.3
cachetools>=5.3.2
psutil>=5.9.7
faker>=22.2.0
prompt_toolkit>=3.0.43
mailjet-rest>=1.3.4
webauthn>=2.0.0
qrcode>=7.4.2
uvicorn[standard]~=0.25.0
python-multipart
playwright~=1.33.0
pydantic~=1.10.8
pyyaml~=6.0
itsdangerous~=2.1.2
reedsolo~=1.7.0
litellm~=1.23.16
aiohttp~=3.8.4
schedule~=1.2.0
Pillow~=9.5.0
SpeechRecognition
langchain-community
pebble~=5.0.3
pydub~=0.25.1
duckduckgo_search
gpt4all~=1.0.12
chromadb~=0.4.13
ddg
scikit-learn~=1.2.2
transformers[torch]~=4.36.2

toolboxv2~=0.1.12
starlette~=0.35.1
langchain~=0.0.295
torch~=2.3.0
inquirer~=3.1.3
networkx~=2.8.8
numpy~=1.24.3
tiktoken~=0.5.1
openai~=1.12.0
pyaudio~=0.2.13
whisper~=1.1.10
gtts~=2.3.2
playsound~=1.3.0
customtkinter~=5.2.2
streamlit~=1.31.1
opencv-python~=4.9.0.80
diffusers~=0.26.3
moviepy~=1.0.3
websockets~=11.0.2
paramiko~=3.4.0
bcrypt~=4.0.1
future~=0.18.3
darkdetect~=0.8.0
packaging~=23.2
yarl~=1.9.2
pygments~=2.15.1

pyaml~=24.4.0
"""
