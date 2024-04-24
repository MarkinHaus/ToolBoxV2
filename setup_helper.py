import os
import sys
import subprocess
from pathlib import Path


def create_virtual_env(path):
    """ Erstellt eine virtuelle Umgebung im angegebenen Pfad. """
    subprocess.check_call([sys.executable, '-m', 'venv', path])


def install_package(env_path, package):
    """ Installiert ein Paket in der virtuellen Umgebung. """
    subprocess.check_call([os.path.join(env_path, 'Scripts', 'python'), '-m', 'pip', 'install', '--upgrade', package])


def run_command(env_path, command):
    """Führt einen Befehl in der virtuellen Umgebung aus und zeigt stdout/stderr an."""
    if not env_path:
        env_path = Path.home() / ".local/share/virtualenvs/toolboxv2_env"
    env_path = Path(env_path)
    process = subprocess.Popen(
        [os.path.join(env_path, 'Scripts', 'python'), '-m'] + command.split(),
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
    _env_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(_env_path)
