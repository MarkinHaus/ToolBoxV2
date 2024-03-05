import sys
import uuid

from toolboxv2 import App, AppArgs

NAME = 'docker'


def get_multiline_input(x):
    print(x)
    print("exit with empty line")
    lines = []

    while True:
        line = input()
        if line:
            lines.append(line)
        else:
            break
    return "\n".join(lines)


def build_docker_image():
    print("Willkommen zum Docker-Image-Erstellungsassistenten!")

    # Schritt 1: Base Image angeben
    base_image = input("Schritt 1: Geben Sie das Base Image an (z.B. ubuntu:latest): ")

    # Schritt 2: Arbeitsverzeichnis (Workdir) festlegen
    workdir = input("Schritt 2: Geben Sie das Arbeitsverzeichnis (Workdir) an: ")

    # Schritt 3: Quellcode-Quelle (Git oder Lokal) wählen
    source_type = input("Schritt 3: Wählen Sie die Quellcode-Quelle (git/local): ").lower()
    git_repository = ""
    local_path = ""
    if 'g' in source_type:
        git_repository = input("Geben Sie die Git-Repository-URL an: ")
    else:
        local_path = input("Geben Sie den Pfad zum lokalen Quellcode an: ")

    # Schritt 4: Exponierte Ports angeben
    exposed_ports = input("Schritt 4: Geben Sie die zu exposenden Ports an (kommagetrennt, z.B. 80,443): ")

    # Schritt 5: Netzwerkeinstellungen
    # network_mode = input("Schritt 5: Geben Sie den Netzwerkmodus an (default/bridge/host): ")

    # Schritt 6: Benutzerdefinierte Befehle für das Image
    custom_commands = get_multiline_input(
        "Schritt 6: Geben Sie benutzerdefinierte Befehle für das Image ein (optional): ")

    # Dockerfile erstellen
    dockerfile_content = f"""
    FROM {base_image}
    # Update aptitude with new repo
    RUN apt-get update
    # Install software
    RUN apt-get install -y git
    WORKDIR {workdir}
    {"RUN git clone " + git_repository + " ." if source_type == "git" else "COPY " + local_path + " ."}
    EXPOSE {exposed_ports}
    USER root
    {custom_commands}
    """

    return dockerfile_content


def run(app: App, args: AppArgs):
    file_data = """# Use an official Python runtime as a parent image
    FROM python:3.9

    MAINTAINER Markin Hausmanns MarkinHausmanns@gmail.com

    # Update aptitude with new repo
    RUN apt-get update

    # Install software
    RUN apt-get install -y git

    # Set the working directory in the container to /app
    WORKDIR /app

    # Clone the git repo into the docker container
    RUN git clone https://github.com/MarkinHaus/ToolBoxV2.git

    #WORKDIR /ToolBoxV2/


    # Install any needed packages specified in requirements.txt
    RUN pip install -e ./ToolBoxV2/

    #RUN npm install ./ToolBoxV2/toolboxv2/web/node_modules/.package-lock.json
    # Make port 5000, 62435 available to the world outside this container
    EXPOSE 5000:5000
    EXPOSE 62435:62435
    """

    try:
        init_args = " ".join(sys.orig_argv)
    except AttributeError:
        init_args = "python3 "
        init_args += " ".join(sys.argv)
    init_args_s = "toolboxv2 " + str(" ".join(init_args.split(' ')[2:])).replace('--docker', '')
    init_args_s = init_args_s.split(' ')
    temp_comm = []
    for i in init_args_s:
        if i:
            temp_comm.append(i)

    init_args_s = temp_comm
    print(init_args_s)
    custom = False
    if not app.id.startswith('main'):
        if 'y' in input("Do you want to build a custom image y/n").lower():
            custom = True
            file_data = build_docker_image()

    if not custom:
        file_data += f"\nCMD "
        for i in init_args_s:
            file_data += f'{i} '

    # Write the string 'x' into the io.StringIO object

    docker_env = app.get_mod('dockerEnv')

    img_name = (app.id + '-dockerImage-' + str(uuid.uuid4())).lower()

    img = docker_env.build_custom_images(img_name, file_data)

    if not custom:
        container_id = docker_env.create_container(img_name, (app.id + '-dockerContainer' + str(uuid.uuid4())).lower(),
                                                   entrypoint=init_args_s, ports={'5000/tcp': 5000, '5000/udp': 5001})
    else:
        container_id = docker_env.create_container(img_name, (app.id + '-dockerContainer' + str(uuid.uuid4())).lower())
