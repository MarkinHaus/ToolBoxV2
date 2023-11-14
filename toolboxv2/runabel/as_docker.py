import io
import os
import sys
import tempfile

from toolboxv2 import App, AppArgs, Spinner
from toolboxv2.mods.dockerEnv import Tools

NAME = 'docker'


def build_docker_file(docker_env, init_args):
    file_data = """# Use an official Python runtime as a parent image
FROM python:3.9-slim

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

#RUN npm install ./ToolBoxV2/toolboxv2/app/node_modules/.package-lock.json
# Make port 5000 available to the world outside this container
EXPOSE 5001:5001
"""

    string_io = io.StringIO()

    # Write the string 'x' into the io.StringIO object
    string_io.write(file_data)


def run(app: App, args: AppArgs):
    file_data = """# Use an official Python runtime as a parent image
    FROM python:3.9-slim

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
    RUN pip install ./ToolBoxV2/

    #RUN npm install ./ToolBoxV2/toolboxv2/app/node_modules/.package-lock.json
    # Make port 5000, 62435 available to the world outside this container
    EXPOSE 5000:5000
    EXPOSE 62435:62435
    """


    try:
        init_args = " ".join(sys.orig_argv)
    except AttributeError:
        init_args = "python3 "
        init_args += " ".join(sys.argv)
    init_args_s = "toolboxv2 "+str(" ".join(init_args.split(' ')[2:])).replace('--docker', '')
    print(init_args_s)
    file_data += f"\nCMD {init_args_s}"

    # Write the string 'x' into the io.StringIO object

    docker_env: Tools = app.get_mod('dockerEnv')

    import tarfile
    import time
    from contextlib import closing
    from io import BytesIO

    filename = 'DockerFile'

    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
        temp_file.write(file_data)
        temp_path = temp_file.name

    # Now you can use temp_path as the path variable

    # For example, let's print the contents of the file
    with Spinner(message=f"Building Docker file {(app.id + '-dockerImage').lower()}", symbols="c"):
        os.system(f"docker build -f {temp_path} -t {(app.id + '-dockerImage').lower()} .")
        # img = docker_env.build_custom_images("./", temp_path, (app.id + '-dockerImage').lower(), False)

#
#    ## Delete the temporary file
    os.remove(temp_path)
    #print(f"Temporary file deleted. {temp_path}")


    container_id = docker_env.create_container((app.id + '-dockerImage').lower(), (app.id + '-dockerContainer').lower(), entrypoint=init_args.split(' '))
