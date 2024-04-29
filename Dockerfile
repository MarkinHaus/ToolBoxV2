# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/engine/reference/builder/

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim as base
# FROM python
# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update
RUN apt-get install libasound-dev libportaudio2 libportaudiocpp0 portaudio19-dev -y
RUN apt-get install gcc npm git -y
RUN npm install bun
# Create a non-privileged user that the app will run under.

# Copy the source code into the container.
# cage for live
# RUN pip install --target ./ ToolBoxV2
# COPY ./toolboxv2/web ./ToolBoxV2/toolboxv2/web
# COPY ./toolboxv2/web/package_docker.json ./ToolBoxV2/toolboxv2/web/package.json
# RUN npx bun install -y ./toolboxv2/web/

COPY ./setup.py ./setup.py
COPY ./requirements.txt ./requirements.txt
COPY ./README.md ./README.md
COPY ./setup.cfg ./setup.cfg
COPY ./MANIFEST.in ./MANIFEST.in
COPY ./toolboxv2/mods ./toolboxv2/mods
COPY ./toolboxv2/runabel ./toolboxv2/runabel
COPY ./toolboxv2/tests ./toolboxv2/tests
COPY ./toolboxv2/web_build ./toolboxv2/web_build
COPY ./toolboxv2/web ./toolboxv2/web
COPY ./toolboxv2/utils ./toolboxv2/utils
COPY ./toolboxv2/__init__.py ./toolboxv2/__init__.py
COPY ./toolboxv2/__main__.py ./toolboxv2/__main__.py
COPY ./toolboxv2/cli.py ./toolboxv2/cli.py
COPY ./toolboxv2/favicon.ico ./toolboxv2/favicon.ico
COPY ./toolboxv2/index.html ./toolboxv2/index.html
COPY ./toolboxv2/index.js ./toolboxv2/index.js
COPY ./toolboxv2/package.json ./toolboxv2/package.json
COPY ./toolboxv2/tbState.yaml ./toolboxv2/tbState.yaml
COPY ./toolboxv2/toolbox.yaml ./toolboxv2/toolbox.yaml
COPY ./toolboxv2/vite.config.js ./toolboxv2/vite.config.js

# Install the local application using pip.
RUN npx bun install -y ./toolboxv2/web/
RUN pip install -e .
# Expose the port that the application listens on.

# Run the application.
CMD toolboxv2 -bgr -l -m bg
