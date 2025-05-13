# syntax=docker/dockerfile:1

FROM toolbox-base:3.11 as base

WORKDIR /app

# Copy only what you need — layer caching helps

COPY ./setup.py ./setup.py
COPY ./requirements.txt ./requirements.txt
COPY ./README.md ./README.md
COPY ./setup.cfg ./setup.cfg
COPY ./MANIFEST.in ./MANIFEST.in
COPY ./pyproject.toml ./pyproject.toml
COPY ./toolboxv2/mods ./toolboxv2/mods
COPY ./toolboxv2/flows ./toolboxv2/flows
COPY ./toolboxv2/tests ./toolboxv2/tests
COPY ./toolboxv2/web ./toolboxv2/web
COPY ./toolboxv2/utils ./toolboxv2/utils
COPY ./toolboxv2/__init__.py ./toolboxv2/__init__.py
COPY ./toolboxv2/__main__.py ./toolboxv2/__main__.py
COPY ./toolboxv2/favicon.ico ./toolboxv2/favicon.ico
COPY ./toolboxv2/index.html ./toolboxv2/index.html
COPY ./toolboxv2/index.js ./toolboxv2/index.js
COPY ./toolboxv2/package.json ./toolboxv2/package.json
COPY ./toolboxv2/src-core/src ./toolboxv2/src-core/src
COPY ./toolboxv2/src-core/config.toml ./toolboxv2/src-core/config.toml
COPY ./toolboxv2/src-core/Cargo.toml ./toolboxv2/src-core/Cargo.toml

# JS deps (cached if package.json unchanged)
RUN npm install --prefix ./toolboxv2/web/
RUN npm install --save-dev webpack-merge --prefix ./toolboxv2/web/

# Install local Python package (torch already installed)
RUN pip install uv
RUN uv pip install -e .[isaa] --system

# Expose ports
EXPOSE 5000/tcp 5000/udp 8000/tcp 8000/udp 8080/tcp 8080/udp 6587/tcp 6587/udp 17334/tcp 17334/udp

# Start
CMD ["tb", "api", "start"]
