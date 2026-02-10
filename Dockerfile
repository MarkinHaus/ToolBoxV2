# syntax=docker/dockerfile:1.4
FROM python:3.11

WORKDIR /app

# ── System deps: Node.js (LTS) + build tools ──
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# ── Copy ToolBoxV2 source ──
COPY ./setup.py ./requirements.txt ./README.md ./setup.cfg ./MANIFEST.in ./pyproject.toml ./
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

# ── JS deps ──
RUN npm install --prefix ./toolboxv2/web/ && \
    npm install --save-dev webpack-merge --prefix ./toolboxv2/web/

# ── Install ToolBoxV2 ──
RUN pip install uv && uv pip install -e .[isaa] --system

# ── User & Permissions ──
RUN useradd -m -s /bin/bash cli && \
    mkdir -p /home/cli/.ssh /data /run/sshd && \
    chmod 700 /home/cli/.ssh && \
    chown -R cli:cli /home/cli /data

# ── SSH Config ──
COPY <<'SSHCFG' /etc/ssh/sshd_config
Port 2222
PubkeyAuthentication yes
AuthorizedKeysFile /home/cli/.ssh/authorized_keys
PasswordAuthentication no
PermitRootLogin no
AllowUsers cli
ForceCommand /usr/local/bin/attach-cli.sh
HostKey /etc/ssh/ssh_host_ed25519_key
PidFile /run/sshd.pid
SSHCFG

# ── 1. Der Loop-Wrapper (NEU: Startet CLI immer neu) ──
COPY <<'LOOP' /usr/local/bin/run-loop.sh
#!/bin/bash
CMD="${DOCKSH_CLI:-tb -m icli}"
while true; do
    echo "--- Starting CLI ---"
    # Befehl ausführen
    $CMD

    # Wenn wir hier sind, ist das Programm gecrasht oder beendet
    echo "!!! CLI crashed or exited. Restarting in 2 seconds..."
    sleep 2
done
LOOP
RUN chmod +x /usr/local/bin/run-loop.sh

# ── 2. Attach Script (Nutzt jetzt den Loop) ──
COPY <<'ATTACH' /usr/local/bin/attach-cli.sh
#!/bin/bash
export TERM=xterm-256color
# Prüfen ob Session existiert
if tmux has-session -t cli 2>/dev/null; then
    exec tmux attach-session -t cli
else
    # Falls Session fehlt (sollte nicht passieren dank entry.sh), neu starten mit Loop
    exec tmux new-session -s cli "/usr/local/bin/run-loop.sh"
fi
ATTACH
RUN chmod +x /usr/local/bin/attach-cli.sh

# ── 3. Entrypoint (Startet tmux im Hintergrund mit Loop) ──
COPY <<'ENTRY' /usr/local/bin/entry.sh
#!/bin/bash
set -e

# Host Key generieren
if [ ! -f /etc/ssh/ssh_host_ed25519_key ]; then
    ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -N "" -q
fi

# Key Injection
if [ ! -z "$SSH_PUBLIC_KEY" ]; then
    echo "$SSH_PUBLIC_KEY" > /home/cli/.ssh/authorized_keys
    chmod 600 /home/cli/.ssh/authorized_keys
    chown cli:cli /home/cli/.ssh/authorized_keys
fi

# Permissions fixen
chown -R cli:cli /home/cli /data
chmod 700 /home/cli/.ssh

# SSHD starten
/usr/sbin/sshd

# Tmux Session als User 'cli' starten (im Hintergrund)
# Wir nutzen hier run-loop.sh, damit es niemals stirbt
su - cli -c "tmux new-session -d -s cli '/usr/local/bin/run-loop.sh'"

echo "Ready. CLI running in loop."

# Container am Leben halten (Polling auf tmux session)
while true; do
    sleep 5
    if ! su - cli -c "tmux has-session -t cli 2>/dev/null"; then
        echo "Tmux died unexpectedly. Restarting..."
        su - cli -c "tmux new-session -d -s cli '/usr/local/bin/run-loop.sh'"
    fi
done
ENTRY
RUN chmod +x /usr/local/bin/entry.sh

EXPOSE 2222
VOLUME /data
ENTRYPOINT ["/usr/local/bin/entry.sh"]
