"""
ContainerManager Web UI - Streamlit Dashboard

Features:
- Admin Key Auth
- User Management
- Container Dashboard
- Live Logs & Stats
"""

import asyncio
import json
import os
import subprocess
import time
from datetime import datetime
from typing import List, Optional

import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from toolboxv2 import get_app, Result
from toolboxv2.mods.ContainerManager import (
    create_container,
    list_containers,
    get_container,
    delete_container,
    start_container,
    stop_container,
    restart_container,
    container_logs,
    container_exec,
    generate_admin_key,
    add_ssh_key_to_container,
    get_container_ssh_info,
    list_ssh_containers,
    register_ssh_key,
    get_my_ssh_info,
)

# ============================================================================
# CONFIG
# ============================================================================

ADMIN_KEY = os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me")

st.set_page_config(
    page_title="Container Manager",
    page_icon="🐳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS STYLING
# ============================================================================

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
<style>
    .stApp { background-color: #08080d; }
    header[data-testid="stHeader"] { background-color: #08080d !important; }

    * {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    code, pre, .stCode {
        font-family: 'IBM Plex Mono', monospace;
    }

    .main-header {
        background: rgba(255,255,255,0.015);
        padding: 1.25rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.04);
    }

    .main-header h1 {
        margin: 0;
        color: #e2e2e8;
        font-size: 22px;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        color: rgba(255,255,255,0.45);
        font-size: 13px;
        font-weight: 400;
    }

    .metric-card {
        background: rgba(255,255,255,0.015);
        padding: 1rem;
        border-radius: 6px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.04);
        transition: all 0.15s;
    }

    .metric-card:hover {
        background: rgba(99,102,241,0.06);
        border-color: rgba(99,102,241,0.2);
    }

    .metric-value {
        font-size: 20px;
        font-weight: 600;
        color: #e2e2e8;
    }

    .metric-label {
        color: rgba(255,255,255,0.25);
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        font-weight: 500;
        margin-top: 0.5rem;
    }

    .status-badge {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 9px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .status-running {
        background: rgba(16,185,129,0.09);
        color: #10b981;
        border: 1px solid rgba(16,185,129,0.18);
    }

    .status-stopped {
        background: rgba(245,158,11,0.09);
        color: #f59e0b;
        border: 1px solid rgba(245,158,11,0.18);
    }

    .status-error {
        background: rgba(239,68,68,0.09);
        color: #ef4444;
        border: 1px solid rgba(239,68,68,0.18);
    }

    .container-row {
        background: rgba(255,255,255,0.015);
        padding: 0.875rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(255,255,255,0.04);
        transition: all 0.15s;
        cursor: pointer;
    }

    .container-row:hover {
        background: rgba(99,102,241,0.06);
        border-color: rgba(99,102,241,0.2);
    }

    .code-block {
        background: rgba(255,255,255,0.02);
        padding: 1rem;
        border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        overflow-x: auto;
        border: 1px solid rgba(255,255,255,0.04);
        color: #a5b4fc;
    }

    .section-label {
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        color: rgba(255,255,255,0.25);
        font-weight: 500;
        margin-bottom: 0.75rem;
    }

    .stInfo, .stSuccess, .stWarning {
        background: rgba(255,255,255,0.015);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 6px;
    }

    .stButton > button {
        border-radius: 6px;
        transition: all 0.15s;
        font-weight: 500;
    }

    .stButton > button:hover {
        opacity: 0.9;
    }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background: rgba(255,255,255,0.015);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 6px;
        color: #e2e2e8;
        font-size: 13px;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: rgba(99,102,241,0.3);
        background: rgba(99,102,241,0.06);
    }

    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.015);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 6px;
        font-size: 13px;
    }

    .streamlit-expanderContent {
        background: rgba(255,255,255,0.01);
        border: 1px solid rgba(255,255,255,0.02);
        border-radius: 0 0 6px 6px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# AUTH
# ============================================================================

def check_auth() -> bool:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 80vh;">
            <div style="background: rgba(255,255,255,0.015); padding: 2rem; border-radius: 6px; border: 1px solid rgba(255,255,255,0.04); max-width: 400px;">
                <h2 style="color: #e2e2e8; text-align: center; font-size: 18px; font-weight: 300; letter-spacing: 0.5px;">Container Manager</h2>
                <p style="color: rgba(255,255,255,0.45); text-align: center; margin-bottom: 1.5rem; font-size: 13px;">Admin authentication required</p>
        """, unsafe_allow_html=True)

        admin_key = st.text_input("Admin Key", type="password", key="auth_input")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Login", use_container_width=True):
                if admin_key == ADMIN_KEY:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid admin key")

        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)
        return False

    return True


def logout():
    st.session_state.authenticated = False
    st.rerun()


# ============================================================================
# HELPERS
# ============================================================================

@st.cache_data(ttl=10)
def get_cached_containers(user_id: Optional[str] = None, all: bool = False) -> dict:
    app = get_app()
    result = asyncio.run(list_containers(app=app, user_id=user_id, admin_key=ADMIN_KEY, all=all))
    if result.is_ok():
        return result.get()
    return {"containers": []}


@st.cache_data(ttl=5)
def get_cached_container_stats(container_id: str) -> dict:
    app = get_app()
    result = asyncio.run(get_container(app=app, container_id=container_id, admin_key=ADMIN_KEY))
    if result.is_ok():
        return result.get()
    return {}


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div class="main-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_metric(value: str, label: str, icon: str = ""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{icon} {value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_status_badge(status: str) -> str:
    status_colors = {
        "running":    ("#10b981", "Running"),
        "exited":     ("#f59e0b", "Stopped"),
        "stopped":    ("#f59e0b", "Stopped"),
        "created":    ("#6366f1", "Created"),
        "restarting": ("#a5b4fc", "Restarting"),
        "dead":       ("#ef4444", "Dead"),
    }
    color, label = status_colors.get(status.lower(), ("#ef4444", status))
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    return (f'<span class="status-badge" style="background: rgba({r},{g},{b},0.09); '
            f'color: {color}; border-color: rgba({r},{g},{b},0.18);">{label}</span>')


# ============================================================================
# PAGES
# ============================================================================

def page_dashboard():
    render_header("Container Dashboard", "Manage your Docker containers")

    containers_data = get_cached_containers(all=True)
    all_containers = containers_data.get("containers", [])

    running = sum(1 for c in all_containers if c.get("status") == "running")
    stopped = sum(1 for c in all_containers if c.get("status") != "running")
    total_users = len(set(c.get("user_id") for c in all_containers))
    ssh_enabled = sum(1 for c in all_containers if c.get("ssh_port"))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric(str(total_users), "Users", "👤")
    with col2:
        render_metric(str(len(all_containers)), "Containers", "🐳")
    with col3:
        render_metric(str(running), "Running", "●")
    with col4:
        render_metric(str(ssh_enabled), "SSH enabled", "🔑")

    st.markdown("---")

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        filter_user = st.text_input("Filter by User ID")
    with col2:
        filter_type = st.selectbox("Filter by Type", ["All", "cli_v4", "project_dev", "preview_server", "custom"])
    with col3:
        filter_status = st.selectbox("Filter by Status", ["All", "running", "stopped", "exited"])

    filtered = [
        c for c in all_containers
        if (not filter_user or filter_user.lower() in c.get("user_id", "").lower())
        and (filter_type == "All" or c.get("container_type") == filter_type)
        and (filter_status == "All" or c.get("status") == filter_status)
    ]

    if not filtered:
        st.info("No containers found")
        return

    for c in filtered:
        with st.expander(
            f"{c['container_id']} — {c.get('container_name', 'N/A')} — {c.get('container_type', 'N/A')}",
            expanded=False
        ):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**User ID:** `{c.get('user_id', 'N/A')}`")
                st.markdown(f"**HTTP Port:** `{c.get('port', 'N/A')}`  |  **URL:** `{c.get('url', 'N/A')}`")
                if c.get("ssh_port"):
                    st.markdown(f"**SSH Port:** `{c.get('ssh_port')}`")
                st.markdown(f"**Created:** `{datetime.fromtimestamp(c.get('created_at', 0)).strftime('%Y-%m-%d %H:%M')}`")
                st.markdown(f"**Status:** {render_status_badge(c.get('status', 'unknown'))}", unsafe_allow_html=True)

            with col2:
                status = c.get("status", "unknown")
                cid = c.get("container_id", "")
                app = get_app()

                if status == "running":
                    if st.button("Stop", key=f"stop_{cid}", use_container_width=True):
                        asyncio.run(stop_container(app=app, container_id=cid, admin_key=ADMIN_KEY))
                        st.cache_data.clear()
                        st.rerun()
                    if st.button("Restart", key=f"restart_{cid}", use_container_width=True):
                        asyncio.run(restart_container(app=app, container_id=cid, admin_key=ADMIN_KEY))
                        st.cache_data.clear()
                        st.rerun()
                else:
                    if st.button("Start", key=f"start_{cid}", use_container_width=True):
                        asyncio.run(start_container(app=app, container_id=cid, admin_key=ADMIN_KEY))
                        st.cache_data.clear()
                        st.rerun()

                if st.button("Logs", key=f"logs_{cid}", use_container_width=True):
                    st.session_state.show_logs = cid
                    st.rerun()

                if st.button("Delete", key=f"delete_{cid}", type="secondary", use_container_width=True):
                    asyncio.run(delete_container(app=app, container_id=cid, admin_key=ADMIN_KEY, force=True))
                    st.cache_data.clear()
                    st.rerun()

    if "show_logs" in st.session_state and st.session_state.show_logs:
        cid = st.session_state.show_logs
        st.markdown("---")
        st.markdown(f"### Logs — `{cid[:12]}`")
        app = get_app()
        result = asyncio.run(container_logs(app=app, container_id=cid, admin_key=ADMIN_KEY, tail=200))
        if result.is_ok():
            st.code(result.get().get("logs", ""), language="text")
        else:
            st.error(result.info)
        if st.button("Close Logs"):
            del st.session_state.show_logs
            st.rerun()


def page_create():
    render_header("Create Container", "Create a new container for a user")

    with st.form("create_container_form"):
        col1, col2 = st.columns(2)

        with col1:
            user_id = st.text_input("User ID", placeholder="usr_xxxxxxxx")
            st.caption("CloudM Auth User ID")

            container_type = st.selectbox(
                "Container Type",
                ["cli_v4", "project_dev", "preview_server", "custom"],
                help="cli_v4: Persistent CLI | project_dev: Streamlit Dev UI | preview_server: HTTP Preview"
            )
            container_name = st.text_input("Container Name (optional)", placeholder="Custom name")

        with col2:
            image = st.text_input("Docker Image (optional)", placeholder="toolboxv2:latest")
            command = st.text_input("Command (optional)", placeholder="Custom entrypoint")
            env_json = st.text_area("Environment Variables (JSON, optional)", placeholder='{"KEY": "value"}')
            ssh_key = st.text_input(
                "SSH Public Key (optional)",
                placeholder="ssh-ed25519 AAAAC3...",
                help="If provided, user can SSH into the container immediately after creation."
            )

        with st.expander("Advanced Settings"):
            col1, col2 = st.columns(2)
            with col1:
                memory_limit = st.text_input("Memory Limit", value="512m")
            with col2:
                cpu_limit = st.text_input("CPU Limit", value="0.5")

        submitted = st.form_submit_button("Create Container", type="primary")

        if submitted:
            if not user_id:
                st.error("User ID is required")
                return

            environment = None
            if env_json:
                try:
                    environment = json.loads(env_json)
                except json.JSONDecodeError:
                    st.error("Invalid JSON in environment variables")
                    return

            app = get_app()
            result = asyncio.run(create_container(
                app=app,
                container_type=container_type,
                user_id=user_id,
                container_name=container_name or None,
                admin_key=ADMIN_KEY,
                image=image or None,
                command=command or None,
                environment=environment,
                memory_limit=memory_limit,
                cpu_limit=cpu_limit,
                ssh_public_key=ssh_key.strip() or None,
            ))

            if result.is_ok():
                data = result.get()
                st.success("Container created successfully!")
                st.json(data)
                if data.get("ssh_connection"):
                    st.info(f"**SSH connection string for user:**\n```\n{data['ssh_connection']}\n```")
            else:
                st.error(f"Failed: {result.info}")


def page_users():
    render_header("User Management", "Manage users and their containers")

    col1, col2 = st.columns([2, 1])
    with col1:
        search_user = st.text_input("Search User by ID")
    with col2:
        st.button("Search", disabled=not search_user)

    containers_data = get_cached_containers(all=True)
    all_containers = containers_data.get("containers", [])

    user_containers: dict = {}
    for c in all_containers:
        uid = c.get("user_id", "unknown")
        user_containers.setdefault(uid, []).append(c)

    for user_id, containers in sorted(user_containers.items()):
        if search_user and search_user.lower() not in user_id.lower():
            continue
        with st.expander(f"👤 {user_id} ({len(containers)} containers)", expanded=False):
            for c in containers:
                ssh_tag = f" · SSH:{c['ssh_port']}" if c.get("ssh_port") else ""
                st.markdown(f"""
                <div class="container-row">
                    <strong>{c.get('container_id')}</strong> — {c.get('container_type')}{ssh_tag}
                    <br>Status: {render_status_badge(c.get('status', 'unknown'))}
                    <br>URL: <code>{c.get('url')}</code>
                </div>
                """, unsafe_allow_html=True)


def page_ssh_keys():
    render_header("SSH Access", "Manage SSH key access to containers")

    st.markdown(
        "SSH access is available on **all container types**. "
        "An SSH port is automatically reserved for every container. "
        "Users register their own public key via the CLI or REST API — no key exchange through the admin required."
    )
    st.markdown("---")

    app = get_app()
    result = asyncio.run(list_ssh_containers(app=app, user_id=None, admin_key=ADMIN_KEY))

    if result.is_error():
        st.error(f"Failed to load containers: {result.info}")
        return

    containers = result.get().get("containers", [])

    if not containers:
        st.info("No SSH-enabled containers found. Create a container to get started.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Inject SSH Key (Admin)")
        st.caption("Use this when the user sends you their public key manually.")

        container_options = {
            f"{c['container_id'][:12]} — {c['container_name']} ({c['user_id']})": c['container_id']
            for c in containers
        }

        selected_label = st.selectbox("Select Container", options=list(container_options.keys()), key="ssh_container_select")

        if selected_label:
            container_id = container_options[selected_label]
            container = next((c for c in containers if c['container_id'] == container_id), None)

            if container:
                st.info(
                    f"**ID:** `{container['container_id']}`  \n"
                    f"**User:** `{container['user_id']}`  \n"
                    f"**SSH Port:** `{container['ssh_port']}`  \n"
                    f"**Connect:** `ssh -p {container['ssh_port']} cli@{container['server_ip']}`"
                )

            st.markdown("#### SSH Public Key")
            ssh_key = st.text_area(
                "Public Key",
                placeholder="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExample...",
                height=90,
                key="ssh_key_input",
            )

            with st.expander("How does the user get their key?"):
                st.markdown("""
                **Self-service (recommended):**
                ```bash
                # User generates key and registers directly
                python -m toolboxv2.mods.ContainerManager.cli setup
                python -m toolboxv2.mods.ContainerManager.cli register-key
                ```
                No admin involvement needed — the user authenticates via their CloudM token.

                **Manual (admin injects key):**
                ```bash
                # User generates key and shows it
                python -m toolboxv2.mods.ContainerManager.cli setup
                # User sends the displayed public key to admin
                # Admin pastes it here
                ```

                **Supported key types:** `ssh-ed25519` (recommended), `ssh-rsa`, `ecdsa-sha2-nistp256`
                """)

            if st.button("Inject Key", type="primary", key="add_ssh_key_btn"):
                if not ssh_key.strip():
                    st.error("Please enter a public key")
                    return
                if not ssh_key.strip().startswith(("ssh-ed25519", "ssh-rsa", "ecdsa-sha2")):
                    st.error("Invalid key format")
                    return

                with st.spinner("Injecting key..."):
                    r = asyncio.run(add_ssh_key_to_container(
                        app=app, container_id=container_id,
                        ssh_public_key=ssh_key.strip(), admin_key=ADMIN_KEY
                    ))

                if r.is_ok():
                    data = r.get()
                    st.success("Key injected successfully!")
                    st.markdown("**Connection string for user:**")
                    st.code(data['ssh_connection'], language="bash")
                    st.markdown("""
                    Send the user this connection string. They connect with:
                    ```bash
                    python -m toolboxv2.mods.ContainerManager.cli connect <container_id>
                    ```
                    """)
                else:
                    st.error(f"Failed: {r.info}")

    with col2:
        st.markdown("### SSH-Enabled Containers")
        for c in containers:
            dot = "●" if c.get("status") == "running" else "○"
            with st.expander(f"{dot} {c['container_id'][:12]} — {c['container_name']}", expanded=False):
                st.markdown(
                    f"**User:** `{c['user_id']}`  \n"
                    f"**Type:** `{c['container_type']}`  \n"
                    f"**SSH Port:** `{c['ssh_port']}`  \n"
                    f"**Status:** {render_status_badge(c.get('status', 'unknown'))}",
                    unsafe_allow_html=True
                )
                st.code(c['connection_string'], language="bash")

    st.markdown("---")

    st.markdown("### User Workflow")
    tab1, tab2, tab3 = st.tabs(["Self-service (recommended)", "Admin-assisted", "Connect"])

    with tab1:
        st.markdown("""
        User handles key setup and registration independently using their CloudM auth token.

        ```bash
        # 1. Generate SSH key pair
        python -m toolboxv2.mods.ContainerManager.cli setup

        # 2. Register key with their container (auth token required)
        python -m toolboxv2.mods.ContainerManager.cli register-key

        # 3. Connect
        python -m toolboxv2.mods.ContainerManager.cli connect
        ```
        """)

    with tab2:
        st.markdown("""
        User generates key and sends public key to admin. Admin injects it above.

        ```bash
        # User runs:
        python -m toolboxv2.mods.ContainerManager.cli setup
        # → copies displayed public key → sends to admin

        # Admin injects key in "Inject SSH Key" panel above.

        # Admin sends connection string back to user.
        ```
        """)

    with tab3:
        st.markdown("""
        Once the key is registered, the user connects with:

        ```bash
        # Using ContainerManager CLI (saves config automatically)
        python -m toolboxv2.mods.ContainerManager.cli connect <container_id>

        # Or plain SSH
        ssh -i ~/.ssh/cm_id_ed25519 -p <ssh-port> cli@<server-ip>
        ```

        The CLI saves the connection info after first use — subsequent calls
        require no arguments:
        ```bash
        python -m toolboxv2.mods.ContainerManager.cli connect
        ```
        """)


def page_settings():
    render_header("Settings", "Configure Container Manager")

    st.markdown("### Admin Key")
    st.info(
        f"Current key ends in: `...{ADMIN_KEY[-4:]}`\n\n"
        "To rotate, set `CONTAINER_ADMIN_KEY` env var and restart."
    )

    st.markdown("---")
    st.markdown("### Generate New Admin Key")
    if st.button("Generate New Admin Key"):
        app = get_app()
        result = asyncio.run(generate_admin_key(app=app))
        if result.is_ok():
            st.json(result.get())

    st.markdown("---")
    st.markdown("### Container Types")
    st.markdown("""
    | Type | Internal Port | Default Image |
    |------|---------------|---------------|
    | `cli_v4` | 8080 | toolboxv2:latest |
    | `project_dev` | 8501 | toolboxv2:dev |
    | `preview_server` | 8600 | toolboxv2:latest |
    | `custom` | 8080 | toolboxv2:latest |

    All types get an SSH port (22000–22500) reserved automatically.
    """)

    st.markdown("---")
    st.markdown("### Port Pools")
    st.info(
        "HTTP containers: `9000 – 9500`  \n"
        "SSH ports: `22000 – 22500`  \n\n"
        "Both pools are tracked in TBEF.DB under `CONTAINER_PORT_POOL` and `CONTAINER_SSH_PORT_POOL`."
    )

    st.markdown("---")
    st.markdown("### Nginx Integration")
    try:
        nginx_ok = subprocess.run(["pgrep", "nginx"], stdout=subprocess.DEVNULL).returncode == 0
        nginx_status = "Active" if nginx_ok else "Not available (Linux only)"
    except Exception:
        nginx_status = "Not available"
    st.info(
        f"Nginx status: **{nginx_status}**\n\n"
        "Configs: `/etc/nginx/box-available/container-<user>-<type>.conf`  \n"
        "Accessible at: `http://your-server/container/<user_id>/<container_type>/`"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not check_auth():
        return

    with st.sidebar:
        st.markdown("# Container Manager")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Dashboard", "Create", "Users", "SSH Access", "Settings"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### Admin")
        if st.button("Logout"):
            logout()

        st.markdown("---")
        st.markdown(
            '<div style="font-size: 0.8rem; color: rgba(255,255,255,0.5);">ContainerManager v1.0.0</div>',
            unsafe_allow_html=True
        )

    if page == "Dashboard":
        page_dashboard()
    elif page == "Create":
        page_create()
    elif page == "Users":
        page_users()
    elif page == "SSH Access":
        page_ssh_keys()
    elif page == "Settings":
        page_settings()


if __name__ == "__main__":
    main()
