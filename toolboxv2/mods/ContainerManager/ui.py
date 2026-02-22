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

# Add parent path to allow imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from toolboxv2 import get_app, Result
# from toolboxv2.mods.CloudM.auth.user_store import _load_user, _find_user_by_email

# Import local module
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

    /* Main Header */
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

    /* Metric Cards */
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

    /* Status Badges */
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

    /* Container Row */
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

    /* Code Block */
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

    /* Section Labels */
    .section-label {
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        color: rgba(255,255,255,0.25);
        font-weight: 500;
        margin-bottom: 0.75rem;
    }

    /* Info Boxes */
    .stInfo, .stSuccess, .stWarning {
        background: rgba(255,255,255,0.015);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 6px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 6px;
        transition: all 0.15s;
        font-weight: 500;
    }

    .stButton > button:hover {
        opacity: 0.9;
    }

    /* Inputs */
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

    /* Expander */
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
    """Check if user is authenticated"""
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
    """Logout"""
    st.session_state.authenticated = False
    st.rerun()


# ============================================================================
# HELPERS
# ============================================================================

@st.cache_data(ttl=10)
def get_cached_containers(user_id: Optional[str] = None, all: bool = False) -> dict:
    """Cached container list"""
    app = get_app()
    result = asyncio.run(list_containers(
        app=app,
        user_id=user_id,
        admin_key=ADMIN_KEY,
        all=all
    ))
    if result.is_ok():
        return result.get()
    return {"containers": []}


@st.cache_data(ttl=5)
def get_cached_container_stats(container_id: str) -> dict:
    """Cached container stats"""
    app = get_app()
    result = asyncio.run(get_container(
        app=app,
        container_id=container_id,
        admin_key=ADMIN_KEY
    ))
    if result.is_ok():
        return result.get()
    return {}


def get_user_list() -> List[str]:
    """Get list of users from CloudM Auth"""
    try:
        app = get_app()
        # Try to get user IDs from container DB (faster)
        result = asyncio.run(app.a_run_any(
            "TBEF.DB.GET",
            query="CONTAINER_USER::*",
            get_results=True
        ))
        # TODO: Implement proper user listing
        return []
    except Exception:
        return []


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header(title: str, subtitle: str = ""):
    """Render page header"""
    st.markdown(f"""
    <div class="main-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_metric(value: str, label: str, icon: str = ""):
    """Render a metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{icon} {value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_status_badge(status: str) -> str:
    """Render status badge (minimal, no emoji)"""
    status_colors = {
        "running": ("#10b981", "Running"),
        "exited": ("#f59e0b", "Stopped"),
        "stopped": ("#f59e0b", "Stopped"),
        "created": ("#6366f1", "Created"),
        "restarting": ("#a5b4fc", "Restarting"),
        "dead": ("#ef4444", "Dead"),
    }

    color, label = status_colors.get(status.lower(), ("#ef4444", status))

    return f'<span class="status-badge" style="background: rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.09); color: {color}; border-color: rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.18);">{label}</span>'

    icon = {
        "running": "🟢",
        "exited": "⚫",
        "stopped": "⏸️",
        "unknown": "❓",
    }.get(status.lower(), "❌")

    return f'<span class="{status_class}">{icon} {status}</span>'


# ============================================================================
# PAGES
# ============================================================================

def page_dashboard():
    """Main dashboard"""
    render_header("Container Dashboard", "Manage your Docker containers")

    # Metrics
    containers_data = get_cached_containers(all=True)
    all_containers = containers_data.get("containers", [])

    running = sum(1 for c in all_containers if c.get("status") == "running")
    stopped = sum(1 for c in all_containers if c.get("status") != "running")
    total_users = len(set(c.get("user_id") for c in all_containers))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric(str(total_users), "Users", "👤")
    with col2:
        render_metric(str(len(all_containers)), "Containers", "🐳")
    with col3:
        render_metric(str(running), "Running", "🟢")
    with col4:
        render_metric(str(stopped), "Stopped", "⚫")

    st.markdown("---")

    # Filter
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        filter_user = st.text_input("Filter by User ID")
    with col2:
        filter_type = st.selectbox("Filter by Type", ["All", "cli_v4", "project_dev", "preview_server", "custom"])
    with col3:
        filter_status = st.selectbox("Filter by Status", ["All", "running", "stopped", "exited"])

    # Container List
    filtered = []
    for c in all_containers:
        if filter_user and filter_user.lower() not in c.get("user_id", "").lower():
            continue
        if filter_type != "All" and c.get("container_type") != filter_type:
            continue
        if filter_status != "All" and c.get("status") != filter_status:
            continue
        filtered.append(c)

    if not filtered:
        st.info("No containers found")
        return

    for c in filtered:
        with st.expander(
            f"{c['container_id']} - {c.get('container_name', 'N/A')} - {c.get('container_type', 'N/A')}",
            expanded=False
        ):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**User ID:** `{c.get('user_id', 'N/A')}`")
                st.markdown(f"**Port:** `{c.get('port', 'N/A')}`")
                st.markdown(f"**URL:** `{c.get('url', 'N/A')}`")
                st.markdown(f"**Created:** `{datetime.fromtimestamp(c.get('created_at', 0)).strftime('%Y-%m-%d %H:%M')}`")
                st.markdown(f"**Status:** {render_status_badge(c.get('status', 'unknown'))}")

            with col2:
                status = c.get("status", "unknown")
                cid = c.get("container_id", "")

                if status == "running":
                    if st.button("⏹ Stop", key=f"stop_{cid}", use_container_width=True):
                        app = get_app()
                        result = asyncio.run(stop_container(
                            app=app,
                            container_id=cid,
                            admin_key=ADMIN_KEY
                        ))
                        st.cache_data.clear()
                        st.rerun()
                    if st.button("🔄 Restart", key=f"restart_{cid}", use_container_width=True):
                        app = get_app()
                        result = asyncio.run(restart_container(
                            app=app,
                            container_id=cid,
                            admin_key=ADMIN_KEY
                        ))
                        st.cache_data.clear()
                        st.rerun()
                else:
                    if st.button("▶ Start", key=f"start_{cid}", use_container_width=True):
                        app = get_app()
                        result = asyncio.run(start_container(
                            app=app,
                            container_id=cid,
                            admin_key=ADMIN_KEY
                        ))
                        st.cache_data.clear()
                        st.rerun()

                if st.button("📋 Logs", key=f"logs_{cid}", use_container_width=True):
                    st.session_state.show_logs = cid
                    st.rerun()

                if st.button("🗑 Delete", key=f"delete_{cid}", type="secondary", use_container_width=True):
                    app = get_app()
                    result = asyncio.run(delete_container(
                        app=app,
                        container_id=cid,
                        admin_key=ADMIN_KEY,
                        force=True
                    ))
                    st.cache_data.clear()
                    st.rerun()

    # Show logs if requested
    if "show_logs" in st.session_state and st.session_state.show_logs:
        cid = st.session_state.show_logs
        st.markdown("---")
        st.markdown(f"### Logs for `{cid[:12]}`")

        app = get_app()
        result = asyncio.run(container_logs(
            app=app,
            container_id=cid,
            admin_key=ADMIN_KEY,
            tail=200
        ))

        if result.is_ok():
            logs = result.get().get("logs", "")
            st.code(logs, language="text")
        else:
            st.error(result.info)

        if st.button("Close Logs"):
            del st.session_state.show_logs
            st.rerun()


def page_create():
    """Create new container"""
    render_header("Create Container", "Create a new container for a user")

    with st.form("create_container_form"):
        col1, col2 = st.columns(2)

        with col1:
            # User selection
            user_id = st.text_input("User ID", placeholder="usr_xxxxxxxx")
            st.caption("Enter the CloudM Auth User ID")

            # Container type
            container_type = st.selectbox(
                "Container Type",
                ["cli_v4", "project_dev", "preview_server", "custom"],
                help="cli_v4: Persistent CLI | project_dev: Streamlit Dev UI | preview_server: HTTP Preview"
            )

            # Container name
            container_name = st.text_input("Container Name (optional)", placeholder="Custom name")

        with col2:
            # Image override
            image = st.text_input("Docker Image (optional)", placeholder="toolboxv2:latest")
            st.caption("Leave empty to use default")

            # Command override
            command = st.text_input("Command (optional)", placeholder="Custom entrypoint")

            # Environment
            env_json = st.text_area("Environment Variables (JSON, optional)", placeholder='{"KEY": "value"}')

        # Advanced settings
        with st.expander("Advanced Settings"):
            col1, col2 = st.columns(2)
            with col1:
                memory_limit = st.text_input("Memory Limit", value="512m", help="e.g., 512m, 1g")
            with col2:
                cpu_limit = st.text_input("CPU Limit", value="0.5", help="e.g., 0.5, 1.0")

        # Submit
        submitted = st.form_submit_button("🚀 Create Container", type="primary")

        if submitted:
            if not user_id:
                st.error("User ID is required")
                return

            # Parse environment
            environment = None
            if env_json:
                try:
                    environment = json.loads(env_json)
                except json.JSONDecodeError:
                    st.error("Invalid JSON in environment variables")
                    return

            # Create container
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
                cpu_limit=cpu_limit
            ))

            if result.is_ok():
                st.success("Container created successfully!")
                st.json(result.get())
            else:
                st.error(f"Failed to create container: {result.info}")


def page_users():
    """User management"""
    render_header("User Management", "Manage users and their containers")

    # User lookup
    col1, col2 = st.columns([2, 1])
    with col1:
        search_user = st.text_input("Search User by ID or Email")
    with col2:
        st.button("🔍 Search", disabled=not search_user)

    # Display users with containers
    app = get_app()

    # Get all containers and group by user
    containers_data = get_cached_containers(all=True)
    all_containers = containers_data.get("containers", [])

    user_containers = {}
    for c in all_containers:
        uid = c.get("user_id", "unknown")
        if uid not in user_containers:
            user_containers[uid] = []
        user_containers[uid].append(c)

    for user_id, containers in sorted(user_containers.items()):
        with st.expander(f"👤 {user_id} ({len(containers)} containers)", expanded=False):
            for c in containers:
                st.markdown(f"""
                <div class="container-row">
                    <strong>{c.get('container_id')}</strong> - {c.get('container_type')}
                    <br>
                    Status: {render_status_badge(c.get('status'))}
                    <br>
                    URL: <code>{c.get('url')}</code>
                </div>
                """, unsafe_allow_html=True)


def page_ssh_keys():
    """SSH Key Management page"""
    render_header("SSH Key Management", "Manage SSH keys for container access")

    st.markdown("""
    This page allows you to add SSH public keys to containers, enabling users to
    access their containers via SSH using the Docksh client.
    """)

    st.markdown("---")

    # Get all SSH-enabled containers
    app = get_app()
    result = asyncio.run(list_ssh_containers(
        app=app,
        user_id=None,  # None = alle Container
        admin_key=ADMIN_KEY
    ))

    if result.is_error():
        st.error(f"Failed to load containers: {result.info}")
        return

    containers = result.get().get("containers", [])

    if not containers:
        st.info("No SSH-enabled containers found. Only `cli_v4` containers support SSH access.")
        return

    # Two-column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Add SSH Key to Container")

        # Container selection
        container_options = {
            f"{c['container_id'][:12]} - {c['container_name']} ({c['user_id']})": c['container_id']
            for c in containers
        }

        selected_container_label = st.selectbox(
            "Select Container",
            options=list(container_options.keys()),
            key="ssh_container_select"
        )

        if selected_container_label:
            container_id = container_options[selected_container_label]

            # Show container info
            container = next((c for c in containers if c['container_id'] == container_id), None)
            if container:
                st.info(f"""
                **Container Details:**
                - **ID:** {container['container_id']}
                - **Name:** {container['container_name']}
                - **User:** {container['user_id']}
                - **SSH Port:** {container['ssh_port']}
                - **Connection:** `ssh -p {container['ssh_port']} cli@{container['server_ip']}`
                """)

            # SSH Key input
            st.markdown("#### Enter SSH Public Key")

            ssh_key = st.text_area(
                "SSH Public Key",
                placeholder="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExample...",
                height=100,
                key="ssh_key_input",
                help="Paste the user's SSH public key here (format: ssh-ed25519 AAAA...)"
            )

            # Key format info
            with st.expander("How to get SSH Public Key"):
                st.markdown("""
                **For Users:**
                1. Run: `python -m toolboxv2.Docksh.docksh setup`
                2. Copy the displayed public key
                3. Send it to the admin
                4. Admin adds it here

                **Supported Key Types:**
                - `ssh-ed25519` (recommended)
                - `ssh-rsa`
                - `ecdsa-sha2-nistp256`
                """)

            # Add key button
            if st.button("Add SSH Key", type="primary", key="add_ssh_key_btn"):
                if not ssh_key.strip():
                    st.error("Please enter an SSH public key")
                    return

                # Validate key format (basic check)
                if not ssh_key.strip().startswith(("ssh-ed25519", "ssh-rsa", "ecdsa-sha2")):
                    st.error("Invalid SSH key format. Key should start with 'ssh-ed25519', 'ssh-rsa', or 'ecdsa-sha2'")
                    return

                # Add the key
                with st.spinner("Adding SSH key to container..."):
                    result = asyncio.run(add_ssh_key_to_container(
                        app=app,
                        container_id=container_id,
                        ssh_public_key=ssh_key.strip(),
                        admin_key=ADMIN_KEY
                    ))

                if result.is_ok():
                    data = result.get()
                    st.success(f"SSH key added successfully!")

                    st.markdown("#### Connection Information for User:")
                    st.code(f"ssh -p {data['ssh_port']} cli@{data['server_ip']}", language="bash")

                    st.markdown("""
                    **Next Steps for User:**
                    1. Save the connection command above
                    2. Connect using: `python -m toolboxv2.Docksh.docksh connect <server-ip> <ssh-port>`
                    3. Or directly: `ssh -p <port> cli@<server-ip>`
                    """)
                else:
                    st.error(f"Failed to add SSH key: {result.info}")

    with col2:
        st.markdown("### SSH-Enabled Containers")

        for container in containers:
            with st.expander(
                f"{container['container_id'][:12]} - {container['container_name']}",
                expanded=False
            ):
                st.markdown(f"""
                **User:** `{container['user_id']}`
                **Type:** `{container['container_type']}`
                **SSH Port:** `{container['ssh_port']}`
                **Status:** {render_status_badge(container.get('status', 'unknown'))}
                """)

                st.markdown("**Connection String:**")
                st.code(container['connection_string'], language="bash")

                # Quick copy buttons
                if st.button(f"Copy", key=f"copy_{container['container_id']}", use_container_width=True):
                    st.toast("Connection string copied!")

    st.markdown("---")

    # Instructions section
    st.markdown("### Complete Workflow")

    tab1, tab2, tab3 = st.tabs(["User Setup", "Admin Action", "User Connect"])

    with tab1:
        st.markdown("""
        **Step 1: User generates SSH key**

        ```bash
        python -m toolboxv2.Docksh.docksh setup
        ```

        This will:
        - Generate a new Ed25519 key pair
        - Display the public key
        - Save the private key to `~/.ssh/docksh_id_ed25519`
        """)

    with tab2:
        st.markdown("""
        **Step 2: User sends public key to admin**

        The user copies the displayed public key and sends it to the administrator.

        **Step 3: Admin adds the key**

        1. Select the container from the dropdown above
        2. Paste the user's public key
        3. Click "Add SSH Key"
        4. Send the connection information to the user
        """)

    with tab3:
        st.markdown("""
        **Step 4: User connects**

        ```bash
        # Option 1: Using Docksh client
        python -m toolboxv2.Docksh.docksh connect <server-ip> <ssh-port>

        # Option 2: Direct SSH
        ssh -i ~/.ssh/docksh_id_ed25519 -p <ssh-port> cli@<server-ip>
        ```

        The user will be dropped into a persistent CLI session inside their container.
        """)


def page_settings():
    """Settings page"""
    render_header("Settings", "Configure Container Manager")

    st.markdown("### Admin Key")

    st.info(f"""
    Current Admin Key: `{'*' * 20}{ADMIN_KEY[-4:]}`

    To change the admin key, set the `CONTAINER_ADMIN_KEY` environment variable
    before starting the ToolBox.
    """)

    st.markdown("---")

    st.markdown("### Generate New Admin Key")
    if st.button("🔑 Generate New Admin Key"):
        app = get_app()
        result = asyncio.run(generate_admin_key(app=app))
        if result.is_ok():
            st.json(result.get())

    st.markdown("---")

    st.markdown("### Container Types")

    st.markdown("""
    | Type | Description | Internal Port | Default Image |
    |------|-------------|---------------|---------------|
    | `cli_v4` | Persistent CLI v4 | 8080 | toolboxv2:latest |
    | `project_dev` | Streamlit Dev UI | 8501 | toolboxv2:dev |
    | `preview_server` | HTTP Preview Server | 8600 | toolboxv2:latest |
    | `custom` | Custom configuration | 8080 | toolboxv2:latest |
    """)

    st.markdown("---")

    st.markdown("### Port Pool")
    st.info(f"""
    Port Range: `{9000}` - `{9500}`

    Containers are automatically assigned ports from this pool.
    Used ports are tracked in TBEF.DB under `CONTAINER_PORT_POOL`.
    """)

    st.markdown("---")

    st.markdown("### Nginx Integration")
    try:
        nginx_status = "✅ Active" if subprocess.run(["pgrep", "nginx"], stdout=subprocess.DEVNULL).returncode == 0 else "⚠️ Not Available (Linux only)"
    except:
        nginx_status = "⚠️ Not Available"
    st.info(f"""
    Nginx Status: {nginx_status}

    Container configs are written to:
    - `/etc/nginx/box-available/container-<user>-<type>.conf`
    - Symlinked from `/etc/nginx/box-enabled/`

    Containers are accessible at:
    - `http://your-server/container/<user_id>/<container_type>/`
    """)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Check auth first
    if not check_auth():
        return

    # Sidebar
    with st.sidebar:
        st.markdown(f"# Container Manager")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Dashboard", "Create", "Users", "SSH Keys", "Settings"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        st.markdown("### Admin")
        if st.button("Logout"):
            logout()

        st.markdown("---")
        st.markdown(f"""
        <div style="font-size: 0.8rem; color: rgba(255,255,255,0.5);">
            Container Manager v1.0.0
        </div>
        """, unsafe_allow_html=True)

    # Render selected page
    if page == "Dashboard":
        page_dashboard()
    elif page == "Create":
        page_create()
    elif page == "Users":
        page_users()
    elif page == "SSH Keys":
        page_ssh_keys()
    elif page == "Settings":
        page_settings()


if __name__ == "__main__":
    main()
