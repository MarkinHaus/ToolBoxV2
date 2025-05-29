# toolboxv2/mods/CloudM/AdminDashboard.py

import uuid
from dataclasses import asdict
import json

from toolboxv2 import TBEF, App, Result, get_app, RequestData
from toolboxv2.mods.CloudM.AuthManager import db_helper_save_user, db_helper_delete_user, get_user_by_name, \
    db_helper_test_exist
from toolboxv2.mods.CloudM import mini
from toolboxv2.mods.CloudM.ModManager import list_modules as list_all_modules
from .types import User
from .UserAccountManager import get_current_user_from_request
# For Waiting List invites, we'll call CloudM.email_services.send_signup_invitation_email
from .email_services import send_signup_invitation_email

Name = 'CloudM.AdminDashboard'
export = get_app(Name + ".Export").tb
version = '0.1.1'  # Incremented version

PID_DIR = "./.info"  # Standardized PID directory


async def _is_admin(app: App, request: RequestData) -> User | None:
    current_user = await get_current_user_from_request(app, request)
    if not current_user:  # Level 0 is admin
        return None
    if current_user.name == 'root' or current_user.name == 'loot':
        return current_user
    return None


@export(mod_name=Name, api=True, version=version, name="main", api_methods=['GET'], request_as_kwarg=True)
async def get_dashboard_main_page(app: App, request: RequestData):
    admin_user = await _is_admin(app, request)
    if not admin_user:
        return Result.html("<h1>Access Denied</h1><p>You do not have permission to view this page.</p>",
                           status=403)

    # Main HTML structure for the admin dashboard
    html_content =  """<div>
<style>
/* Refactored styles for Admin Dashboard Page, based on tbjs-main.css principles */

body {
    margin: 0;
    font-family: var(--font-family-base);
    background-color: var(--theme-bg);
    color: var(--theme-text);
    transition: background-color var(--transition-medium), color var(--transition-medium);
}

#admin-dashboard {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
}

#admin-header {
    /* Using a distinct dark accent for admin header, different from user-facing primary */
    background-color: var(--dark-acent); /* e.g., #011b33 from main styles */
    color: var(--anti-text-clor);       /* e.g., #fff from main styles */
    padding: var(--spacing) calc(var(--spacing) * 1.5);
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 4px color-mix(in srgb, var(--theme-text) 10%, transparent); /* Adapting shadow */
}

#admin-header h1 {
    margin: 0;
    font-size: var(--font-size-xl);
    font-weight: var(--font-weight-semibold);
    display: flex;
    align-items: center;
}

#admin-header h1 .material-symbols-outlined {
    vertical-align: middle;
    font-size: 1.5em;
    margin-right: 0.3em;
}

#admin-header .header-actions {
    display: flex;
    align-items: center;
}

#admin-nav ul {
    list-style: none;
    padding: 0;
    margin: 0 0 0 calc(var(--spacing) * 1.5);
    display: flex;
}

#admin-nav li {
    margin-left: var(--spacing);
    cursor: pointer;
    padding: calc(var(--spacing) * 0.6) var(--spacing);
    border-radius: var(--radius-sm);
    transition: background-color var(--transition-fast);
    font-weight: var(--font-weight-medium);
    display: flex;
    align-items: center;
}

#admin-nav li .material-symbols-outlined {
    vertical-align: text-bottom;
    margin-right: 0.3em;
}

#admin-nav li:hover {
    /* Subtle hover on the dark header, using the light text color from header */
    background-color: color-mix(in srgb, var(--anti-text-clor) 15%, transparent);
}

#admin-container {
    display: flex;
    flex-grow: 1;
}

#admin-sidebar {
    width: 240px;
    background-color: var(--input-bg); /* Good for side panels, adapts to theme */
    padding: calc(var(--spacing) * 1.5) var(--spacing);
    border-right: 1px solid var(--theme-border);
    box-shadow: 1px 0 3px color-mix(in srgb, var(--theme-text) 5%, transparent);
    transition: background-color var(--transition-medium), border-color var(--transition-medium);
    /* Responsive behavior handled by @media query below */
}
/* Removed body[data-theme="dark"] #admin-sidebar as --input-bg and --theme-border handle themes */

#admin-sidebar ul {
    list-style: none;
    padding: 0;
    margin: 0;
}

#admin-sidebar li {
    padding: calc(var(--spacing) * 0.9) var(--spacing);
    margin-bottom: calc(var(--spacing) * 0.6);
    cursor: pointer;
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    font-weight: var(--font-weight-medium);
    color: var(--theme-text); /* Sidebar items use main text color */
    transition: background-color var(--transition-fast), color var(--transition-fast);
}
/* Removed body[data-theme="dark"] #admin-sidebar li */

#admin-sidebar li .material-symbols-outlined {
    margin-right: calc(var(--spacing) * 0.85);
    font-size: 1.4rem;
}

#admin-sidebar li:hover {
    background-color: color-mix(in srgb, var(--theme-primary) 15%, transparent);
    color: var(--theme-primary);
}
/* Removed body[data-theme="dark"] #admin-sidebar li:hover */

#admin-sidebar li.active {
    background-color: var(--theme-primary);
    color: var(--theme-text-on-primary) !important;
    font-weight: var(--font-weight-semibold);
    box-shadow: 0 2px 8px color-mix(in srgb, var(--theme-primary) 30%, transparent);
}
/* Removed body[data-theme="dark"] #admin-sidebar li.active */

#admin-content {
    flex-grow: 1;
    padding: calc(var(--spacing) * 2);
    overflow-y: auto; /* Ensure content within can scroll if it overflows */
}

.content-section {
    display: none;
}

.content-section.active {
    display: block;
    animation: fadeIn 0.5s ease-out;
}

@keyframes fadeIn { /* Kept original keyframes */
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.content-section h2 {
    font-size: var(--font-size-3xl);
    font-weight: var(--font-weight-semibold);
    color: var(--theme-text);
    margin-bottom: calc(var(--spacing) * 1.8);
    padding-bottom: var(--spacing);
    border-bottom: 1px solid var(--theme-border);
    display: flex;
    align-items: center;
}

.content-section h2 .material-symbols-outlined {
    font-size: 1.3em;
    margin-right: 0.5em;
}
/* Removed body[data-theme="dark"] .content-section h2 */

.content-section h3 {
    font-size: var(--font-size-xl); /* Mapped 1.5rem to xl */
    font-weight: var(--font-weight-medium);
    margin-top: calc(var(--spacing) * 2);
    margin-bottom: var(--spacing);
    color: var(--theme-text);
}
/* Removed body[data-theme="dark"] .content-section h3 */

table {
    width: 100%;
    border-collapse: collapse; /* Main theme sets border-collapse */
    margin-top: calc(var(--spacing) * 1.5);
    font-size: var(--font-size-sm); /* Main theme sets table font-size */
    box-shadow: 0 1px 3px color-mix(in srgb, var(--theme-text) 5%, transparent);
    border-radius: var(--radius-md);
    overflow: hidden; /* For border-radius to apply to table contents properly */
    border: 1px solid var(--theme-border); /* Main theme sets this */
}

th, td {
    /* Padding and text-align from main theme or override here if different */
    padding: calc(var(--spacing) * 0.75) var(--spacing); /* 12px, 15px. Using 0.75 & 1 for approximation */
    text-align: left; /* Main theme sets this */
    border: 1px solid var(--theme-border); /* Main theme sets this */
}
/* Removed body[data-theme="dark"] th, body[data-theme="dark"] td */

th {
    /* Background, font-weight from main theme or override */
    background-color: color-mix(in srgb, var(--theme-text) 3%, transparent); /* Main theme th bg */
    font-weight: var(--font-weight-semibold); /* Main theme th font-weight */
}
/* Removed body[data-theme="dark"] th */

.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
}

.status-green { background-color: var(--color-success); }
.status-yellow { background-color: var(--color-warning); }
.status-red { background-color: var(--color-error); }

.action-btn {
    padding: calc(var(--spacing) * 0.5) calc(var(--spacing) * 0.9); /* 8px, 15px roughly */
    margin: calc(var(--spacing) * 0.25);
    border: none; /* Main theme button has border: 1px solid transparent, this is fine */
    border-radius: var(--radius-sm); /* 6px, sm is 4px. md is 8px. Keep sm for smaller buttons */
    cursor: pointer;
    font-size: var(--font-size-sm); /* 0.875rem */
    font-weight: var(--font-weight-medium);
    transition: background-color var(--transition-fast), transform var(--transition-fast), box-shadow var(--transition-fast), color var(--transition-fast);
    display: inline-flex;
    align-items: center;
    line-height: var(--line-height-tight); /* Ensure icon and text align well */
}

.action-btn .material-symbols-outlined {
    margin-right: 6px;
    font-size: 1.2em;
}

.action-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 5px color-mix(in srgb, var(--theme-text) 10%, transparent);
}

.action-btn:active {
    transform: translateY(0px);
    box-shadow: inset 0 1px 3px color-mix(in srgb, var(--theme-text) 10%, transparent);
}

.btn-restart { background-color: var(--color-warning); color: var(--theme-text); /* Black on yellow is often readable */ }
.btn-restart:hover { background-color: color-mix(in srgb, var(--color-warning) 85%, black 15%); }

.btn-edit { background-color: var(--color-info); color: var(--theme-text-on-primary); }
.btn-edit:hover { background-color: color-mix(in srgb, var(--color-info) 85%, black 15%); }

.btn-delete { background-color: var(--color-error); color: var(--theme-text-on-primary); }
.btn-delete:hover { background-color: color-mix(in srgb, var(--color-error) 85%, black 15%); }

.btn-send-invite { background-color: var(--color-success); color: var(--theme-text-on-primary); }
.btn-send-invite:hover { background-color: color-mix(in srgb, var(--color-success) 85%, black 15%); }

.btn-open-link { background-color: var(--theme-secondary); color: var(--theme-text-on-primary); }
.btn-open-link:hover { background-color: color-mix(in srgb, var(--theme-secondary) 85%, black 15%); }


.frosted-glass-pane {
    background-color: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border-radius: var(--radius-lg);
    padding: calc(var(--spacing) * 2);
    border: 1px solid var(--glass-border);
    box-shadow: var(--glass-shadow);
}
/* Removed body[data-theme="dark"] .frosted-glass-pane */

.tb-input { /* Re-using from previous answer, ensure it's consistent */
    display: block;
    width: 100%;
    padding: calc(var(--spacing) * 0.75) var(--spacing);
    font-family: inherit;
    font-size: var(--font-size-base);
    line-height: var(--line-height-normal);
    color: var(--theme-text);
    background-color: var(--input-bg);
    background-clip: padding-box;
    border: 1px solid var(--input-border);
    border-radius: var(--radius-md); /* Original was 6px, md is 8px */
    transition: border-color var(--transition-fast), box-shadow var(--transition-fast);
    appearance: none;
    box-sizing: border-box;
}
.tb-input:focus {
    outline: none;
    border-color: var(--input-focus-border);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--theme-primary) 25%, transparent);
}
/* Removed body[data-theme="dark"] .tb-input */

.tb-label {
    font-weight: var(--font-weight-medium);
    margin-bottom: calc(var(--spacing) * 0.5);
    display: block;
    font-size: var(--font-size-sm);
}

.tb-checkbox { /* Actual input element */
    /* Inherits from main input[type="checkbox"] styles */
    margin-right: calc(var(--spacing) * 0.5);
}

.tb-btn { /* Re-using from previous answer */
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: calc(var(--spacing) * 0.6) calc(var(--spacing) * 1.2);
    border-radius: var(--radius-md);
    font-weight: var(--font-weight-medium);
    cursor: pointer;
    transition: background-color var(--transition-fast), box-shadow var(--transition-fast), border-color var(--transition-fast), color var(--transition-fast);
    border: 1px solid transparent;
    text-align: center;
    vertical-align: middle;
    user-select: none;
}
.tb-btn:focus-visible {
    outline: none;
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--theme-primary) 35%, transparent);
}
.tb-btn .material-symbols-outlined {
    margin-right: 0.4em;
    font-size: 1.2em;
}

.tb-btn-primary {
    background-color: var(--button-bg);
    color: var(--button-text);
    border-color: var(--button-bg);
}
.tb-btn-primary:hover {
    background-color: var(--button-hover-bg);
    border-color: var(--button-hover-bg);
}

.tb-space-y-6 > *:not([hidden]) ~ *:not([hidden]) { margin-top: calc(var(--spacing) * 1.5); }
.tb-mt-2 { margin-top: calc(var(--spacing) * 0.5); }
.tb-mb-1 { margin-bottom: calc(var(--spacing) * 0.25); }
.tb-mb-2 { margin-bottom: calc(var(--spacing) * 0.5); }
.tb-mt-6 { margin-top: calc(var(--spacing) * 1.5); }
.tb-mr-1 { margin-right: calc(var(--spacing) * 0.25); }

.md\\:tb-w-2\\/3 { width: 66.666667%; } /* Kept as is */

.tb-text-red-500 { color: var(--color-error); }
.tb-text-green-500 { color: var(--color-success); } /* Assuming 500 maps to base success */
.tb-text-yellow-500 { color: var(--color-warning); } /* Assuming 500 maps to base warning */
.tb-text-gray-500 { color: var(--theme-text-muted); }
/* Removed body[data-theme="dark"] .tb-text-gray-500 */

.tb-text-sm { font-size: var(--font-size-sm); }
.tb-text-md { font-size: var(--font-size-base); }
.tb-text-lg { font-size: var(--font-size-lg); }
.tb-font-semibold { font-weight: var(--font-weight-semibold); }
.tb-flex { display: flex; }
.tb-items-center { align-items: center; }
.tb-cursor-pointer { cursor: pointer; }

/* This was globally hidden in your provided styles for admin. If a toggle button is needed for mobile,
   its styling would be similar to #sidebar-toggle-btn in the user dashboard example,
   and this global hide would be removed or scoped to desktop. */
#sidebar-toggle-btn {
    display: none;
}

@media (max-width: 767.98px) {
    #admin-sidebar {
        position: fixed;
        left: -250px; /* Start off-screen */
        top: 0;
        bottom: 0;
        width: 240px;
        z-index: var(--z-modal); /* Above backdrop */
        transition: left var(--transition-medium);
        overflow-y: auto;
        /* background-color and border-right will use the default #admin-sidebar styles which are theme-aware */
    }
    /* Removed body[data-theme="dark"] #admin-sidebar as it's covered by main styles */

    #admin-sidebar.open {
        left: 0; /* Slide in */
        box-shadow: 2px 0 10px color-mix(in srgb, var(--theme-text) 20%, transparent);
    }

    #sidebar-backdrop-admin {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: color-mix(in srgb, var(--theme-bg) 50%, black 50%);
        opacity: 0; /* Start transparent */
        z-index: calc(var(--z-modal) - 1); /* Below sidebar */
        transition: opacity var(--transition-medium);
    }

    #sidebar-backdrop-admin.active {
        display: block;
        opacity: 0.7; /* Fade in to 70% opacity */
    }
    /* If #sidebar-toggle-btn IS used on mobile for admin, its display:block/inline-flex would go here */
}
         @media (max-width: 767.98px) {
            #admin-sidebar { position: fixed; left: -250px; top: 0; bottom: 0; width: 240px; z-index: 1000; transition: left 0.3s ease-in-out; overflow-y: auto; background-color: var(--sidebar-bg, var(--tb-color-neutral-100, #ffffff)); border-right: 1px solid var(--sidebar-border, var(--tb-color-neutral-300, #e0e0e0)); }
            body[data-theme="dark"] #admin-sidebar { background-color: var(--sidebar-bg-dark, var(--tb-color-neutral-850, #232b33)); border-right-color: var(--sidebar-border-dark, var(--tb-color-neutral-700, #374151));}
            #admin-sidebar.open { left: 0; box-shadow: 2px 0 10px rgba(0,0,0,0.2); }
            #sidebar-backdrop-admin { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.5); z-index: 999; opacity: 0; transition: opacity 0.3s ease-in-out; }
            #sidebar-backdrop-admin.active { display: block; opacity: 1; }
        }
    </style>
</head>
<body data-theme="system">
    <div id="admin-dashboard">
        <div id="admin-header">

            <h1><span class="material-symbols-outlined">shield_person</span>CloudM Admin</h1>
            <button id="sidebar-toggle-btn" class="tb-btn" style="margin-right: 1rem; background: none; border: none; color: white;">
                <span class="material-symbols-outlined">menu</span>
            </button>
            <div class="header-actions">
                 <div id="darkModeToggleContainer" style="display: inline-flex; align-items: center; margin-right: 1.5rem;"></div>
                <nav id="admin-nav">
                    <ul>
                        <li id="logoutButton"><span class="material-symbols-outlined">logout</span>Logout</li>
                    </ul>
                </nav>
            </div>
        </div>
        <div id="admin-container">
            <aside id="admin-sidebar">
                 <ul>
                    <li data-section="system-status" class="active"><span class="material-symbols-outlined">monitoring</span>System Status</li>
                    <li data-section="user-management"><span class="material-symbols-outlined">group</span>User Management</li>
                    <li data-section="module-management"><span class="material-symbols-outlined">extension</span>Module Management</li>
                    <li data-section="spp-management"><span class="material-symbols-outlined">web_stories</span>SPP Management</li>
                    <li data-section="my-account"><span class="material-symbols-outlined">manage_accounts</span>My Account</li>
                </ul>
            </aside>
            <main id="admin-content">
                <section id="system-status-section" class="content-section active frosted-glass-pane">
                    <h2><span class="material-symbols-outlined">bar_chart_4_bars</span>System Status</h2>
                    <div id="system-status-content"><p class="tb-text-gray-500">Loading system status...</p></div>
                </section>
                <section id="user-management-section" class="content-section frosted-glass-pane">
                    <h2><span class="material-symbols-outlined">manage_history</span>User Management</h2>
                    <div id="user-management-content-main"><p class="tb-text-gray-500">Loading user data...</p></div>
                    <h3 class="tb-mt-6"><span class="material-symbols-outlined">person_add</span>Users on Waiting List</h3>
                    <div id="user-waiting-list-content"><p class="tb-text-gray-500">Loading waiting list...</p></div>
                </section>
                <section id="module-management-section" class="content-section frosted-glass-pane">
                    <h2><span class="material-symbols-outlined">view_module</span>Module Management</h2>
                    <div id="module-management-content"><p class="tb-text-gray-500">Loading module list...</p></div>
                </section>
                <section id="spp-management-section" class="content-section frosted-glass-pane">
                    <h2><span class="material-symbols-outlined">apps</span>Registered SPPs / UI Panels</h2>
                    <div id="spp-management-content"><p class="tb-text-gray-500">Loading registered SPPs...</p></div>
                </section>
                <section id="my-account-section" class="content-section frosted-glass-pane">
                    <h2><span class="material-symbols-outlined">account_circle</span>My Account Settings</h2>
                    <div id="my-account-content"><p class="tb-text-gray-500">Loading account details...</p></div>
                </section>
            </main>
        </div>
        <div id="sidebar-backdrop-admin"></div>
    </div>

    <script type="module">
        if (typeof TB === 'undefined' || !TB.ui || !TB.api || !TB.user || !TB.utils) {
            console.error('CRITICAL: TB (tbjs) or its core modules are not defined.');
            document.body.innerHTML = '<div style="padding:20px; text-align:center; font-size:1.2em; color:red;">Critical Error: Frontend library (tbjs) failed to load.</div>';
        } else {
            console.log('TB object found. Initializing Admin Dashboard.');
            let currentAdminUser = null;

            function _waitForTbInitAdmin(callback) {
                if (window.TB?.events && window.TB.config?.get('appRootId')) {
                    callback();
                } else {
                    document.addEventListener('tbjs:initialized', callback, { once: true });
                }
            }

            async function initializeAdminDashboard() {
                console.log("Admin Dashboard Initializing with tbjs...");
                TB.ui.DarkModeToggle.init();
                setupNavigation();
                await setupLogout();
                setupMobileSidebarAdmin();

                try {
                    const userRes = await TB.api.request('CloudM.UserAccountManager', 'get_current_user_from_request_api_wrapper', null, 'GET');
                    if (userRes.error === TB.ToolBoxError.none && userRes.get()) {
                        currentAdminUser = userRes.get();
                        if (currentAdminUser.name) {
                            const adminTitleElement = document.querySelector('#admin-header h1');
                            if (adminTitleElement) {
                                adminTitleElement.innerHTML = `<span class="material-symbols-outlined">shield_person</span>CloudM Admin (${TB.utils.escapeHtml(currentAdminUser.name)})`;
                            }
                        }
                        await loadMyAccountSection();
                        await showSection('system-status');
                    } else {
                        console.error("Failed to load current admin user:", userRes.info.help_text);
                        document.getElementById('admin-content').innerHTML = '<p class="tb-text-red-500">Error: Could not verify admin user. Please login.</p>';
                    }
                } catch (e) {
                    console.error("Error fetching current admin user:", e);
                    document.getElementById('admin-content').innerHTML = '<p class="tb-text-red-500">Network error verifying admin user.</p>';
                }
            }
            _waitForTbInitAdmin(initializeAdminDashboard);

            function setupMobileSidebarAdmin() {
                // ... (Implementation from previous response, ensure it correctly uses 'admin-sidebar' and 'sidebar-backdrop-admin')
                const sidebar = document.getElementById('admin-sidebar');
                const toggleBtn = document.getElementById('sidebar-toggle-btn');
                const backdrop = document.getElementById('sidebar-backdrop-admin');

                if (!sidebar || !toggleBtn || !backdrop) {
                    console.warn("Admin mobile sidebar elements not found. Mobile navigation might not work.");
                    if(toggleBtn) toggleBtn.style.display = 'none'; // Ensure it's hidden if setup fails
                    return;
                }

                function updateToggleBtnVisibility() {
                    toggleBtn.style.display = window.innerWidth < 768 ? 'inline-flex' : 'none';
                     if (window.innerWidth >= 768 && sidebar.classList.contains('open')) {
                        sidebar.classList.remove('open');
                        backdrop.classList.remove('active');
                        document.body.style.overflow = '';
                    }
                }
                updateToggleBtnVisibility();
                window.addEventListener('resize', updateToggleBtnVisibility);

                toggleBtn.addEventListener('click', (e) => {
                    e.stopPropagation(); // Important to prevent unintended closes if backdrop is also under button somehow
                    sidebar.classList.toggle('open');
                    backdrop.classList.toggle('active');
                    document.body.style.overflow = sidebar.classList.contains('open') ? 'hidden' : '';
                });
                backdrop.addEventListener('click', () => {
                    sidebar.classList.remove('open');
                    backdrop.classList.remove('active');
                    document.body.style.overflow = '';
                });
                // Also close sidebar when a nav item is clicked on mobile
                sidebar.querySelectorAll('li[data-section]').forEach(item => {
                    item.addEventListener('click', () => {
                        if (window.innerWidth < 768 && sidebar.classList.contains('open')) {
                            sidebar.classList.remove('open');
                            backdrop.classList.remove('active');
                            document.body.style.overflow = '';
                        }
                    });
                });
                console.log("Admin mobile sidebar setup complete.");
            }

             // Using event delegation for dynamically added buttons inside content sections
            document.getElementById('admin-content').addEventListener('click', async function(event) {
                const target = event.target.closest('button.action-btn'); // Find closest action button
                if (!target) return; // Not an action button click

                console.log("Action button clicked:", target.dataset);

                // System Status Restart Button
                if (target.classList.contains('btn-restart') && target.dataset.service) {
                    const serviceName = target.dataset.service;
                    TB.ui.Toast.showInfo(`Restart for ${serviceName} (placeholder).`);
                }
                // User Management Edit/Delete Buttons
                else if (target.dataset.uid && target.classList.contains('btn-edit')) {
                    const usersData = JSON.parse(target.closest('table').dataset.users || '[]'); // Requires storing users data on table
                    showUserEditModal(target.dataset.uid, usersData);
                }
                else if (target.dataset.uid && target.classList.contains('btn-delete')) {
                    handleDeleteUser(target.dataset.uid, target.dataset.name);
                }
                // Waiting List Buttons
                else if (target.dataset.email && target.classList.contains('btn-send-invite')) {
                    const email = target.dataset.email;
                    const proposedUsername = prompt(`Enter a proposed username for ${email} (e.g., derived from email prefix):`, email.split('@')[0]);
                    if (!proposedUsername) { TB.ui.Toast.showWarning("Username is required."); return; }
                    TB.ui.Loader.show(`Sending invite...`);
                    try {
                        const inviteRes = await TB.api.request('CloudM.AdminDashboard', 'send_invite_to_waiting_list_user_admin', { email, username: proposedUsername }, 'POST');
                        if (inviteRes.error === TB.ToolBoxError.none) {
                            TB.ui.Toast.showSuccess(inviteRes.info.help_text || `Invite sent.`);
                            await loadWaitingListUsers('user-waiting-list-content');
                        } else { TB.ui.Toast.showError(`Invite failed: ${TB.utils.escapeHtml(inviteRes.info.help_text)}`); }
                    } catch (err) { TB.ui.Toast.showError("Network error."); }
                    finally { TB.ui.Loader.hide(); }
                }
                else if (target.dataset.email && target.classList.contains('btn-delete')) { // Waiting list remove
                    const email = target.dataset.email;
                    if (!confirm(`Remove ${email} from waiting list?`)) return;
                    TB.ui.Loader.show(`Removing...`);
                    try {
                        const removeRes = await TB.api.request('CloudM.AdminDashboard', 'remove_from_waiting_list_admin', { email }, 'POST');
                         if (removeRes.error === TB.ToolBoxError.none) {
                            TB.ui.Toast.showSuccess(`${email} removed.`);
                            await loadWaitingListUsers('user-waiting-list-content');
                        } else { TB.ui.Toast.showError(`Failed: ${TB.utils.escapeHtml(removeRes.info.help_text)}`); }
                    } catch (err) { TB.ui.Toast.showError("Network error."); }
                    finally { TB.ui.Loader.hide(); }
                }
                // Module Management Reload Button
                else if (target.dataset.module && target.classList.contains('btn-restart')) {
                     const modName = target.dataset.module;
                    TB.ui.Toast.showInfo(`Reloading ${modName}...`);
                    TB.ui.Loader.show(`Reloading ${modName}...`);
                    try {
                        const res = await TB.api.request('CloudM.AdminDashboard', 'reload_module_admin', { module_name: modName }, 'POST');
                        if (res.error === TB.ToolBoxError.none) { TB.ui.Toast.showSuccess(`${modName} reload: ${TB.utils.escapeHtml(res.get() || 'OK')}`); }
                        else { TB.ui.Toast.showError(`Error reloading ${modName}: ${TB.utils.escapeHtml(res.info.help_text)}`); }
                    } catch (err) { TB.ui.Toast.showError('Network error.'); }
                    finally { TB.ui.Loader.hide(); }
                }
                // SPP Management Open Link Button
                else if (target.dataset.path && target.classList.contains('btn-open-link')) {
                    const path = target.dataset.path;
                    if (path.startsWith("http") || path.startsWith("/api/")) { window.open(path, '_blank'); }
                    else { TB.router.navigateTo(path); }
                }
            });

            // Navigation, ShowSection, Logout Setup (Functions from previous responses, ensure they are complete)
            function setupNavigation() {
                const navItems = document.querySelectorAll('#admin-sidebar li[data-section]');
                navItems.forEach(item => {
                    item.addEventListener('click', async () => {
                        navItems.forEach(i => i.classList.remove('active'));
                        item.classList.add('active');
                        await showSection(item.getAttribute('data-section'));
                    });
                });
            }

                        async function showSection(sectionId) {
                console.log(`Showing section: ${sectionId}`);
                document.querySelectorAll('#admin-content .content-section').forEach(s => s.classList.remove('active'));
                const activeSectionElement = document.getElementById(`${sectionId}-section`); // The <section> element

                if (activeSectionElement) {
                    activeSectionElement.classList.add('active');

                    // Handle content loading for each specific section
                    if (sectionId === 'system-status') {
                        const contentDiv = document.getElementById('system-status-content');
                        if (contentDiv) {
                            contentDiv.innerHTML = `<p class="tb-text-gray-500">Loading system status...</p>`;
                            await loadSystemStatus('system-status-content');
                        } else { console.error('Content div for system-status not found.'); }
                    }
                    else if (sectionId === 'user-management') {
                        // This section has two distinct content areas
                        const mainUsersContentDiv = document.getElementById('user-management-content-main');
                        const waitingListContentDiv = document.getElementById('user-waiting-list-content');

                        if (mainUsersContentDiv) {
                            mainUsersContentDiv.innerHTML = `<p class="tb-text-gray-500">Loading user data...</p>`;
                            await loadUserManagement('user-management-content-main');
                        } else { console.error('Content div for main user management not found.'); }

                        if (waitingListContentDiv) {
                            waitingListContentDiv.innerHTML = `<p class="tb-text-gray-500">Loading waiting list...</p>`;
                            await loadWaitingListUsers('user-waiting-list-content');
                        } else { console.error('Content div for user waiting list not found.'); }
                    }
                    else if (sectionId === 'module-management') {
                        const contentDiv = document.getElementById('module-management-content');
                        if (contentDiv) {
                            contentDiv.innerHTML = `<p class="tb-text-gray-500">Loading module list...</p>`;
                            await loadModuleManagement('module-management-content');
                        } else { console.error('Content div for module-management not found.'); }
                    }
                    else if (sectionId === 'spp-management') {
                        const contentDiv = document.getElementById('spp-management-content');
                        if (contentDiv) {
                            contentDiv.innerHTML = `<p class="tb-text-gray-500">Loading registered SPPs...</p>`;
                            await loadSppManagement('spp-management-content');
                        } else { console.error('Content div for spp-management not found.'); }
                    }
                    else if (sectionId === 'my-account') {
                        const contentDiv = document.getElementById('my-account-content');
                        if (contentDiv) {
                            contentDiv.innerHTML = `<p class="tb-text-gray-500">Loading account details...</p>`;
                            await loadMyAccountSection('my-account-content');
                        } else { console.error('Content div for my-account not found.'); }
                    }
                    console.log(`Section ${sectionId} content loading initiated or already handled.`);
                } else {
                    console.error(`Section element ${sectionId}-section not found.`);
                }
            }

            async function setupLogout() {
                document.getElementById('logoutButton')?.addEventListener('click', async () => {
                    TB.ui.Loader.show("Logging out...");
                    await TB.user.logout();
                    window.location.href = '/';
                    TB.ui.Loader.hide();
                });
            }

            async function loadSystemStatus(targetDivId) {
                const contentDiv = document.getElementById(targetDivId);
                if (!contentDiv) return;
                try {
                    const response = await TB.api.request('CloudM.AdminDashboard', 'get_system_status', null, 'GET');
                    if (response.error === TB.ToolBoxError.none) {
                        renderSystemStatus(response.get(), contentDiv);
                    } else {
                        contentDiv.innerHTML = `<p class="tb-text-red-500">Error: ${TB.utils.escapeHtml(response.info.help_text)}</p>`;
                    }
                } catch (e) { contentDiv.innerHTML = '<p class="tb-text-red-500">Network error.</p>'; console.error(e); }
            }

            function renderSystemStatus(statusData, contentDiv) {
                if (!contentDiv) return;
                if (!statusData || Object.keys(statusData).length === 0 || (Object.keys(statusData).length === 1 && statusData["unknown_service_format"])) {
                     if (statusData && statusData["unknown_service_format"]) {
                        contentDiv.innerHTML = `<p class="tb-text-yellow-500">Service status format error: ${TB.utils.escapeHtml(statusData["unknown_service_format"].details)}</p>`;
                     } else { contentDiv.innerHTML = '<p class="tb-text-gray-500">No services found or status unavailable.</p>';}
                    return;
                }
                let html = '<table><thead><tr><th>Service</th><th>Status</th><th>PID</th><th>Actions</th></tr></thead><tbody>';
                for (const [name, data] of Object.entries(statusData)) {
                    let sClass = data.status_indicator === '🟢' ? 'status-green' : (data.status_indicator === '🔴' ? 'status-red' : 'status-yellow');
                    html += `<tr><td>${TB.utils.escapeHtml(name)}</td><td><span class="status-indicator ${sClass}"></span> ${data.status_indicator}</td><td>${TB.utils.escapeHtml(data.pid)}</td><td><button class="action-btn btn-restart" data-service="${TB.utils.escapeHtml(name)}"><span class="material-symbols-outlined">restart_alt</span>Restart</button></td></tr>`;
                }
                html += '</tbody></table>';
                contentDiv.innerHTML = html;
                contentDiv.querySelectorAll('.btn-restart').forEach(btn => btn.addEventListener('click', e => TB.ui.Toast.showInfo(`Restart for ${e.currentTarget.dataset.service} (placeholder).`)));
            }

             async function loadUserManagement(targetDivId = 'user-management-content-main') {
                const contentDiv = document.getElementById(targetDivId);
                if (!contentDiv) return;
                try {
                    const response = await TB.api.request('CloudM.AdminDashboard', 'list_users_admin', null, 'GET');
                    if (response.error === TB.ToolBoxError.none) {
                        renderUserManagement(response.get(), contentDiv);
                    } else { contentDiv.innerHTML = `<p class="tb-text-red-500">Error: ${TB.utils.escapeHtml(response.info.help_text)}</p>`; }
                } catch (e) { contentDiv.innerHTML = '<p class="tb-text-red-500">Network error.</p>'; console.error(e); }
            }
            function renderUserManagement(users, contentDiv) {
                if (!contentDiv) return;
                if (!users || users.length === 0) { contentDiv.innerHTML = '<p class="tb-text-gray-500">No users.</p>'; return; }
                // Store users data on the table for easier access by modal
                let tableHtml = `<table data-users='${TB.utils.escapeHtml(JSON.stringify(users))}'><thead><tr><th>Name</th><th>Email</th><th>Level</th><th>UID</th><th>Actions</th></tr></thead><tbody>`;
                users.forEach(user => {
                    tableHtml += `<tr><td>${TB.utils.escapeHtml(user.name)}</td><td>${TB.utils.escapeHtml(user.email || 'N/A')}</td><td>${user.level} ${user.level === 0 ? '(Admin)' : ''}</td><td>${TB.utils.escapeHtml(user.uid)}</td><td><button class="action-btn btn-edit" data-uid="${user.uid}"><span class="material-symbols-outlined">edit</span>Edit</button>${(currentAdminUser && currentAdminUser.uid !== user.uid) ? `<button class="action-btn btn-delete" data-uid="${user.uid}" data-name="${TB.utils.escapeHtml(user.name)}"><span class="material-symbols-outlined">delete</span>Delete</button>` : ''}</td></tr>`;
                });
                tableHtml += '</tbody></table>';
                contentDiv.innerHTML = tableHtml;
            }

            function showUserEditModal(userId, allUsers) {
                const user = allUsers.find(u => u.uid === userId);
                if (!user) { TB.ui.Toast.showError("User not found for editing."); return; }
                console.log(`Showing edit modal for user:`, user);

                const modalContent = `
                    <form id="editUserFormAdmin" class="tb-space-y-4">
                        <input type="hidden" name="uid" value="${user.uid}">
                        <div><label class="tb-label" for="editUserNameAdminModal">Name:</label><input type="text" id="editUserNameAdminModal" name="name" class="tb-input" value="${TB.utils.escapeHtml(user.name)}" readonly></div>
                        <div><label class="tb-label" for="editUserEmailAdminModal">Email:</label><input type="email" id="editUserEmailAdminModal" name="email" class="tb-input" value="${TB.utils.escapeHtml(user.email || '')}"></div>
                        <div><label class="tb-label" for="editUserLevelAdminModal">Level:</label><input type="number" id="editUserLevelAdminModal" name="level" class="tb-input" value="${user.level}"></div>
                        <div><label class="tb-label tb-flex tb-items-center"><input type="checkbox" name="experimental_features" class="tb-checkbox tb-mr-2" ${user.settings && user.settings.experimental_features ? 'checked' : ''}>Experimental Features</label></div>
                    </form>`;

                TB.ui.Modal.show({
                    title: `Edit User: ${TB.utils.escapeHtml(user.name)}`,
                    content: modalContent,
                    buttons: [
                        { text: 'Cancel', action: modal => modal.close(), variant: 'secondary' },
                        {
                            text: 'Save Changes',
                            action: async modal => {
                                const form = document.getElementById('editUserFormAdmin');
                                if (!form) { console.error("Edit user form not found in modal."); return; }
                                const updatedData = {
                                    uid: form.uid.value,
                                    name: form.name.value,
                                    email: form.email.value,
                                    level: parseInt(form.level.value),
                                    settings: { experimental_features: form.experimental_features.checked }
                                };
                                console.log("Saving user data:", updatedData);
                                TB.ui.Loader.show('Saving user data...');
                                try {
                                    const resp = await TB.api.request('CloudM.AdminDashboard', 'update_user_admin', updatedData, 'POST');
                                    console.log("Update user response:", resp);
                                    if (resp.error === TB.ToolBoxError.none) {
                                        TB.ui.Toast.showSuccess('User updated successfully!');
                                        await loadUserManagement('user-management-content');
                                        modal.close();
                                    } else {
                                        TB.ui.Toast.showError(`Error updating user: ${TB.utils.escapeHtml(resp.info.help_text)}`);
                                    }
                                } catch (e) {
                                    TB.ui.Toast.showError('Network error while saving user.');
                                    console.error("Update user error:", e);
                                } finally {
                                    TB.ui.Loader.hide();
                                }
                            },
                            variant: 'primary'
                        }
                    ]
                });
            }

            async function handleDeleteUser(userId, userName) {
                if (currentAdminUser && currentAdminUser.uid === userId) {
                    TB.ui.Toast.showError("Administrators cannot delete their own account through this panel.");
                    return;
                }
                console.log(`Confirming delete for user: ${userName} (UID: ${userId})`);
                TB.ui.Modal.show({
                    title: 'Confirm Deletion',
                    content: `<p>Are you sure you want to delete user <strong>${TB.utils.escapeHtml(userName)}</strong> (UID: ${TB.utils.escapeHtml(userId)})? This action cannot be undone.</p>`,
                    buttons: [
                        { text: 'Cancel', action: m => m.close(), variant: 'secondary' },
                        {
                            text: 'Delete User',
                            variant: 'danger',
                            action: async m => {
                                console.log(`Deleting user: ${userId}`);
                                TB.ui.Loader.show('Deleting user...');
                                try {
                                    const resp = await TB.api.request('CloudM.AdminDashboard', 'delete_user_admin', { uid: userId }, 'POST');
                                    console.log("Delete user response:", resp);
                                    if (resp.error === TB.ToolBoxError.none) {
                                        TB.ui.Toast.showSuccess('User deleted successfully!');
                                        await loadUserManagement('user-management-content');
                                    } else {
                                        TB.ui.Toast.showError(`Error deleting user: ${TB.utils.escapeHtml(resp.info.help_text)}`);
                                    }
                                } catch (e) {
                                    TB.ui.Toast.showError('Network error while deleting user.');
                                    console.error("Delete user error:", e);
                                } finally {
                                    TB.ui.Loader.hide();
                                    m.close();
                                }
                            }
                        }
                    ]
                });
            }

             async function loadWaitingListUsers(targetDivId) {
                const contentDiv = document.getElementById(targetDivId);
                if (!contentDiv) return;
                contentDiv.innerHTML = '<p class="tb-text-gray-500">Fetching waiting list...</p>';
                try {
                    const response = await TB.api.request('CloudM.AdminDashboard', 'get_waiting_list_users_admin', null, 'GET');
                    if (response.error === TB.ToolBoxError.none) {
                        renderWaitingListUsers(response.get(), contentDiv);
                    } else {
                        contentDiv.innerHTML = `<p class="tb-text-red-500">Error: ${TB.utils.escapeHtml(response.info.help_text)}</p>`;
                    }
                } catch (e) { contentDiv.innerHTML = '<p class="tb-text-red-500">Network error.</p>'; console.error(e); }
            }

            function renderWaitingListUsers(waitingUsers, contentDiv) {
                if (!waitingUsers || waitingUsers.length === 0) {
                    contentDiv.innerHTML = '<p class="tb-text-gray-500">No users currently on the waiting list.</p>'; return;
                }
                let html = '<table><thead><tr><th>Email</th><th>Actions</th></tr></thead><tbody>';
                waitingUsers.forEach(entry => { // Assuming entry is just an email string for now
                    const email = (typeof entry === 'string') ? entry : (entry.email || 'Invalid Entry');
                    html += `<tr>
                        <td>${TB.utils.escapeHtml(email)}</td>
                        <td>
                            <button class="action-btn btn-send-invite" data-email="${TB.utils.escapeHtml(email)}"><span class="material-symbols-outlined">outgoing_mail</span>Send Invite</button>
                            <button class="action-btn btn-delete" data-email="${TB.utils.escapeHtml(email)}"><span class="material-symbols-outlined">person_remove</span>Remove</button>
                        </td>
                    </tr>`;
                });
                html += '</tbody></table>';
                contentDiv.innerHTML = html;

                contentDiv.querySelectorAll('.btn-send-invite').forEach(btn => {
                    btn.addEventListener('click', async (e) => {
                        const email = e.currentTarget.dataset.email;
                        const proposedUsername = prompt(`Enter a proposed username for ${email} (e.g., derived from email prefix):`, email.split('@')[0]);
                        if (!proposedUsername) { TB.ui.Toast.showWarning("Username is required to send an invite."); return; }

                        TB.ui.Loader.show(`Sending invite to ${email}...`);
                        try {
                            const inviteRes = await TB.api.request('CloudM.AdminDashboard', 'send_invite_to_waiting_list_user_admin',
                                { email: email, username: proposedUsername }, 'POST');
                            if (inviteRes.error === TB.ToolBoxError.none) {
                                TB.ui.Toast.showSuccess(inviteRes.info.help_text || `Invite sent to ${email}.`);
                                await loadWaitingListUsers('user-waiting-list-content'); // Refresh list
                            } else {
                                TB.ui.Toast.showError(`Failed to send invite: ${TB.utils.escapeHtml(inviteRes.info.help_text)}`);
                            }
                        } catch (err) { TB.ui.Toast.showError("Network error sending invite."); console.error(err); }
                        finally { TB.ui.Loader.hide(); }
                    });
                });
                contentDiv.querySelectorAll('.btn-delete').forEach(btn => {
                    btn.addEventListener('click', async (e) => {
                        const email = e.currentTarget.dataset.email;
                        if (!confirm(`Are you sure you want to remove ${email} from the waiting list?`)) return;

                        TB.ui.Loader.show(`Removing ${email}...`);
                        try {
                            const removeRes = await TB.api.request('CloudM.AdminDashboard', 'remove_from_waiting_list_admin', { email: email }, 'POST');
                             if (removeRes.error === TB.ToolBoxError.none) {
                                TB.ui.Toast.showSuccess(`${email} removed from waiting list.`);
                                await loadWaitingListUsers('user-waiting-list-content'); // Refresh list
                            } else {
                                TB.ui.Toast.showError(`Failed to remove: ${TB.utils.escapeHtml(removeRes.info.help_text)}`);
                            }
                        } catch (err) { TB.ui.Toast.showError("Network error removing from list."); console.error(err); }
                        finally { TB.ui.Loader.hide(); }
                    });
                });
            }

            async function loadModuleManagement(targetDivId) {
                const contentDiv = document.getElementById(targetDivId);
                 if (!contentDiv) return;
                try {
                    console.log("Requesting module list from API: CloudM.AdminDashboard/list_modules_admin");
                    const response = await TB.api.request('CloudM.AdminDashboard', 'list_modules_admin', null, 'GET');
                    console.log("Module list response:", response);
                    if (response.error === TB.ToolBoxError.none) {
                        renderModuleManagement(response.get(), contentDiv);
                    } else {
                        contentDiv.innerHTML = `<p class="tb-text-red-500">Error loading modules: ${TB.utils.escapeHtml(response.info.help_text)}</p>`;
                    }
                } catch (e) {
                    contentDiv.innerHTML = '<p class="tb-text-red-500">Network error while fetching modules.</p>';
                    console.error("loadModuleManagement error:", e);
                }
            }

            function renderModuleManagement(modules, contentDiv) {
                if (!contentDiv) return;
                console.log("Rendering module management:", modules);
                if (!modules || modules.length === 0) {
                    contentDiv.innerHTML = '<p class="tb-text-gray-500">No modules found or loaded in the system.</p>';
                    return;
                }
                let html = '<table><thead><tr><th>Module Name</th><th>Actions</th></tr></thead><tbody>';
                modules.forEach(modName => {
                    html += `<tr>
                        <td>${TB.utils.escapeHtml(modName)}</td>
                        <td><button class="action-btn btn-restart" data-module="${TB.utils.escapeHtml(modName)}"><span class="material-symbols-outlined">refresh</span>Reload</button></td>
                        </tr>`;
                });
                html += '</tbody></table>';
                contentDiv.innerHTML = html;
                contentDiv.querySelectorAll('.btn-restart').forEach(btn => {
                    btn.addEventListener('click', async e => {
                        const modName = e.currentTarget.dataset.module;
                        console.log(`Reload button clicked for module: ${modName}`);
                        TB.ui.Toast.showInfo(`Attempting to reload ${modName}...`);
                        TB.ui.Loader.show(`Reloading ${modName}...`);
                        try {
                            const res = await TB.api.request('CloudM.AdminDashboard', 'reload_module_admin', { module_name: modName }, 'POST');
                            console.log(`Reload module ${modName} response:`, res);
                            if (res.error === TB.ToolBoxError.none) {
                                TB.ui.Toast.showSuccess(`${modName} reload status: ${TB.utils.escapeHtml(res.get())}`);
                            } else {
                                TB.ui.Toast.showError(`Error reloading ${modName}: ${TB.utils.escapeHtml(res.info.help_text)}`);
                            }
                        } catch (err) {
                            TB.ui.Toast.showError('Network error during module reload.');
                            console.error(`Reload module ${modName} error:`, err);
                        } finally {
                            TB.ui.Loader.hide();
                        }
                    });
                });
            }

             async function loadSppManagement(targetDivId) {
                const contentDiv = document.getElementById(targetDivId);
                if (!contentDiv) return;
                contentDiv.innerHTML = '<p class="tb-text-gray-500">Fetching registered SPPs/UI Panels...</p>';
                try {
                    const response = await TB.api.request('CloudM.AdminDashboard', 'list_spps_admin', null, 'GET');
                    if (response.error === TB.ToolBoxError.none) {
                        renderSppManagement(response.get(), contentDiv);
                    } else {
                        contentDiv.innerHTML = `<p class="tb-text-red-500">Error: ${TB.utils.escapeHtml(response.info.help_text)}</p>`;
                    }
                } catch (e) { contentDiv.innerHTML = '<p class="tb-text-red-500">Network error.</p>'; console.error(e); }
            }

            function renderSppManagement(spps, contentDiv) {
                if (!spps || spps.length === 0) {
                    contentDiv.innerHTML = '<p class="tb-text-gray-500">No SPPs or UI Panels are currently registered.</p>'; return;
                }
                let html = '<table><thead><tr><th>Title</th><th>Name/ID</th><th>Path</th><th>Description</th><th>Auth Req.</th><th>Actions</th></tr></thead><tbody>';
                spps.forEach(spp => {
                    // spp structure is: {"auth":auth,"path": path, "title": title, "description": description, "name": name}
                    // Note: 'name' key might not be in original dict if 'add_ui' doesn't add it. Assuming 'title' or path can be unique identifier.
                    // Let's assume `openui` adds the `name` key to the dictionary for easier identification.
                    const name = spp.name || spp.title; // Fallback if name isn't explicitly set
                    html += `<tr>
                        <td>${TB.utils.escapeHtml(spp.title)}</td>
                        <td>${TB.utils.escapeHtml(name)}</td>
                        <td>${TB.utils.escapeHtml(spp.path)}</td>
                        <td>${TB.utils.escapeHtml(spp.description)}</td>
                        <td>${spp.auth ? 'Yes' : 'No'}</td>
                        <td>
                            <button class="action-btn btn-open-link" data-path="${spp.path}"><span class="material-symbols-outlined">open_in_new</span>Open</button>
                            <!-- Conceptual buttons for Start/Stop/Logs - require backend implementation -->
                            <!-- <button class="action-btn btn-restart" data-spp-name="${TB.utils.escapeHtml(name)}">Start/Stop</button> -->
                            <!-- <button class="action-btn btn-view" data-spp-name="${TB.utils.escapeHtml(name)}">Logs</button> -->
                        </td>
                    </tr>`;
                });
                html += '</tbody></table>';
                contentDiv.innerHTML = html;
                contentDiv.querySelectorAll('.btn-open-link').forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        const path = e.currentTarget.dataset.path;
                        if (path) {
                           // If path is absolute or starts with /api/, open in new tab
                           // Otherwise, try to use TB.router if it's a relative client-side path
                           if (path.startsWith("http") || path.startsWith("/api/")) {
                               window.open(path, '_blank');
                           } else {
                               TB.router.navigateTo(path); // Assumes path is router-compatible
                           }
                        }
                    });
                });
            }

            async function loadMyAccountSection(targetDivId = 'my-account-content') {
                const contentDiv = document.getElementById(targetDivId);
                if (!contentDiv) { console.error("My Account content div not found."); return; }

                if (!currentAdminUser) {
                    contentDiv.innerHTML = "<p class='tb-text-red-500'>Account details not available. Please ensure you are logged in.</p>";
                    console.warn("loadMyAccountSection called but currentAdminUser is null.");
                    return;
                }
                console.log("Loading My Account section for:", currentAdminUser);
                const user = currentAdminUser;
                const emailSectionId = `email-updater-${TB.utils.uniqueId()}`;
                const expFeaturesId = `exp-features-${TB.utils.uniqueId()}`;
                const personaStatusId = `persona-status-${TB.utils.uniqueId()}`;

                let personaBtnHtml = !user.is_persona ?
                    `<button id="registerPersonaBtnAdmin" class="tb-btn tb-btn-success tb-mt-2"><span class="material-symbols-outlined tb-mr-1">fingerprint</span>Add Persona Device</button><div id="${personaStatusId}" class="tb-text-sm tb-mt-1"></div>` :
                    `<p class='tb-text-md tb-text-green-600 dark:tb-text-green-400'><span class="material-symbols-outlined tb-mr-1" style="vertical-align: text-bottom;">verified_user</span>Persona (WebAuthn) is configured for this account.</p>`;

                contentDiv.innerHTML = `
                    <div class="tb-space-y-6">
                        <div>
                            <h4 class="tb-text-lg tb-font-semibold tb-mb-2">Email Address</h4>
                            <div id="${emailSectionId}" class="tb-space-y-2">
                                 <p class="tb-text-md"><strong>Current Email:</strong> ${user.email ? TB.utils.escapeHtml(user.email) : "Not set"}</p>
                                 <input type="email" name="new_email_admin" value="${user.email ? TB.utils.escapeHtml(user.email) : ''}" class="tb-input md:tb-w-2/3" placeholder="Enter new email">
                                 <button class="tb-btn tb-btn-primary tb-mt-2"
                                    data-hx-post="/api/CloudM.UserAccountManager/update_email"
                                    data-hx-include="[name='new_email_admin']"
                                    data-hx-target="#${emailSectionId}" data-hx-swap="innerHTML"><span class="material-symbols-outlined tb-mr-1">save</span>Update Email</button>
                            </div>
                        </div>
                        <div>
                            <h4 class="tb-text-lg tb-font-semibold tb-mb-2">Persona Device (WebAuthn)</h4>
                            ${personaBtnHtml}
                        </div>
                        <div>
                            <h4 class="tb-text-lg tb-font-semibold tb-mb-1">User Level</h4>
                            <p class="tb-text-md">${user.level} ${user.level === 0 ? '(Administrator)' : ''}</p>
                        </div>
                        <div>
                            <h4 class="tb-text-lg tb-font-semibold tb-mb-2">Application Settings</h4>
                            <div id="${expFeaturesId}">
                                <label class="tb-label tb-flex tb-items-center tb-cursor-pointer">
                                    <input type="checkbox" name="exp_features_admin_val" ${user.settings && user.settings.experimental_features ? "checked" : ""}
                                           class="tb-checkbox tb-mr-2"
                                           data-hx-post="/api/CloudM.UserAccountManager/update_setting"
                                           data-hx-vals='{"setting_key": "experimental_features", "setting_value": event.target.checked ? "true" : "false"}'
                                           data-hx-target="#${expFeaturesId}" data-hx-swap="innerHTML">
                                    <span class="tb-text-md">Enable Experimental Features</span>
                                </label>
                            </div>
                        </div>
                    </div>`;

                if (window.htmx) {
                    console.log("Processing HTMX for My Account section.");
                    window.htmx.process(contentDiv);
                } else {
                    console.warn("HTMX not found. Dynamic updates in 'My Account' section via htmx attributes will not work. Ensure HTMX is loaded if desired.");
                }

                const personaBtnAdmin = document.getElementById('registerPersonaBtnAdmin');
                if (personaBtnAdmin) {
                    console.log("Persona registration button found, attaching listener.");
                    personaBtnAdmin.addEventListener('click', async () => {
                        const statusDiv = document.getElementById(personaStatusId);
                        if (!statusDiv) { console.error("Persona status div not found."); return; }
                        statusDiv.innerHTML = '<p class="tb-text-sm tb-text-blue-500">Initiating WebAuthn registration...</p>';
                        console.log("Attempting WebAuthn registration for user:", user.name);
                         if (window.TB && window.TB.user && user.name) {
                            const result = await window.TB.user.registerWebAuthnForCurrentUser(user.name);
                            console.log("WebAuthn registration result:", result);
                            if (result.success) {
                                statusDiv.innerHTML = `<p class="tb-text-sm tb-text-green-500">${TB.utils.escapeHtml(result.message)} Refreshing account details to reflect changes.</p>`;
                                TB.ui.Toast.showSuccess("Persona registered! Refreshing account details...");
                                setTimeout(async () => {
                                    console.log("Re-fetching admin user data after persona registration.");
                                    const updatedUserRes = await TB.api.request('CloudM.UserAccountManager', 'get_current_user_from_request_api_wrapper', null, 'GET');
                                    if (updatedUserRes.error === TB.ToolBoxError.none && updatedUserRes.get()) {
                                        currentAdminUser = updatedUserRes.get();
                                        await loadMyAccountSection(); // Re-render "My Account" section
                                        console.log("My Account section re-rendered after persona update.");
                                    } else {
                                        console.error("Failed to re-fetch admin user data after persona registration:", updatedUserRes.info.help_text);
                                    }
                                }, 1500);
                            } else {
                                statusDiv.innerHTML = `<p class="tb-text-sm tb-text-red-500">Error: ${TB.utils.escapeHtml(result.message)}</p>`;
                            }
                        } else {
                            statusDiv.innerHTML = '<p class="tb-text-sm tb-text-red-500">TB.user or current username not available for WebAuthn registration.</p>';
                            console.error("TB.user or currentAdminUser.name is not available for WebAuthn.");
                        }
                    });
                } else if (!user.is_persona) {
                    console.warn("Persona registration button (registerPersonaBtnAdmin) not found, though user is not persona.");
                }
            }
        }
    </script>
    <a href="/api/CloudM.UserDashboard/main">User Dashboard</a>
</div>"""
    return Result.html(html_content)


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def get_system_status(app: App, request: RequestData):
    admin_user = await _is_admin(app, request)
    if not admin_user: return Result.default_user_error(info="Permission denied", exec_code=403)

    status_str = mini.get_service_status(PID_DIR)
    services_data = {}
    lines = status_str.split('\n')
    if lines and lines[0].startswith("Service(s):"):
        for line in lines[1:]:
            if not line.strip(): continue
            parts = line.split('(PID:')
            name_part = parts[0].strip()
            pid_part = "N/A"
            if len(parts) > 1 and parts[1]: pid_part = parts[1].replace(')', '').strip()
            status_indicator = name_part[0] if len(name_part) > 0 else "🟡"
            service_full_name = name_part[2:].strip() if len(name_part) > 1 else "Unknown Service"
            services_data[service_full_name] = {"status_indicator": status_indicator, "pid": pid_part}
    elif status_str == "No services found":
        services_data = {}
    else:
        if '(PID:' in status_str:
            parts = status_str.split('(PID:')
            name_part = parts[0].strip()
            pid_part = "N/A"
            if len(parts) > 1 and parts[1]: pid_part = parts[1].replace(')', '').strip()
            status_indicator = name_part[0] if len(name_part) > 0 else "🟡"
            service_full_name = name_part[2:].strip() if len(name_part) > 1 else "Unknown Service"
            services_data[service_full_name] = {"status_indicator": status_indicator, "pid": pid_part}
        elif status_str.strip():
            services_data["unknown_service_format"] = {"status_indicator": "🟡", "pid": "N/A", "details": status_str}
        else:
            services_data = {}
    return Result.json(data=services_data)


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def list_users_admin(app: App, request: RequestData):
    admin_user = await _is_admin(app, request)
    if not admin_user: return Result.default_user_error(info="Permission denied", exec_code=403)
    all_users_result = await app.a_run_any(TBEF.DB.GET, query="USER::*", get_results=True)
    if all_users_result.is_error():
        return Result.default_internal_error(info="Failed to fetch users: " + str(all_users_result.info))
    users_data = []
    user_list_raw = all_users_result.get()

    def helper(user_bytes):
        try:
            user_str = user_bytes.decode() if isinstance(user_bytes, bytes) else str(user_bytes)
            user_dict = {}
            try:
                user_dict = json.loads(user_str)
            except json.JSONDecodeError:
                app.print("Warning: User data (admin list) not valid JSON, falling back to eval: " + str(
                    user_str[:100]) + "...", "WARNING")
                user_dict = eval(user_str)
            users_data.append({"uid": user_dict.get("uid", "N/A"), "name": user_dict.get("name", "N/A"),
                               "email": user_dict.get("email"), "level": user_dict.get("level", -1),
                               "is_persona": user_dict.get("is_persona", False),
                               "settings": user_dict.get("settings", {})})
        except Exception as e:
            app.print("Error parsing user data for admin list: " + str(user_bytes[:100]) + "... - Error: " + str(e),
                      "ERROR")
    if user_list_raw:
        if isinstance(user_list_raw, list):
            for user_bytes_ in user_list_raw:
                helper(user_bytes_)
        else:
            helper(user_list_raw)

    return Result.json(data=users_data)


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def list_modules_admin(app: App, request: RequestData):
    admin_user = await _is_admin(app, request)
    if not admin_user: return Result.default_user_error(info="Permission denied", exec_code=403)
    modules = list_all_modules(app)
    return Result.json(data=modules)


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def update_user_admin(app: App, request: RequestData, data: dict):
    admin_user = await _is_admin(app, request)
    if not admin_user: return Result.default_user_error(info="Permission denied", exec_code=403)
    uid_to_update = data.get("uid")
    name_to_update = data.get("name")
    if not uid_to_update or not name_to_update: return Result.default_user_error(info="User UID and Name are required.")

    user_res = await app.a_run_any(TBEF.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=name_to_update,
                                   uid=uid_to_update, get_results=True)  # Use a_run_any for TBEF
    if user_res.is_error() or not user_res.get(): return Result.default_user_error(
        info="User " + str(name_to_update) + " (UID: " + str(uid_to_update) + ") not found.")

    user_to_update = user_res.get()
    if "email" in data: user_to_update.email = data["email"]
    if "level" in data:
        try:
            user_to_update.level = int(data["level"])
        except ValueError:
            return Result.default_user_error(info="Invalid level format.")
    if "settings" in data and isinstance(data["settings"], dict):
        if user_to_update.settings is None: user_to_update.settings = {}
        user_to_update.settings.update(data["settings"])

    save_result =  db_helper_save_user(app, asdict(user_to_update))
    if save_result.is_error(): return Result.default_internal_error(
        info="Failed to save user: " + str(save_result.info))
    return Result.ok(info="User updated successfully.")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def delete_user_admin(app: App, request: RequestData, data: dict):
    admin_user = await _is_admin(app, request)
    if not admin_user: return Result.default_user_error(info="Permission denied", exec_code=403)
    uid_to_delete = data.get("uid")
    if not uid_to_delete: return Result.default_user_error(info="User UID is required.")
    if admin_user.uid == uid_to_delete: return Result.default_user_error(info="Admin cannot delete themselves.")

    user_to_delete_res = await app.a_run_any(TBEF.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username='*', uid=uid_to_delete, get_results=True)
    username_to_delete = None
    if not user_to_delete_res.is_error() and user_to_delete_res.get():
        username_to_delete = user_to_delete_res.get().name
    else:
        all_users_raw_res = await app.a_run_any(TBEF.DB.GET, query="USER::*::" + str(uid_to_delete), get_results=True)
        if not all_users_raw_res.is_error() and all_users_raw_res.get():
            try:
                user_bytes = all_users_raw_res.get()[0];
                user_str = user_bytes.decode() if isinstance(user_bytes, bytes) else str(user_bytes)
                user_dict_raw = {};
                try:
                    user_dict_raw = json.loads(user_str)
                except json.JSONDecodeError:
                    user_dict_raw = eval(user_str)
                username_to_delete = user_dict_raw.get("name")
            except Exception as e:
                return Result.default_internal_error(info="Error parsing user data for deletion: " + str(e))
    if not username_to_delete: return Result.default_user_error(
        info="User with UID " + str(uid_to_delete) + " not found or name indeterminate.")

    delete_result = db_helper_delete_user(app, username_to_delete, uid_to_delete, matching=True)
    #delete_result = await app.a_run_any(TBEF.CLOUDM_AUTHMANAGER.CREATE_USER, username=username_to_delete,
    #                                    uid=uid_to_delete, get_results=True)
    if delete_result.is_error(): return Result.default_internal_error(
        info="Failed to delete user: " + str(delete_result.info))
    return Result.ok(info="User " + str(username_to_delete) + " deleted.")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def reload_module_admin(app: App, request: RequestData, data: dict):
    admin_user = await _is_admin(app, request)
    if not admin_user: return Result.redirect("/web/core0/index.html")
    module_name = data.get("module_name")
    if not module_name: return Result.default_user_error(info="Module name is required.")
    app.print("Admin request to reload module: " + str(module_name))
    try:
        if module_name in app.get_all_mods():
            if hasattr(app, 'reload_mod'):
                app.reload_mod(module_name)  # Assuming reload_mod could be async
            else:
                app.remove_mod(module_name)
                app.save_load(module_name)
            return Result.ok(info="Module " + str(module_name) + " reload process completed.")
        else:
            return Result.default_user_error(info="Module " + str(module_name) + " not found.")
    except Exception as e:
        app.print("Error during module reload for " + str(module_name) + ": " + str(e), "ERROR")
        return Result.default_internal_error(info="Error during reload: " + str(e))


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def get_waiting_list_users_admin(app: App, request: RequestData):
    admin_user = await _is_admin(app, request)
    if not admin_user: return Result.redirect("/web/core0/index.html")
    waiting_list_res = await app.a_run_any(TBEF.DB.GET, query="email_waiting_list", get_results=True)
    waiting_list_res.print()
    if waiting_list_res.is_error() or not waiting_list_res.get(): return Result.json(data=[])
    raw_data = waiting_list_res.get()
    try:
        if isinstance(raw_data, bytes): raw_data = raw_data.decode()
        if isinstance(raw_data, list) and len(raw_data) > 0: raw_data = raw_data[0]
        waiting_list_emails = json.loads(raw_data.replace("'", '"')).get("set")
        app.print(f"DARA::, {waiting_list_emails}, {type(waiting_list_emails)}")
        if not isinstance(waiting_list_emails, list): return Result.json(data=[])
        return Result.json(data=list(waiting_list_emails))
    except (json.JSONDecodeError, TypeError, IndexError) as e:
        app.print("Error parsing waiting list data: " + str(e), "ERROR");
        return Result.json(data=[])


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def remove_from_waiting_list_admin(app: App, request: RequestData, data: dict):
    admin_user = await _is_admin(app, request)
    if not admin_user: return Result.redirect("/web/core0/index.html")
    email_to_remove = data.get("email")
    if not email_to_remove: return Result.default_user_error(info="Email is required.")
    waiting_list_res = await app.a_run_any(TBEF.DB.GET, query="email_waiting_list", get_results=True)
    updated_list = []
    if not waiting_list_res.is_error() and waiting_list_res.get():
        raw_data = waiting_list_res.get()
        try:
            if isinstance(raw_data, bytes): raw_data = raw_data.decode()
            if isinstance(raw_data, list) and len(raw_data) > 0: raw_data = raw_data[0]
            current_list = json.loads(raw_data)
            if isinstance(current_list, list):
                updated_list = [email for email in current_list if email != email_to_remove]
        except (json.JSONDecodeError, TypeError, IndexError):
            pass
    save_res = await app.a_run_any(TBEF.DB.SET, query="email_waiting_list", data=json.dumps(updated_list),
                                   get_results=True)
    if save_res.is_error(): return Result.default_internal_error(
        info="Failed to update waiting list: " + str(save_res.info))
    return Result.ok(info="Email removed from waiting list.")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def send_invite_to_waiting_list_user_admin(app: App, request: RequestData, data: dict):
    admin_user = await _is_admin(app, request)
    if not admin_user: return Result.redirect("/web/core0/index.html")
    email_to_invite = data.get("email")
    proposed_username = data.get("username")
    if not email_to_invite or not proposed_username: return Result.default_user_error(
        info="Email and proposed username are required.")


    # db_helper_test_exist is sync, wrap it if true async desired or use a_run_any
    if db_helper_test_exist(app, username=proposed_username):
        return Result.default_user_error(info="Proposed username '" + str(proposed_username) + "' already exists.")

    # send_signup_invitation_email is sync. Assuming it's okay or would be wrapped.
    invite_result = send_signup_invitation_email(app, invited_user_email=email_to_invite,
                                                 invited_username=proposed_username, inviter_username=admin_user.name)
    if not invite_result.is_error():
        return Result.ok(
            info="Invitation email sent to " + str(email_to_invite) + " for username " + str(proposed_username) + ".")
    else:
        return Result.default_internal_error(info="Failed to send invitation: " + str(invite_result.info))


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def list_spps_admin(app: App, request: RequestData):
    admin_user = await _is_admin(app, request)
    if not admin_user: return Result.redirect("/web/core0/index.html")
    spp_list = []
    try:
        # Directly use the imported dictionary from extras
        for name_key, details in json.loads(app.config_fh.get_file_handler("CloudM::UI", "{}")).items():
            spp_list.append({
                "name": name_key,
                "title": details.get("title"), "path": details.get("path"),
                "description": details.get("description"), "auth": details.get("auth", False)
            })
    except Exception as e:
        app.print("Error fetching SPP list from CloudM.extras.uis: " + str(e), "ERROR")
        return Result.default_internal_error(info="Could not retrieve SPP list.")
    return Result.json(data=spp_list)
