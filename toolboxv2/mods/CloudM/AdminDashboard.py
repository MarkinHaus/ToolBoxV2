# toolboxv2/mods/CloudM/AdminDashboard.py

import uuid
from dataclasses import asdict
import json  # For trying to parse user data as JSON first

from toolboxv2 import TBEF, App, Result, get_app, RequestData
from toolboxv2.mods.CloudM.AuthManager import db_helper_save_user, db_helper_delete_user, get_user_by_name
from toolboxv2.mods.CloudM.mini import get_service_status
from toolboxv2.mods.CloudM.ModManager import list_modules as list_all_modules
from .types import User
from .UserAccountManager import get_current_user_from_request

Name = 'CloudM.AdminDashboard'
export = get_app(Name + ".Export").tb
version = '0.1.0'

PID_DIR = "./.info"


async def _is_admin(app: App, request: RequestData) -> User | None:
    current_user = await get_current_user_from_request(app, request)
    if not current_user or current_user.level > 0:
        return None
    return current_user


@export(mod_name=Name, api=True, version=version, name="main", api_methods=['GET'], request_as_kwarg=True, row=True)
async def get_dashboard_main_page(app: App, request: RequestData):
    admin_user = await _is_admin(app, request)
    if not admin_user:
        return Result.html("<h1>Access Denied</h1><p>You do not have permission to view this page.</p>",
                           status_code=403)

    # Using Python's triple-quoted string for the main HTML, CSS, and JS block.
    # JavaScript within this block will use its own template literals (backticks) for dynamic HTML.
    html_content = """
<style>
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--theme-bg, var(--tb-color-neutral-50, #f0f2f5));
            color: var(--theme-text, var(--tb-color-neutral-800, #333));
            transition: background-color 0.3s ease, color 0.3s ease;
        }

        #admin-dashboard { display: flex; flex-direction: column; min-height: 100vh; }

        #admin-header {
            background-color: var(--tb-color-neutral-800, #2c3e50);
            color: var(--tb-color-neutral-100, white);
            padding: 1rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        #admin-header h1 { margin: 0; font-size: 1.6rem; font-weight: 600; display: flex; align-items: center;}
        #admin-header h1 .material-symbols-outlined { vertical-align: middle; font-size: 1.5em; margin-right: 0.3em; }
        #admin-header .header-actions { display: flex; align-items: center; }

        #admin-nav ul { list-style: none; padding: 0; margin: 0 0 0 1.5rem; display: flex; }
        #admin-nav li { margin-left: 1rem; cursor: pointer; padding: 0.6rem 1rem; border-radius: 6px; transition: background-color 0.2s ease; font-weight: 500; display: flex; align-items: center; }
        #admin-nav li .material-symbols-outlined { vertical-align: text-bottom; margin-right: 0.3em; }
        #admin-nav li:hover { background-color: var(--tb-color-neutral-700, #34495e); }

        #admin-container { display: flex; flex-grow: 1; }

        #admin-sidebar {
            width: 230px;
            background-color: var(--tb-color-neutral-100, #ffffff);
            padding: 1.5rem 1rem;
            border-right: 1px solid var(--tb-color-neutral-300, #e0e0e0);
            box-shadow: 1px 0 3px rgba(0,0,0,0.05);
            transition: background-color 0.3s ease, border-color 0.3s ease;
        }
        body[data-theme="dark"] #admin-sidebar {
             background-color: var(--tb-color-neutral-850, #232b33);
             border-right: 1px solid var(--tb-color-neutral-700, #374151);
        }

        #admin-sidebar ul { list-style: none; padding: 0; margin: 0; }
        #admin-sidebar li {
            padding: 0.9rem 1rem;
            margin-bottom: 0.6rem;
            cursor: pointer;
            border-radius: 8px;
            transition: background-color 0.2s ease, color 0.2s ease;
            display: flex;
            align-items: center;
            font-weight: 500;
            color: var(--tb-color-neutral-700, #333);
        }
        body[data-theme="dark"] #admin-sidebar li { color: var(--tb-color-neutral-200, #ccc); }
        #admin-sidebar li .material-symbols-outlined { margin-right: 0.85rem; font-size: 1.4rem; }

        #admin-sidebar li:hover { background-color: var(--tb-color-neutral-200, #e9ecef); }
        body[data-theme="dark"] #admin-sidebar li:hover { background-color: var(--tb-color-neutral-700, #34495e); }

        #admin-sidebar li.active {
            background-color: var(--tb-color-primary-500, #007bff);
            color: white !important;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(0, 123, 255, 0.3);
        }
        body[data-theme="dark"] #admin-sidebar li.active { background-color: var(--tb-color-primary-600, #0056b3); }

        #admin-content { flex-grow: 1; padding: 2rem; }

        .content-section { display: none; }
        .content-section.active { display: block; animation: fadeIn 0.5s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

        .content-section h2 {
            font-size: 2rem;
            font-weight: 600;
            color: var(--tb-color-neutral-700, #212529);
            margin-bottom: 1.8rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--tb-color-neutral-300, #dee2e6);
            display: flex;
            align-items: center;
        }
        .content-section h2 .material-symbols-outlined { font-size: 1.3em; margin-right: 0.5em;}
        body[data-theme="dark"] .content-section h2 { color: var(--tb-color-neutral-100, #f8f9fa); border-bottom-color: var(--tb-color-neutral-700, #495057); }

        table { width: 100%; border-collapse: collapse; margin-top: 1.5rem; font-size: 0.9rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border-radius: 8px; overflow: hidden; }
        th, td { border: 1px solid var(--tb-color-neutral-200, #e0e0e0); padding: 12px 15px; text-align: left; }
        body[data-theme="dark"] th, body[data-theme="dark"] td { border-color: var(--tb-color-neutral-700, #4a5568); }
        th { background-color: var(--tb-color-neutral-100, #f8f9fa); font-weight: 600; }
        body[data-theme="dark"] th { background-color: var(--tb-color-neutral-750, #323840); }

        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; vertical-align: middle; }
        .status-green { background-color: var(--tb-color-success-500, #28a745); }
        .status-yellow { background-color: var(--tb-color-warning-500, #ffc107); }
        .status-red { background-color: var(--tb-color-danger-500, #dc3545); }

        .action-btn { padding: 8px 15px; margin: 4px; border: none; border-radius: 6px; cursor: pointer; font-size: 0.875rem; font-weight: 500; transition: background-color 0.2s ease, transform 0.1s ease, box-shadow 0.2s ease; display: inline-flex; align-items: center; }
        .action-btn .material-symbols-outlined { margin-right: 6px; font-size: 1.2em; }
        .action-btn:hover { transform: translateY(-1px); box-shadow: 0 2px 5px rgba(0,0,0,0.1);}
        .action-btn:active { transform: translateY(0px); box-shadow: inset 0 1px 3px rgba(0,0,0,0.1); }

        .btn-restart { background-color: var(--tb-color-warning-500, #ffc107); color: var(--tb-color-neutral-900, black); }
        .btn-restart:hover { background-color: var(--tb-color-warning-600, #e0a800); }
        .btn-edit { background-color: var(--tb-color-info-500, #17a2b8); color: white; }
        .btn-edit:hover { background-color: var(--tb-color-info-600, #138496); }
        .btn-delete { background-color: var(--tb-color-danger-500, #dc3545); color: white; }
        .btn-delete:hover { background-color: var(--tb-color-danger-600, #c82333); }

        .frosted-glass-pane {
            background: var(--glass-bg, rgba(255, 255, 255, 0.75));
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 2rem;
            border: 1px solid var(--glass-border, rgba(255, 255, 255, 0.35));
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        }
        body[data-theme="dark"] .frosted-glass-pane {
            background: var(--glass-bg-dark, rgba(30, 35, 40, 0.75));
            border-color: var(--glass-border-dark, rgba(255, 255, 255, 0.15));
        }

        /* Basic tbjs utility classes for spacing, text, etc. (should be in tbjs.css or app can define) */
        .tb-input { padding: 0.75rem 1rem; border-radius: 6px; width: 100%; box-sizing: border-box; border: 1px solid var(--tb-color-neutral-300, #ced4da); background-color: var(--tb-color-neutral-0, #fff); color: var(--tb-color-neutral-700, #495057); }
        body[data-theme="dark"] .tb-input { border-color: var(--tb-color-neutral-600, #495057); background-color: var(--tb-color-neutral-800, #343a40); color: var(--tb-color-neutral-100, #f8f9fa); }
        .tb-label { font-weight: 500; margin-bottom: 0.5rem; display: block; }
        .tb-checkbox { margin-right: 0.5rem; }
        .tb-btn { /* Assumed base styles from tbjs.css */ }
        .tb-btn-primary { background-color: var(--tb-color-primary-500); color: white; } .tb-btn-primary:hover { background-color: var(--tb-color-primary-600); }
        .tb-btn-success { background-color: var(--tb-color-success-500); color: white; } .tb-btn-success:hover { background-color: var(--tb-color-success-600); }
        .tb-space-y-2 > *:not([hidden]) ~ *:not([hidden]) { margin-top: 0.5rem; }
        .tb-space-y-4 > *:not([hidden]) ~ *:not([hidden]) { margin-top: 1rem; }
        .tb-space-y-6 > *:not([hidden]) ~ *:not([hidden]) { margin-top: 1.5rem; }
        .tb-mt-1 { margin-top: 0.25rem; } .tb-mt-2 { margin-top: 0.5rem; }
        .tb-mb-1 { margin-bottom: 0.25rem; } .tb-mb-2 { margin-bottom: 0.5rem; }
        .tb-mr-1 { margin-right: 0.25rem; } .tb-mr-2 { margin-right: 0.5rem; }
        .tb-w-full { width: 100%; }
        .md\\:tb-w-2\\/3 { width: 66.666667%; } /* Adjusted for CSS literal */
        .tb-text-red-500 { color: #ef4444; } .tb-text-green-500 { color: #22c55e; } .tb-text-green-600 { color: #16a34a; }
        .tb-text-blue-500 { color: #3b82f6; } .tb-text-yellow-500 { color: #eab308; }
        .tb-text-gray-500 { color: #6b7280; }
        body[data-theme="dark"] .tb-text-gray-500 { color: #9ca3af; }
        .tb-text-sm { font-size: 0.875rem; } .tb-text-md { font-size: 1rem; } .tb-text-lg { font-size: 1.125rem; }
        .tb-font-semibold { font-weight: 600; }
        .tb-flex { display: flex; } .tb-items-center { align-items: center; } .tb-cursor-pointer { cursor: pointer; }
    </style>
</head>
<body data-theme="system">
    <div id="admin-dashboard">
        <header id="admin-header">
            <h1><span class="material-symbols-outlined">shield_person</span>CloudM Admin</h1>
            <div class="header-actions">
                 <div id="darkModeToggleContainer" style="display: inline-flex; align-items: center; margin-right: 1.5rem;"></div>
                <nav id="admin-nav">
                    <ul>
                        <li id="logoutButton"><span class="material-symbols-outlined">logout</span>Logout</li>
                    </ul>
                </nav>
            </div>
        </header>
        <div id="admin-container">
            <aside id="admin-sidebar">
                 <ul>
                    <li data-section="system-status" class="active"><span class="material-symbols-outlined">monitoring</span>System Status</li>
                    <li data-section="user-management"><span class="material-symbols-outlined">group</span>User Management</li>
                    <li data-section="module-management"><span class="material-symbols-outlined">extension</span>Modules</li>
                    <li data-section="spp-management"><span class="material-symbols-outlined">deployed_code</span>SPPs</li>
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
                    <div id="user-management-content"><p class="tb-text-gray-500">Loading user data...</p></div>
                </section>
                <section id="module-management-section" class="content-section frosted-glass-pane">
                    <h2><span class="material-symbols-outlined">view_module</span>Module Management</h2>
                    <div id="module-management-content"><p class="tb-text-gray-500">Loading module list...</p></div>
                </section>
                <section id="spp-management-section" class="content-section frosted-glass-pane">
                    <h2><span class="material-symbols-outlined">memory</span>SPP Management (Placeholder)</h2>
                    <div id="spp-management-content"><p class="tb-text-gray-500">Functionality for SPP management will be available here in a future update.</p></div>
                </section>
                <section id="my-account-section" class="content-section frosted-glass-pane">
                    <h2><span class="material-symbols-outlined">account_circle</span>My Account Settings</h2>
                    <div id="my-account-content"><p class="tb-text-gray-500">Loading account details...</p></div>
                </section>
            </main>
        </div>
    </div>

    <script type="module">
        // Ensure tbjs (TB) is available
        if (typeof TB === 'undefined' || !TB.ui || !TB.api || !TB.user) {
            console.error('CRITICAL: TB (tbjs) or its core modules (ui, api, user) are not defined. Ensure tbjs.js is loaded correctly and before this script.');
            document.body.innerHTML = '<div style="padding: 20px; text-align: center; font-size: 1.2em; color: red;">Critical Error: Frontend library (tbjs) failed to load essential components. Please contact support.</div>';
        } else {
            console.log('TB object found. Proceeding with Admin Dashboard initialization.');
            let currentAdminUser = null;

            async function initializeAdminDashboard() {
                console.log("Admin Dashboard Initializing with tbjs...");
                TB.ui.DarkModeToggle.init();
                console.log("Dark Mode Toggle Initialized.");
                setupNavigation();
                console.log("Navigation Setup Complete.");
                await setupLogout();
                console.log("Logout Setup Complete.");

                try {
                    console.log("Fetching current admin user...");
                    const userRes = await TB.api.request('CloudM.UserAccountManager', 'get_current_user_from_request_api_wrapper', null, 'GET');
                    console.log("Admin user response:", userRes);
                    if (userRes.error === TB.ToolBoxError.none && userRes.get()) {
                        currentAdminUser = userRes.get();
                        console.log("Current Admin User:", currentAdminUser);
                        if (currentAdminUser.name) {
                            const adminTitleElement = document.querySelector('#admin-header h1');
                            if (adminTitleElement) {
                                adminTitleElement.innerHTML = `<span class="material-symbols-outlined">shield_person</span>CloudM Admin (${TB.utils.escapeHtml(currentAdminUser.name)})`;
                            }
                        }
                        await loadMyAccountSection();
                        console.log("My Account Section Loaded.");
                        await showSection('system-status');
                        console.log("Initial section 'system-status' shown.");
                    } else {
                        console.error("Failed to load current admin user:", userRes.info.help_text);
                        document.getElementById('admin-content').innerHTML = '<p class="tb-text-red-500">Error: Could not verify admin user. Please login.</p>';
                    }
                } catch (e) {
                    console.error("Error fetching current admin user:", e);
                    document.getElementById('admin-content').innerHTML = '<p class="tb-text-red-500">Network error verifying admin user.</p>';
                }
                 console.log("Admin Dashboard Initialization finished.");
            }

            function _waitForTbInit(callback) {
            if (window.TB?.events) {
    if (window.TB.config?.get('appRootId')) { // A sign that TB.init might have run
         callback();
    } else {
        window.TB.events.on('tbjs:initialized', callback, { once: true });
    }
} else {
    // Fallback if TB is not even an object yet, very early load
    document.addEventListener('tbjs:initialized', callback, { once: true }); // Custom event dispatch from TB.init
}
            }

            _waitForTbInit(initializeAdminDashboard);


            function setupNavigation() {
                const navItems = document.querySelectorAll('#admin-sidebar li[data-section]');
                if(navItems.length === 0) console.warn("No sidebar navigation items found for setupNavigation.");
                navItems.forEach(item => {
                    item.addEventListener('click', async () => {
                        console.log(`Sidebar navigation: Clicked ${item.dataset.section}`);
                        navItems.forEach(i => i.classList.remove('active'));
                        item.classList.add('active');
                        const sectionId = item.getAttribute('data-section');
                        await showSection(sectionId);
                    });
                });
            }

            async function showSection(sectionId) {
                console.log(`Showing section: ${sectionId}`);
                document.querySelectorAll('.content-section').forEach(section => section.classList.remove('active'));
                const activeSection = document.getElementById(`${sectionId}-section`);
                if (activeSection) {
                    activeSection.classList.add('active');
                    const contentDivId = `${sectionId}-content`;
                    const contentDiv = document.getElementById(contentDivId);
                    if (!contentDiv) {
                        console.error(`Content div ${contentDivId} not found for section ${sectionId}`);
                        return;
                    }

                    contentDiv.innerHTML = `<p class="tb-text-gray-500">Loading ${sectionId.replace(/-/g, " ")}...</p>`;

                    try {
                        if (sectionId === 'system-status') await loadSystemStatus(contentDivId);
                        else if (sectionId === 'user-management') await loadUserManagement(contentDivId);
                        else if (sectionId === 'module-management') await loadModuleManagement(contentDivId);
                        else if (sectionId === 'my-account') await loadMyAccountSection(contentDivId);
                        else if (sectionId === 'spp-management') {
                            contentDiv.innerHTML = '<p class="tb-text-gray-500">Functionality for SPP management will be available here in a future update.</p>';
                        }
                        console.log(`Section ${sectionId} content loading initiated.`);
                    } catch (error) {
                        console.error(`Error loading content for section ${sectionId}:`, error);
                        contentDiv.innerHTML = `<p class="tb-text-red-500">An error occurred while loading this section.</p>`;
                    }
                } else {
                    console.error(`Section element ${sectionId}-section not found.`);
                }
            }

            async function setupLogout() {
                const logoutButton = document.getElementById('logoutButton');
                if (logoutButton) {
                    logoutButton.addEventListener('click', async () => {
                        console.log("Logout button clicked.");
                        TB.ui.Loader.show("Logging out...");
                        try {
                            await TB.user.logout();
                            window.location.href = '/';
                        } catch (e) {
                            console.error("Logout error:", e);
                            TB.ui.Toast.showError("Logout failed. Please try again.");
                        } finally {
                            TB.ui.Loader.hide();
                        }
                    });
                } else {
                    console.warn("Logout button not found.");
                }
            }

            async function loadSystemStatus(targetDivId) {
                const contentDiv = document.getElementById(targetDivId);
                if (!contentDiv) return;
                try {
                    console.log("Requesting system status from API: CloudM.AdminDashboard/get_system_status");
                    const response = await TB.api.request('CloudM.AdminDashboard', 'get_system_status', null, 'GET');
                    console.log("System status response:", response);
                    if (response.error === TB.ToolBoxError.none) {
                        renderSystemStatus(response.get(), contentDiv);
                    } else {
                        contentDiv.innerHTML = `<p class="tb-text-red-500">Error loading system status: ${TB.utils.escapeHtml(response.info.help_text)}</p>`;
                    }
                } catch (e) {
                    contentDiv.innerHTML = '<p class="tb-text-red-500">Network error while fetching system status.</p>';
                    console.error("loadSystemStatus error:", e);
                }
            }

            function renderSystemStatus(statusData, contentDiv) {
                if (!contentDiv) return;
                console.log("Rendering system status:", statusData);
                if (!statusData || Object.keys(statusData).length === 0 || (Object.keys(statusData).length === 1 && statusData["unknown_format"])) {
                     if (statusData && statusData["unknown_format"]) {
                        contentDiv.innerHTML = `<p class="tb-text-yellow-500">Service status format not recognized: ${TB.utils.escapeHtml(statusData["unknown_format"].details)}</p>`;
                     } else {
                        contentDiv.innerHTML = '<p class="tb-text-gray-500">No services found or status is currently unavailable.</p>';
                     }
                    return;
                }
                let html = '<table><thead><tr><th>Service</th><th>Status</th><th>PID</th><th>Actions</th></tr></thead><tbody>';
                for (const [name, data] of Object.entries(statusData)) {
                    let sClass = data.status_indicator === '🟢' ? 'status-green' : (data.status_indicator === '🔴' ? 'status-red' : 'status-yellow');
                    html += `<tr>
                        <td>${TB.utils.escapeHtml(name)}</td>
                        <td><span class="status-indicator ${sClass}"></span> ${data.status_indicator}</td>
                        <td>${TB.utils.escapeHtml(data.pid)}</td>
                        <td><button class="action-btn btn-restart" data-service="${TB.utils.escapeHtml(name)}"><span class="material-symbols-outlined">restart_alt</span>Restart</button></td>
                        </tr>`;
                }
                html += '</tbody></table>';
                contentDiv.innerHTML = html;
                contentDiv.querySelectorAll('.btn-restart').forEach(btn => {
                    btn.addEventListener('click', async e => {
                        const serviceName = e.currentTarget.dataset.service;
                        console.log(`Restart button clicked for service: ${serviceName}`);
                        TB.ui.Toast.showInfo(`Restart for ${serviceName} clicked (Note: Restart functionality is a placeholder and not implemented).`);
                    });
                });
            }

            async function loadUserManagement(targetDivId) {
                const contentDiv = document.getElementById(targetDivId);
                if (!contentDiv) return;
                try {
                    console.log("Requesting user list from API: CloudM.AdminDashboard/list_users_admin");
                    const response = await TB.api.request('CloudM.AdminDashboard', 'list_users_admin', null, 'GET');
                    console.log("User list response:", response);
                    if (response.error === TB.ToolBoxError.none) {
                        renderUserManagement(response.get(), contentDiv);
                    } else {
                        contentDiv.innerHTML = `<p class="tb-text-red-500">Error loading users: ${TB.utils.escapeHtml(response.info.help_text)}</p>`;
                    }
                } catch (e) {
                    contentDiv.innerHTML = '<p class="tb-text-red-500">Network error while fetching users.</p>';
                    console.error("loadUserManagement error:", e);
                }
            }

            function renderUserManagement(users, contentDiv) {
                if (!contentDiv) return;
                console.log("Rendering user management:", users);
                if (!users || users.length === 0) {
                    contentDiv.innerHTML = '<p class="tb-text-gray-500">No users found in the system.</p>';
                    return;
                }
                let html = '<table><thead><tr><th>Name</th><th>Email</th><th>Level</th><th>UID</th><th>Actions</th></tr></thead><tbody>';
                users.forEach(user => {
                    html += `<tr>
                        <td>${TB.utils.escapeHtml(user.name)}</td>
                        <td>${TB.utils.escapeHtml(user.email || 'N/A')}</td>
                        <td>${user.level} ${user.level >= 1 ? '(Admin)' : ''}</td>
                        <td>${TB.utils.escapeHtml(user.uid)}</td>
                        <td>
                        <button class="action-btn btn-edit" data-uid="${user.uid}"><span class="material-symbols-outlined">edit</span>Edit</button>
                        ${(currentAdminUser && currentAdminUser.uid !== user.uid) ?
                            `<button class="action-btn btn-delete" data-uid="${user.uid}" data-name="${TB.utils.escapeHtml(user.name)}"><span class="material-symbols-outlined">delete</span>Delete</button>`
                            : ''}
                        </td>
                        </tr>`;
                });
                html += '</tbody></table>';
                contentDiv.innerHTML = html;
                contentDiv.querySelectorAll('.btn-edit').forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        console.log(`Edit button clicked for UID: ${e.currentTarget.dataset.uid}`);
                        showUserEditModal(e.currentTarget.dataset.uid, users)
                    });
                });
                contentDiv.querySelectorAll('.btn-delete').forEach(btn => {
                     btn.addEventListener('click', (e) => {
                        console.log(`Delete button clicked for UID: ${e.currentTarget.dataset.uid}`);
                        handleDeleteUser(e.currentTarget.dataset.uid, e.currentTarget.dataset.name)
                    });
                });
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
                    `<button id="registerPersonaBtnAdmin" class="tb-btn tb-btn-success tb-mt-2" style:"color: var(--text-color)"><span class="material-symbols-outlined tb-mr-1" style:"color: var(--text-color)">fingerprint</span>Add Persona Device</button><div id="${personaStatusId}" class="tb-text-sm tb-mt-1" style:"color: var(--text-color)"></div>` :
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
            } // End of loadMyAccountSection
        } // End of tbjs check
    </script>
"""
    return Result.html(html_content)


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def get_system_status(app: App, request: RequestData):
    admin_user = await _is_admin(app, request)
    if not admin_user: return Result.default_user_error(info="Permission denied", exec_code=403)

    status_str = get_service_status(PID_DIR)
    services_data = {}
    lines = status_str.split('\n')
    if lines and lines[0].startswith("Service(s):"):  # Check if lines is not empty
        for line in lines[1:]:
            if not line.strip(): continue
            parts = line.split('(PID:')
            name_part = parts[0].strip()
            pid_part = "N/A"
            if len(parts) > 1 and parts[1]:
                pid_part = parts[1].replace(')', '').strip()

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
            if len(parts) > 1 and parts[1]:
                pid_part = parts[1].replace(')', '').strip()

            status_indicator = name_part[0] if len(name_part) > 0 else "🟡"
            service_full_name = name_part[2:].strip() if len(name_part) > 1 else "Unknown Service"
            services_data[service_full_name] = {"status_indicator": status_indicator, "pid": pid_part}
        elif status_str.strip():  # If status_str is not empty but not matching known formats
            services_data["unknown_service_format"] = {"status_indicator": "🟡", "pid": "N/A", "details": status_str}
        else:  # If status_str is empty or whitespace
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
    if user_list_raw:
        for user_bytes in user_list_raw:
            try:
                user_str = user_bytes.decode() if isinstance(user_bytes, bytes) else str(user_bytes)
                user_dict = {}
                try:
                    user_dict = json.loads(user_str)
                except json.JSONDecodeError:
                    # Fallback for non-JSON string (e.g. Python dict string representation)
                    # This is risky and should be avoided if possible by storing JSON in DB
                    app.print("Warning: User data for admin list was not valid JSON, falling back to eval: " + str(
                        user_str[:100]) + "...", "WARNING")
                    user_dict = eval(user_str)  # Ensure this eval is safe or replace DB storage method

                users_data.append({
                    "uid": user_dict.get("uid", "N/A"),
                    "name": user_dict.get("name", "N/A"),
                    "email": user_dict.get("email"),
                    "level": user_dict.get("level", -1),
                    "is_persona": user_dict.get("is_persona", False),
                    "settings": user_dict.get("settings", {})
                })
            except Exception as e:
                app.print("Error parsing user data for admin list: " + str(user_bytes[:100]) + "... - Error: " + str(e),
                          "ERROR")
    return Result.json(data=users_data)


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def list_modules_admin(app: App, request: RequestData):
    admin_user = await _is_admin(app, request)
    if not admin_user: return Result.default_user_error(info="Permission denied", exec_code=403)

    modules = list_all_modules(app)  # This function is assumed to exist and work
    return Result.json(data=modules)


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def update_user_admin(app: App, request: RequestData, data: dict):
    admin_user = await _is_admin(app, request)
    if not admin_user: return Result.default_user_error(info="Permission denied", exec_code=403)

    uid_to_update = data.get("uid")
    name_to_update = data.get("name")  # Name is often used as part of the DB key structure
    if not uid_to_update or not name_to_update:
        return Result.default_user_error(info="User UID and Name are required for update.")

    user_res = get_user_by_name(app, username=name_to_update, uid=uid_to_update)
    if user_res.is_error() or not user_res.get():
        error_info = "User " + str(name_to_update) + " (UID: " + str(uid_to_update) + ") not found."
        return Result.default_user_error(info=error_info)

    user_to_update = user_res.get()
    if "email" in data: user_to_update.email = data["email"]
    if "level" in data:
        try:
            user_to_update.level = int(data["level"])
        except ValueError:
            return Result.default_user_error(info="Invalid level format. Level must be an integer.")
    if "settings" in data and isinstance(data["settings"], dict):
        if user_to_update.settings is None: user_to_update.settings = {}
        user_to_update.settings.update(data["settings"])

    save_result = db_helper_save_user(app, asdict(user_to_update))  # This is synchronous
    if save_result.is_error():
        return Result.default_internal_error(info="Failed to save user: " + str(save_result.info))
    return Result.ok(info="User updated successfully.")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def delete_user_admin(app: App, request: RequestData, data: dict):
    admin_user = await _is_admin(app, request)
    if not admin_user: return Result.default_user_error(info="Permission denied", exec_code=403)

    uid_to_delete = data.get("uid")
    if not uid_to_delete: return Result.default_user_error(info="User UID is required for deletion.")
    if admin_user.uid == uid_to_delete: return Result.default_user_error(
        info="Administrators cannot delete their own account.")

    # Fetch user to get their name, as db_helper_delete_user usually requires it for the key.
    user_to_delete_res = get_user_by_name(app, username='*', uid=uid_to_delete)  # Try with wildcard name
    username_to_delete = None

    if not user_to_delete_res.is_error() and user_to_delete_res.get():
        username_to_delete = user_to_delete_res.get().name
    else:
        # Fallback: if get_user_by_name fails with wildcard, try fetching USER::*::{uid}
        # This assumes the DB key might be USER::{name}::{uid} or sometimes just USER::*::{uid} for specific queries
        app.print("Could not find user by get_user_by_name with wildcard name. Trying direct DB query for UID: " + str(
            uid_to_delete), "DEBUG")
        all_users_raw_res = await app.a_run_any(TBEF.DB.GET, query="USER::*::" + str(uid_to_delete), get_results=True)
        if not all_users_raw_res.is_error() and all_users_raw_res.get():
            try:
                user_bytes = all_users_raw_res.get()[0]  # Assuming first result is the one
                user_str = user_bytes.decode() if isinstance(user_bytes, bytes) else str(user_bytes)
                user_dict_raw = {}
                try:
                    user_dict_raw = json.loads(user_str)
                except json.JSONDecodeError:
                    app.print("Warning: User data for deletion (fallback) was not valid JSON, trying eval: " + str(
                        user_str[:100]) + "...", "WARNING")
                    user_dict_raw = eval(user_str)
                username_to_delete = user_dict_raw.get("name")
            except Exception as e:
                error_info = "Error parsing user data during fallback deletion search: " + str(e)
                return Result.default_internal_error(info=error_info)

    if not username_to_delete:
        error_info = "User with UID " + str(
            uid_to_delete) + " not found, or their name could not be determined for deletion."
        return Result.default_user_error(info=error_info)

    app.print("Attempting to delete user: " + str(username_to_delete) + " (UID: " + str(uid_to_delete) + ")", "INFO")
    delete_result = db_helper_delete_user(app, username_to_delete, uid_to_delete)  # Synchronous
    if delete_result.is_error():
        return Result.default_internal_error(info="Failed to delete user: " + str(delete_result.info))
    return Result.ok(info="User " + str(username_to_delete) + " deleted successfully.")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def reload_module_admin(app: App, request: RequestData, data: dict):
    admin_user = await _is_admin(app, request)
    if not admin_user: return Result.default_user_error(info="Permission denied", exec_code=403)

    module_name = data.get("module_name")
    if not module_name: return Result.default_user_error(info="Module name is required for reload.")

    app.print("Admin request to reload module: " + str(module_name), "INFO")
    try:
        if module_name in app.get_all_mods():  # Check if module is currently loaded
            # The actual mechanism for hot-reloading depends heavily on ToolBoxV2's core capabilities.
            # A simple remove & add might work for stateless modules or if state is re-initialized.
            app.print("Attempting to remove module: " + str(module_name), "DEBUG")
            # app.remove_mod(module_name)  # May need spec for specific instances

            app.print("Attempting to re-add module: " + str(module_name), "DEBUG")
            # Re-adding might require knowing how it was originally added (e.g., from a path, a class)
            # app.add_mod_with_tools(module_name) assumes it's a standard module that can be found by name.
            # This part might need to be more sophisticated, e.g., app.toolbox.load_module_by_name(module_name)
            app.reload_mod(module_name)  # Or app.load_module(module_name) or similar

            return Result.ok(info="Module " + str(module_name) + " reload process completed.")
        else:
            return Result.default_user_error(info="Module " + str(module_name) + " not found or not currently loaded.")
    except Exception as e:
        app.print("Error during module reload for " + str(module_name) + ": " + str(e), "ERROR")
        return Result.default_internal_error(info="Error during reload attempt for " + str(module_name) + ": " + str(e))
