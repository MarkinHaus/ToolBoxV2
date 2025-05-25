# toolboxv2/mods/CloudM/UserDashboard.py

import uuid
from dataclasses import asdict
import json

from toolboxv2 import TBEF, App, Result, get_app, RequestData
from toolboxv2.mods.CloudM.AuthManager import db_helper_save_user, get_magic_link_email as request_magic_link_backend
from .types import User
from .UserAccountManager import get_current_user_from_request
from .UserInstances import get_user_instance as get_user_instance_internal, \
    close_user_instance as close_user_instance_internal

# We'll need a new function in UserInstances or a dedicated module manager for user instances
# from .UserInstanceManager import update_active_modules_for_user_instance # Placeholder

Name = 'CloudM.UserDashboard'
export = get_app(Name + ".Export").tb
version = '0.1.1'  # Incremented version


@export(mod_name=Name, api=True, version=version, name="main", api_methods=['GET'], request_as_kwarg=True, row=True)
async def get_user_dashboard_main_page(app: App, request: RequestData):
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.html("<h1>Access Denied</h1><p>Please log in to view your dashboard.</p>", status_code=401)

    # HTML structure for the User Dashboard
    # Using Python's triple-quoted string for the main HTML block.
    html_content = """
    <style>
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--theme-bg, var(--tb-color-neutral-50, #f0f2f5));
            color: var(--theme-text, var(--tb-color-neutral-800, #333));
            transition: background-color 0.3s ease, color 0.3s ease;
        }
        #user-dashboard { display: flex; flex-direction: column; min-height: 100vh; }
        #user-header {
            background-color: var(--theme-primary, var(--tb-color-primary-600, #0056b3));
            color: var(--tb-color-neutral-100, white);
            padding: 1rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        #user-header h1 { margin: 0; font-size: 1.6rem; font-weight: 600; display: flex; align-items: center;}
        #user-header h1 .material-symbols-outlined { vertical-align: middle; font-size: 1.5em; margin-right: 0.3em; }
        #user-header .header-actions { display: flex; align-items: center; }

        #user-nav ul { list-style: none; padding: 0; margin: 0 0 0 1.5rem; display: flex; }
        #user-nav li { margin-left: 1rem; cursor: pointer; padding: 0.6rem 1rem; border-radius: 6px; transition: background-color 0.2s ease; font-weight: 500; display: flex; align-items: center; }
        #user-nav li .material-symbols-outlined { vertical-align: text-bottom; margin-right: 0.3em; }
        #user-nav li:hover { background-color: var(--theme-primary-darker, var(--tb-color-primary-700, #004085)); } /* Use theme var */

        #user-container { display: flex; flex-grow: 1; }
        #user-sidebar {
            display: none; /* Hidden by default on small screens */
            position: fixed; /* Or absolute, depending on desired behavior */
            left: -250px; /* Start off-screen */
            top: 0;
            bottom: 0;
            width: 240px; /* Or your desired width */
            z-index: 1000;
            transition: left 0.3s ease-in-out;
            overflow-y: auto; /* If content might overflow */
        }

        #user-sidebar.open {
            display: flex; /* Or block */
            left: 0; /* Slide in */
        }

        #sidebar-toggle-btn { /* Your hamburger button */
            display: inline-flex; /* Or block, depending on header layout */
            /* Styling for the button */
            padding: 0.5rem;
            cursor: pointer;
            z-index: 1001; /* Above sidebar when closed, or managed differently */
        }

        @media (min-width: 768px) { /* Or your preferred breakpoint for tablet/desktop */
            #sidebar-toggle-btn {
                display: none; /* Hide hamburger on larger screens */
            }
            #user-sidebar {
                display: flex; /* Show sidebar in its normal position */
                position: static; /* Revert to static positioning */
                left: auto;
                width: 230px; /* Your desktop width */
                transition: none; /* No transition needed for static display */
            }
            /* Adjust #user-container or #user-content if needed when sidebar is static */
            #user-container.sidebar-present #user-content {
                margin-left: 230px; /* Example: if sidebar is fixed and content needs to shift */
            }
        }

        /* Backdrop for mobile sidebar */
        #sidebar-backdrop {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
            z-index: 999; /* Below sidebar, above content */
        }
        #sidebar-backdrop.active {
            display: block;
        }
        #user-sidebar.open { /* This class is toggled by JS */
            /* ... your existing .open styles ... */
            left: 0;
            box-shadow: 2px 0 5px rgba(0,0,0,0.2); /* Add shadow when open */
        }
        @media (max-width: 767.98px) { /* Apply only on smaller screens */
            #user-sidebar {
                display: flex; /* Keep it flex for content alignment */
                position: fixed;
                left: -250px; /* Start off-screen (adjust width if needed) */
                top: 0;
                bottom: 0;
                width: 240px; /* Consistent width */
                z-index: 1000; /* Above backdrop */
                transition: left 0.3s ease-in-out;
                overflow-y: auto;
                 /* Make sure background and border are explicitly set for mobile overlay */
                background-color: var(--sidebar-bg, var(--tb-color-neutral-100, #ffffff));
                border-right: 1px solid var(--sidebar-border, var(--tb-color-neutral-300, #e0e0e0));
            }
            body[data-theme="dark"] #user-sidebar {
                 background-color: var(--sidebar-bg-dark, var(--tb-color-neutral-850, #232b33));
                 border-right-color: var(--sidebar-border-dark, var(--tb-color-neutral-700, #374151));
            }
        }
        body[data-theme="dark"] #user-sidebar {
             background-color: var(--sidebar-bg-dark, var(--tb-color-neutral-850, #232b33));
             border-right-color: var(--sidebar-border-dark, var(--tb-color-neutral-700, #374151));
        }
        #user-sidebar ul { list-style: none; padding: 0; margin: 0; }
        #user-sidebar li { padding: 0.9rem 1rem; margin-bottom: 0.6rem; cursor: pointer; border-radius: 8px; display: flex; align-items: center; font-weight: 500; color: var(--sidebar-text, var(--tb-color-neutral-700, #333)); transition: background-color 0.2s ease, color 0.2s ease; }
        body[data-theme="dark"] #user-sidebar li { color: var(--sidebar-text-dark, var(--tb-color-neutral-200, #ccc)); }
        #user-sidebar li .material-symbols-outlined { margin-right: 0.85rem; font-size: 1.4rem; }
        #user-sidebar li:hover { background-color: var(--sidebar-hover-bg, var(--tb-color-neutral-200, #e9ecef)); }
        body[data-theme="dark"] #user-sidebar li:hover { background-color: var(--sidebar-hover-bg-dark, var(--tb-color-neutral-700, #34495e)); }
        #user-sidebar li.active { background-color: var(--theme-primary, var(--tb-color-primary-500, #007bff)); color: var(--sidebar-active-text, white) !important; font-weight: 600; box-shadow: 0 2px 8px rgba(0, 123, 255, 0.3); }
        body[data-theme="dark"] #user-sidebar li.active { background-color: var(--theme-primary-darker, var(--tb-color-primary-600, #0056b3)); }

        #user-content { flex-grow: 1; padding: 2rem; }
        .content-section { display: none; }
        .content-section.active { display: block; animation: fadeIn 0.5s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .content-section h2 { font-size: 2rem; font-weight: 600; color: var(--theme-text, var(--tb-color-neutral-700, #212529)); margin-bottom: 1.8rem; padding-bottom: 1rem; border-bottom: 1px solid var(--section-border, var(--tb-color-neutral-300, #dee2e6)); display: flex; align-items: center; }
        .content-section h2 .material-symbols-outlined { font-size: 1.3em; margin-right: 0.5em;}
        body[data-theme="dark"] .content-section h2 { color: var(--theme-text-dark, var(--tb-color-neutral-100, #f8f9fa)); border-bottom-color: var(--section-border-dark, var(--tb-color-neutral-700, #495057)); }

        .frosted-glass-pane {
            background: var(--glass-bg, rgba(255, 255, 255, 0.75));
            backdrop-filter: blur(var(--glass-blur, 10px)); -webkit-backdrop-filter: blur(var(--glass-blur, 10px));
            border-radius: 12px; padding: 2rem;
            border: 1px solid var(--glass-border, rgba(255, 255, 255, 0.35));
            box-shadow: var(--glass-shadow, 0 4px 15px rgba(0, 0, 0, 0.08));
        }
        body[data-theme="dark"] .frosted-glass-pane {
            background: var(--glass-bg-dark, rgba(30, 35, 40, 0.75));
            border-color: var(--glass-border-dark, rgba(255, 255, 255, 0.15));
        }
        .instance-card, .module-card {
            border: 1px solid var(--card-border, var(--tb-color-neutral-300, #ddd));
            border-radius: 8px; padding: 1rem; margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            background-color: var(--card-bg, var(--tb-color-neutral-0, #fff));
        }
        body[data-theme="dark"] .instance-card, body[data-theme="dark"] .module-card {
            border-color: var(--card-border-dark, var(--tb-color-neutral-700, #444));
            background-color: var(--card-bg-dark, var(--tb-color-neutral-800, #2d3748));
        }
        .instance-card h4, .module-card h4 { margin-top: 0; color: var(--theme-primary); }
        .instance-card .module-list, .module-card .module-status { list-style: disc; margin-left: 1.5rem; font-size: 0.9em;}
        .module-card { display: flex; justify-content: space-between; align-items: center; }

        .settings-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; }
        .setting-item { padding: 1rem; border: 1px solid var(--tb-color-neutral-200, #eee); border-radius: 8px; }
        body[data-theme="dark"] .setting-item { border-color: var(--tb-color-neutral-700, #444); }
        .setting-item label { display: block; margin-bottom: 0.5rem; font-weight: 500; }
        .setting-item input[type="color"], .setting-item input[type="text"], .setting-item input[type="number"], .setting-item select { width:100%; padding:0.5rem; border-radius:4px; border:1px solid var(--tb-color-neutral-400); margin-bottom:0.5rem; }

        .tb-input { padding: 0.75rem 1rem; border-radius: 6px; width: 100%; box-sizing: border-box; border: 1px solid var(--tb-color-neutral-300, #ced4da); background-color: var(--tb-color-neutral-0, #fff); color: var(--tb-color-neutral-700, #495057); }
        body[data-theme="dark"] .tb-input { border-color: var(--tb-color-neutral-600, #495057); background-color: var(--tb-color-neutral-800, #343a40); color: var(--tb-color-neutral-100, #f8f9fa); }
        .tb-label { font-weight: 500; margin-bottom: 0.5rem; display: block; }
        .tb-checkbox-label { display: flex; align-items: center; cursor: pointer; }
        .tb-checkbox { margin-right: 0.5rem; }
        .tb-btn { display:inline-flex; align-items:center; justify-content:center; padding:0.6rem 1.2rem; border-radius:6px; font-weight:500; cursor:pointer; transition: background-color 0.2s ease, box-shadow 0.2s ease; border:none; }
        .tb-btn .material-symbols-outlined { margin-right: 0.4em; font-size: 1.2em; }
        .tb-btn-primary { background-color: var(--theme-primary, var(--tb-color-primary-500)); color: white; } .tb-btn-primary:hover { background-color: var(--theme-primary-darker, var(--tb-color-primary-600)); }
        .tb-btn-secondary { background-color: var(--theme-secondary, #6c757d); color: white; } .tb-btn-secondary:hover { background-color: var(--theme-secondary-darker, #5a6268); }
        .tb-btn-danger { background-color: var(--tb-color-danger-500, #dc3545); color: white; } .tb-btn-danger:hover { background-color: var(--tb-color-danger-600, #c82333); }
        .tb-btn-success { background-color: var(--tb-color-success-500, #28a745); color: white; } .tb-btn-success:hover { background-color: var(--tb-color-success-600, #218838); }

        .tb-space-y-6 > *:not([hidden]) ~ *:not([hidden]) { margin-top: 1.5rem; }
        .tb-mt-2 { margin-top: 0.5rem; } .tb-mb-1 { margin-bottom: 0.25rem; } .tb-mb-2 { margin-bottom: 0.5rem; }
        .tb-mr-1 { margin-right: 0.25rem; }
        .md\\:tb-w-2\\/3 { width: 66.666667%; } /* Adjusted for CSS literal */
        .tb-text-red-500 { color: #ef4444; } .tb-text-green-600 { color: #16a34a; } .tb-text-blue-500 { color: #3b82f6; }
        .tb-text-gray-500 { color: #6b7280; }
        body[data-theme="dark"] .tb-text-gray-500 { color: #9ca3af; }
        .tb-text-sm { font-size: 0.875rem; } .tb-text-md { font-size: 1rem; } .tb-text-lg { font-size: 1.125rem; }
        .tb-font-semibold { font-weight: 600; }
        .tb-flex { display: flex; } .tb-items-center { align-items: center; } .tb-cursor-pointer { cursor: pointer; }
        .tb-space-x-2 > *:not([hidden]) ~ *:not([hidden]) { margin-left: 0.5rem; }
        .toggle-switch { display: inline-flex; align-items: center; cursor: pointer; }
        .toggle-switch input { opacity: 0; width: 0; height: 0; }
        .toggle-slider { width: 40px; height: 20px; background-color: #ccc; border-radius: 20px; position: relative; transition: background-color 0.2s; }
        .toggle-slider:before { content: ""; position: absolute; height: 16px; width: 16px; left: 2px; bottom: 2px; background-color: white; border-radius: 50%; transition: transform 0.2s; }
        input:checked + .toggle-slider { background-color: var(--theme-primary, #007bff); }
        input:checked + .toggle-slider:before { transform: translateX(20px); }

    </style>
</head>
<body data-theme="system">
    <div id="user-dashboard">
        <div id="user-header">
            <button id="sidebar-toggle-btn" class="tb-btn" style="margin-right: 1rem; display: none; background: none; border: none; color: white;">
                <span class="material-symbols-outlined">menu</span>
            </button>
            <h1><span class="material-symbols-outlined">dashboard</span>User Dashboard</h1>
            <div class="header-actions">
                 <div id="darkModeToggleContainer" style="display: inline-flex; align-items: center; margin-right: 1.5rem;"></div>
                <nav id="user-nav">
                    <ul>
                        <li id="logoutButtonUser"><span class="material-symbols-outlined">logout</span>Logout</li>
                    </ul>
                </nav>
            </div>
        </div>
        <div id="user-container">
            <aside id="user-sidebar">
                 <ul>
                    <li data-section="my-profile" class="active"><span class="material-symbols-outlined">account_box</span>My Profile</li>
                    <li data-section="my-instances"><span class="material-symbols-outlined">dns</span>My Instances & Modules</li>
                    <li data-section="app-appearance"><span class="material-symbols-outlined">palette</span>Appearance</li>
                    <li data-section="user-settings"><span class="material-symbols-outlined">tune</span>Settings</li>
                </ul>
            </aside>
            <main id="user-content">
                <section id="my-profile-section" class="content-section active frosted-glass-pane">
                    <h2><span class="material-symbols-outlined">badge</span>My Profile</h2>
                    <div id="my-profile-content"><p class="tb-text-gray-500">Loading profile...</p></div>
                </section>
                <section id="my-instances-section" class="content-section frosted-glass-pane">
                    <h2><span class="material-symbols-outlined">developer_board</span>Active Instances & Modules</h2>
                    <div id="my-instances-content"><p class="tb-text-gray-500">Loading active instances and modules...</p></div>
                </section>
                <section id="app-appearance-section" class="content-section frosted-glass-pane">
                    <h2><span class="material-symbols-outlined">visibility</span>Application Appearance</h2>
                    <div id="app-appearance-content"><p class="tb-text-gray-500">Loading appearance settings...</p></div>
                </section>
                <section id="user-settings-section" class="content-section frosted-glass-pane">
                    <h2><span class="material-symbols-outlined">settings_applications</span>User Settings</h2>
                    <div id="user-settings-content"><p class="tb-text-gray-500">Loading user settings...</p></div>
                </section>
            </main>
        </div>
        <div id="sidebar-backdrop"></div>
    </div>

    <script type="module">
        if (typeof TB === 'undefined' || !TB.ui || !TB.api || !TB.user || !TB.utils) {
            console.error('CRITICAL: TB (tbjs) or its core modules are not defined.');
            document.body.innerHTML = '<div style="padding: 20px; text-align: center; font-size: 1.2em; color: red;">Critical Error: Frontend library (tbjs) failed to load.</div>';
        } else {
            console.log('TB object found. Initializing User Dashboard.');
            let currentUserDetails = null;
            let allAvailableModules = []; // To store modules from app.get_all_mods()

            async function initializeUserDashboard() {
                console.log("User Dashboard Initializing with tbjs...");
                TB.ui.DarkModeToggle.init();
                setupUserNavigation();
                await setupUserLogout();

                try {
                    const userRes = await TB.api.request('CloudM.UserAccountManager', 'get_current_user_from_request_api_wrapper', null, 'GET');
                    if (userRes.error === TB.ToolBoxError.none && userRes.get()) {
                        currentUserDetails = userRes.get();
                        if (currentUserDetails.name) {
                            const userHeaderTitle = document.querySelector('#user-header h1');
                            if (userHeaderTitle) {
                                userHeaderTitle.innerHTML = `<span class="material-symbols-outlined">dashboard</span>Welcome, ${TB.utils.escapeHtml(currentUserDetails.name)}!`;
                            }
                        }
                        // Load all available modules for the "My Instances" section
                        const modulesRes = await TB.api.request('CloudM.UserDashboard', 'get_all_available_modules', null, 'GET');
                        if (modulesRes.error === TB.ToolBoxError.none) {
                            allAvailableModules = modulesRes.get() || [];
                        } else {
                            console.warn("Could not fetch all available modules list:", modulesRes.info.help_text);
                        }

                        await showUserSection('my-profile'); // Default section
                    } else {
                        console.error("Failed to load current user for dashboard:", userRes.info.help_text);
                        document.getElementById('user-content').innerHTML = '<p class="tb-text-red-500">Error: Could not load your details. Please try logging in again.</p>';
                    }
                } catch (e) {
                    console.error("Error fetching current user for dashboard:", e);
                    document.getElementById('user-content').innerHTML = '<p class="tb-text-red-500">Network error loading your details.</p>';
                }
                setupMobileSidebar();
            }


            function setupMobileSidebar() {
            const sidebar = document.getElementById('user-sidebar');
            const toggleBtn = document.getElementById('sidebar-toggle-btn');
            const backdrop = document.getElementById('sidebar-backdrop'); // Get the backdrop

            if (!sidebar || !toggleBtn || !backdrop) { // Check for backdrop too
                console.warn("Mobile sidebar elements (sidebar, toggle button, or backdrop) not found. Skipping mobile sidebar setup.");
                // If toggle button isn't there, ensure it's not shown via CSS either if it was meant to be.
                if(toggleBtn) toggleBtn.style.display = 'none';
                return;
            }

            // Show toggle button only on smaller screens
            function updateToggleBtnVisibility() {
                if (window.innerWidth < 768) { // Your mobile breakpoint
                    toggleBtn.style.display = 'inline-flex';
                } else {
                    toggleBtn.style.display = 'none';
                    sidebar.classList.remove('open'); // Ensure sidebar is closed if screen resizes to desktop
                    backdrop.classList.remove('active');
                    document.body.style.overflow = '';
                }
            }

            updateToggleBtnVisibility(); // Initial check
            window.addEventListener('resize', updateToggleBtnVisibility);


            toggleBtn.addEventListener('click', (e) => {
                e.stopPropagation(); // Prevent click from bubbling if needed
                sidebar.classList.toggle('open');
                backdrop.classList.toggle('active');
                document.body.style.overflow = sidebar.classList.contains('open') ? 'hidden' : '';
            });

            backdrop.addEventListener('click', () => {
                sidebar.classList.remove('open');
                backdrop.classList.remove('active');
                document.body.style.overflow = '';
            });

            sidebar.querySelectorAll('li[data-section]').forEach(item => {
                item.addEventListener('click', () => {
                    // Only close if it's mobile view and sidebar is open
                    if (window.innerWidth < 768 && sidebar.classList.contains('open')) {
                        sidebar.classList.remove('open');
                        backdrop.classList.remove('active');
                        document.body.style.overflow = '';
                    }
                });
            });
             console.log("Mobile sidebar setup complete.");
        }

            function _waitForTbInitUser(callback) {
                 if (window.TB?.events && window.TB.config?.get('appRootId')) {
                    callback();
                } else {
                    document.addEventListener('tbjs:initialized', callback, { once: true });
                }
            }
            _waitForTbInitUser(initializeUserDashboard);

            function setupUserNavigation() {
                const navItems = document.querySelectorAll('#user-sidebar li[data-section]');
                navItems.forEach(item => {
                    item.addEventListener('click', async () => {
                        navItems.forEach(i => i.classList.remove('active'));
                        item.classList.add('active');
                        const sectionId = item.getAttribute('data-section');
                        await showUserSection(sectionId);
                    });
                });
            }

            async function showUserSection(sectionId) {
                document.querySelectorAll('#user-content .content-section').forEach(s => s.classList.remove('active'));
                const activeSection = document.getElementById(`${sectionId}-section`);
                if (activeSection) {
                    activeSection.classList.add('active');
                    const contentDivId = `${sectionId}-content`;
                    const contentDiv = document.getElementById(contentDivId);
                    if (!contentDiv) { console.error(`Content div ${contentDivId} not found.`); return; }

                    contentDiv.innerHTML = `<p class="tb-text-gray-500">Loading ${sectionId.replace(/-/g, " ")}...</p>`;
                    if (sectionId === 'my-profile') await loadMyProfileSection(contentDivId);
                    else if (sectionId === 'my-instances') await loadMyInstancesAndModulesSection(contentDivId);
                    else if (sectionId === 'app-appearance') await loadAppearanceSection(contentDivId);
                    else if (sectionId === 'user-settings') await loadGenericUserSettingsSection(contentDivId);
                }
            }

            async function setupUserLogout() {
                const logoutButton = document.getElementById('logoutButtonUser');
                if (logoutButton) {
                    logoutButton.addEventListener('click', async () => {
                        TB.ui.Loader.show("Logging out...");
                        await TB.user.logout();
                        window.location.href = '/';
                        TB.ui.Loader.hide();
                    });
                }
            }

            async function loadMyProfileSection(targetDivId = 'my-profile-content') {
                const contentDiv = document.getElementById(targetDivId);
                if (!currentUserDetails) { contentDiv.innerHTML = "<p class='tb-text-red-500'>Profile details not available.</p>"; return; }
                const user = currentUserDetails;
                const emailSectionId = `user-email-updater-${TB.utils.uniqueId()}`;
                const expFeaturesIdUser = `user-exp-features-${TB.utils.uniqueId()}`;
                const personaStatusIdUser = `user-persona-status-${TB.utils.uniqueId()}`;
                const magicLinkIdUser = `user-magic-link-${TB.utils.uniqueId()}`;

                let personaBtnHtmlUser = !user.is_persona ?
                    `<button id="registerPersonaBtnUser" class="tb-btn tb-btn-success tb-mt-2"><span class="material-symbols-outlined tb-mr-1">fingerprint</span>Add Persona Device</button><div id="${personaStatusIdUser}" class="tb-text-sm tb-mt-1"></div>` :
                    `<p class='tb-text-md tb-text-green-600 dark:tb-text-green-400'><span class="material-symbols-outlined tb-mr-1" style="vertical-align: text-bottom;">verified_user</span>Persona (WebAuthn) is configured.</p>`;

                contentDiv.innerHTML = `
                    <div class="tb-space-y-6">
                        <div>
                            <h4 class="tb-text-lg tb-font-semibold tb-mb-2">Email Address</h4>
                            <div id="${emailSectionId}" class="tb-space-y-2">
                                 <p class="tb-text-md"><strong>Current Email:</strong> ${user.email ? TB.utils.escapeHtml(user.email) : "Not set"}</p>
                                 <input type="email" name="new_email_user" value="${user.email ? TB.utils.escapeHtml(user.email) : ''}" class="tb-input md:tb-w-2/3" placeholder="Enter new email">
                                 <button class="tb-btn tb-btn-primary tb-mt-2"
                                    data-hx-post="/api/CloudM.UserAccountManager/update_email"
                                    data-hx-include="[name='new_email_user']"
                                    data-hx-target="#${emailSectionId}" data-hx-swap="innerHTML"><span class="material-symbols-outlined tb-mr-1">save</span>Update Email</button>
                            </div>
                        </div>
                        <div>
                            <h4 class="tb-text-lg tb-font-semibold tb-mb-2">Magic Link</h4>
                            <div id="${magicLinkIdUser}">
                                <button id="requestMagicLinkBtnUser" class="tb-btn tb-btn-secondary"><span class="material-symbols-outlined tb-mr-1">link</span>Request New Magic Link</button>
                                <p class="tb-text-sm tb-mt-1">Request a new magic link to log in on other devices.</p>
                            </div>
                        </div>
                        <div>
                            <h4 class="tb-text-lg tb-font-semibold tb-mb-2">Persona Device (WebAuthn)</h4>
                            ${personaBtnHtmlUser}
                        </div>
                        <div>
                            <h4 class="tb-text-lg tb-font-semibold tb-mb-2">Application Settings</h4>
                            <div id="${expFeaturesIdUser}">
                                <label class="tb-label tb-checkbox-label">
                                    <input type="checkbox" name="exp_features_user_val" ${user.settings && user.settings.experimental_features ? "checked" : ""}
                                           class="tb-checkbox"
                                           data-hx-post="/api/CloudM.UserAccountManager/update_setting"
                                           data-hx-vals='{"setting_key": "experimental_features", "setting_value": event.target.checked ? "true" : "false"}'
                                           data-hx-target="#${expFeaturesIdUser}" data-hx-swap="innerHTML">
                                    <span class="tb-text-md">Enable Experimental Features</span>
                                </label>
                            </div>
                        </div>
                    </div>`;
                if (window.htmx) window.htmx.process(contentDiv);

                document.getElementById('requestMagicLinkBtnUser')?.addEventListener('click', async () => {
                    TB.ui.Loader.show("Requesting magic link...");
                    const magicLinkRes = await TB.api.request('CloudM.UserDashboard', 'request_my_magic_link', null, 'POST');
                    TB.ui.Loader.hide();
                    if (magicLinkRes.error === TB.ToolBoxError.none) {
                        TB.ui.Toast.showSuccess(magicLinkRes.info.help_text || "Magic link request sent to your email.");
                    } else {
                        TB.ui.Toast.showError(`Failed to request magic link: ${TB.utils.escapeHtml(magicLinkRes.info.help_text)}`);
                    }
                });

                const personaBtnUsr = document.getElementById('registerPersonaBtnUser');
                if (personaBtnUsr) {
                    personaBtnUsr.addEventListener('click', async () => {
                        const statusDiv = document.getElementById(personaStatusIdUser);
                        if (!statusDiv) return;
                        statusDiv.innerHTML = '<p class="tb-text-sm tb-text-blue-500">Initiating WebAuthn registration...</p>';
                        if (window.TB?.user && user.name) {
                            const result = await window.TB.user.registerWebAuthnForCurrentUser(user.name);
                            if (result.success) {
                                statusDiv.innerHTML = `<p class="tb-text-sm tb-text-green-500">${TB.utils.escapeHtml(result.message)} Refreshing details...</p>`;
                                TB.ui.Toast.showSuccess("Persona registered! Refreshing...");
                                setTimeout(async () => {
                                    const updatedUserRes = await TB.api.request('CloudM.UserAccountManager', 'get_current_user_from_request_api_wrapper', null, 'GET');
                                    if (updatedUserRes.error === TB.ToolBoxError.none && updatedUserRes.get()) {
                                        currentUserDetails = updatedUserRes.get();
                                        await loadMyProfileSection();
                                    }
                                }, 1500);
                            } else {
                                statusDiv.innerHTML = `<p class="tb-text-sm tb-text-red-500">Error: ${TB.utils.escapeHtml(result.message)}</p>`;
                            }
                        } else { statusDiv.innerHTML = '<p class="tb-text-sm tb-text-red-500">User details unavailable for WebAuthn.</p>'; }
                    });
                }
            }

            async function loadMyInstancesAndModulesSection(targetDivId = 'my-instances-content') {
                const contentDiv = document.getElementById(targetDivId);
                if (!contentDiv) return;
                try {
                    const response = await TB.api.request('CloudM.UserDashboard', 'get_my_active_instances', null, 'GET');
                    if (response.error === TB.ToolBoxError.none) {
                        renderMyInstancesAndModules(response.get(), contentDiv);
                    } else {
                        contentDiv.innerHTML = `<p class="tb-text-red-500">Error loading instances: ${TB.utils.escapeHtml(response.info.help_text)}</p>`;
                    }
                } catch(e) {
                    contentDiv.innerHTML = '<p class="tb-text-red-500">Network error fetching instances.</p>'; console.error(e);
                }
            }

            function renderMyInstancesAndModules(instances, contentDiv) {
                if (!instances || instances.length === 0) {
                    contentDiv.innerHTML = '<p class="tb-text-gray-500">You have no active instances. Modules can be activated once an instance is present.</p>'; return;
                }
                // Assuming one primary instance for now for module management simplicity
                const primaryInstance = instances[0];
                let html = `<div class="instance-card">
                                <h4>Primary Instance (Session ID: ${TB.utils.escapeHtml(primaryInstance.SiID)})</h4>
                                <p class="tb-text-sm">WebSocket ID: ${TB.utils.escapeHtml(primaryInstance.webSocketID)}</p>
                                <button class="tb-btn tb-btn-danger tb-mt-2" data-instance-siid="${primaryInstance.SiID}"><span class="material-symbols-outlined tb-mr-1">close</span>Close This Instance</button>
                            </div>
                            <h3 class="tb-text-lg tb-font-semibold tb-mt-4 tb-mb-2">Available Modules</h3>
                            <div class="settings-grid">`;

                const activeModuleNames = primaryInstance.live_modules.map(m => m.name);

                allAvailableModules.forEach(modName => {
                    const isActive = activeModuleNames.includes(modName);
                    html += `
                        <div class="module-card">
                            <h4>${TB.utils.escapeHtml(modName)}</h4>
                            <label class="toggle-switch">
                                <input type="checkbox" data-module-name="${TB.utils.escapeHtml(modName)}" ${isActive ? 'checked' : ''}>
                                <span class="toggle-slider"></span>
                            </label>
                        </div>`;
                });
                html += '</div>';
                contentDiv.innerHTML = html;

                contentDiv.querySelector(`button[data-instance-siid="${primaryInstance.SiID}"]`)?.addEventListener('click', async (e) => {
                    // Same close instance logic as before
                    const siidToClose = e.currentTarget.dataset.instanceSiid;
                        TB.ui.Modal.show({
                            title: "Confirm Instance Closure",
                            content: `<p>Are you sure you want to close instance <strong>${siidToClose}</strong>? This may log you out from that session.</p>`,
                            buttons: [
                                { text: 'Cancel', action: m => m.close(), variant: 'secondary' },
                                { text: 'Close Instance', variant: 'danger', action: async m => {
                                    TB.ui.Loader.show("Closing instance...");
                                    const closeRes = await TB.api.request('CloudM.UserDashboard', 'close_my_instance', { siid: siidToClose }, 'POST');
                                    if (closeRes.error === TB.ToolBoxError.none) {
                                        TB.ui.Toast.showSuccess("Instance closed successfully.");
                                        await loadMyInstancesAndModulesSection();
                                    } else { TB.ui.Toast.showError(`Error closing instance: ${TB.utils.escapeHtml(closeRes.info.help_text)}`); }
                                    TB.ui.Loader.hide(); m.close();
                                }}
                            ]
                        });
                });

                contentDiv.querySelectorAll('.toggle-switch input[data-module-name]').forEach(toggle => {
                    toggle.addEventListener('change', async (e) => {
                        const moduleName = e.target.dataset.moduleName;
                        const activate = e.target.checked;
                        TB.ui.Loader.show(`${activate ? 'Activating' : 'Deactivating'} ${moduleName}...`);
                        const apiPayload = { module_name: moduleName, activate: activate, siid: primaryInstance.SiID };
                        const modUpdateRes = await TB.api.request('CloudM.UserDashboard', 'update_my_instance_modules', apiPayload, 'POST');
                        TB.ui.Loader.hide();
                        if (modUpdateRes.error === TB.ToolBoxError.none) {
                            TB.ui.Toast.showSuccess(`Module ${moduleName} ${activate ? 'activated' : 'deactivated'}.`);
                            // Refresh instance data to reflect change
                            const updatedInstanceRes = await TB.api.request('CloudM.UserDashboard', 'get_my_active_instances', null, 'GET');
                             if (updatedInstanceRes.error === TB.ToolBoxError.none) {
                                const updatedInstances = updatedInstanceRes.get();
                                if (updatedInstances && updatedInstances.length > 0) {
                                    // Update the displayed active modules without full re-render for smoother UX if possible
                                    const currentInstanceDisplay = updatedInstances.find(inst => inst.SiID === primaryInstance.SiID);
                                    if (currentInstanceDisplay) {
                                        primaryInstance.live_modules = currentInstanceDisplay.live_modules; // Update local cache
                                        // For a full refresh: renderMyInstancesAndModules(updatedInstances, contentDiv);
                                    }
                                }
                            }
                        } else {
                            TB.ui.Toast.showError(`Failed to update module ${moduleName}: ${TB.utils.escapeHtml(modUpdateRes.info.help_text)}`);
                            e.target.checked = !activate; // Revert toggle on error
                        }
                    });
                });
            }

            async function loadAppearanceSection(targetDivId = 'app-appearance-content') {
                const contentDiv = document.getElementById(targetDivId);
                if (!contentDiv || !currentUserDetails) return;

                const userSettings = currentUserDetails.settings || {};
                const themeOverrides = userSettings.theme_overrides || {};
                const graphicsSettings = userSettings.graphics_settings || {};

                const themeVars = [
                    { name: 'Theme Background', key: '--theme-bg', type: 'color', default: '#f0f2f5' },
                    { name: 'Theme Text', key: '--theme-text', type: 'color', default: '#333333' },
                    { name: 'Theme Primary', key: '--theme-primary', type: 'color', default: '#0056b3' },
                    { name: 'Theme Secondary', key: '--theme-secondary', type: 'color', default: '#537FE7' },
                    { name: 'Theme Accent', key: '--theme-accent', type: 'color', default: '#045fab' },
                    { name: 'Glass BG', key: '--glass-bg', type: 'text', placeholder: 'rgba(255,255,255,0.6)', default: 'rgba(255,255,255,0.75)'},
                    // Add more theme variables as needed
                ];

                let themeVarsHtml = themeVars.map(v => `
                    <div class="setting-item">
                        <label for="theme-var-${v.key}">${v.name} (${v.key}):</label>
                        <input type="${v.type}" id="theme-var-${v.key}" data-var-key="${v.key}"
                               value="${TB.utils.escapeHtml(themeOverrides[v.key] || v.default)}"
                               ${v.placeholder ? `placeholder="${v.placeholder}"` : ''}>
                    </div>
                `).join('');

                contentDiv.innerHTML = `
                    <div class="tb-space-y-6">
                        <div>
                            <h4 class="tb-text-lg tb-font-semibold tb-mb-2">Site Theme Preference</h4>
                            <p class="tb-text-sm tb-mb-2">Current site theme preference: <strong id="currentThemePreferenceText">${TB.ui.theme.getPreference()}</strong></p>
                            <div class="tb-flex tb-space-x-2">
                                <button class="tb-btn" data-theme-set="light">Light</button>
                                <button class="tb-btn" data-theme-set="dark">Dark</button>
                                <button class="tb-btn" data-theme-set="system">System Default</button>
                            </div>
                        </div>
                        <div>
                            <h4 class="tb-text-lg tb-font-semibold tb-mb-2">Custom Colors & Styles</h4>
                            <div class="settings-grid">${themeVarsHtml}</div>
                        </div>
                        <div>
                            <h4 class="tb-text-lg tb-font-semibold tb-mb-2">Background Settings</h4>
                            <div class="settings-grid">
                                <div class="setting-item">
                                    <label for="bgTypeSelect">Background Type:</label>
                                    <select id="bgTypeSelect" class="tb-input">
                                        <option value="color" ${(graphicsSettings.type || 'color') === 'color' ? 'selected':''}>Color</option>
                                        <option value="image" ${graphicsSettings.type === 'image' ? 'selected':''}>Image</option>
                                        <option value="3d" ${graphicsSettings.type === '3d' ? 'selected':''}>3D Animated</option>
                                    </select>
                                </div>
                                <div class="setting-item" id="bgColorSetting" style="display: ${(graphicsSettings.type || 'color') === 'color' ? 'block':'none'};">
                                    <label for="bgColorInput">Background Color (Light Mode):</label>
                                    <input type="color" id="bgColorInputLight" value="${graphicsSettings.bgColorLight || '#FFFFFF'}">
                                    <label for="bgColorInputDark" class="tb-mt-2">Background Color (Dark Mode):</label>
                                    <input type="color" id="bgColorInputDark" value="${graphicsSettings.bgColorDark || '#121212'}">
                                </div>
                                <div class="setting-item" id="bgImageSetting" style="display: ${graphicsSettings.type === 'image' ? 'block':'none'};">
                                    <label for="bgImageUrlInputLight">Background Image URL (Light Mode):</label>
                                    <input type="text" id="bgImageUrlInputLight" class="tb-input" value="${graphicsSettings.bgImageUrlLight || ''}" placeholder="https://example.com/light.jpg">
                                     <label for="bgImageUrlInputDark" class="tb-mt-2">Background Image URL (Dark Mode):</label>
                                    <input type="text" id="bgImageUrlInputDark" class="tb-input" value="${graphicsSettings.bgImageUrlDark || ''}" placeholder="https://example.com/dark.jpg">
                                </div>
                                <div class="setting-item" id="bg3dSetting" style="display: ${graphicsSettings.type === '3d' ? 'block':'none'};">
                                    <label for="sierpinskiDepthInput">3D Sierpinski Depth (0-5):</label>
                                    <input type="number" id="sierpinskiDepthInput" class="tb-input" min="0" max="5" value="${graphicsSettings.sierpinskiDepth || 2}">
                                    <label for="animationSpeedFactorInput" class="tb-mt-2">3D Animation Speed Factor (0.1-2.0):</label>
                                    <input type="number" id="animationSpeedFactorInput" class="tb-input" min="0.1" max="2.0" step="0.1" value="${graphicsSettings.animationSpeedFactor || 1.0}">
                                </div>
                            </div>
                        </div>
                        <button id="saveAppearanceSettingsBtn" class="tb-btn tb-btn-primary tb-mt-4"><span class="material-symbols-outlined">save</span>Save Appearance Settings</button>
                    </div>
                `;

                document.getElementById('bgTypeSelect').addEventListener('change', function() {
                    document.getElementById('bgColorSetting').style.display = this.value === 'color' ? 'block' : 'none';
                    document.getElementById('bgImageSetting').style.display = this.value === 'image' ? 'block' : 'none';
                    document.getElementById('bg3dSetting').style.display = this.value === '3d' ? 'block' : 'none';
                });

                contentDiv.querySelectorAll('button[data-theme-set]').forEach(btn => {
                    btn.addEventListener('click', () => {
                        const newPref = btn.dataset.themeSet;
                        TB.ui.theme.setPreference(newPref); // This updates tbjs internal and applies system-wide
                        document.getElementById('currentThemePreferenceText').textContent = newPref;
                        TB.ui.Toast.showInfo(`Site theme preference set to ${newPref}`);
                    });
                });

                document.getElementById('saveAppearanceSettingsBtn').addEventListener('click', async () => {
                    TB.ui.Loader.show("Saving appearance settings...");
                    let newThemeOverrides = {};
                    themeVars.forEach(v => {
                        newThemeOverrides[v.key] = document.getElementById(`theme-var-${v.key}`).value;
                    });

                    let newGraphicsSettings = {
                        type: document.getElementById('bgTypeSelect').value,
                        bgColorLight: document.getElementById('bgColorInputLight').value,
                        bgColorDark: document.getElementById('bgColorInputDark').value,
                        bgImageUrlLight: document.getElementById('bgImageUrlInputLight').value,
                        bgImageUrlDark: document.getElementById('bgImageUrlInputDark').value,
                        sierpinskiDepth: parseInt(document.getElementById('sierpinskiDepthInput').value),
                        animationSpeedFactor: parseFloat(document.getElementById('animationSpeedFactorInput').value)
                    };

                    const payload = {
                        settings: { // Assuming settings are namespaced on the backend
                            theme_overrides: newThemeOverrides,
                            graphics_settings: newGraphicsSettings,
                            // Important: include the current site theme preference from TB.ui.theme
                            site_theme_preference: TB.ui.theme.getPreference()
                        }
                    };

                    const result = await TB.api.request('CloudM.UserDashboard', 'update_my_appearance_settings', payload, 'POST');
                    TB.ui.Loader.hide();
                    if (result.error === TB.ToolBoxError.none) {
                        TB.ui.Toast.showSuccess("Appearance settings saved!");
                        // Update currentUserDetails and re-apply settings client-side
                        if(currentUserDetails.settings) {
                            currentUserDetails.settings.theme_overrides = newThemeOverrides;
                            currentUserDetails.settings.graphics_settings = newGraphicsSettings;
                            currentUserDetails.settings.site_theme_preference = TB.ui.theme.getPreference();
                        } else {
                            currentUserDetails.settings = payload.settings;
                        }
                        // Trigger tbjs to re-apply these settings from TB.state if it supports it,
                        // or manually apply them now.
                        _applyCustomThemeVariables(newThemeOverrides);
                        _applyCustomBackgroundSettings(newGraphicsSettings); // Requires TB.graphics to use these
                         TB.ui.theme.setPreference(newGraphicsSettings.site_theme_preference || 'system'); // Re-apply base theme choice

                    } else {
                        TB.ui.Toast.showError(`Failed to save settings: ${TB.utils.escapeHtml(result.info.help_text)}`);
                    }
                });
            }

            // Inside your _applyCustomThemeVariables function in the User Dashboard JS
            function _applyCustomThemeVariables(themeOverrides) {
                if (!themeOverrides) return;
                const rootStyle = document.documentElement.style;
                for (const [key, value] of Object.entries(themeOverrides)) {
                    if (key.startsWith('--theme-') || key.startsWith('--glass-') || key.startsWith('--sidebar-')) { // Be specific
                        rootStyle.setProperty(key, value);
                    }
                }
                console.log("Applied custom theme variables:", themeOverrides);
                // Additionally, you might need to inform tbjs.ui.theme if its internal state
                // or background calculations depend on these specific CSS variables.
                // This might involve emitting an event or calling a theme refresh method if available.
                // For example, if TB.ui.theme._applyBackground() reads these vars.
                if (TB.ui.theme?._applyBackground) {
                    TB.ui.theme._applyBackground(); // If safe to call directly
                } else if (TB.events) {
                    TB.events.emit('customTheme:variablesApplied', themeOverrides);
                }
            }

            // Inside your _applyCustomBackgroundSettings function
            function _applyCustomBackgroundSettings(graphicsSettings) {
                if (!graphicsSettings) return;
                console.log("Applying user-defined background settings:", graphicsSettings);

                // 1. Update tbjs's internal config for theme/background so it persists across reloads/theme toggles
                // This part is tricky as TB.config is usually set at init.
                // A better approach would be for TB.ui.theme to listen to state changes for these settings.
                // For now, let's assume TB.state is the source of truth for these overrides.

                // Example: Saving to TB.state (which should then be read by TB.ui.theme)
                TB.state.set('user.settings.graphics_settings', graphicsSettings, { persist: true });


                // 2. Directly trigger TB.ui.theme to re-evaluate its background
                //    This requires TB.ui.theme to be able to consume these settings.
                if (TB.ui.theme && typeof TB.ui.theme.updateBackgroundConfiguration === 'function') {
                    // Ideal: TB.ui.theme has a method to accept new background config parts
                    TB.ui.theme.updateBackgroundConfiguration({
                        type: graphicsSettings.type,
                        light: {
                            color: graphicsSettings.bgColorLight,
                            image: graphicsSettings.bgImageUrlLight
                        },
                        dark: {
                            color: graphicsSettings.bgColorDark,
                            image: graphicsSettings.bgImageUrlDark
                        }
                        // ... and potentially placeholder settings if user can configure them
                    });
                } else if (TB.ui.theme?._applyBackground) {
                    // Less ideal, but might work if _applyBackground re-reads from a source affected by the state update
                    TB.ui.theme._applyBackground();
                }


                // 3. Directly interact with TB.graphics for 3D specific settings
                if (graphicsSettings.type === '3d' && TB.graphics) {
                    if (typeof graphicsSettings.sierpinskiDepth === 'number' && TB.graphics.setSierpinskiDepth) {
                        TB.graphics.setSierpinskiDepth(graphicsSettings.sierpinskiDepth);
                    }
                    if (typeof graphicsSettings.animationSpeedFactor === 'number' && TB.graphics.setAnimationSpeed) {
                        // Assuming default base speeds and only factor is changed by user
                        TB.graphics.setAnimationSpeed( // Use existing animParams or defaults if available
                            TB.graphics.animParams?.x || 0.0001,
                            TB.graphics.animParams?.y || 0.0002,
                            TB.graphics.animParams?.z || 0.00005,
                            graphicsSettings.animationSpeedFactor
                        );
                    }
                }
                TB.logger.log("[UserDashboard] User background settings applied/updated.");
            }


            async function loadGenericUserSettingsSection(targetDivId = 'user-settings-content') {
                 const contentDiv = document.getElementById(targetDivId);
                 if (!contentDiv || !currentUserDetails) return;
                 const userSettings = currentUserDetails.settings || {};

                 // Example: Storing arbitrary JSON data by the user
                 const customJsonData = userSettings.custom_json_data || {};

                 contentDiv.innerHTML = `
                    <div class="tb-space-y-4">
                        <h4 class="tb-text-lg tb-font-semibold">Custom User Data (JSON)</h4>
                        <p class="tb-text-sm">Store arbitrary JSON data for your use. This is saved to your account.</p>
                        <textarea id="customJsonDataInput" class="tb-input" rows="8" placeholder='${JSON.stringify({"myKey": "myValue", "nested": {"num": 123}}, null, 2)}'>${TB.utils.escapeHtml(JSON.stringify(customJsonData, null, 2))}</textarea>
                        <button id="saveCustomJsonDataBtn" class="tb-btn tb-btn-primary tb-mt-2"><span class="material-symbols-outlined">save</span>Save Custom Data</button>
                        <p id="customJsonDataStatus" class="tb-text-sm tb-mt-1"></p>
                    </div>
                 `;
                 document.getElementById('saveCustomJsonDataBtn').addEventListener('click', async () => {
                    const textarea = document.getElementById('customJsonDataInput');
                    const statusEl = document.getElementById('customJsonDataStatus');
                    let jsonData;
                    try {
                        jsonData = JSON.parse(textarea.value);
                    } catch (err) {
                        statusEl.textContent = 'Error: Invalid JSON format.';
                        TB.ui.Toast.showError('Invalid JSON format provided.');
                        return;
                    }
                    statusEl.textContent = 'Saving...';

                    const saveResult = await TB.api.request('CloudM.UserAccountManager', 'update_user_specific_setting',
                        { setting_key: 'custom_json_data', setting_value: jsonData }, // Send parsed JSON
                        'POST'
                    );

                    if (saveResult.error === TB.ToolBoxError.none) {
                        statusEl.textContent = 'Custom data saved successfully!';
                        TB.ui.Toast.showSuccess('Custom data saved.');
                        if(currentUserDetails.settings) currentUserDetails.settings['custom_json_data'] = jsonData;
                        else currentUserDetails.settings = {'custom_json_data': jsonData};
                    } else {
                        statusEl.textContent = `Error: ${TB.utils.escapeHtml(saveResult.info.help_text)}`;
                        TB.ui.Toast.showError('Failed to save custom data.');
                    }
                 });
            }
        } // End of TB check
    </script>
"""
    return Result.html(html_content)


# --- API Endpoints for UserDashboard ---

@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def get_my_active_instances(app: App, request: RequestData):
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="User not authenticated.", exec_code=401)

    instance_data_result = get_user_instance_internal(current_user.uid,
                                                      hydrate=True)  # hydrate=True to get live modules

    active_instances_output = []
    if instance_data_result and isinstance(instance_data_result, dict):
        live_modules_info = []
        if instance_data_result.get("live"):  # 'live' contains {mod_name: spec, ...}
            for mod_name, spec_val in instance_data_result.get("live").items():
                live_modules_info.append({"name": mod_name, "spec": str(spec_val)})

        instance_summary = {
            "SiID": instance_data_result.get("SiID"),
            "VtID": instance_data_result.get("VtID"),  # Important for module context
            "webSocketID": instance_data_result.get("webSocketID"),
            "live_modules": live_modules_info,  # List of {"name": "ModName", "spec": "spec_id"}
            "saved_modules": instance_data_result.get("save", {}).get("mods", [])  # List of mod_names
        }
        active_instances_output.append(instance_summary)

    return Result.json(data=active_instances_output)


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def get_all_available_modules(app: App, request: RequestData):
    current_user = await get_current_user_from_request(app, request)
    if not current_user:  # Minimal auth check, could be stricter if needed
        return Result.default_user_error(info="User not authenticated.", exec_code=401)

    all_mods = app.get_all_mods()  # This is synchronous
    return Result.json(data=all_mods)


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def update_my_instance_modules(app: App, request: RequestData, data: dict):
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="User not authenticated.", exec_code=401)

    module_name = data.get("module_name")
    activate = data.get("activate", False)  # boolean
    instance_siid = data.get("siid")  # The SiID of the instance to modify

    if not module_name or not instance_siid:
        return Result.default_user_error(info="Module name and instance SIID are required.")

    # --- Placeholder for actual logic ---
    # This is where the complex part of dynamically loading/unloading modules for a *specific user instance*
    # (identified by SiID or its associated VtID) would go.
    # 1. Validate that the instance_siid belongs to current_user.
    # 2. Get the VtID associated with this SiID from UserInstances.
    # 3. If activating:
    #    - Check if module is already active for this VtID.
    #    - Call something like `app.get_mod(module_name, spec=VtID)` or a dedicated
    #      `UserInstanceManager.activate_module_for_instance(VtID, module_name)`
    #    - Update the 'live' and 'save.mods' in the user's instance data in UserInstances and DB.
    # 4. If deactivating:
    #    - Call `app.remove_mod(module_name, spec=VtID)` or
    #      `UserInstanceManager.deactivate_module_for_instance(VtID, module_name)`
    #    - Update 'live' and 'save.mods'.
    # This requires `UserInstances.py` to be significantly enhanced or a new manager.
    app.print(
        f"User '{current_user.name}' requested to {'activate' if activate else 'deactivate'} module '{module_name}' for instance '{instance_siid}'. (Placeholder)",
        "INFO")

    # Simulate success for UI testing
    # In reality, update the user's instance in UserInstances and persist it.
    # The `get_user_instance_internal` should be modified or a new setter created
    # to update the live and saved mods for the specific instance.

    # Example of how UserInstances might be updated (needs methods in UserInstances.py):
    # from .UserInstances import update_module_in_instance
    # update_success = update_module_in_instance(app, current_user.uid, instance_siid, module_name, activate)
    # if update_success:
    #    return Result.ok(info=f"Module {module_name} {'activated' if activate else 'deactivated'}.")
    # else:
    #    return Result.default_internal_error(info=f"Failed to update module {module_name}.")

    return Result.ok(
        info=f"Module {module_name} {'activation' if activate else 'deactivation'} request processed (simulated).")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def close_my_instance(app: App, request: RequestData, data: dict):
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="User not authenticated.", exec_code=401)

    siid_to_close = data.get("siid")
    if not siid_to_close:
        return Result.default_user_error(info="Instance SIID is required.")

    # More robust check: Get the instance by UID, then check if the passed SIID matches that instance's SIID.
    user_instance = get_user_instance_internal(current_user.uid, hydrate=False)
    if not user_instance or user_instance.get("SiID") != siid_to_close:
        return Result.default_user_error(info="Instance not found or does not belong to the current user.")

    result_msg = close_user_instance_internal(
        current_user.uid)  # Assumes this closes the instance associated with siid_to_close

    if result_msg == "User instance not found" or result_msg == "No modules to close":
        return Result.ok(info="Instance already closed or not found: " + str(result_msg))
    elif result_msg is None:
        return Result.ok(info="Instance closed successfully.")
    else:
        return Result.default_internal_error(info="Could not close instance: " + str(result_msg))


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def request_my_magic_link(app: App, request: RequestData):
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="User not authenticated.", exec_code=401)

    # Call the existing magic link function from AuthManager
    magic_link_result = await request_magic_link_backend(app, username=current_user.name)

    if not magic_link_result.as_result().is_error():
        return Result.ok(info="Magic link request sent to your email: " + current_user.email)
    else:
        return Result.default_internal_error(info="Failed to send magic link: " + str(magic_link_result.info))


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def update_my_appearance_settings(app: App, request: RequestData, data: dict):
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="User not authenticated.", exec_code=401)

    settings_payload = data.get("settings")
    if not isinstance(settings_payload, dict):
        return Result.default_user_error(info="Invalid settings payload.")

    # Validate and sanitize settings_payload before saving
    # Example: theme_overrides should be a dict of string:string
    # graphics_settings should have known keys with correct types

    if current_user.settings is None:
        current_user.settings = {}

    # Merge carefully, don't just overwrite all settings
    if "theme_overrides" in settings_payload:
        current_user.settings["theme_overrides"] = settings_payload["theme_overrides"]
    if "graphics_settings" in settings_payload:
        current_user.settings["graphics_settings"] = settings_payload["graphics_settings"]
    if "site_theme_preference" in settings_payload:  # Save the general theme choice
        current_user.settings["site_theme_preference"] = settings_payload["site_theme_preference"]

    save_result = db_helper_save_user(app, asdict(current_user))
    if save_result.is_error():
        return Result.default_internal_error(info="Failed to save appearance settings: " + str(save_result.info))

    return Result.ok(info="Appearance settings saved successfully.",
                     data=current_user.settings)  # Return updated settings
