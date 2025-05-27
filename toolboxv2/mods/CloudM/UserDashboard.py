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
        /* Refactored styles for Ansyb Page, based on tbjs-main.css principles */

body {
    margin: 0;
    font-family: var(--font-family-base); /* Use main style's base font family */
    background-color: var(--theme-bg);
    color: var(--theme-text);
    transition: background-color var(--transition-medium), color var(--transition-medium);
}

#user-dashboard {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
}

#user-header {
    background-color: var(--theme-primary);
    color: var(--theme-text-on-primary); /* Text on primary background */
    padding: var(--spacing) calc(var(--spacing) * 1.5);
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 4px color-mix(in srgb, var(--theme-text) 10%, transparent); /* Subtle shadow adapting to theme */
}

#user-header h1 {
    margin: 0;
    font-size: var(--font-size-xl); /* Using responsive font size from main styles */
    font-weight: var(--font-weight-semibold);
    display: flex;
    align-items: center;
}

#user-header h1 .material-symbols-outlined {
    vertical-align: middle;
    font-size: 1.5em; /* Keeps original icon sizing relative to h1 */
    margin-right: 0.3em;
}

#user-header .header-actions {
    display: flex;
    align-items: center;
}

#user-nav ul {
    list-style: none;
    padding: 0;
    margin: 0 0 0 calc(var(--spacing) * 1.5);
    display: flex;
}

#user-nav li {
    margin-left: var(--spacing);
    cursor: pointer;
    padding: calc(var(--spacing) * 0.6) var(--spacing);
    border-radius: var(--radius-sm); /* Using 4px radius from main styles */
    transition: background-color var(--transition-fast);
    font-weight: var(--font-weight-medium);
    display: flex;
    align-items: center;
}

#user-nav li .material-symbols-outlined {
    vertical-align: text-bottom;
    margin-right: 0.3em;
}

#user-nav li:hover {
    /* Subtle hover effect, derived from the text color on primary */
    background-color: color-mix(in srgb, var(--theme-text-on-primary) 15%, transparent);
}

#user-container {
    display: flex;
    flex-grow: 1;
}

#user-sidebar {
    display: none;
    position: fixed;
    left: -250px;
    top: 0;
    bottom: 0;
    width: 240px;
    z-index: var(--z-navigation); /* Using z-index from main styles */
    transition: left var(--transition-medium); /* Using transition speed from main styles */
    overflow-y: auto;
    /* Background and border will be applied below, respecting media queries and themes */
}

#user-sidebar.open {
    display: flex;
    left: 0;
    box-shadow: 2px 0 5px color-mix(in srgb, var(--theme-text) 20%, transparent); /* Adapting shadow */
}

#sidebar-toggle-btn {
    display: inline-flex;
    padding: calc(var(--spacing) * 0.5);
    cursor: pointer;
    z-index: var(--z-nav-controls); /* Ensure it's above sidebar when closed */
}

@media (min-width: 768px) {
    #sidebar-toggle-btn {
        display: none;
    }
    #user-sidebar {
        display: flex;
        position: static;
        left: auto;
        width: 230px;
        transition: none;
        background-color: var(--theme-bg); /* Sidebar background for desktop */
        border-right: 1px solid var(--theme-border); /* Sidebar border for desktop */
    }
    /* This rule was : display:table - kept it as per instruction but display:grid is in the base .settings-grid */
    .settings-grid {
        display: table;
    }
    #user-container.sidebar-present #user-content {
        margin-left: 230px;
    }
}

#sidebar-backdrop {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: color-mix(in srgb, var(--theme-bg) 50%, black 50%); /* Semi-transparent backdrop */
    opacity: 0.7; /* Adjust opacity as needed */
    z-index: calc(var(--z-navigation) - 1); /* Below sidebar, above content */
}

#sidebar-backdrop.active {
    display: block;
}

@media (max-width: 767.98px) {
    #user-sidebar { /* Styles for mobile overlay sidebar */
        display: flex;
        position: fixed;
        left: -250px;
        top: 0;
        bottom: 0;
        width: 240px;
        z-index: var(--z-modal); /* Higher z-index for mobile overlay */
        transition: left var(--transition-medium);
        overflow-y: auto;
        background-color: var(--theme-bg); /* Mobile sidebar uses theme background */
        border-right: 1px solid var(--theme-border); /* And theme border */
    }
}
/* No specific body[data-theme="dark"] #user-sidebar needed if using --theme-bg and --theme-border */

#user-sidebar ul {
    list-style: none;
    padding: 0;
    margin: 0;
    width: 100%; /* Ensure list takes full width for items */
}

#user-sidebar li {
    padding: calc(var(--spacing) * 0.9) var(--spacing);
    margin-bottom: calc(var(--spacing) * 0.6);
    cursor: pointer;
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    font-weight: var(--font-weight-medium);
    color: var(--theme-text); /* Text color from theme */
    transition: background-color var(--transition-fast), color var(--transition-fast);
}

#user-sidebar li .material-symbols-outlined {
    margin-right: calc(var(--spacing) * 0.85);
    font-size: 1.4rem; /* Original size */
}

#user-sidebar li:hover {
    background-color: color-mix(in srgb, var(--theme-primary) 15%, transparent); /* Subtle primary hover */
    color: var(--theme-primary); /* Text color changes to primary on hover */
}

#user-sidebar li.active {
    background-color: var(--theme-primary);
    color: var(--theme-text-on-primary) !important;
    font-weight: var(--font-weight-semibold);
    box-shadow: 0 2px 8px color-mix(in srgb, var(--theme-primary) 30%, transparent);
}
/* No specific body[data-theme="dark"] needed for .active li if vars handle it */

#user-content {
    flex-grow: 1;
    padding: calc(var(--spacing) * 2);
}

.content-section {
    display: none;
}

.content-section.active {
    display: block;
    animation: fadeIn 0.5s ease-out; /* Original animation */
}

@keyframes fadeIn { /* Kept original keyframes */
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.content-section h2 {
    font-size: var(--font-size-3xl); /* Using responsive font size */
    font-weight: var(--font-weight-semibold);
    color: var(--theme-text);
    margin-bottom: calc(var(--spacing) * 1.8);
    padding-bottom: var(--spacing);
    border-bottom: 1px solid var(--theme-border);
    display: flex;
    align-items: center;
}

.content-section h2 .material-symbols-outlined {
    font-size: 1.3em; /* Original size */
    margin-right: 0.5em;
}
/* No specific body[data-theme="dark"] needed for h2 if vars handle it */

.frosted-glass-pane {
    background-color: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border-radius: var(--radius-lg); /* Using main style's large radius */
    padding: calc(var(--spacing) * 2);
    border: 1px solid var(--glass-border);
    box-shadow: var(--glass-shadow);
}
/* No specific body[data-theme="dark"] needed for .frosted-glass-pane */

.instance-card,
.module-card {
    border: 1px solid var(--theme-border);
    border-radius: var(--radius-md);
    padding: var(--spacing);
    margin-bottom: var(--spacing);
    /* Using input-bg as a slightly offset background, common for cards */
    background-color: var(--input-bg);
    box-shadow: 0 1px 3px color-mix(in srgb, var(--theme-text) 10%, transparent); /* Softer, theme-adaptive shadow */
}
/* No specific body[data-theme="dark"] needed for cards if vars handle it */

.instance-card h4,
.module-card h4 {
    margin-top: 0;
    color: var(--theme-primary); /* Titles in primary color */
}

.instance-card .module-list,
.module-card .module-status {
    list-style: disc;
    margin-left: calc(var(--spacing) * 1.5);
    font-size: var(--font-size-sm); /* Using responsive small font size */
}

.module-card {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.settings-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: calc(var(--spacing) * 1.5);
}

.setting-item {
    padding: var(--spacing);
    border: 1px solid var(--theme-border);
    border-radius: var(--radius-md);
    background-color: var(--input-bg); /* To make it consistent with other interactive element backgrounds */
}
/* No specific body[data-theme="dark"] needed for .setting-item */

.setting-item label {
    display: block;
    margin-bottom: calc(var(--spacing) * 0.5); /* Original was 0.5rem */
    font-weight: var(--font-weight-medium);
}

/* Inputs within setting-item will inherit from global input styles (section 4 of main.css) */
/* Overrides for sizing or specific padding if needed: */
.setting-item input[type="text"],
.setting-item input[type="number"],
.setting-item select,
.setting-item input[type="color"] { /* Apply consistent styling to color input too if desired */
    width: 100%;
    padding: calc(var(--spacing) * 0.5); /* Smaller padding as per original */
    margin-bottom: calc(var(--spacing) * 0.5);
    /* border, border-radius, focus will come from global input styles */
    /* If input[type="color"] needs specific dimensions different from main styles: */
}


.tb-label { /* Corresponds to main 'label' style */
    font-weight: var(--font-weight-medium);
    margin-bottom: calc(var(--spacing) * 0.3); /* From main 'label' style */
    display: block; /* From main 'label' style */
    font-size: var(--font-size-sm); /* From main 'label' style */
}

/* .tb-input should leverage main input styles and customize if needed */
.tb-input {
    display: block;
    width: 100%;
    padding: calc(var(--spacing) * 0.75) var(--spacing); /* Custom padding */
    font-family: inherit;
    font-size: var(--font-size-base);
    line-height: var(--line-height-normal);
    color: var(--theme-text); /* Text color from theme, will be on --input-bg */
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
/* No specific body[data-theme="dark"] needed for .tb-input */


.tb-checkbox-label {
    display: flex;
    align-items: center;
    cursor: pointer;
    /* For styling the checkbox itself, refer to main styles input[type="checkbox"] */
}

.tb-checkbox { /* The actual input element */
    /* Inherits from main input[type="checkbox"] styles */
    margin-right: calc(var(--spacing) * 0.5); /* Original was 0.5rem */
}

.tb-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: calc(var(--spacing) * 0.6) calc(var(--spacing) * 1.2); /* Matches main button */
    border-radius: var(--radius-md); /* Matches main button */
    font-weight: var(--font-weight-medium); /* Matches main button */
    cursor: pointer;
    transition: background-color var(--transition-fast), box-shadow var(--transition-fast), border-color var(--transition-fast), color var(--transition-fast);
    border: 1px solid transparent; /* Matches main button */
    text-align: center;
    vertical-align: middle;
    user-select: none;
}
.tb-btn:focus-visible { /* From main button style */
    outline: none;
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--theme-primary) 35%, transparent);
}
.tb-btn .material-symbols-outlined {
    margin-right: 0.4em;
    font-size: 1.2em; /* Original size */
}

.tb-btn-primary {
    background-color: var(--button-bg); /* Uses main button vars */
    color: var(--button-text);
    border-color: var(--button-bg);
}
.tb-btn-primary:hover {
    background-color: var(--button-hover-bg);
    border-color: var(--button-hover-bg);
}

.tb-btn-secondary {
    background-color: var(--theme-secondary);
    color: var(--theme-text-on-primary); /* Assuming contrast similar to primary */
    border-color: var(--theme-secondary);
}
.tb-btn-secondary:hover {
    background-color: color-mix(in srgb, var(--theme-secondary) 80%, var(--theme-bg) 20%); /* Mix with bg for hover */
    border-color: color-mix(in srgb, var(--theme-secondary) 80%, var(--theme-bg) 20%);
}

.tb-btn-danger {
    background-color: var(--color-error);
    color: var(--theme-text-on-primary); /* Standard white text on status colors */
    border-color: var(--color-error);
}
.tb-btn-danger:hover {
    background-color: color-mix(in srgb, var(--color-error) 80%, black 20%);
    border-color: color-mix(in srgb, var(--color-error) 80%, black 20%);
}

.tb-btn-success {
    background-color: var(--color-success);
    color: var(--theme-text-on-primary);
    border-color: var(--color-success);
}
.tb-btn-success:hover {
    background-color: color-mix(in srgb, var(--color-success) 80%, black 20%);
    border-color: color-mix(in srgb, var(--color-success) 80%, black 20%);
}

/* Utility Classes - map to main styles if possible, or keep if specific */
.tb-space-y-6 > *:not([hidden]) ~ *:not([hidden]) { margin-top: calc(var(--spacing) * 1.5); } /* Original was 1.5rem */
.tb-mt-2 { margin-top: calc(var(--spacing) * 0.5); }
.tb-mb-1 { margin-bottom: calc(var(--spacing) * 0.25); }
.tb-mb-2 { margin-bottom: calc(var(--spacing) * 0.5); }
.tb-mr-1 { margin-right: calc(var(--spacing) * 0.25); }

/* This class name is kept as is. If responsive behavior is needed, a media query wrapping would be required. */
.md\\:tb-w-2\\/3 { width: 66.666667%; }

.tb-text-red-500 { color: var(--color-error); } /* Using main error color */
.tb-text-green-600 { color: var(--color-success); } /* Using main success color */
.tb-text-blue-500 { color: var(--tb-color-primary-500); } /* Using specific primary shade from main palette */
.tb-text-gray-500 { color: var(--theme-text-muted); } /* Using main muted text color */
/* No specific body[data-theme="dark"] needed for .tb-text-gray-500 */

.tb-text-sm { font-size: var(--font-size-sm); }
.tb-text-md { font-size: var(--font-size-base); } /* base is the equivalent of md */
.tb-text-lg { font-size: var(--font-size-lg); }

.tb-font-semibold { font-weight: var(--font-weight-semibold); }

.tb-flex { display: flex; }
.tb-items-center { align-items: center; }
.tb-cursor-pointer { cursor: pointer; }
.tb-space-x-2 > *:not([hidden]) ~ *:not([hidden]) { margin-left: calc(var(--spacing) * 0.5); } /* Original was 0.5rem */

.toggle-switch {
    display: inline-flex;
    align-items: center;
    cursor: pointer;
}
.toggle-switch input { /* Hidden checkbox */
    opacity: 0;
    width: 0;
    height: 0;
    position: absolute;
}
.toggle-slider {
    width: 40px; /* Original size */
    height: 20px; /* Original size */
    background-color: var(--input-border); /* Off state using input border color */
    border-radius: 20px; /* Original radius */
    position: relative;
    transition: background-color var(--transition-fast);
}
.toggle-slider:before { /* The knob */
    content: "";
    position: absolute;
    height: 16px; /* Original size */
    width: 16px; /* Original size */
    left: 2px;   /* Original position */
    bottom: 2px; /* Original position */
    background-color: var(--theme-bg-sun); /* Knob color, light in light-mode, darkish in dark-mode */
    border-radius: 50%;
    transition: transform var(--transition-fast), background-color var(--transition-fast);
}
.toggle-switch input:checked + .toggle-slider {
    background-color: var(--theme-primary); /* On state using theme primary */
}
.toggle-switch input:checked + .toggle-slider:before {
    transform: translateX(20px); /* Original translation */
    background-color: var(--theme-text-on-primary); /* Knob color when active */
}
    </style>
</head>
<body data-theme="system">
    <div id="user-dashboard">
        <div id="user-header">
            <h1><span class="material-symbols-outlined">dashboard</span>User Dashboard</h1>
            <button id="sidebar-toggle-btn" class="tb-btn" style="margin-right: 1rem; display: none; background: none; border: none; color: white;">
                <span class="material-symbols-outlined">menu</span>
            </button>

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
                    { name: 'Theme Background', key: '--theme-bg', type: 'color', default: '#f8f9fa' },
                    { name: 'Theme Text', key: '--theme-text', type: 'color', default: '#181823' },
                    { name: 'Theme Primary', key: '--theme-primary', type: 'color', default: '#3a5fcd' },
                    { name: 'Theme Secondary', key: '--theme-secondary', type: 'color', default: '#537FE7' },
                    { name: 'Theme Accent', key: '--theme-accent', type: 'color', default: '#045fab' },
                    { name: 'Theme Accent', key: '--theme-bg-light', type: 'color', default: '#537FE7' },
                    { name: 'Theme Accent', key: '--theme-bg-sun', type: 'color', default: '#ffffff' },
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
