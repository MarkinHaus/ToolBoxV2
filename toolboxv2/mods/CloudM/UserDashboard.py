# toolboxv2/mods/CloudM/UserDashboard.py
"""
ToolBox V2 - Enhanced User Dashboard
Benutzerfreundliches Dashboard für:
- Profil-Verwaltung
- Mod-Interaktion und Konfiguration
- Einstellungen ohne technisches Wissen
- Appearance/Theme-Customization
"""

import json
from dataclasses import asdict

from toolboxv2 import App, RequestData, Result, get_app
from toolboxv2.mods.CloudM.AuthManager import db_helper_save_user
from toolboxv2.mods.CloudM.AuthManager import (
    get_magic_link_email as request_magic_link_backend,
)

from .UserAccountManager import get_current_user_from_request
from .UserInstances import close_user_instance as close_user_instance_internal
from .UserInstances import get_user_instance as get_user_instance_internal

Name = 'CloudM.UserDashboard'
export = get_app(Name + ".Export").tb
version = '0.2.0'


# =================== Haupt-Dashboard ===================

@export(mod_name=Name, api=True, version=version, name="main", api_methods=['GET'], request_as_kwarg=True, row=True)
async def get_user_dashboard_main_page(app: App, request: RequestData):
    """Haupt-Dashboard Seite - vollständig responsive und benutzerfreundlich"""

    html_content = """
<style>
/* ========== User Dashboard Styles ========== */
body {
    margin: 0;
    font-family: var(--font-family-base);
    background-color: var(--theme-bg);
    color: var(--theme-text);
    transition: background-color var(--transition-medium), color var(--transition-medium);
}

#user-dashboard {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
}

/* Header */
#user-header {
    background: linear-gradient(135deg, var(--theme-primary) 0%, var(--theme-secondary) 100%);
    color: var(--theme-text-on-primary);
    padding: var(--spacing) calc(var(--spacing) * 1.5);
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 12px color-mix(in srgb, var(--theme-text) 15%, transparent);
}

#user-header h1 {
    margin: 0;
    font-size: var(--font-size-xl);
    font-weight: var(--font-weight-semibold);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.user-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: var(--theme-text-on-primary);
    color: var(--theme-primary);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 1.2rem;
}

/* Sidebar */
#user-sidebar {
    width: 260px;
    background-color: var(--input-bg);
    padding: calc(var(--spacing) * 1.5) var(--spacing);
    border-right: 1px solid var(--theme-border);
    transition: all var(--transition-medium);
}

#user-sidebar ul {
    list-style: none;
    padding: 0;
    margin: 0;
}

#user-sidebar li {
    padding: calc(var(--spacing) * 0.9) var(--spacing);
    margin-bottom: calc(var(--spacing) * 0.5);
    cursor: pointer;
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-weight: var(--font-weight-medium);
    color: var(--theme-text);
    transition: all var(--transition-fast);
}

#user-sidebar li:hover {
    background-color: color-mix(in srgb, var(--theme-primary) 12%, transparent);
    color: var(--theme-primary);
    transform: translateX(4px);
}

#user-sidebar li.active {
    background: linear-gradient(135deg, var(--theme-primary) 0%, var(--theme-secondary) 100%);
    color: var(--theme-text-on-primary) !important;
    font-weight: var(--font-weight-semibold);
    box-shadow: 0 4px 12px color-mix(in srgb, var(--theme-primary) 30%, transparent);
}

#user-sidebar li .material-symbols-outlined {
    font-size: 1.3rem;
}

/* Main Container */
#user-container {
    display: flex;
    flex-grow: 1;
}

#user-content {
    flex-grow: 1;
    padding: calc(var(--spacing) * 2);
    overflow-y: auto;
}

/* Content Sections */
.content-section {
    display: none;
}

.content-section.active {
    display: block;
    animation: fadeSlideIn 0.4s ease-out;
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

.content-section h2 {
    font-size: var(--font-size-2xl);
    font-weight: var(--font-weight-semibold);
    color: var(--theme-text);
    margin-bottom: calc(var(--spacing) * 1.5);
    padding-bottom: var(--spacing);
    border-bottom: 2px solid var(--theme-primary);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Cards */
.dashboard-card {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    border-radius: var(--radius-lg);
    padding: calc(var(--spacing) * 1.5);
    border: 1px solid var(--glass-border);
    box-shadow: var(--glass-shadow);
    margin-bottom: calc(var(--spacing) * 1.5);
    transition: all var(--transition-fast);
}

.dashboard-card:hover {
    box-shadow: 0 8px 24px color-mix(in srgb, var(--theme-text) 12%, transparent);
    transform: translateY(-2px);
}

.dashboard-card h3 {
    font-size: var(--font-size-lg);
    font-weight: var(--font-weight-semibold);
    margin: 0 0 var(--spacing) 0;
    color: var(--theme-primary);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Module Cards */
.module-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: var(--spacing);
}

.module-card {
    background: var(--input-bg);
    border-radius: var(--radius-md);
    padding: var(--spacing);
    border: 1px solid var(--theme-border);
    transition: all var(--transition-fast);
}

.module-card:hover {
    border-color: var(--theme-primary);
    box-shadow: 0 4px 12px color-mix(in srgb, var(--theme-primary) 15%, transparent);
}

.module-card.active {
    border-color: var(--color-success);
    background: color-mix(in srgb, var(--color-success) 8%, var(--input-bg));
}

.module-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: calc(var(--spacing) * 0.5);
}

.module-name {
    font-weight: var(--font-weight-semibold);
    color: var(--theme-text);
}

.module-status {
    font-size: var(--font-size-xs);
    padding: 2px 8px;
    border-radius: var(--radius-full);
    font-weight: var(--font-weight-medium);
}

.module-status.loaded {
    background: var(--color-success);
    color: white;
}

.module-status.available {
    background: var(--theme-border);
    color: var(--theme-text-muted);
}

.module-description {
    font-size: var(--font-size-sm);
    color: var(--theme-text-muted);
    margin-bottom: calc(var(--spacing) * 0.75);
}

.module-actions {
    display: flex;
    gap: calc(var(--spacing) * 0.5);
    flex-wrap: wrap;
}

/* Settings Grid */
.settings-section {
    margin-bottom: calc(var(--spacing) * 2);
}

.settings-section h4 {
    font-size: var(--font-size-base);
    font-weight: var(--font-weight-semibold);
    margin-bottom: var(--spacing);
    color: var(--theme-text);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.setting-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--spacing);
    background: var(--input-bg);
    border-radius: var(--radius-md);
    margin-bottom: calc(var(--spacing) * 0.5);
    border: 1px solid var(--theme-border);
}

.setting-item:hover {
    border-color: var(--theme-primary);
}

.setting-info {
    flex: 1;
}

.setting-label {
    font-weight: var(--font-weight-medium);
    color: var(--theme-text);
    margin-bottom: 2px;
}

.setting-description {
    font-size: var(--font-size-sm);
    color: var(--theme-text-muted);
}

/* Toggle Switch */
.toggle-switch {
    position: relative;
    width: 52px;
    height: 28px;
    flex-shrink: 0;
}

.toggle-switch input {
    opacity: 0;
    width: 0;
    height: 0;
}

.toggle-slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: var(--theme-border);
    transition: var(--transition-fast);
    border-radius: 28px;
}

.toggle-slider:before {
    position: absolute;
    content: "";
    height: 22px;
    width: 22px;
    left: 3px;
    bottom: 3px;
    background-color: white;
    transition: var(--transition-fast);
    border-radius: 50%;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

input:checked + .toggle-slider {
    background: linear-gradient(135deg, var(--theme-primary) 0%, var(--theme-secondary) 100%);
}

input:checked + .toggle-slider:before {
    transform: translateX(24px);
}

/* Buttons */
.tb-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: calc(var(--spacing) * 0.5) calc(var(--spacing) * 1);
    border-radius: var(--radius-md);
    font-weight: var(--font-weight-medium);
    font-size: var(--font-size-sm);
    cursor: pointer;
    transition: all var(--transition-fast);
    border: 1px solid transparent;
    gap: 0.35rem;
}

.tb-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px color-mix(in srgb, var(--theme-text) 12%, transparent);
}

.tb-btn-primary {
    background: linear-gradient(135deg, var(--theme-primary) 0%, var(--theme-secondary) 100%);
    color: var(--theme-text-on-primary);
}

.tb-btn-secondary {
    background: var(--input-bg);
    color: var(--theme-text);
    border-color: var(--theme-border);
}

.tb-btn-success {
    background: var(--color-success);
    color: white;
}

.tb-btn-danger {
    background: var(--color-error);
    color: white;
}

.tb-btn-sm {
    padding: calc(var(--spacing) * 0.35) calc(var(--spacing) * 0.75);
    font-size: var(--font-size-xs);
}

/* Input Styles */
.tb-input {
    width: 100%;
    padding: calc(var(--spacing) * 0.75) var(--spacing);
    font-size: var(--font-size-base);
    color: var(--theme-text);
    background-color: var(--input-bg);
    border: 1px solid var(--input-border);
    border-radius: var(--radius-md);
    transition: all var(--transition-fast);
}

.tb-input:focus {
    outline: none;
    border-color: var(--theme-primary);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--theme-primary) 20%, transparent);
}

/* Stats Cards */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: var(--spacing);
    margin-bottom: calc(var(--spacing) * 1.5);
}

.stat-card {
    background: var(--input-bg);
    border-radius: var(--radius-md);
    padding: var(--spacing);
    text-align: center;
    border: 1px solid var(--theme-border);
}

.stat-value {
    font-size: var(--font-size-2xl);
    font-weight: var(--font-weight-bold);
    color: var(--theme-primary);
}

.stat-label {
    font-size: var(--font-size-sm);
    color: var(--theme-text-muted);
}

/* Mod Data Panel */
.mod-data-panel {
    background: var(--input-bg);
    border-radius: var(--radius-md);
    border: 1px solid var(--theme-border);
    overflow: hidden;
}

.mod-data-header {
    background: color-mix(in srgb, var(--theme-primary) 10%, var(--input-bg));
    padding: calc(var(--spacing) * 0.75) var(--spacing);
    font-weight: var(--font-weight-semibold);
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
}

.mod-data-content {
    padding: var(--spacing);
    display: none;
}

.mod-data-content.open {
    display: block;
}

.mod-data-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: calc(var(--spacing) * 0.5) 0;
    border-bottom: 1px solid var(--theme-border);
}

.mod-data-item:last-child {
    border-bottom: none;
}

.mod-data-key {
    font-weight: var(--font-weight-medium);
    color: var(--theme-text-muted);
}

.mod-data-value {
    color: var(--theme-text);
}

/* Mobile Responsive */
@media (max-width: 768px) {
    #user-sidebar {
        position: fixed;
        left: -280px;
        top: 0;
        bottom: 0;
        z-index: var(--z-modal);
        transition: left var(--transition-medium);
    }

    #user-sidebar.open {
        left: 0;
        box-shadow: 4px 0 20px color-mix(in srgb, var(--theme-text) 25%, transparent);
    }

    #sidebar-toggle-btn {
        display: inline-flex !important;
    }

    #sidebar-backdrop.active {
        display: block;
    }

    .module-grid {
        grid-template-columns: 1fr;
    }

    .stats-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (min-width: 769px) {
    #sidebar-toggle-btn {
        display: none !important;
    }

    #sidebar-backdrop {
        display: none !important;
    }
}

#sidebar-backdrop {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.5);
    z-index: calc(var(--z-modal) - 1);
}

/* Toast/Notification */
.toast-success {
    background: var(--color-success) !important;
}

.toast-error {
    background: var(--color-error) !important;
}

/* Loading Spinner */
.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 2px solid var(--theme-border);
    border-top-color: var(--theme-primary);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Helper Classes */
.text-muted { color: var(--theme-text-muted); }
.text-success { color: var(--color-success); }
.text-error { color: var(--color-error); }
.text-sm { font-size: var(--font-size-sm); }
.mt-2 { margin-top: calc(var(--spacing) * 0.5); }
.mt-4 { margin-top: var(--spacing); }
.mb-2 { margin-bottom: calc(var(--spacing) * 0.5); }
.mb-4 { margin-bottom: var(--spacing); }
.flex { display: flex; }
.gap-2 { gap: calc(var(--spacing) * 0.5); }
.items-center { align-items: center; }
</style>

<div id="user-dashboard">
    <header id="user-header">
        <div class="flex items-center gap-2">
            <button id="sidebar-toggle-btn" class="tb-btn tb-btn-secondary" style="display:none;">
                <span class="material-symbols-outlined">menu</span>
            </button>
            <h1>
                <span class="material-symbols-outlined">dashboard</span>
                <span id="welcome-text">Willkommen!</span>
            </h1>
        </div>
        <div class="flex items-center gap-2">
            <div id="darkModeToggleContainer"></div>
            <button id="logoutButtonUser" class="tb-btn tb-btn-secondary">
                <span class="material-symbols-outlined">logout</span>
                <span class="logout-text">Abmelden</span>
            </button>
        </div>
    </header>

    <div id="user-container">
        <aside id="user-sidebar">
            <ul>
                <li data-section="overview" class="active">
                    <span class="material-symbols-outlined">home</span>
                    Übersicht
                </li>
                <li data-section="my-modules">
                    <span class="material-symbols-outlined">extension</span>
                    Meine Module
                </li>
                <li data-section="mod-data">
                    <span class="material-symbols-outlined">database</span>
                    Mod-Daten
                </li>
                <li data-section="settings">
                    <span class="material-symbols-outlined">settings</span>
                    Einstellungen
                </li>
                <li data-section="appearance">
                    <span class="material-symbols-outlined">palette</span>
                    Erscheinungsbild
                </li>
                <li data-section="profile">
                    <span class="material-symbols-outlined">person</span>
                    Mein Profil
                </li>
            </ul>
        </aside>

        <main id="user-content">
            <!-- Übersicht -->
            <section id="overview-section" class="content-section active">
                <h2><span class="material-symbols-outlined">home</span>Übersicht</h2>
                <div id="overview-content">
                    <p class="text-muted">Lädt...</p>
                </div>
            </section>

            <!-- Meine Module -->
            <section id="my-modules-section" class="content-section">
                <h2><span class="material-symbols-outlined">extension</span>Meine Module</h2>
                <div id="my-modules-content">
                    <p class="text-muted">Lädt Module...</p>
                </div>
            </section>

            <!-- Mod-Daten -->
            <section id="mod-data-section" class="content-section">
                <h2><span class="material-symbols-outlined">database</span>Mod-Daten</h2>
                <div id="mod-data-content">
                    <p class="text-muted">Lädt Mod-Daten...</p>
                </div>
            </section>

            <!-- Einstellungen -->
            <section id="settings-section" class="content-section">
                <h2><span class="material-symbols-outlined">settings</span>Einstellungen</h2>
                <div id="settings-content">
                    <p class="text-muted">Lädt Einstellungen...</p>
                </div>
            </section>

            <!-- Erscheinungsbild -->
            <section id="appearance-section" class="content-section">
                <h2><span class="material-symbols-outlined">palette</span>Erscheinungsbild</h2>
                <div id="appearance-content">
                    <p class="text-muted">Lädt Theme-Einstellungen...</p>
                </div>
            </section>

            <!-- Profil -->
            <section id="profile-section" class="content-section">
                <h2><span class="material-symbols-outlined">person</span>Mein Profil</h2>
                <div id="profile-content">
                    <p class="text-muted">Lädt Profil...</p>
                </div>
            </section>
        </main>
    </div>

    <div id="sidebar-backdrop"></div>
</div>

<script type="module">
if (typeof TB === 'undefined' || !TB.ui || !TB.api) {
    console.error('CRITICAL: TB (tbjs) not loaded.');
    document.body.innerHTML = '<div style="padding:40px; text-align:center; color:red;">Fehler: Frontend-Bibliothek konnte nicht geladen werden.</div>';
} else {
    console.log('TB object found. Initializing User Dashboard v2...');

    let currentUser = null;
    let allModules = [];
    let userInstance = null;
    let modDataCache = {};

    // ========== Initialization ==========
    async function initDashboard() {
        console.log("Dashboard wird initialisiert...");
        TB.ui.DarkModeToggle.init();
        setupNavigation();
        setupMobileSidebar();
        setupLogout();

        try {
            // Benutzer laden
            const userRes = await TB.api.request('CloudM.UserAccountManager', 'get_current_user', null, 'GET');
            if (userRes.error === TB.ToolBoxError.none && userRes.get()) {
                currentUser = userRes.get();
                updateWelcomeText();

                // Module laden
                const modulesRes = await TB.api.request('CloudM.UserDashboard', 'get_all_available_modules', null, 'GET');
                if (modulesRes.error === TB.ToolBoxError.none) {
                    allModules = modulesRes.get() || [];
                }

                // Instanz laden
                const instanceRes = await TB.api.request('CloudM.UserDashboard', 'get_my_active_instances', null, 'GET');
                if (instanceRes.error === TB.ToolBoxError.none && instanceRes.get()?.length > 0) {
                    userInstance = instanceRes.get()[0];
                }

                // Erste Sektion laden
                await showSection('overview');
            } else {
                document.getElementById('user-content').innerHTML = `
                    <div class="dashboard-card" style="text-align:center; padding:40px;">
                        <span class="material-symbols-outlined" style="font-size:64px; color:var(--color-warning);">login</span>
                        <h3 style="margin-top:16px;">Nicht angemeldet</h3>
                        <p class="text-muted">Bitte melden Sie sich an, um fortzufahren.</p>
                        <button onclick="TB.user.signIn()" class="tb-btn tb-btn-primary mt-4">
                            <span class="material-symbols-outlined">login</span>
                            Anmelden
                        </button>
                    </div>
                `;
            }
        } catch (e) {
            console.error("Fehler beim Initialisieren:", e);
            document.getElementById('user-content').innerHTML = `
                <div class="dashboard-card" style="text-align:center; padding:40px;">
                    <span class="material-symbols-outlined" style="font-size:64px; color:var(--color-error);">error</span>
                    <h3 style="margin-top:16px;">Verbindungsfehler</h3>
                    <p class="text-muted">Die Verbindung zum Server konnte nicht hergestellt werden.</p>
                </div>
            `;
        }
    }

    function updateWelcomeText() {
        const welcomeEl = document.getElementById('welcome-text');
        if (welcomeEl && currentUser) {
            const name = currentUser.username || currentUser.name || 'Benutzer';
            welcomeEl.textContent = `Willkommen, ${name}!`;
        }
    }

    // ========== Navigation ==========
    function setupNavigation() {
        document.querySelectorAll('#user-sidebar li[data-section]').forEach(item => {
            item.addEventListener('click', async () => {
                document.querySelectorAll('#user-sidebar li').forEach(i => i.classList.remove('active'));
                item.classList.add('active');
                await showSection(item.dataset.section);

                // Mobile: Sidebar schließen
                if (window.innerWidth < 769) {
                    document.getElementById('user-sidebar').classList.remove('open');
                    document.getElementById('sidebar-backdrop').classList.remove('active');
                }
            });
        });
    }

    function setupMobileSidebar() {
        const toggleBtn = document.getElementById('sidebar-toggle-btn');
        const sidebar = document.getElementById('user-sidebar');
        const backdrop = document.getElementById('sidebar-backdrop');

        toggleBtn.addEventListener('click', () => {
            sidebar.classList.toggle('open');
            backdrop.classList.toggle('active');
        });

        backdrop.addEventListener('click', () => {
            sidebar.classList.remove('open');
            backdrop.classList.remove('active');
        });
    }

    function setupLogout() {
        document.getElementById('logoutButtonUser').addEventListener('click', async () => {
            TB.ui.Loader.show("Abmelden...");
            await TB.user.logout();
            window.location.href = '/';
        });
    }

    // ========== Section Loading ==========
    async function showSection(sectionId) {
        document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
        const section = document.getElementById(`${sectionId}-section`);
        if (section) {
            section.classList.add('active');

            switch(sectionId) {
                case 'overview': await loadOverview(); break;
                case 'my-modules': await loadModules(); break;
                case 'mod-data': await loadModData(); break;
                case 'settings': await loadSettings(); break;
                case 'appearance': await loadAppearance(); break;
                case 'profile': await loadProfile(); break;
            }
        }
    }

    // ========== Übersicht ==========
    async function loadOverview() {
        const content = document.getElementById('overview-content');
        const loadedModsCount = userInstance?.live_modules?.length || 0;
        const savedModsCount = userInstance?.saved_modules?.length || 0;
        const cliSessions = userInstance?.active_cli_sessions || 0;

        content.innerHTML = `
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">${loadedModsCount}</div>
                    <div class="stat-label">Aktive Module</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${savedModsCount}</div>
                    <div class="stat-label">Gespeicherte Module</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${cliSessions}</div>
                    <div class="stat-label">CLI Sitzungen</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${currentUser?.level || 1}</div>
                    <div class="stat-label">Benutzer-Level</div>
                </div>
            </div>

            <div class="dashboard-card">
                <h3><span class="material-symbols-outlined">speed</span>Schnellzugriff</h3>
                <div class="flex gap-2" style="flex-wrap:wrap;">
                    <button class="tb-btn tb-btn-primary" onclick="showSection('my-modules')">
                        <span class="material-symbols-outlined">extension</span>
                        Module verwalten
                    </button>
                    <button class="tb-btn tb-btn-secondary" onclick="showSection('settings')">
                        <span class="material-symbols-outlined">settings</span>
                        Einstellungen
                    </button>
                    <button class="tb-btn tb-btn-secondary" onclick="showSection('appearance')">
                        <span class="material-symbols-outlined">palette</span>
                        Theme ändern
                    </button>
                </div>
            </div>

            ${userInstance?.live_modules?.length > 0 ? `
                <div class="dashboard-card">
                    <h3><span class="material-symbols-outlined">play_circle</span>Aktive Module</h3>
                    <div class="module-grid">
                        ${userInstance.live_modules.map(mod => `
                            <div class="module-card active">
                                <div class="module-header">
                                    <span class="module-name">${TB.utils.escapeHtml(mod.name)}</span>
                                    <span class="module-status loaded">Aktiv</span>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}

            <div class="dashboard-card">
                <h3><span class="material-symbols-outlined">info</span>Konto-Info</h3>
                <table style="width:100%;">
                    <tr><td class="text-muted">Benutzername</td><td><strong>${TB.utils.escapeHtml(currentUser?.username || currentUser?.name || '-')}</strong></td></tr>
                    <tr><td class="text-muted">E-Mail</td><td>${TB.utils.escapeHtml(currentUser?.email || 'Nicht angegeben')}</td></tr>
                    <tr><td class="text-muted">Level</td><td>${currentUser?.level || 1}</td></tr>
                </table>
            </div>
        `;
    }

    // ========== Module ==========
    async function loadModules() {
        const content = document.getElementById('my-modules-content');

        // Aktuelle Instanz-Daten neu laden
        try {
            const instanceRes = await TB.api.request('CloudM.UserDashboard', 'get_my_active_instances', null, 'GET');
            if (instanceRes.error === TB.ToolBoxError.none && instanceRes.get()?.length > 0) {
                userInstance = instanceRes.get()[0];
            }
        } catch(e) {}

        const liveModNames = (userInstance?.live_modules || []).map(m => m.name);
        const savedModNames = userInstance?.saved_modules || [];

        // Kategorien erstellen
        const categories = {};
        allModules.forEach(mod => {
            const category = mod.split('.')[0] || 'Andere';
            if (!categories[category]) categories[category] = [];
            categories[category].push(mod);
        });

        content.innerHTML = `
            <div class="dashboard-card">
                <h3><span class="material-symbols-outlined">info</span>Hinweis</h3>
                <p class="text-sm text-muted">
                    Hier können Sie Module aktivieren oder deaktivieren. Aktive Module stehen Ihnen sofort zur Verfügung.
                    Gespeicherte Module werden bei der nächsten Anmeldung automatisch geladen.
                </p>
            </div>

            <div class="dashboard-card">
                <h3><span class="material-symbols-outlined">bookmark</span>Meine gespeicherten Module (${savedModNames.length})</h3>
                ${savedModNames.length > 0 ? `
                    <div class="module-grid">
                        ${savedModNames.map(modName => {
                            const isLive = liveModNames.includes(modName);
                            return `
                                <div class="module-card ${isLive ? 'active' : ''}">
                                    <div class="module-header">
                                        <span class="module-name">${TB.utils.escapeHtml(modName)}</span>
                                        <span class="module-status ${isLive ? 'loaded' : 'available'}">${isLive ? 'Aktiv' : 'Gespeichert'}</span>
                                    </div>
                                    <div class="module-actions mt-2">
                                        ${!isLive ? `
                                            <button class="tb-btn tb-btn-success tb-btn-sm" onclick="loadModule('${TB.utils.escapeHtml(modName)}')">
                                                <span class="material-symbols-outlined">play_arrow</span>
                                                Laden
                                            </button>
                                        ` : `
                                            <button class="tb-btn tb-btn-secondary tb-btn-sm" onclick="unloadModule('${TB.utils.escapeHtml(modName)}')">
                                                <span class="material-symbols-outlined">stop</span>
                                                Entladen
                                            </button>
                                        `}
                                        <button class="tb-btn tb-btn-danger tb-btn-sm" onclick="removeFromSaved('${TB.utils.escapeHtml(modName)}')">
                                            <span class="material-symbols-outlined">delete</span>
                                        </button>
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                ` : '<p class="text-muted">Keine Module gespeichert.</p>'}
            </div>

            <div class="dashboard-card">
                <h3><span class="material-symbols-outlined">apps</span>Verfügbare Module (${allModules.length})</h3>
                <div class="mb-4">
                    <input type="text" id="module-search" class="tb-input" placeholder="Module durchsuchen..." oninput="filterModules(this.value)">
                </div>
                <div id="module-categories">
                    ${Object.entries(categories).map(([cat, mods]) => `
                        <details class="mb-4" ${cat === 'CloudM' ? 'open' : ''}>
                            <summary style="cursor:pointer; font-weight:var(--font-weight-semibold); padding:var(--spacing) 0;">
                                <span class="material-symbols-outlined" style="vertical-align:middle;">folder</span>
                                ${TB.utils.escapeHtml(cat)} (${mods.length})
                            </summary>
                            <div class="module-grid" style="margin-top:var(--spacing);">
                                ${mods.map(modName => {
                                    const isLive = liveModNames.includes(modName);
                                    const isSaved = savedModNames.includes(modName);
                                    return `
                                        <div class="module-card module-item ${isLive ? 'active' : ''}" data-name="${modName.toLowerCase()}">
                                            <div class="module-header">
                                                <span class="module-name">${TB.utils.escapeHtml(modName)}</span>
                                                ${isLive ? '<span class="module-status loaded">Aktiv</span>' :
                                                  isSaved ? '<span class="module-status available">Gespeichert</span>' : ''}
                                            </div>
                                            <div class="module-actions mt-2">
                                                ${!isSaved ? `
                                                    <button class="tb-btn tb-btn-primary tb-btn-sm" onclick="addToSaved('${TB.utils.escapeHtml(modName)}')">
                                                        <span class="material-symbols-outlined">bookmark_add</span>
                                                        Speichern
                                                    </button>
                                                ` : ''}
                                                ${!isLive ? `
                                                    <button class="tb-btn tb-btn-success tb-btn-sm" onclick="loadModule('${TB.utils.escapeHtml(modName)}')">
                                                        <span class="material-symbols-outlined">play_arrow</span>
                                                        Laden
                                                    </button>
                                                ` : `
                                                    <button class="tb-btn tb-btn-secondary tb-btn-sm" onclick="unloadModule('${TB.utils.escapeHtml(modName)}')">
                                                        <span class="material-symbols-outlined">stop</span>
                                                        Entladen
                                                    </button>
                                                `}
                                            </div>
                                        </div>
                                    `;
                                }).join('')}
                            </div>
                        </details>
                    `).join('')}
                </div>
            </div>
        `;
    }

    // Modul-Funktionen global machen
    window.filterModules = function(query) {
        const q = query.toLowerCase();
        document.querySelectorAll('.module-item').forEach(item => {
            const name = item.dataset.name;
            item.style.display = name.includes(q) ? '' : 'none';
        });
    };

    window.loadModule = async function(modName) {
        TB.ui.Loader.show(`Lade ${modName}...`);
        try {
            const res = await TB.api.request('CloudM.UserDashboard', 'add_module_to_instance', {module_name: modName}, 'POST');
            TB.ui.Loader.hide();
            if (res.error === TB.ToolBoxError.none) {
                TB.ui.Toast.showSuccess(`${modName} wurde geladen`);
                await loadModules();
            } else {
                TB.ui.Toast.showError(res.info.help_text || 'Fehler beim Laden');
            }
        } catch(e) {
            TB.ui.Loader.hide();
            TB.ui.Toast.showError('Netzwerkfehler');
        }
    };

    window.unloadModule = async function(modName) {
        TB.ui.Loader.show(`Entlade ${modName}...`);
        try {
            const res = await TB.api.request('CloudM.UserDashboard', 'remove_module_from_instance', {module_name: modName}, 'POST');
            TB.ui.Loader.hide();
            if (res.error === TB.ToolBoxError.none) {
                TB.ui.Toast.showSuccess(`${modName} wurde entladen`);
                await loadModules();
            } else {
                TB.ui.Toast.showError(res.info.help_text || 'Fehler beim Entladen');
            }
        } catch(e) {
            TB.ui.Loader.hide();
            TB.ui.Toast.showError('Netzwerkfehler');
        }
    };

    window.addToSaved = async function(modName) {
        TB.ui.Loader.show('Speichere...');
        try {
            const res = await TB.api.request('CloudM.UserDashboard', 'add_module_to_saved', {module_name: modName}, 'POST');
            TB.ui.Loader.hide();
            if (res.error === TB.ToolBoxError.none) {
                TB.ui.Toast.showSuccess(`${modName} gespeichert`);
                await loadModules();
            } else {
                TB.ui.Toast.showError(res.info.help_text || 'Fehler');
            }
        } catch(e) {
            TB.ui.Loader.hide();
            TB.ui.Toast.showError('Netzwerkfehler');
        }
    };

    window.removeFromSaved = async function(modName) {
        if (!confirm(`Möchten Sie "${modName}" wirklich aus den gespeicherten Modulen entfernen?`)) return;
        TB.ui.Loader.show('Entferne...');
        try {
            const res = await TB.api.request('CloudM.UserDashboard', 'remove_module_from_saved', {module_name: modName}, 'POST');
            TB.ui.Loader.hide();
            if (res.error === TB.ToolBoxError.none) {
                TB.ui.Toast.showSuccess(`${modName} entfernt`);
                await loadModules();
            } else {
                TB.ui.Toast.showError(res.info.help_text || 'Fehler');
            }
        } catch(e) {
            TB.ui.Loader.hide();
            TB.ui.Toast.showError('Netzwerkfehler');
        }
    };

    // ========== Mod-Daten ==========
    async function loadModData() {
        const content = document.getElementById('mod-data-content');

        // Mod-Daten laden
        try {
            const res = await TB.api.request('CloudM.UserDashboard', 'get_all_mod_data', null, 'GET');
            if (res.error === TB.ToolBoxError.none) {
                modDataCache = res.get() || {};
            }
        } catch(e) {}

        const modNames = Object.keys(modDataCache);

        content.innerHTML = `
            <div class="dashboard-card">
                <h3><span class="material-symbols-outlined">info</span>Was sind Mod-Daten?</h3>
                <p class="text-sm text-muted">
                    Jedes Modul kann eigene Daten für Sie speichern. Hier können Sie diese Daten einsehen und bei Bedarf bearbeiten.
                    Diese Daten sind nur für Sie sichtbar und werden sicher gespeichert.
                </p>
            </div>

            ${modNames.length > 0 ? modNames.map(modName => {
                const data = modDataCache[modName] || {};
                const entries = Object.entries(data);
                return `
                    <div class="mod-data-panel mb-4">
                        <div class="mod-data-header" onclick="this.nextElementSibling.classList.toggle('open'); this.querySelector('.material-symbols-outlined').textContent = this.nextElementSibling.classList.contains('open') ? 'expand_less' : 'expand_more';">
                            <span>
                                <span class="material-symbols-outlined" style="vertical-align:middle;">extension</span>
                                ${TB.utils.escapeHtml(modName)}
                            </span>
                            <span class="material-symbols-outlined">expand_more</span>
                        </div>
                        <div class="mod-data-content">
                            ${entries.length > 0 ? entries.map(([key, value]) => `
                                <div class="mod-data-item">
                                    <span class="mod-data-key">${TB.utils.escapeHtml(key)}</span>
                                    <span class="mod-data-value">
                                        ${typeof value === 'boolean' ?
                                            `<span class="${value ? 'text-success' : 'text-error'}">${value ? 'Ja' : 'Nein'}</span>` :
                                            TB.utils.escapeHtml(String(value).substring(0, 100))}
                                    </span>
                                </div>
                            `).join('') : '<p class="text-muted text-sm">Keine Daten gespeichert.</p>'}
                            <div class="mt-4 flex gap-2">
                                <button class="tb-btn tb-btn-secondary tb-btn-sm" onclick="editModData('${TB.utils.escapeHtml(modName)}')">
                                    <span class="material-symbols-outlined">edit</span>
                                    Bearbeiten
                                </button>
                                <button class="tb-btn tb-btn-danger tb-btn-sm" onclick="clearModData('${TB.utils.escapeHtml(modName)}')">
                                    <span class="material-symbols-outlined">delete</span>
                                    Löschen
                                </button>
                            </div>
                        </div>
                    </div>
                `;
            }).join('') : `
                <div class="dashboard-card" style="text-align:center; padding:40px;">
                    <span class="material-symbols-outlined" style="font-size:48px; color:var(--theme-text-muted);">folder_off</span>
                    <p class="text-muted mt-4">Noch keine Mod-Daten vorhanden.</p>
                    <p class="text-sm text-muted">Module speichern hier automatisch Ihre Einstellungen und Fortschritte.</p>
                </div>
            `}
        `;
    }

    window.editModData = async function(modName) {
        const data = modDataCache[modName] || {};
        const json = JSON.stringify(data, null, 2);

        TB.ui.Modal.show({
            title: `${modName} - Daten bearbeiten`,
            content: `
                <p class="text-sm text-muted mb-4">Vorsicht: Änderungen können die Funktionalität des Moduls beeinflussen.</p>
                <textarea id="mod-data-editor" style="width:100%; height:200px; font-family:monospace; padding:8px; border:1px solid var(--theme-border); border-radius:var(--radius-md);">${TB.utils.escapeHtml(json)}</textarea>
            `,
            buttons: [
                { text: 'Abbrechen', action: m => m.close(), variant: 'secondary' },
                {
                    text: 'Speichern',
                    variant: 'primary',
                    action: async m => {
                        try {
                            const newData = JSON.parse(document.getElementById('mod-data-editor').value);
                            TB.ui.Loader.show('Speichere...');
                            const res = await TB.api.request('CloudM.UserAccountManager', 'update_mod_data', {mod_name: modName, data: newData}, 'POST');
                            TB.ui.Loader.hide();
                            if (res.error === TB.ToolBoxError.none) {
                                TB.ui.Toast.showSuccess('Daten gespeichert');
                                modDataCache[modName] = newData;
                                m.close();
                                await loadModData();
                            } else {
                                TB.ui.Toast.showError('Fehler beim Speichern');
                            }
                        } catch(e) {
                            TB.ui.Toast.showError('Ungültiges JSON-Format');
                        }
                    }
                }
            ]
        });
    };

    window.clearModData = async function(modName) {
        if (!confirm(`Möchten Sie wirklich alle Daten von "${modName}" löschen?`)) return;
        TB.ui.Loader.show('Lösche...');
        try {
            const res = await TB.api.request('CloudM.UserAccountManager', 'update_mod_data', {mod_name: modName, data: {}}, 'POST');
            TB.ui.Loader.hide();
            if (res.error === TB.ToolBoxError.none) {
                TB.ui.Toast.showSuccess('Daten gelöscht');
                modDataCache[modName] = {};
                await loadModData();
            } else {
                TB.ui.Toast.showError('Fehler beim Löschen');
            }
        } catch(e) {
            TB.ui.Loader.hide();
            TB.ui.Toast.showError('Netzwerkfehler');
        }
    };

    // ========== Einstellungen ==========
    async function loadSettings() {
        const content = document.getElementById('settings-content');
        const settings = currentUser?.settings || {};

        content.innerHTML = `
            <div class="dashboard-card">
                <h3><span class="material-symbols-outlined">tune</span>Allgemeine Einstellungen</h3>

                <div class="settings-section">
                    <div class="setting-item">
                        <div class="setting-info">
                            <div class="setting-label">Experimentelle Funktionen</div>
                            <div class="setting-description">Aktiviert neue Funktionen, die sich noch in der Testphase befinden</div>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" ${settings.experimental_features ? 'checked' : ''}
                                   onchange="updateSetting('experimental_features', this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>

                    <div class="setting-item">
                        <div class="setting-info">
                            <div class="setting-label">Benachrichtigungen</div>
                            <div class="setting-description">Erhalten Sie Benachrichtigungen über wichtige Ereignisse</div>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" ${settings.notifications !== false ? 'checked' : ''}
                                   onchange="updateSetting('notifications', this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>

                    <div class="setting-item">
                        <div class="setting-info">
                            <div class="setting-label">Auto-Laden von Modulen</div>
                            <div class="setting-description">Gespeicherte Module beim Anmelden automatisch laden</div>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" ${settings.auto_load_modules !== false ? 'checked' : ''}
                                   onchange="updateSetting('auto_load_modules', this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>

                    <div class="setting-item">
                        <div class="setting-info">
                            <div class="setting-label">Detaillierte Protokolle</div>
                            <div class="setting-description">Ausführliche Protokollierung für Fehlerbehebung aktivieren</div>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" ${settings.verbose_logging ? 'checked' : ''}
                                   onchange="updateSetting('verbose_logging', this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                </div>
            </div>

            <div class="dashboard-card">
                <h3><span class="material-symbols-outlined">language</span>Sprache & Region</h3>
                <div class="settings-section">
                    <div class="setting-item">
                        <div class="setting-info">
                            <div class="setting-label">Sprache</div>
                            <div class="setting-description">Wählen Sie Ihre bevorzugte Sprache</div>
                        </div>
                        <select class="tb-input" style="width:auto;" onchange="updateSetting('language', this.value)">
                            <option value="de" ${settings.language === 'de' || !settings.language ? 'selected' : ''}>Deutsch</option>
                            <option value="en" ${settings.language === 'en' ? 'selected' : ''}>English</option>
                        </select>
                    </div>
                </div>
            </div>

            <div class="dashboard-card">
                <h3><span class="material-symbols-outlined">security</span>Datenschutz</h3>
                <div class="settings-section">
                    <div class="setting-item">
                        <div class="setting-info">
                            <div class="setting-label">Nutzungsstatistiken</div>
                            <div class="setting-description">Anonyme Statistiken zur Verbesserung der Anwendung senden</div>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" ${settings.analytics !== false ? 'checked' : ''}
                                   onchange="updateSetting('analytics', this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                </div>
            </div>
        `;
    }

    window.updateSetting = async function(key, value) {
        try {
            const res = await TB.api.request('CloudM.UserAccountManager', 'update_setting', {
                setting_key: key,
                setting_value: String(value)
            }, 'POST');
            if (res.error === TB.ToolBoxError.none) {
                if (!currentUser.settings) currentUser.settings = {};
                currentUser.settings[key] = value;
                TB.ui.Toast.showSuccess('Einstellung gespeichert');
            } else {
                TB.ui.Toast.showError('Fehler beim Speichern');
            }
        } catch(e) {
            TB.ui.Toast.showError('Netzwerkfehler');
        }
    };

    // ========== Erscheinungsbild ==========
    async function loadAppearance() {
        const content = document.getElementById('appearance-content');
        const themePreference = TB.ui.theme?.getPreference() || 'system';

        content.innerHTML = `
            <div class="dashboard-card">
                <h3><span class="material-symbols-outlined">contrast</span>Farbschema</h3>
                <p class="text-sm text-muted mb-4">Wählen Sie Ihr bevorzugtes Farbschema für die Anwendung.</p>

                <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(140px, 1fr)); gap:var(--spacing);">
                    <button class="theme-option ${themePreference === 'light' ? 'active' : ''}" onclick="setTheme('light')" style="padding:20px; border-radius:var(--radius-lg); border:2px solid ${themePreference === 'light' ? 'var(--theme-primary)' : 'var(--theme-border)'}; background:var(--input-bg); cursor:pointer;">
                        <span class="material-symbols-outlined" style="font-size:32px; display:block; margin-bottom:8px;">light_mode</span>
                        <span>Hell</span>
                    </button>
                    <button class="theme-option ${themePreference === 'dark' ? 'active' : ''}" onclick="setTheme('dark')" style="padding:20px; border-radius:var(--radius-lg); border:2px solid ${themePreference === 'dark' ? 'var(--theme-primary)' : 'var(--theme-border)'}; background:var(--input-bg); cursor:pointer;">
                        <span class="material-symbols-outlined" style="font-size:32px; display:block; margin-bottom:8px;">dark_mode</span>
                        <span>Dunkel</span>
                    </button>
                    <button class="theme-option ${themePreference === 'system' ? 'active' : ''}" onclick="setTheme('system')" style="padding:20px; border-radius:var(--radius-lg); border:2px solid ${themePreference === 'system' ? 'var(--theme-primary)' : 'var(--theme-border)'}; background:var(--input-bg); cursor:pointer;">
                        <span class="material-symbols-outlined" style="font-size:32px; display:block; margin-bottom:8px;">computer</span>
                        <span>System</span>
                    </button>
                </div>
            </div>

            <div class="dashboard-card">
                <h3><span class="material-symbols-outlined">format_size</span>Schriftgröße</h3>
                <p class="text-sm text-muted mb-4">Passen Sie die Schriftgröße nach Ihren Bedürfnissen an.</p>

                <div class="flex items-center gap-2">
                    <span class="text-sm">A</span>
                    <input type="range" min="80" max="120" value="${currentUser?.settings?.font_scale || 100}"
                           style="flex:1;" onchange="updateSetting('font_scale', this.value); document.documentElement.style.fontSize = this.value + '%';">
                    <span style="font-size:1.2em;">A</span>
                </div>
            </div>
        `;
    }

    window.setTheme = function(theme) {
        if (TB.ui.theme?.setPreference) {
            TB.ui.theme.setPreference(theme);
            TB.ui.Toast.showSuccess(`Theme auf "${theme === 'system' ? 'System' : theme === 'dark' ? 'Dunkel' : 'Hell'}" gesetzt`);
            loadAppearance();
        }
    };

    // ========== Profil ==========
    async function loadProfile() {
        const content = document.getElementById('profile-content');

        content.innerHTML = `
            <div class="dashboard-card">
                <h3><span class="material-symbols-outlined">account_circle</span>Profil-Informationen</h3>

                <div class="settings-section">
                    <div class="setting-item">
                        <div class="setting-info">
                            <div class="setting-label">Benutzername</div>
                            <div class="setting-description">${TB.utils.escapeHtml(currentUser?.username || currentUser?.name || '-')}</div>
                        </div>
                    </div>

                    <div class="setting-item">
                        <div class="setting-info">
                            <div class="setting-label">E-Mail-Adresse</div>
                            <div class="setting-description">${TB.utils.escapeHtml(currentUser?.email || 'Nicht angegeben')}</div>
                        </div>
                        <button class="tb-btn tb-btn-secondary tb-btn-sm" onclick="TB.user?.getClerkInstance?.()?.openUserProfile?.() || TB.ui.Toast.showInfo('Profil-Einstellungen werden geladen...')">
                            <span class="material-symbols-outlined">edit</span>
                            Ändern
                        </button>
                    </div>

                    <div class="setting-item">
                        <div class="setting-info">
                            <div class="setting-label">Benutzer-Level</div>
                            <div class="setting-description">Level ${currentUser?.level || 1}</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="dashboard-card">
                <h3><span class="material-symbols-outlined">key</span>Sicherheit</h3>

                <div class="flex gap-2" style="flex-wrap:wrap;">
                    <button class="tb-btn tb-btn-secondary" onclick="requestMagicLink()">
                        <span class="material-symbols-outlined">link</span>
                        Magic Link anfordern
                    </button>
                    <button class="tb-btn tb-btn-secondary" onclick="TB.user?.getClerkInstance?.()?.openUserProfile?.()">
                        <span class="material-symbols-outlined">security</span>
                        Sicherheitseinstellungen
                    </button>
                </div>
            </div>

            <div class="dashboard-card">
                <h3><span class="material-symbols-outlined">devices</span>Aktive Sitzungen</h3>
                <p class="text-sm text-muted mb-4">Ihre aktuell angemeldeten Geräte und Sitzungen.</p>

                <div id="sessions-list">
                    <div class="setting-item">
                        <div class="setting-info">
                            <div class="setting-label">Diese Sitzung</div>
                            <div class="setting-description">${navigator.userAgent.includes('Mobile') ? 'Mobiles Gerät' : 'Desktop-Browser'}</div>
                        </div>
                        <span class="module-status loaded">Aktiv</span>
                    </div>
                    ${userInstance?.cli_sessions?.length > 0 ? userInstance.cli_sessions.map(s => `
                        <div class="setting-item">
                            <div class="setting-info">
                                <div class="setting-label">CLI Sitzung</div>
                                <div class="setting-description">Gestartet: ${new Date(s.created_at * 1000).toLocaleString()}</div>
                            </div>
                            <button class="tb-btn tb-btn-danger tb-btn-sm" onclick="closeCLISession('${s.cli_session_id}')">
                                <span class="material-symbols-outlined">close</span>
                            </button>
                        </div>
                    `).join('') : ''}
                </div>
            </div>

            <div class="dashboard-card" style="border-color:var(--color-error);">
                <h3 style="color:var(--color-error);"><span class="material-symbols-outlined">warning</span>Gefahrenzone</h3>

                <button class="tb-btn tb-btn-danger" onclick="TB.user.logout().then(() => window.location.href = '/')">
                    <span class="material-symbols-outlined">logout</span>
                    Abmelden
                </button>
            </div>
        `;
    }

    window.requestMagicLink = async function() {
        TB.ui.Loader.show('Magic Link wird angefordert...');
        try {
            const res = await TB.api.request('CloudM.UserDashboard', 'request_my_magic_link', null, 'POST');
            TB.ui.Loader.hide();
            if (res.error === TB.ToolBoxError.none) {
                TB.ui.Toast.showSuccess('Magic Link wurde an Ihre E-Mail gesendet');
            } else {
                TB.ui.Toast.showError(res.info.help_text || 'Fehler beim Anfordern');
            }
        } catch(e) {
            TB.ui.Loader.hide();
            TB.ui.Toast.showError('Netzwerkfehler');
        }
    };

    window.closeCLISession = async function(sessionId) {
        if (!confirm('Möchten Sie diese CLI-Sitzung wirklich beenden?')) return;
        TB.ui.Loader.show('Beende Sitzung...');
        try {
            const res = await TB.api.request('CloudM.UserDashboard', 'close_cli_session', {cli_session_id: sessionId}, 'POST');
            TB.ui.Loader.hide();
            if (res.error === TB.ToolBoxError.none) {
                TB.ui.Toast.showSuccess('Sitzung beendet');
                await loadProfile();
            } else {
                TB.ui.Toast.showError('Fehler');
            }
        } catch(e) {
            TB.ui.Loader.hide();
            TB.ui.Toast.showError('Netzwerkfehler');
        }
    };

    // Make showSection global
    window.showSection = showSection;

    // ========== Start ==========
    if (window.TB?.events && window.TB.config?.get('appRootId')) {
        initDashboard();
    } else {
        document.addEventListener('tbjs:initialized', initDashboard, { once: true });
    }
}
</script>
"""
    return Result.html(html_content)


# =================== API Endpoints für Modul-Verwaltung ===================

@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['GET'])
async def get_all_available_modules(app: App, request: RequestData):
    """Liste aller verfügbaren Module für den Benutzer"""
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    try:
        all_mods = app.get_all_mods()
        # Filter basierend auf Benutzer-Level
        user_level = getattr(current_user, 'level', 1)
        # Für jetzt alle Module zurückgeben
        return Result.ok(data=list(all_mods))
    except Exception as e:
        return Result.default_internal_error(f"Fehler beim Laden der Module: {e}")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['GET'])
async def get_my_active_instances(app: App, request: RequestData):
    """Aktive Instanzen des aktuellen Benutzers abrufen"""
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    uid = getattr(current_user, 'uid', None) or getattr(current_user, 'clerk_user_id', None)
    if not uid:
        return Result.default_user_error(info="Benutzer-ID nicht gefunden")

    from .UserInstances import get_user_instance_with_cli_sessions, get_user_cli_sessions

    instance_data = get_user_instance_with_cli_sessions(uid, hydrate=True)
    cli_sessions = get_user_cli_sessions(uid)

    active_instances = []
    if instance_data and isinstance(instance_data, dict):
        live_modules = []
        if instance_data.get("live"):
            for mod_name, spec_val in instance_data.get("live").items():
                live_modules.append({"name": mod_name, "spec": str(spec_val)})

        instance_summary = {
            "SiID": instance_data.get("SiID"),
            "VtID": instance_data.get("VtID"),
            "webSocketID": instance_data.get("webSocketID"),
            "live_modules": live_modules,
            "saved_modules": instance_data.get("save", {}).get("mods", []),
            "cli_sessions": cli_sessions,
            "active_cli_sessions": len([s for s in cli_sessions if s.get('status') == 'active'])
        }
        active_instances.append(instance_summary)

    return Result.ok(data=active_instances)


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def add_module_to_instance(app: App, request: RequestData, data: dict):
    """Modul zur Benutzer-Instanz hinzufügen und laden"""
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    module_name = data.get("module_name")
    if not module_name:
        return Result.default_user_error(info="Modulname erforderlich")

    uid = getattr(current_user, 'uid', None) or getattr(current_user, 'clerk_user_id', None)

    try:
        instance = get_user_instance_internal(uid, hydrate=False)
        if not instance:
            return Result.default_internal_error("Instanz nicht gefunden")

        # Modul laden
        if module_name not in app.get_all_mods():
            return Result.default_user_error(f"Modul '{module_name}' nicht verfügbar")

        spec = app.save_load(module_name)
        if spec:
            if 'live' not in instance:
                instance['live'] = {}
            instance['live'][module_name] = spec

            from .UserInstances import save_user_instances
            save_user_instances(instance)

            return Result.ok(info=f"Modul '{module_name}' geladen")
        else:
            return Result.default_internal_error(f"Fehler beim Laden von '{module_name}'")
    except Exception as e:
        return Result.default_internal_error(f"Fehler: {e}")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def remove_module_from_instance(app: App, request: RequestData, data: dict):
    """Modul aus Benutzer-Instanz entladen"""
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    module_name = data.get("module_name")
    if not module_name:
        return Result.default_user_error(info="Modulname erforderlich")

    uid = getattr(current_user, 'uid', None) or getattr(current_user, 'clerk_user_id', None)

    try:
        instance = get_user_instance_internal(uid, hydrate=False)
        if not instance:
            return Result.default_internal_error("Instanz nicht gefunden")

        if 'live' in instance and module_name in instance['live']:
            spec = instance['live'][module_name]
            app.remove_mod(mod_name=module_name, spec=spec, delete=False)
            del instance['live'][module_name]

            from .UserInstances import save_user_instances
            save_user_instances(instance)

            return Result.ok(info=f"Modul '{module_name}' entladen")
        else:
            return Result.default_user_error(f"Modul '{module_name}' nicht geladen")
    except Exception as e:
        return Result.default_internal_error(f"Fehler: {e}")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def add_module_to_saved(app: App, request: RequestData, data: dict):
    """Modul zu den gespeicherten Modulen hinzufügen"""
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    module_name = data.get("module_name")
    if not module_name:
        return Result.default_user_error(info="Modulname erforderlich")

    uid = getattr(current_user, 'uid', None) or getattr(current_user, 'clerk_user_id', None)

    try:
        instance = get_user_instance_internal(uid, hydrate=False)
        if not instance:
            return Result.default_internal_error("Instanz nicht gefunden")

        if 'save' not in instance:
            instance['save'] = {'mods': [], 'uid': uid}
        if 'mods' not in instance['save']:
            instance['save']['mods'] = []

        if module_name not in instance['save']['mods']:
            instance['save']['mods'].append(module_name)

            from .UserInstances import save_user_instances
            save_user_instances(instance)

            # In DB speichern
            app.run_any('DB', 'set',
                        query=f"User::Instance::{uid}",
                        data=json.dumps({"saves": instance['save']}))

            return Result.ok(info=f"Modul '{module_name}' gespeichert")
        else:
            return Result.ok(info=f"Modul '{module_name}' bereits gespeichert")
    except Exception as e:
        return Result.default_internal_error(f"Fehler: {e}")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def remove_module_from_saved(app: App, request: RequestData, data: dict):
    """Modul aus den gespeicherten Modulen entfernen"""
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    module_name = data.get("module_name")
    if not module_name:
        return Result.default_user_error(info="Modulname erforderlich")

    uid = getattr(current_user, 'uid', None) or getattr(current_user, 'clerk_user_id', None)

    try:
        instance = get_user_instance_internal(uid, hydrate=False)
        if not instance:
            return Result.default_internal_error("Instanz nicht gefunden")

        if 'save' in instance and 'mods' in instance['save']:
            if module_name in instance['save']['mods']:
                instance['save']['mods'].remove(module_name)

                from .UserInstances import save_user_instances
                save_user_instances(instance)

                # In DB speichern
                app.run_any('DB', 'set',
                            query=f"User::Instance::{uid}",
                            data=json.dumps({"saves": instance['save']}))

                return Result.ok(info=f"Modul '{module_name}' entfernt")

        return Result.default_user_error(f"Modul '{module_name}' nicht in gespeicherten Modulen")
    except Exception as e:
        return Result.default_internal_error(f"Fehler: {e}")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['GET'])
async def get_all_mod_data(app: App, request: RequestData):
    """Alle Mod-Daten des aktuellen Benutzers abrufen"""
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    mod_data = {}
    if hasattr(current_user, 'mod_data') and current_user.mod_data:
        mod_data = current_user.mod_data
    elif hasattr(current_user, 'settings') and current_user.settings:
        mod_data = current_user.settings.get('mod_data', {})

    return Result.ok(data=mod_data)


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def request_my_magic_link(app: App, request: RequestData):
    """Magic Link für den aktuellen Benutzer anfordern"""
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    username = getattr(current_user, 'username', None) or getattr(current_user, 'name', None)
    if not username:
        return Result.default_user_error(info="Benutzername nicht gefunden")

    magic_link_result = await request_magic_link_backend(app, username=username)

    if not magic_link_result.as_result().is_error():
        email = getattr(current_user, 'email', 'Ihre E-Mail')
        return Result.ok(info=f"Magic Link wurde an {email} gesendet")
    else:
        return Result.default_internal_error(f"Fehler: {magic_link_result.info}")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def update_my_settings(app: App, request: RequestData, data: dict):
    """Benutzereinstellungen aktualisieren"""
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    settings_payload = data.get("settings")
    if not isinstance(settings_payload, dict):
        return Result.default_user_error(info="Ungültige Einstellungen")

    if current_user.settings is None:
        current_user.settings = {}

    current_user.settings.update(settings_payload)

    save_result = db_helper_save_user(app, asdict(current_user))
    if save_result.is_error():
        return Result.default_internal_error(f"Fehler beim Speichern: {save_result.info}")

    return Result.ok(info="Einstellungen gespeichert", data=current_user.settings)


# =================== CLI Session Management ===================

@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def close_cli_session(app: App, request: RequestData, data: dict):
    """CLI-Sitzung schließen"""
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    cli_session_id = data.get("cli_session_id")
    if not cli_session_id:
        return Result.default_user_error(info="Session-ID erforderlich")

    from .UserInstances import close_cli_session as close_cli_session_internal, UserInstances

    uid = getattr(current_user, 'uid', None) or getattr(current_user, 'clerk_user_id', None)

    # Überprüfen ob Sitzung dem Benutzer gehört
    if cli_session_id in UserInstances().cli_sessions:
        session_data = UserInstances().cli_sessions[cli_session_id]
        if session_data['uid'] != uid:
            return Result.default_user_error(info="Nicht berechtigt, diese Sitzung zu schließen")

    result = close_cli_session_internal(cli_session_id)
    return Result.ok(info=result)
