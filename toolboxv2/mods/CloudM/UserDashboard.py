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


@export(
    mod_name=Name,
    api=True,
    version=version,
    name="main",
    api_methods=["GET"],
    request_as_kwarg=True,
    row=True,
)
async def get_user_dashboard_main_page(app: App, request: RequestData):
    """Haupt-Dashboard Seite - Modern, Tab-basiert, vollständig responsive"""

    html_content = """
<style>
/* ============================================================
   User Dashboard Styles (nutzen TBJS v2 Variablen)
   ============================================================ */

.dashboard {
    max-width: 1200px;
    margin: 0 auto;
    padding: var(--space-6) var(--space-5);
}

/* ========== Header ========== */
.dashboard-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: var(--space-6);
    flex-wrap: wrap;
    gap: var(--space-4);
}

.dashboard-title {
    display: flex;
    align-items: center;
    gap: var(--space-3);
}

.dashboard-title h1 {
    font-size: var(--text-3xl);
    font-weight: var(--weight-bold);
    color: var(--text-primary);
    margin: 0;
}

.user-avatar {
    width: 48px;
    height: 48px;
    border-radius: var(--radius-full);
    background: linear-gradient(135deg, var(--color-primary-400), var(--color-primary-600));
    color: var(--text-inverse);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: var(--weight-bold);
    font-size: var(--text-lg);
    flex-shrink: 0;
    box-shadow: var(--shadow-sm);
}

.header-actions {
    display: flex;
    align-items: center;
    gap: var(--space-3);
}

/* ========== Tab Navigation ========== */
.tab-navigation {
    display: flex;
    gap: var(--space-2);
    margin-bottom: var(--space-6);
    padding-bottom: var(--space-2);
    border-bottom: var(--border-width) solid var(--border-default);
    overflow-x: auto;
    scrollbar-width: none;
    -ms-overflow-style: none;
    -webkit-overflow-scrolling: touch;
}

.tab-navigation::-webkit-scrollbar {
    display: none;
}

.tab-btn {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-3) var(--space-4);
    background: transparent;
    border: none;
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    font-size: var(--text-sm);
    font-weight: var(--weight-medium);
    font-family: inherit;
    cursor: pointer;
    white-space: nowrap;
    flex-shrink: 0;
    transition: all var(--duration-fast) var(--ease-default);
    width: max-content; !important;
}

.tab-btn:hover {
    color: var(--text-primary);
    background: var(--interactive-muted);
}

.tab-btn.active {
    color: var(--text-inverse);
    background: linear-gradient(135deg, var(--color-primary-400), var(--color-primary-600));
    box-shadow: var(--shadow-primary);
}

.tab-btn .material-symbols-outlined {
    font-size: 20px;
}

/* Mobile Tab Scroll Indicator */
.tab-scroll-hint {
    display: none;
    position: absolute;
    right: 0;
    top: 0;
    bottom: 0;
    width: 40px;
    background: linear-gradient(to left, var(--bg-surface), transparent);
    pointer-events: none;
}

/* ========== Content Sections ========== */
.content-section {
    display: none;
    animation: fadeSlideIn 0.3s var(--ease-out);
}

.content-section.active {
    display: block;
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.section-header {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    margin-bottom: var(--space-5);
}

.section-header h2 {
    font-size: var(--text-2xl);
    font-weight: var(--weight-semibold);
    color: var(--text-primary);
    margin: 0;
}

.section-header .material-symbols-outlined {
    font-size: 28px;
    color: var(--interactive);
}

/* ========== Stats Grid ========== */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: var(--space-4);
    margin-bottom: var(--space-6);
}

.stat-card {
    background: var(--bg-surface);
    border: var(--border-width) solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: var(--space-5);
    text-align: center;
    box-shadow: var(--highlight-subtle), var(--shadow-sm);
    transition: all var(--duration-fast) var(--ease-default);
}

.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--highlight-subtle), var(--shadow-md);
}

.stat-value {
    font-size: var(--text-3xl);
    font-weight: var(--weight-bold);
    color: var(--interactive);
    line-height: var(--leading-tight);
}

.stat-label {
    font-size: var(--text-sm);
    color: var(--text-muted);
    margin-top: var(--space-1);
}

/* ========== Dashboard Cards ========== */
.dashboard-card {
    background: var(--bg-surface);
    border: var(--border-width) solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: var(--space-5);
    margin-bottom: var(--space-5);
    box-shadow: var(--highlight-subtle), var(--shadow-sm);
    transition: all var(--duration-fast) var(--ease-default);
}

.dashboard-card:hover {
    box-shadow: var(--highlight-subtle), var(--shadow-md);
}

.dashboard-card h3 {
    font-size: var(--text-lg);
    font-weight: var(--weight-semibold);
    color: var(--text-primary);
    margin: 0 0 var(--space-4) 0;
    display: flex;
    align-items: center;
    gap: var(--space-2);
}

.dashboard-card h3 .material-symbols-outlined {
    color: var(--interactive);
    font-size: 22px;
}

/* ========== Module Grid ========== */
.module-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: var(--space-4);
}

.module-card {
    background: var(--bg-elevated);
    border: var(--border-width) solid var(--border-default);
    border-radius: var(--radius-md);
    padding: var(--space-4);
    transition: all var(--duration-fast) var(--ease-default);
}

.module-card:hover {
    border-color: var(--interactive);
    box-shadow: var(--shadow-sm);
}

.module-card.active {
    border-color: var(--color-success);
    background: oklch(from var(--color-success) l c h / 0.08);
}

.module-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--space-2);
}

.module-name {
    font-weight: var(--weight-semibold);
    color: var(--text-primary);
    font-size: var(--text-sm);
}

.module-status {
    font-size: var(--text-xs);
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius-full);
    font-weight: var(--weight-medium);
}

.module-status.loaded {
    background: var(--color-success);
    color: white;
}

.module-status.available {
    background: var(--border-default);
    color: var(--text-muted);
}

.module-actions {
    display: flex;
    gap: var(--space-2);
    flex-wrap: wrap;
    margin-top: var(--space-3);
}

/* ========== Settings ========== */
.settings-section {
    margin-bottom: var(--space-6);
}

.settings-section h4 {
    font-size: var(--text-base);
    font-weight: var(--weight-semibold);
    margin-bottom: var(--space-4);
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: var(--space-2);
}

.setting-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--space-4);
    background: var(--bg-elevated);
    border: var(--border-width) solid var(--border-subtle);
    border-radius: var(--radius-md);
    margin-bottom: var(--space-2);
    transition: border-color var(--duration-fast) var(--ease-default);
}

.setting-item:hover {
    border-color: var(--border-strong);
}

.setting-info {
    flex: 1;
    min-width: 0;
}

.setting-label {
    font-weight: var(--weight-medium);
    color: var(--text-primary);
    margin-bottom: var(--space-1);
}

.setting-description {
    font-size: var(--text-sm);
    color: var(--text-muted);
}

/* ========== Toggle Switch ========== */
.toggle-switch {
    position: relative;
    width: 48px;
    height: 26px;
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
    inset: 0;
    background-color: var(--border-default);
    transition: var(--duration-fast) var(--ease-default);
    border-radius: var(--radius-full);
}

.toggle-slider::before {
    position: absolute;
    content: "";
    height: 20px;
    width: 20px;
    left: 3px;
    bottom: 3px;
    background-color: white;
    transition: var(--duration-fast) var(--ease-default);
    border-radius: var(--radius-full);
    box-shadow: var(--shadow-xs);
}

input:checked + .toggle-slider {
    background: linear-gradient(135deg, var(--color-primary-400), var(--color-primary-600));
}

input:checked + .toggle-slider::before {
    transform: translateX(22px);
}

/* ========== Buttons ========== */
.tb-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-4);
    border-radius: var(--radius-md);
    font-weight: var(--weight-medium);
    font-size: var(--text-sm);
    font-family: inherit;
    cursor: pointer;
    border: var(--border-width) solid transparent;
    transition: all var(--duration-fast) var(--ease-default);
}

.tb-btn:hover {
    transform: translateY(-1px);
}

.tb-btn-primary {
    background: linear-gradient(135deg, var(--color-primary-400), var(--color-primary-600));
    color: var(--text-inverse);
    box-shadow: var(--shadow-primary);
}

.tb-btn-primary:hover {
    box-shadow: 0 6px 20px oklch(55% 0.18 230 / 0.4);
}

.tb-btn-secondary {
    background: var(--bg-surface);
    color: var(--text-primary);
    border-color: var(--border-default);
    box-shadow: var(--shadow-xs);
}

.tb-btn-secondary:hover {
    background: var(--bg-elevated);
    border-color: var(--border-strong);
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
    padding: var(--space-1) var(--space-3);
    font-size: var(--text-xs);
}

.tb-btn .material-symbols-outlined {
    font-size: 18px;
}

/* ========== Inputs ========== */
.tb-input {
    width: 100%;
    padding: var(--space-3) var(--space-4);
    font-size: var(--text-base);
    font-family: inherit;
    color: var(--text-primary);
    background-color: var(--input-bg);
    border: var(--border-width) solid var(--input-border);
    border-radius: var(--radius-md);
    transition: all var(--duration-fast) var(--ease-default);
    margin-bottom: 0;
}

.tb-input:focus {
    outline: none;
    border-color: var(--input-focus);
    box-shadow: 0 0 0 3px oklch(from var(--input-focus) l c h / 0.15);
}

/* ========== Mod Data Panel ========== */
.mod-data-panel {
    background: var(--bg-elevated);
    border: var(--border-width) solid var(--border-default);
    border-radius: var(--radius-md);
    overflow: hidden;
    margin-bottom: var(--space-4);
}

.mod-data-header {
    background: var(--interactive-muted);
    padding: var(--space-3) var(--space-4);
    font-weight: var(--weight-semibold);
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    transition: background var(--duration-fast) var(--ease-default);
}

.mod-data-header:hover {
    background: oklch(from var(--interactive) l c h / 0.15);
}

.mod-data-content {
    padding: var(--space-4);
    display: none;
}

.mod-data-content.open {
    display: block;
}

.mod-data-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--space-3) 0;
    border-bottom: var(--border-width) solid var(--border-subtle);
}

.mod-data-item:last-child {
    border-bottom: none;
}

.mod-data-key {
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
    font-size: var(--text-sm);
}

.mod-data-value {
    color: var(--text-primary);
    font-size: var(--text-sm);
}

/* ========== Theme Selector ========== */
.theme-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: var(--space-4);
}

.theme-option {
    padding: var(--space-5);
    border-radius: var(--radius-lg);
    border: 2px solid var(--border-default);
    background: var(--bg-surface);
    cursor: pointer;
    text-align: center;
    transition: all var(--duration-fast) var(--ease-default);
}

.theme-option:hover {
    border-color: var(--border-strong);
}

.theme-option.active {
    border-color: var(--interactive);
    box-shadow: 0 0 0 3px oklch(from var(--interactive) l c h / 0.15);
}

.theme-option .material-symbols-outlined {
    font-size: 32px;
    display: block;
    margin-bottom: var(--space-2);
    color: var(--interactive);
}

/* ========== Info Table ========== */
.info-table {
    width: 100%;
    border-collapse: collapse;
}

.info-table td {
    padding: var(--space-3) var(--space-2);
    border-bottom: var(--border-width) solid var(--border-subtle);
}

.info-table tr:last-child td {
    border-bottom: none;
}

.info-table td:first-child {
    color: var(--text-muted);
    font-size: var(--text-sm);
    width: 40%;
}

/* ========== Quick Actions ========== */
.quick-actions {
    display: flex;
    gap: var(--space-3);
    flex-wrap: wrap;
}

/* ========== Empty State ========== */
.empty-state {
    text-align: center;
    padding: var(--space-10) var(--space-6);
    color: var(--text-muted);
}

.empty-state .material-symbols-outlined {
    font-size: 56px;
    margin-bottom: var(--space-4);
    opacity: 0.4;
}

.empty-state p {
    margin: 0;
    font-size: var(--text-lg);
}

/* ========== Loading Spinner ========== */
.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 2px solid var(--border-default);
    border-top-color: var(--interactive);
    border-radius: var(--radius-full);
    animation: spin 0.8s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* ========== Responsive ========== */
@media screen and (max-width: 767px) {
    .dashboard {
        padding: var(--space-4) var(--space-3);
    }

    .dashboard-header {
        flex-direction: column;
        align-items: stretch;
        gap: var(--space-3);
    }

    .dashboard-title h1 {
        font-size: var(--text-2xl);
    }

    .header-actions {
        justify-content: center;
    }

    .tab-navigation {
        margin-left: calc(var(--space-3) * -1);
        margin-right: calc(var(--space-3) * -1);
        padding-left: var(--space-3);
        padding-right: var(--space-3);
        position: relative;
    }

    .tab-btn {
        padding: var(--space-2) var(--space-3);
    }

    .tab-btn span:not(.material-symbols-outlined) {
        display: none;
    }

    .stats-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .module-grid {
        grid-template-columns: 1fr;
    }

    .setting-item {
        flex-direction: column;
        align-items: flex-start;
        gap: var(--space-3);
    }

    .toggle-switch {
        align-self: flex-end;
    }

    .quick-actions {
        flex-direction: column;
    }

    .quick-actions .tb-btn {
        width: 100%;
    }

    .theme-grid {
        grid-template-columns: repeat(3, 1fr);
    }
}

/* ========== Utility Classes ========== */
.text-muted { color: var(--text-muted); }
.text-success { color: var(--color-success); }
.text-error { color: var(--color-error); }
.text-sm { font-size: var(--text-sm); }
.mt-2 { margin-top: var(--space-2); }
.mt-4 { margin-top: var(--space-4); }
.mb-2 { margin-bottom: var(--space-2); }
.mb-4 { margin-bottom: var(--space-4); }
.flex { display: flex; }
.gap-2 { gap: var(--space-2); }
.gap-4 { gap: var(--space-4); }
.items-center { align-items: center; }
.justify-between { justify-content: space-between; }
</style>

<div class="content-wrapper">
    <main class="dashboard main-content glass">
        <!-- Header -->
        <header class="dashboard-header">
            <div class="dashboard-title">
                <div class="user-avatar" id="user-avatar">?</div>
                <div>
                    <h1 id="welcome-text">Dashboard</h1>
                    <span class="text-sm text-muted" id="user-email"></span>
                </div>
            </div>
            <div class="header-actions">
                <div id="darkModeToggleContainer"></div>
                <button id="logoutButtonUser" class="tb-btn tb-btn-secondary">
                    <span class="material-symbols-outlined">logout</span>
                    <span class="logout-text">Abmelden</span>
                </button>
            </div>
        </header>

        <!-- Tab Navigation -->
        <nav class="tab-navigation" id="tab-navigation" role="tablist">
            <button class="tab-btn active" data-section="overview" role="tab" aria-selected="true">
                <span class="material-symbols-outlined">home</span>
                <span>Übersicht</span>
            </button>
            <button class="tab-btn" data-section="my-modules" role="tab" aria-selected="false">
                <span class="material-symbols-outlined">extension</span>
                <span>Module</span>
            </button>
            <button class="tab-btn" data-section="mod-data" role="tab" aria-selected="false">
                <span class="material-symbols-outlined">database</span>
                <span>Daten</span>
            </button>
            <button class="tab-btn" data-section="settings" role="tab" aria-selected="false">
                <span class="material-symbols-outlined">settings</span>
                <span>Einstellungen</span>
            </button>
            <button class="tab-btn" data-section="appearance" role="tab" aria-selected="false">
                <span class="material-symbols-outlined">palette</span>
                <span>Theme</span>
            </button>
            <button class="tab-btn" data-section="profile" role="tab" aria-selected="false">
                <span class="material-symbols-outlined">person</span>
                <span>Profil</span>
            </button>
        </nav>

        <!-- Content Sections -->
        <div id="dashboard-content">
            <!-- Übersicht -->
            <section id="overview-section" class="content-section active">
                <div id="overview-content">
                    <p class="text-muted">Lädt...</p>
                </div>
            </section>

            <!-- Meine Module -->
            <section id="my-modules-section" class="content-section">
                <div id="my-modules-content">
                    <p class="text-muted">Lädt Module...</p>
                </div>
            </section>

            <!-- Mod-Daten -->
            <section id="mod-data-section" class="content-section">
                <div id="mod-data-content">
                    <p class="text-muted">Lädt Mod-Daten...</p>
                </div>
            </section>

            <!-- Einstellungen -->
            <section id="settings-section" class="content-section">
                <div id="settings-content">
                    <p class="text-muted">Lädt Einstellungen...</p>
                </div>
            </section>

            <!-- Erscheinungsbild -->
            <section id="appearance-section" class="content-section">
                <div id="appearance-content">
                    <p class="text-muted">Lädt Theme-Einstellungen...</p>
                </div>
            </section>

            <!-- Profil -->
            <section id="profile-section" class="content-section">
                <div id="profile-content">
                    <p class="text-muted">Lädt Profil...</p>
                </div>
            </section>
        </div>
    </main>
</div>

<script type="module">
if (typeof TB === 'undefined' || !TB.ui || !TB.api) {
    console.error('CRITICAL: TB (tbjs) not loaded.');
    document.body.innerHTML = '<div style="padding:40px; text-align:center; color:var(--color-error);">Fehler: Frontend-Bibliothek konnte nicht geladen werden.</div>';
} else {
    console.log('TB object found. Initializing User Dashboard v3...');

    let currentUser = null;
    let allModules = [];
    let userInstance = null;
    let modDataCache = {};

    // ========== Initialization ==========
    async function initDashboard() {
        console.log("Dashboard wird initialisiert...");
        TB.ui.DarkModeToggle.init();
        setupNavigation();
        setupLogout();

        try {
            const userRes = await TB.api.request('CloudM.UserAccountManager', 'get_current_user', null, 'GET');
            if (userRes.error === TB.ToolBoxError.none && userRes.get()) {
                currentUser = userRes.get();
                updateHeader();

                const modulesRes = await TB.api.request('CloudM.UserDashboard', 'get_all_available_modules', null, 'GET');
                if (modulesRes.error === TB.ToolBoxError.none) {
                    allModules = modulesRes.get() || [];
                }

                const instanceRes = await TB.api.request('CloudM.UserDashboard', 'get_my_active_instances', null, 'GET');
                if (instanceRes.error === TB.ToolBoxError.none && instanceRes.get()?.length > 0) {
                    userInstance = instanceRes.get()[0];
                }

                await showSection('overview');
            } else {
                showNotAuthenticated();
            }
        } catch (e) {
            console.error("Fehler beim Initialisieren:", e);
            showConnectionError();
        }
    }

    function updateHeader() {
        const avatarEl = document.getElementById('user-avatar');
        const welcomeEl = document.getElementById('welcome-text');
        const emailEl = document.getElementById('user-email');

        if (currentUser) {
            const name = currentUser.username || currentUser.name || 'Benutzer';
            const initial = name.charAt(0).toUpperCase();

            if (avatarEl) avatarEl.textContent = initial;
            if (welcomeEl) welcomeEl.textContent = `Hallo, ${name}!`;
            if (emailEl) emailEl.textContent = currentUser.email || '';
        }
    }

    function showNotAuthenticated() {
        document.getElementById('dashboard-content').innerHTML = `
            <div class="empty-state">
                <span class="material-symbols-outlined">login</span>
                <h3 style="margin-top:var(--space-4);">Nicht angemeldet</h3>
                <p class="text-muted">Bitte melden Sie sich an, um fortzufahren.</p>
                <button onclick="TB.router.navigateTo('/web/assets/login.html')" class="tb-btn tb-btn-primary mt-4">
                    <span class="material-symbols-outlined">login</span>
                    Anmelden
                </button>
            </div>
        `;
    }

    function showConnectionError() {
        document.getElementById('dashboard-content').innerHTML = `
            <div class="empty-state">
                <span class="material-symbols-outlined">cloud_off</span>
                <h3 style="margin-top:var(--space-4);">Verbindungsfehler</h3>
                <p class="text-muted">Die Verbindung zum Server konnte nicht hergestellt werden.</p>
            </div>
        `;
    }

    // ========== Navigation ==========
    function setupNavigation() {
        document.querySelectorAll('#tab-navigation .tab-btn').forEach(btn => {
            btn.addEventListener('click', async () => {
                document.querySelectorAll('#tab-navigation .tab-btn').forEach(b => {
                    b.classList.remove('active');
                    b.setAttribute('aria-selected', 'false');
                });
                btn.classList.add('active');
                btn.setAttribute('aria-selected', 'true');
                await showSection(btn.dataset.section);
            });
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
                    <div class="stat-label">Gespeichert</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${cliSessions}</div>
                    <div class="stat-label">CLI Sitzungen</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${currentUser?.level || 1}</div>
                    <div class="stat-label">Level</div>
                </div>
            </div>

            <div class="dashboard-card">
                <h3><span class="material-symbols-outlined">bolt</span>Schnellzugriff</h3>
                <div class="quick-actions">
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
                <h3><span class="material-symbols-outlined">account_circle</span>Konto-Info</h3>
                <table class="info-table">
                    <tr><td>Benutzername</td><td><strong>${TB.utils.escapeHtml(currentUser?.username || currentUser?.name || '-')}</strong></td></tr>
                    <tr><td>E-Mail</td><td>${TB.utils.escapeHtml(currentUser?.email || 'Nicht angegeben')}</td></tr>
                    <tr><td>Level</td><td>${currentUser?.level || 1}</td></tr>
                </table>
            </div>
        `;
    }

    // ========== Module ==========
    async function loadModules() {
        const content = document.getElementById('my-modules-content');

        try {
            const instanceRes = await TB.api.request('CloudM.UserDashboard', 'get_my_active_instances', null, 'GET');
            if (instanceRes.error === TB.ToolBoxError.none && instanceRes.get()?.length > 0) {
                userInstance = instanceRes.get()[0];
            }
        } catch(e) {}

        const liveModNames = (userInstance?.live_modules || []).map(m => m.name);
        const savedModNames = userInstance?.saved_modules || [];

        const categories = {};
        allModules.forEach(mod => {
            const category = mod.split('.')[0] || 'Andere';
            if (!categories[category]) categories[category] = [];
            categories[category].push(mod);
        });

        content.innerHTML = `
            <div class="dashboard-card">
                <h3><span class="material-symbols-outlined">info</span>Hinweis</h3>
                <p class="text-sm text-muted" style="margin:0;">
                    Aktivieren oder deaktivieren Sie Module nach Bedarf. Gespeicherte Module werden beim nächsten Login automatisch geladen.
                </p>
            </div>

            <div class="dashboard-card">
                <h3><span class="material-symbols-outlined">bookmark</span>Gespeicherte Module (${savedModNames.length})</h3>
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
                                    <div class="module-actions">
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
                            <summary style="cursor:pointer; font-weight:var(--weight-semibold); padding:var(--space-3) 0; color:var(--text-primary);">
                                <span class="material-symbols-outlined" style="vertical-align:middle; margin-right:var(--space-2);">folder</span>
                                ${TB.utils.escapeHtml(cat)} (${mods.length})
                            </summary>
                            <div class="module-grid" style="margin-top:var(--space-3);">
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
                                            <div class="module-actions">
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
                <p class="text-sm text-muted" style="margin:0;">
                    Jedes Modul kann eigene Daten für Sie speichern. Hier können Sie diese einsehen und bearbeiten.
                </p>
            </div>

            ${modNames.length > 0 ? modNames.map(modName => {
                const data = modDataCache[modName] || {};
                const entries = Object.entries(data);
                return `
                    <div class="mod-data-panel">
                        <div class="mod-data-header" onclick="this.nextElementSibling.classList.toggle('open'); this.querySelector('.expand-icon').textContent = this.nextElementSibling.classList.contains('open') ? 'expand_less' : 'expand_more';">
                            <span class="flex items-center gap-2">
                                <span class="material-symbols-outlined">extension</span>
                                ${TB.utils.escapeHtml(modName)}
                            </span>
                            <span class="material-symbols-outlined expand-icon">expand_more</span>
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
                <div class="empty-state">
                    <span class="material-symbols-outlined">folder_off</span>
                    <p>Noch keine Mod-Daten vorhanden.</p>
                    <p class="text-sm text-muted mt-2">Module speichern hier automatisch Ihre Einstellungen.</p>
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
                <textarea id="mod-data-editor" style="width:100%; height:200px; font-family:var(--font-mono); padding:var(--space-3); border:var(--border-width) solid var(--border-default); border-radius:var(--radius-md); background:var(--input-bg); color:var(--text-primary);">${TB.utils.escapeHtml(json)}</textarea>
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
                            <div class="setting-description">Aktiviert neue Funktionen in der Testphase</div>
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
                            <div class="setting-description">Benachrichtigungen über wichtige Ereignisse</div>
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
                            <div class="setting-description">Gespeicherte Module beim Login automatisch laden</div>
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
                            <div class="setting-description">Ausführliche Protokollierung für Fehlerbehebung</div>
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
                <div class="setting-item">
                    <div class="setting-info">
                        <div class="setting-label">Sprache</div>
                        <div class="setting-description">Bevorzugte Sprache</div>
                    </div>
                    <select class="tb-input" style="width:auto; margin-bottom:0;" onchange="updateSetting('language', this.value)">
                        <option value="de" ${settings.language === 'de' || !settings.language ? 'selected' : ''}>Deutsch</option>
                        <option value="en" ${settings.language === 'en' ? 'selected' : ''}>English</option>
                    </select>
                </div>
            </div>

            <div class="dashboard-card">
                <h3><span class="material-symbols-outlined">security</span>Datenschutz</h3>
                <div class="setting-item">
                    <div class="setting-info">
                        <div class="setting-label">Nutzungsstatistiken</div>
                        <div class="setting-description">Anonyme Statistiken zur Verbesserung senden</div>
                    </div>
                    <label class="toggle-switch">
                        <input type="checkbox" ${settings.analytics !== false ? 'checked' : ''}
                               onchange="updateSetting('analytics', this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
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
                <p class="text-sm text-muted mb-4">Wählen Sie Ihr bevorzugtes Farbschema.</p>

                <div class="theme-grid">
                    <button class="theme-option ${themePreference === 'light' ? 'active' : ''}" onclick="setTheme('light')">
                        <span class="material-symbols-outlined">light_mode</span>
                        <span>Hell</span>
                    </button>
                    <button class="theme-option ${themePreference === 'dark' ? 'active' : ''}" onclick="setTheme('dark')">
                        <span class="material-symbols-outlined">dark_mode</span>
                        <span>Dunkel</span>
                    </button>
                    <button class="theme-option ${themePreference === 'system' ? 'active' : ''}" onclick="setTheme('system')">
                        <span class="material-symbols-outlined">computer</span>
                        <span>System</span>
                    </button>
                </div>
            </div>

            <div class="dashboard-card">
                <h3><span class="material-symbols-outlined">format_size</span>Schriftgröße</h3>
                <p class="text-sm text-muted mb-4">Passen Sie die Schriftgröße an.</p>

                <div class="flex items-center gap-4">
                    <span class="text-sm">A</span>
                    <input type="range" min="80" max="120" value="${currentUser?.settings?.font_scale || 100}"
                           style="flex:1; accent-color:var(--interactive);"
                           onchange="updateSetting('font_scale', this.value); document.documentElement.style.fontSize = this.value + '%';">
                    <span style="font-size:1.25em;">A</span>
                </div>
            </div>
        `;
    }

    window.setTheme = function(theme) {
        if (TB.ui.theme?.setPreference) {
            TB.ui.theme.setPreference(theme);
            TB.ui.Toast.showSuccess(`Theme: ${theme === 'system' ? 'System' : theme === 'dark' ? 'Dunkel' : 'Hell'}`);
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
                        <button class="tb-btn tb-btn-secondary tb-btn-sm" onclick="TB.user?.getClerkInstance?.()?.openUserProfile?.() || TB.ui.Toast.showInfo('Profil wird geladen...')">
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

                <div class="quick-actions">
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
                <p class="text-sm text-muted mb-4">Ihre aktuell angemeldeten Geräte.</p>

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
