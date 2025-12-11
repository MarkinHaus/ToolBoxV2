# toolboxv2/mods/CloudM/AdminDashboard.py

import json
from dataclasses import asdict

from toolboxv2 import TBEF, App, RequestData, Result, get_app
from toolboxv2.mods.CloudM import mini
from toolboxv2.mods.CloudM.AuthManager import (
    db_helper_delete_user,
    db_helper_save_user,
    db_helper_test_exist,
)
from toolboxv2.mods.CloudM.ModManager import list_modules as list_all_modules

# For Waiting List invites, we'll call CloudM.email_services.send_signup_invitation_email
from .email_services import send_signup_invitation_email
from .types import User
from .UserAccountManager import get_current_user_from_request

Name = 'CloudM.AdminDashboard'
export = get_app(Name + ".Export").tb
version = '0.1.1'  # Incremented version

PID_DIR = "./.info"  # Standardized PID directory


async def _is_admin(app: App, request: RequestData) -> User | None:
    """Check if user is admin. Admin = level -1 or username 'root'/'loot'"""
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return None

    # Get username from either attribute (Clerk uses 'username', legacy uses 'name')
    username = getattr(current_user, 'username', None) or getattr(current_user, 'name', None)
    level = getattr(current_user, 'level', 1)

    # Admin check: level -1 OR special usernames
    if level == -1 or username == 'root' or username == 'loot':
        return current_user
    return None


@export(
    mod_name=Name,
    api=True,
    version=version,
    name="main",
    api_methods=["GET"],
    request_as_kwarg=True,
)
async def get_dashboard_main_page(app: App, request: RequestData):
    admin_user = await _is_admin(app, request)
    if not admin_user:
        return Result.html(
            "<h1>Access Denied</h1><p>You do not have permission to view this page.</p>",
            status=403,
        )

    html_content = """
<style>
/* ============================================================
   Admin Dashboard Styles (nutzen TBJS v2 Variablen)
   ============================================================ */

.dashboard {
    max-width: 100%;
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
    display: flex;
    align-items: center;
    gap: var(--space-2);
}

.dashboard-title h1 .material-symbols-outlined {
    color: var(--color-warning);
    font-size: 1.2em;
}

.admin-badge {
    background: linear-gradient(135deg, var(--color-warning), oklch(65% 0.2 45));
    color: var(--color-neutral-900);
    padding: var(--space-1) var(--space-3);
    border-radius: var(--radius-md);
    font-size: var(--text-xs);
    font-weight: var(--weight-bold);
    text-transform: uppercase;
    letter-spacing: var(--tracking-wide);
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
    color: var(--color-neutral-900);
    background: linear-gradient(135deg, var(--color-warning), oklch(65% 0.2 45));
    box-shadow: 0 4px 14px oklch(75% 0.18 85 / 0.35);
}

.tab-btn .material-symbols-outlined {
    font-size: 20px;
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
    padding-bottom: var(--space-4);
    border-bottom: var(--border-width) solid var(--border-default);
}

.section-header h2 {
    font-size: var(--text-2xl);
    font-weight: var(--weight-semibold);
    color: var(--text-primary);
    margin: 0;
    display: flex;
    align-items: center;
    gap: var(--space-2);
}

.section-header h2 .material-symbols-outlined {
    font-size: 28px;
    color: var(--color-warning);
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

/* ========== Tables ========== */
.admin-table {
    width: 100%;
    border-collapse: collapse;
    font-size: var(--text-sm);
    margin-top: var(--space-4);
}

.admin-table th,
.admin-table td {
    padding: var(--space-3) var(--space-4);
    text-align: left;
    border-bottom: var(--border-width) solid var(--border-subtle);
}

.admin-table th {
    background: var(--bg-sunken);
    font-weight: var(--weight-semibold);
    color: var(--text-primary);
    white-space: nowrap;
}

.admin-table tr:hover {
    background: oklch(from var(--interactive) l c h / 0.05);
}

.admin-table td {
    color: var(--text-secondary);
}

/* Table Responsive Wrapper */
.table-wrapper {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    border: var(--border-width) solid var(--border-subtle);
    border-radius: var(--radius-md);
}

.table-wrapper .admin-table {
    margin-top: 0;
}

.table-wrapper .admin-table th:first-child,
.table-wrapper .admin-table td:first-child {
    position: sticky;
    left: 0;
    background: var(--bg-surface);
    z-index: 1;
}

.table-wrapper .admin-table th:first-child {
    background: var(--bg-sunken);
}

/* ========== Stats Grid ========== */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: var(--space-4);
    margin-bottom: var(--space-5);
}

.stat-card {
    background: var(--bg-surface);
    border: var(--border-width) solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: var(--space-4);
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
    line-height: 1.2;
}

.stat-label {
    font-size: var(--text-sm);
    color: var(--text-muted);
    margin-top: var(--space-1);
}

/* ========== Status Indicators ========== */
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
}

.status-dot {
    width: 10px;
    height: 10px;
    border-radius: var(--radius-full);
    flex-shrink: 0;
}

.status-dot.green { background: var(--color-success); box-shadow: 0 0 8px var(--color-success); }
.status-dot.yellow { background: var(--color-warning); box-shadow: 0 0 8px var(--color-warning); }
.status-dot.red { background: var(--color-error); box-shadow: 0 0 8px var(--color-error); }

/* ========== Action Buttons ========== */
.action-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-1);
    padding: var(--space-1) var(--space-3);
    border-radius: var(--radius-sm);
    font-weight: var(--weight-medium);
    font-size: var(--text-xs);
    font-family: inherit;
    cursor: pointer;
    border: none;
    transition: all var(--duration-fast) var(--ease-default);
    white-space: nowrap;
}

.action-btn:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-sm);
}

.action-btn .material-symbols-outlined {
    font-size: 16px;
}

.btn-restart {
    background: var(--color-warning);
    color: var(--color-neutral-900);
}

.btn-restart:hover {
    background: oklch(from var(--color-warning) calc(l - 0.1) c h);
}

.btn-edit {
    background: var(--color-info);
    color: white;
}

.btn-edit:hover {
    background: oklch(from var(--color-info) calc(l - 0.1) c h);
}

.btn-delete {
    background: var(--color-error);
    color: white;
}

.btn-delete:hover {
    background: oklch(from var(--color-error) calc(l - 0.1) c h);
}

.btn-send-invite {
    background: var(--color-success);
    color: white;
}

.btn-send-invite:hover {
    background: oklch(from var(--color-success) calc(l - 0.1) c h);
}

.btn-open-link {
    background: var(--interactive);
    color: white;
}

.btn-open-link:hover {
    background: var(--interactive-hover);
}

.action-group {
    display: flex;
    gap: var(--space-2);
    flex-wrap: wrap;
}

/* ========== General Buttons ========== */
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

.tb-label {
    display: block;
    font-size: var(--text-sm);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
    margin-bottom: var(--space-1);
}

.tb-checkbox {
    margin-right: var(--space-2);
    accent-color: var(--interactive);
}

/* ========== Settings Section ========== */
.settings-section {
    margin-bottom: var(--space-6);
}

.settings-section h4 {
    font-size: var(--text-base);
    font-weight: var(--weight-semibold);
    margin-bottom: var(--space-4);
    color: var(--text-primary);
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

/* ========== Loading State ========== */
.loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-4);
    padding: var(--space-10) var(--space-6);
    color: var(--text-secondary);
}

.spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border-default);
    border-top-color: var(--interactive);
    border-radius: var(--radius-full);
    animation: spin 0.8s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* ========== User Level Badge ========== */
.level-badge {
    display: inline-flex;
    align-items: center;
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius-sm);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
}

.level-badge.admin {
    background: oklch(from var(--color-warning) l c h / 0.2);
    color: oklch(from var(--color-warning) calc(l - 0.2) c h);
}

[data-theme="dark"] .level-badge.admin {
    background: oklch(from var(--color-warning) l c h / 0.15);
    color: var(--color-warning);
}

/* ========== Quick Actions ========== */
.quick-actions {
    display: flex;
    gap: var(--space-3);
    flex-wrap: wrap;
}

/* ========== Form Spacing ========== */
.form-group {
    margin-bottom: var(--space-4);
}

.form-row {
    display: flex;
    gap: var(--space-4);
    flex-wrap: wrap;
}

.form-row > * {
    flex: 1;
    min-width: 200px;
}

/* ========== Module Cards ========== */
.module-card {
    background: var(--bg-elevated);
    border: var(--border-width) solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: var(--space-4);
    margin-bottom: var(--space-3);
    transition: all var(--duration-fast) var(--ease-default);
}

.module-card:hover {
    border-color: var(--border-strong);
}

.module-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: var(--space-3);
}

.module-name {
    font-weight: var(--weight-semibold);
    color: var(--text-primary);
}

.module-actions {
    display: flex;
    gap: var(--space-2);
    flex-wrap: wrap;
}

/* ========== SPP Cards ========== */
.spp-card {
    background: var(--bg-elevated);
    border: var(--border-width) solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: var(--space-4);
    margin-bottom: var(--space-3);
    transition: all var(--duration-fast) var(--ease-default);
}

.spp-card:hover {
    border-color: var(--border-strong);
}

.spp-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: var(--space-3);
}

.spp-info {
    flex: 1;
    min-width: 0;
}

.spp-title {
    font-weight: var(--weight-semibold);
    color: var(--text-primary);
    margin-bottom: var(--space-1);
}

.spp-path {
    font-size: var(--text-sm);
    color: var(--text-muted);
    word-break: break-all;
}

.spp-actions {
    display: flex;
    gap: var(--space-2);
    flex-shrink: 0;
}

/* ========== Empty State ========== */
.empty-state {
    text-align: center;
    padding: var(--space-8);
    color: var(--text-muted);
}

.empty-state .material-symbols-outlined {
    font-size: 64px;
    margin-bottom: var(--space-4);
    opacity: 0.5;
}

.empty-state p {
    font-size: var(--text-lg);
}

/* ========== Responsive - Tablet ========== */
@media screen and (max-width: 1024px) {
    .dashboard {
        padding: var(--space-5) var(--space-4);
    }

    .stats-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .admin-table {
        font-size: var(--text-sm);
    }

    .dashboard-card {
        padding: var(--space-4);
    }
}

/* ========== Responsive - Mobile ========== */
@media screen and (max-width: 767px) {
    .dashboard {
        padding: var(--space-3) var(--space-2);
    }

    .dashboard-header {
        flex-direction: column;
        align-items: stretch;
        gap: var(--space-3);
        padding-bottom: var(--space-3);
    }

    .dashboard-title {
        text-align: center;
    }

    .dashboard-title h1 {
        font-size: var(--text-xl);
    }

    .dashboard-title p {
        font-size: var(--text-xs);
    }

    .header-actions {
        justify-content: center;
        flex-wrap: wrap;
        gap: var(--space-2);
    }

    .header-actions .tb-btn {
        flex: 1;
        min-width: 120px;
        justify-content: center;
    }

    /* Tab Navigation - Horizontal Scroll */
    .tab-navigation {
        margin-left: calc(var(--space-2) * -1);
        margin-right: calc(var(--space-2) * -1);
        padding-left: var(--space-2);
        padding-right: var(--space-2);
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: none;
    }

    .tab-navigation::-webkit-scrollbar {
        display: none;
    }

    .tab-btn {
        padding: var(--space-2) var(--space-3);
        flex-shrink: 0;
        min-width: auto;
    }

    .tab-btn span:not(.material-symbols-outlined) {
        display: none;
    }

    .tab-btn .material-symbols-outlined {
        margin-right: 0;
    }

    /* Section Headers */
    .section-header {
        flex-direction: column;
        align-items: stretch;
        gap: var(--space-3);
    }

    .section-header h2 {
        font-size: var(--text-lg);
        text-align: center;
    }

    .section-header .tb-input {
        width: 100%;
    }

    /* Stats Grid */
    .stats-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: var(--space-2);
    }

    .stat-card {
        padding: var(--space-3);
    }

    .stat-value {
        font-size: var(--text-xl);
    }

    .stat-label {
        font-size: var(--text-xs);
    }

    /* Dashboard Cards */
    .dashboard-card {
        padding: var(--space-3);
        margin-bottom: var(--space-3);
    }

    .dashboard-card h3 {
        font-size: var(--text-base);
    }

    /* Tables - Responsive */
    .table-container {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        margin: 0 calc(var(--space-3) * -1);
        padding: 0 var(--space-3);
    }

    .admin-table {
        font-size: var(--text-xs);
        min-width: 500px;
    }

    .admin-table th,
    .admin-table td {
        padding: var(--space-2);
        white-space: nowrap;
    }

    .admin-table th:first-child,
    .admin-table td:first-child {
        position: sticky;
        left: 0;
        background: var(--bg-surface);
        z-index: 1;
    }

    /* Action Buttons */
    .action-group {
        flex-direction: column;
        gap: var(--space-2);
    }

    .action-btn {
        width: 100%;
        justify-content: center;
        padding: var(--space-2) var(--space-3);
    }

    /* Settings Items */
    .setting-item {
        flex-direction: column;
        align-items: stretch;
        gap: var(--space-3);
        padding: var(--space-3);
    }

    .setting-info {
        text-align: left;
    }

    /* Quick Actions */
    .quick-actions {
        flex-direction: column;
        gap: var(--space-2);
    }

    .quick-actions .tb-btn {
        width: 100%;
        justify-content: center;
    }

    /* Forms */
    .form-row {
        flex-direction: column;
        gap: var(--space-3);
    }

    .form-row > * {
        min-width: 100%;
    }

    /* Level Badges */
    .level-badge {
        font-size: var(--text-xs);
        padding: 2px 6px;
    }

    /* Module Cards */
    .module-card {
        padding: var(--space-3);
    }

    .module-header {
        flex-direction: column;
        align-items: flex-start;
        gap: var(--space-2);
    }

    .module-actions {
        width: 100%;
        flex-wrap: wrap;
    }

    .module-actions .tb-btn {
        flex: 1;
        min-width: calc(50% - var(--space-1));
    }

    /* SPP Cards */
    .spp-card {
        padding: var(--space-3);
    }

    .spp-header {
        flex-direction: column;
        align-items: flex-start;
        gap: var(--space-2);
    }

    .spp-actions {
        width: 100%;
    }

    .spp-actions .tb-btn {
        width: 100%;
        justify-content: center;
    }

    /* Empty State */
    .empty-state {
        padding: var(--space-6) var(--space-3);
    }

    .empty-state .material-symbols-outlined {
        font-size: 48px;
    }
}

/* ========== Responsive - Small Mobile ========== */
@media screen and (max-width: 480px) {
    .dashboard {
        padding: var(--space-2);
    }

    .dashboard-title h1 {
        font-size: var(--text-lg);
    }

    .stats-grid {
        grid-template-columns: 1fr 1fr;
        gap: var(--space-2);
    }

    .stat-card {
        padding: var(--space-2);
    }

    .stat-value {
        font-size: var(--text-lg);
    }

    .tab-btn {
        padding: var(--space-2);
    }

    .admin-table {
        min-width: 400px;
    }

    .header-actions .tb-btn {
        font-size: var(--text-xs);
        padding: var(--space-2) var(--space-3);
    }

    .header-actions .tb-btn span:not(.material-symbols-outlined) {
        display: none;
    }
}

/* ========== Utility Classes ========== */
.text-muted { color: var(--text-muted); }
.text-success { color: var(--color-success); }
.text-error { color: var(--color-error); }
.text-warning { color: var(--color-warning); }
.text-sm { font-size: var(--text-sm); }
.text-xs { font-size: var(--text-xs); }
.font-semibold { font-weight: var(--weight-semibold); }
.mt-2 { margin-top: var(--space-2); }
.mt-4 { margin-top: var(--space-4); }
.mt-6 { margin-top: var(--space-6); }
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
                <h1>
                    <span class="material-symbols-outlined">shield_person</span>
                    <span id="admin-title-text">Admin Panel</span>
                </h1>
                <span class="admin-badge">Admin</span>
            </div>
            <div class="header-actions">
                <div id="darkModeToggleContainer"></div>
                <button id="logoutButton" class="tb-btn tb-btn-secondary">
                    <span class="material-symbols-outlined">logout</span>
                    <span>Logout</span>
                </button>
            </div>
        </header>

        <!-- Tab Navigation -->
        <nav class="tab-navigation" id="tab-navigation" role="tablist">
            <button class="tab-btn active" data-section="system-status" role="tab" aria-selected="true">
                <span class="material-symbols-outlined">monitoring</span>
                <span>System</span>
            </button>
            <button class="tab-btn" data-section="user-management" role="tab" aria-selected="false">
                <span class="material-symbols-outlined">group</span>
                <span>Users</span>
            </button>
            <button class="tab-btn" data-section="module-management" role="tab" aria-selected="false">
                <span class="material-symbols-outlined">extension</span>
                <span>Modules</span>
            </button>
            <button class="tab-btn" data-section="spp-management" role="tab" aria-selected="false">
                <span class="material-symbols-outlined">web_stories</span>
                <span>SPPs</span>
            </button>
        </nav>

        <!-- Content Sections -->
        <div id="admin-content">
            <!-- System Status -->
            <section id="system-status-section" class="content-section active">
                <div class="section-header">
                    <h2><span class="material-symbols-outlined">bar_chart_4_bars</span>System Status</h2>
                </div>
                <div id="system-status-content">
                    <div class="loading-state">
                        <div class="spinner"></div>
                        <span>Loading system status...</span>
                    </div>
                </div>
            </section>

            <!-- User Management -->
            <section id="user-management-section" class="content-section">
                <div class="section-header">
                    <h2><span class="material-symbols-outlined">manage_history</span>User Management</h2>
                </div>
                <div id="user-management-content-main">
                    <div class="loading-state">
                        <div class="spinner"></div>
                        <span>Loading users...</span>
                    </div>
                </div>

                <div class="dashboard-card mt-6">
                    <h3><span class="material-symbols-outlined">person_add</span>Waiting List</h3>
                    <div id="user-waiting-list-content">
                        <p class="text-muted">Loading waiting list...</p>
                    </div>
                </div>
            </section>

            <!-- Module Management -->
            <section id="module-management-section" class="content-section">
                <div class="section-header">
                    <h2><span class="material-symbols-outlined">view_module</span>Module Management</h2>
                </div>
                <div id="module-management-content">
                    <div class="loading-state">
                        <div class="spinner"></div>
                        <span>Loading modules...</span>
                    </div>
                </div>
            </section>

            <!-- SPP Management -->
            <section id="spp-management-section" class="content-section">
                <div class="section-header">
                    <h2><span class="material-symbols-outlined">apps</span>Registered SPPs / UI Panels</h2>
                </div>
                <div id="spp-management-content">
                    <div class="loading-state">
                        <div class="spinner"></div>
                        <span>Loading SPPs...</span>
                    </div>
                </div>
            </section>
        </div>

        <!-- Footer Link -->
        <div class="mt-6" style="text-align: center;">
            <a href="/api/CloudM.UserDashboard/main" class="tb-btn tb-btn-secondary">
                <span class="material-symbols-outlined">dashboard</span>
                User Dashboard
            </a>
        </div>
    </main>
</div>

<script type="module">
if (typeof TB === 'undefined' || !TB.ui || !TB.api || !TB.user || !TB.utils) {
    console.error('CRITICAL: TB (tbjs) not loaded.');
    document.body.innerHTML = '<div style="padding:40px; text-align:center; color:var(--color-error);">Critical Error: Frontend library failed to load.</div>';
} else {
    console.log('TB object found. Initializing Admin Dashboard v3...');

    var currentAdminUser = null;

    // ========== Initialization ==========
    async function initializeAdminDashboard() {
        console.log("Admin Dashboard Initializing...");
        TB.ui.DarkModeToggle.init();
        setupNavigation();
        setupLogout();
        setupEventDelegation();

        try {
            var userRes = await TB.api.request('CloudM.UserAccountManager', 'get_current_user_from_request_api_wrapper', null, 'GET');
            if (userRes.error === TB.ToolBoxError.none && userRes.get()) {
                currentAdminUser = userRes.get();
                updateHeader();
                await showSection('system-status');
            } else {
                console.error("Failed to load admin user:", userRes.info && userRes.info.help_text ? userRes.info.help_text : 'Unknown error');
                showAccessDenied();
            }
        } catch (e) {
            console.error("Error fetching admin user:", e);
            showConnectionError();
        }
    }

    function updateHeader() {
        if (currentAdminUser && currentAdminUser.name) {
            var titleEl = document.getElementById('admin-title-text');
            if (titleEl) {
                titleEl.textContent = 'Admin (' + currentAdminUser.name + ')';
            }
        }
    }

    function showAccessDenied() {
        document.getElementById('admin-content').innerHTML = '<div class="empty-state">' +
            '<span class="material-symbols-outlined">block</span>' +
            '<h3 style="margin-top:var(--space-4);">Access Denied</h3>' +
            '<p class="text-muted">Could not verify admin privileges. Please login.</p>' +
        '</div>';
    }

    function showConnectionError() {
        document.getElementById('admin-content').innerHTML = '<div class="empty-state">' +
            '<span class="material-symbols-outlined">cloud_off</span>' +
            '<h3 style="margin-top:var(--space-4);">Connection Error</h3>' +
            '<p class="text-muted">Could not connect to server.</p>' +
        '</div>';
    }

    // ========== Navigation ==========
    function setupNavigation() {
        document.querySelectorAll('#tab-navigation .tab-btn').forEach(function(btn) {
            btn.addEventListener('click', async function() {
                document.querySelectorAll('#tab-navigation .tab-btn').forEach(function(b) {
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
        var logoutBtn = document.getElementById('logoutButton');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', async function() {
                TB.ui.Loader.show("Logging out...");
                await TB.user.logout();
                window.location.href = '/';
            });
        }
    }

    // ========== Event Delegation ==========
    function setupEventDelegation() {
        document.getElementById('admin-content').addEventListener('click', async function(event) {
            var target = event.target.closest('button.action-btn');
            if (!target) return;

            // System Status Restart
            if (target.classList.contains('btn-restart') && target.dataset.service) {
                TB.ui.Toast.showInfo('Restart for ' + target.dataset.service + ' (placeholder).');
            }
            // User Edit
            else if (target.dataset.uid && target.classList.contains('btn-edit')) {
                var tableEl = target.closest('.table-wrapper');
                var usersData = [];
                if (tableEl) {
                    var tbl = tableEl.querySelector('table');
                    if (tbl && tbl.dataset.users) {
                        try { usersData = JSON.parse(tbl.dataset.users); } catch(e) {}
                    }
                }
                showUserEditModal(target.dataset.uid, usersData);
            }
            // User Delete
            else if (target.dataset.uid && target.classList.contains('btn-delete')) {
                handleDeleteUser(target.dataset.uid, target.dataset.name);
            }
            // Waiting List Send Invite
            else if (target.dataset.email && target.classList.contains('btn-send-invite')) {
                var email = target.dataset.email;
                var proposedUsername = prompt('Enter username for ' + email + ':', email.split('@')[0]);
                if (!proposedUsername) { TB.ui.Toast.showWarning("Username required."); return; }

                TB.ui.Loader.show('Sending invite...');
                try {
                    var res = await TB.api.request('CloudM.AdminDashboard', 'send_invite_to_waiting_list_user_admin',
                        { email: email, username: proposedUsername }, 'POST');
                    TB.ui.Loader.hide();
                    if (res.error === TB.ToolBoxError.none) {
                        TB.ui.Toast.showSuccess(res.info.help_text || 'Invite sent.');
                        await loadWaitingListUsers();
                    } else {
                        TB.ui.Toast.showError('Failed: ' + TB.utils.escapeHtml(res.info.help_text));
                    }
                } catch (e) {
                    TB.ui.Loader.hide();
                    TB.ui.Toast.showError("Network error.");
                }
            }
            // Waiting List Remove
            else if (target.dataset.email && target.classList.contains('btn-delete')) {
                var emailToRemove = target.dataset.email;
                if (!confirm('Remove ' + emailToRemove + ' from waiting list?')) return;

                TB.ui.Loader.show('Removing...');
                try {
                    var res = await TB.api.request('CloudM.AdminDashboard', 'remove_from_waiting_list_admin', { email: emailToRemove }, 'POST');
                    TB.ui.Loader.hide();
                    if (res.error === TB.ToolBoxError.none) {
                        TB.ui.Toast.showSuccess(emailToRemove + ' removed.');
                        await loadWaitingListUsers();
                    } else {
                        TB.ui.Toast.showError('Failed: ' + TB.utils.escapeHtml(res.info.help_text));
                    }
                } catch (e) {
                    TB.ui.Loader.hide();
                    TB.ui.Toast.showError("Network error.");
                }
            }
            // Module Reload
            else if (target.dataset.module && target.classList.contains('btn-restart')) {
                var modName = target.dataset.module;
                TB.ui.Loader.show('Reloading ' + modName + '...');
                try {
                    var res = await TB.api.request('CloudM.AdminDashboard', 'reload_module_admin', { module_name: modName }, 'POST');
                    TB.ui.Loader.hide();
                    if (res.error === TB.ToolBoxError.none) {
                        TB.ui.Toast.showSuccess(modName + ': ' + TB.utils.escapeHtml(res.get() || 'OK'));
                    } else {
                        TB.ui.Toast.showError('Error: ' + TB.utils.escapeHtml(res.info.help_text));
                    }
                } catch (e) {
                    TB.ui.Loader.hide();
                    TB.ui.Toast.showError('Network error.');
                }
            }
            // SPP Open Link
            else if (target.dataset.path && target.classList.contains('btn-open-link')) {
                var path = target.dataset.path;
                if (path.indexOf("http") === 0 || path.indexOf("/api/") === 0) {
                    window.open(path, '_blank');
                } else {
                    TB.router.navigateTo(path);
                }
            }
        });
    }

    // ========== Section Loading ==========
    async function showSection(sectionId) {
        console.log('Showing section: ' + sectionId);
        document.querySelectorAll('.content-section').forEach(function(s) { s.classList.remove('active'); });
        var section = document.getElementById(sectionId + '-section');

        if (section) {
            section.classList.add('active');

            switch(sectionId) {
                case 'system-status':
                    await loadSystemStatus();
                    break;
                case 'user-management':
                    await loadUserManagement();
                    await loadWaitingListUsers();
                    break;
                case 'module-management':
                    await loadModuleManagement();
                    break;
                case 'spp-management':
                    await loadSppManagement();
                    break;
            }
        }
    }

    // ========== System Status ==========
    async function loadSystemStatus() {
        var content = document.getElementById('system-status-content');
        if (!content) return;

        try {
            var res = await TB.api.request('CloudM.AdminDashboard', 'get_system_status', null, 'GET');
            if (res.error === TB.ToolBoxError.none) {
                renderSystemStatus(res.get(), content);
            } else {
                content.innerHTML = '<p class="text-error">Error: ' + TB.utils.escapeHtml(res.info.help_text) + '</p>';
            }
        } catch (e) {
            content.innerHTML = '<p class="text-error">Network error.</p>';
            console.error(e);
        }
    }

    function renderSystemStatus(data, content) {
        if (!data || Object.keys(data).length === 0) {
            content.innerHTML = '<div class="empty-state">' +
                '<span class="material-symbols-outlined">dns</span>' +
                '<p>No services found or status unavailable.</p>' +
            '</div>';
            return;
        }

        // Count services by status
        var serviceKeys = Object.keys(data);
        var totalServices = serviceKeys.length;
        var runningServices = 0;
        var stoppedServices = 0;

        serviceKeys.forEach(function(name) {
            var info = data[name];
            if (info.status_indicator === '🟢') {
                runningServices++;
            } else if (info.status_indicator === '🔴') {
                stoppedServices++;
            }
        });

        // Stats Grid
        var html = '<div class="stats-grid">' +
            '<div class="stat-card">' +
                '<div class="stat-value">' + totalServices + '</div>' +
                '<div class="stat-label">Total Services</div>' +
            '</div>' +
            '<div class="stat-card">' +
                '<div class="stat-value" style="color:var(--color-success);">' + runningServices + '</div>' +
                '<div class="stat-label">Running</div>' +
            '</div>' +
            '<div class="stat-card">' +
                '<div class="stat-value" style="color:var(--color-error);">' + stoppedServices + '</div>' +
                '<div class="stat-label">Stopped</div>' +
            '</div>' +
            '<div class="stat-card">' +
                '<div class="stat-value" style="color:var(--color-warning);">' + (totalServices - runningServices - stoppedServices) + '</div>' +
                '<div class="stat-label">Other</div>' +
            '</div>' +
        '</div>';

        // Services Table
        html += '<div class="dashboard-card">' +
            '<h3><span class="material-symbols-outlined">dns</span>Running Services</h3>' +
            '<div class="table-wrapper">' +
                '<table class="admin-table">' +
                    '<thead><tr>' +
                        '<th>Service</th>' +
                        '<th>Status</th>' +
                        '<th>PID</th>' +
                        '<th>Actions</th>' +
                    '</tr></thead>' +
                    '<tbody>';

        serviceKeys.forEach(function(name) {
            var info = data[name];
            var statusClass = info.status_indicator === '🟢' ? 'green' :
                             (info.status_indicator === '🔴' ? 'red' : 'yellow');
            html += '<tr>' +
                '<td><strong>' + TB.utils.escapeHtml(name) + '</strong></td>' +
                '<td>' +
                    '<span class="status-indicator">' +
                        '<span class="status-dot ' + statusClass + '"></span>' +
                        info.status_indicator +
                    '</span>' +
                '</td>' +
                '<td class="text-muted">' + TB.utils.escapeHtml(info.pid || 'N/A') + '</td>' +
                '<td>' +
                    '<button class="action-btn btn-restart" data-service="' + TB.utils.escapeHtml(name) + '">' +
                        '<span class="material-symbols-outlined">restart_alt</span>' +
                        'Restart' +
                    '</button>' +
                '</td>' +
            '</tr>';
        });

        html += '</tbody></table></div></div>';
        content.innerHTML = html;
    }

    // ========== User Management ==========
    async function loadUserManagement() {
        var content = document.getElementById('user-management-content-main');
        if (!content) return;

        try {
            var res = await TB.api.request('CloudM.AdminDashboard', 'list_users_admin', null, 'GET');
            if (res.error === TB.ToolBoxError.none) {
                renderUserManagement(res.get(), content);
            } else {
                content.innerHTML = '<p class="text-error">Error: ' + TB.utils.escapeHtml(res.info.help_text) + '</p>';
            }
        } catch (e) {
            content.innerHTML = '<p class="text-error">Network error.</p>';
            console.error(e);
        }
    }

    // Level Mapping: -1 = Admin, 0 = Nicht eingeloggt, 1 = Eingeloggt, 2 = Spezial Nutzer
    function getLevelInfo(level) {
        switch(level) {
            case -1: return { name: 'Admin', badge: 'admin', color: 'var(--color-warning)' };
            case 0: return { name: 'Gast', badge: 'guest', color: 'var(--text-muted)' };
            case 1: return { name: 'User', badge: 'user', color: 'var(--color-success)' };
            case 2: return { name: 'Spezial', badge: 'special', color: 'var(--color-info)' };
            default: return { name: 'Level ' + level, badge: '', color: 'var(--text-secondary)' };
        }
    }

    function renderUserManagement(users, content) {
        if (!users || users.length === 0) {
            content.innerHTML = '<p class="text-muted">No users found. Users will appear here after they sign in via Clerk.</p>';
            return;
        }

        var usersJson = TB.utils.escapeHtml(JSON.stringify(users));
        var html = '<div class="dashboard-card">' +
            '<h3><span class="material-symbols-outlined">group</span>All Users (' + users.length + ')</h3>' +
            '<input type="text" id="user-search-admin" class="tb-input mb-4" placeholder="Search users..." oninput="filterAdminUsers(this.value)">' +
            '<div class="table-wrapper">' +
            '<table class="admin-table" data-users="' + usersJson + '">' +
            '<thead><tr>' +
            '<th>Name</th>' +
            '<th>Email</th>' +
            '<th>Level</th>' +
            '<th>Source</th>' +
            '<th>UID</th>' +
            '<th>Actions</th>' +
            '</tr></thead><tbody>';

        users.forEach(function(user) {
            var levelInfo = getLevelInfo(user.level);
            var canDelete = currentAdminUser && currentAdminUser.uid !== user.uid;
            var sourceLabel = user.source === 'clerk' ? 'Clerk DB' : (user.source === 'clerk_api' ? 'Clerk API' : 'Legacy');
            var sourceColor = user.source === 'clerk' || user.source === 'clerk_api' ? 'var(--color-info)' : 'var(--text-muted)';

            html += '<tr class="user-row-admin" data-name="' + (user.name || '').toLowerCase() + '" data-email="' + (user.email || '').toLowerCase() + '">' +
                '<td><strong>' + TB.utils.escapeHtml(user.name || 'N/A') + '</strong></td>' +
                '<td class="text-muted">' + TB.utils.escapeHtml(user.email || 'N/A') + '</td>' +
                '<td>' +
                    '<span class="level-badge ' + levelInfo.badge + '" style="background:oklch(from ' + levelInfo.color + ' l c h / 0.2); color:' + levelInfo.color + ';">' +
                        levelInfo.name +
                    '</span>' +
                '</td>' +
                '<td><span class="text-xs" style="color:' + sourceColor + ';">' + sourceLabel + '</span></td>' +
                '<td class="text-xs text-muted">' + TB.utils.escapeHtml(user.uid || 'N/A') + '</td>' +
                '<td>' +
                    '<div class="action-group">' +
                        '<button class="action-btn btn-edit" data-uid="' + user.uid + '">' +
                            '<span class="material-symbols-outlined">edit</span>' +
                            'Edit' +
                        '</button>' +
                        (canDelete ? '<button class="action-btn btn-delete" data-uid="' + user.uid + '" data-name="' + TB.utils.escapeHtml(user.name || 'N/A') + '">' +
                            '<span class="material-symbols-outlined">delete</span>' +
                            'Delete' +
                        '</button>' : '') +
                    '</div>' +
                '</td>' +
            '</tr>';
        });

        html += '</tbody></table></div></div>';
        content.innerHTML = html;
    }

    window.filterAdminUsers = function(query) {
        var q = query.toLowerCase();
        document.querySelectorAll('.user-row-admin').forEach(function(row) {
            var name = row.dataset.name || '';
            var email = row.dataset.email || '';
            row.style.display = (name.includes(q) || email.includes(q)) ? '' : 'none';
        });
    };

    function showUserEditModal(userId, allUsers) {
        var user = allUsers.find(function(u) { return u.uid === userId; });
        if (!user) { TB.ui.Toast.showError("User not found."); return; }

        var formContent = '<form id="editUserFormAdmin">' +
            '<input type="hidden" name="uid" value="' + user.uid + '">' +
            '<div class="form-group">' +
                '<label class="tb-label">Name</label>' +
                '<input type="text" name="name" class="tb-input" value="' + TB.utils.escapeHtml(user.name) + '" readonly style="opacity:0.6;">' +
            '</div>' +
            '<div class="form-group">' +
                '<label class="tb-label">Email</label>' +
                '<input type="email" name="email" class="tb-input" value="' + TB.utils.escapeHtml(user.email || '') + '">' +
            '</div>' +
            '<div class="form-group">' +
                '<label class="tb-label">Level</label>' +
                '<select name="level" class="tb-input">' +
                    '<option value="-1"' + (user.level === -1 ? ' selected' : '') + '>-1: Admin</option>' +
                    '<option value="0"' + (user.level === 0 ? ' selected' : '') + '>0: Gast (nicht eingeloggt)</option>' +
                    '<option value="1"' + (user.level === 1 ? ' selected' : '') + '>1: User (eingeloggt)</option>' +
                    '<option value="2"' + (user.level === 2 ? ' selected' : '') + '>2: Spezial Nutzer</option>' +
                '</select>' +
                '<p class="text-xs text-muted mt-2">-1=Admin, 0=Gast, 1=User, 2=Spezial</p>' +
            '</div>' +
            '<div class="form-group">' +
                '<label class="flex items-center" style="cursor:pointer;">' +
                    '<input type="checkbox" name="experimental_features" class="tb-checkbox"' + (user.settings && user.settings.experimental_features ? ' checked' : '') + '>' +
                    '<span>Experimental Features</span>' +
                '</label>' +
            '</div>' +
        '</form>';

        TB.ui.Modal.show({
            title: 'Edit User: ' + TB.utils.escapeHtml(user.name),
            content: formContent,
            buttons: [
                { text: 'Cancel', action: function(m) { m.close(); }, variant: 'secondary' },
                {
                    text: 'Save Changes',
                    variant: 'primary',
                    action: async function(m) {
                        var form = document.getElementById('editUserFormAdmin');
                        if (!form) return;

                        var updatedData = {
                            uid: form.uid.value,
                            name: form.name.value,
                            email: form.email.value,
                            level: parseInt(form.level.value),
                            settings: { experimental_features: form.experimental_features.checked }
                        };

                        TB.ui.Loader.show('Saving...');
                        try {
                            var res = await TB.api.request('CloudM.AdminDashboard', 'update_user_admin', updatedData, 'POST');
                            TB.ui.Loader.hide();
                            if (res.error === TB.ToolBoxError.none) {
                                TB.ui.Toast.showSuccess('User updated!');
                                await loadUserManagement();
                                m.close();
                            } else {
                                TB.ui.Toast.showError('Error: ' + TB.utils.escapeHtml(res.info.help_text));
                            }
                        } catch (e) {
                            TB.ui.Loader.hide();
                            TB.ui.Toast.showError('Network error.');
                        }
                    }
                }
            ]
        });
    }

    async function handleDeleteUser(userId, userName) {
        if (currentAdminUser && currentAdminUser.uid === userId) {
            TB.ui.Toast.showError("Cannot delete your own account.");
            return;
        }

        var modalContent = '<p>Delete user <strong>' + TB.utils.escapeHtml(userName) + '</strong>?<br><span class="text-sm text-muted">This action cannot be undone.</span></p>';

        TB.ui.Modal.show({
            title: 'Confirm Deletion',
            content: modalContent,
            buttons: [
                { text: 'Cancel', action: function(m) { m.close(); }, variant: 'secondary' },
                {
                    text: 'Delete User',
                    variant: 'danger',
                    action: async function(m) {
                        TB.ui.Loader.show('Deleting...');
                        try {
                            var res = await TB.api.request('CloudM.AdminDashboard', 'delete_user_admin', { uid: userId }, 'POST');
                            TB.ui.Loader.hide();
                            if (res.error === TB.ToolBoxError.none) {
                                TB.ui.Toast.showSuccess('User deleted!');
                                await loadUserManagement();
                            } else {
                                TB.ui.Toast.showError('Error: ' + TB.utils.escapeHtml(res.info.help_text));
                            }
                        } catch (e) {
                            TB.ui.Loader.hide();
                            TB.ui.Toast.showError('Network error.');
                        }
                        m.close();
                    }
                }
            ]
        });
    }

    // ========== Waiting List ==========
    async function loadWaitingListUsers() {
        var content = document.getElementById('user-waiting-list-content');
        if (!content) return;

        try {
            var res = await TB.api.request('CloudM.AdminDashboard', 'get_waiting_list_users_admin', null, 'GET');
            if (res.error === TB.ToolBoxError.none) {
                renderWaitingList(res.get(), content);
            } else {
                content.innerHTML = '<p class="text-error">Error: ' + TB.utils.escapeHtml(res.info.help_text) + '</p>';
            }
        } catch (e) {
            content.innerHTML = '<p class="text-error">Network error.</p>';
            console.error(e);
        }
    }

    function renderWaitingList(waitingUsers, content) {
        if (!waitingUsers || waitingUsers.length === 0) {
            content.innerHTML = '<p class="text-muted">No users on waiting list.</p>';
            return;
        }

        var html = '<div class="table-wrapper">' +
            '<table class="admin-table">' +
            '<thead><tr><th>Email</th><th>Actions</th></tr></thead>' +
            '<tbody>';

        waitingUsers.forEach(function(entry) {
            var email = typeof entry === 'string' ? entry : (entry.email || 'Invalid');
            var escapedEmail = TB.utils.escapeHtml(email);
            html += '<tr>' +
                '<td>' + escapedEmail + '</td>' +
                '<td>' +
                    '<div class="action-group">' +
                        '<button class="action-btn btn-send-invite" data-email="' + escapedEmail + '">' +
                            '<span class="material-symbols-outlined">outgoing_mail</span>' +
                            'Invite' +
                        '</button>' +
                        '<button class="action-btn btn-delete" data-email="' + escapedEmail + '">' +
                            '<span class="material-symbols-outlined">person_remove</span>' +
                            'Remove' +
                        '</button>' +
                    '</div>' +
                '</td>' +
            '</tr>';
        });

        html += '</tbody></table></div>';
        content.innerHTML = html;
    }

    // ========== Module Management ==========
    async function loadModuleManagement() {
        var content = document.getElementById('module-management-content');
        if (!content) return;

        try {
            var res = await TB.api.request('CloudM.AdminDashboard', 'list_modules_admin', null, 'GET');
            if (res.error === TB.ToolBoxError.none) {
                renderModuleManagement(res.get(), content);
            } else {
                content.innerHTML = '<p class="text-error">Error: ' + TB.utils.escapeHtml(res.info.help_text) + '</p>';
            }
        } catch (e) {
            content.innerHTML = '<p class="text-error">Network error.</p>';
            console.error(e);
        }
    }

    function renderModuleManagement(modules, content) {
        if (!modules || modules.length === 0) {
            content.innerHTML = '<div class="empty-state">' +
                '<span class="material-symbols-outlined">extension_off</span>' +
                '<p>No modules loaded.</p>' +
            '</div>';
            return;
        }

        // Group modules by prefix
        var groups = {};
        modules.forEach(function(mod) {
            var prefix = mod.split('.')[0] || 'Other';
            if (!groups[prefix]) groups[prefix] = [];
            groups[prefix].push(mod);
        });

        var html = '<div class="dashboard-card">' +
            '<h3><span class="material-symbols-outlined">inventory_2</span>Loaded Modules (' + modules.length + ')</h3>' +
            '<input type="text" id="module-search-admin" class="tb-input mb-4" placeholder="Search modules..." oninput="filterAdminModules(this.value)">';

        Object.keys(groups).forEach(function(group) {
            var mods = groups[group];
            var isOpen = group === 'CloudM' ? ' open' : '';
            html += '<details class="mb-4"' + isOpen + '>' +
                '<summary style="cursor:pointer; font-weight:var(--weight-semibold); padding:var(--space-3) 0; color:var(--text-primary);">' +
                    '<span class="material-symbols-outlined" style="vertical-align:middle; margin-right:var(--space-2);">folder</span>' +
                    TB.utils.escapeHtml(group) + ' (' + mods.length + ')' +
                '</summary>' +
                '<div class="table-wrapper" style="margin-top:var(--space-2);">' +
                    '<table class="admin-table">' +
                        '<thead><tr><th>Module Name</th><th>Actions</th></tr></thead>' +
                        '<tbody>';

            mods.forEach(function(modName) {
                html += '<tr class="module-row-admin" data-name="' + modName.toLowerCase() + '">' +
                    '<td>' + TB.utils.escapeHtml(modName) + '</td>' +
                    '<td>' +
                        '<button class="action-btn btn-restart" data-module="' + TB.utils.escapeHtml(modName) + '">' +
                            '<span class="material-symbols-outlined">refresh</span>' +
                            'Reload' +
                        '</button>' +
                    '</td>' +
                '</tr>';
            });

            html += '</tbody></table></div></details>';
        });

        html += '</div>';
        content.innerHTML = html;
    }

    window.filterAdminModules = function(query) {
        var q = query.toLowerCase();
        document.querySelectorAll('.module-row-admin').forEach(function(row) {
            row.style.display = row.dataset.name.includes(q) ? '' : 'none';
        });
    };

    // ========== SPP Management ==========
    async function loadSppManagement() {
        var content = document.getElementById('spp-management-content');
        if (!content) return;

        try {
            var res = await TB.api.request('CloudM.AdminDashboard', 'list_spps_admin', null, 'GET');
            if (res.error === TB.ToolBoxError.none) {
                renderSppManagement(res.get(), content);
            } else {
                content.innerHTML = '<p class="text-error">Error: ' + TB.utils.escapeHtml(res.info.help_text) + '</p>';
            }
        } catch (e) {
            content.innerHTML = '<p class="text-error">Network error.</p>';
            console.error(e);
        }
    }

    function renderSppManagement(spps, content) {
        if (!spps || spps.length === 0) {
            content.innerHTML = '<div class="empty-state">' +
                '<span class="material-symbols-outlined">web_asset_off</span>' +
                '<p>No SPPs registered.</p>' +
            '</div>';
            return;
        }

        var html = '<div class="dashboard-card">' +
            '<h3><span class="material-symbols-outlined">web</span>UI Panels (' + spps.length + ')</h3>' +
            '<div class="table-wrapper">' +
                '<table class="admin-table">' +
                    '<thead><tr>' +
                        '<th>Title</th>' +
                        '<th>Name</th>' +
                        '<th>Path</th>' +
                        '<th>Auth</th>' +
                        '<th>Actions</th>' +
                    '</tr></thead>' +
                    '<tbody>';

        spps.forEach(function(spp) {
            var authLabel = spp.auth ? '<span class="text-success">Yes</span>' : '<span class="text-muted">No</span>';
            html += '<tr>' +
                '<td><strong>' + TB.utils.escapeHtml(spp.title) + '</strong></td>' +
                '<td class="text-muted text-sm">' + TB.utils.escapeHtml(spp.name || spp.title) + '</td>' +
                '<td class="text-xs text-muted">' + TB.utils.escapeHtml(spp.path) + '</td>' +
                '<td>' + authLabel + '</td>' +
                '<td>' +
                    '<button class="action-btn btn-open-link" data-path="' + TB.utils.escapeHtml(spp.path) + '">' +
                        '<span class="material-symbols-outlined">open_in_new</span>' +
                        'Open' +
                    '</button>' +
                '</td>' +
            '</tr>';
        });

        html += '</tbody></table></div></div>';
        content.innerHTML = html;
    }

    // Make showSection global
    window.showSection = showSection;

    // ========== Toggle Switch Style (reuse from UserDashboard) ==========
    var style = document.createElement('style');
    style.textContent = '.toggle-switch { position: relative; width: 48px; height: 26px; flex-shrink: 0; } ' +
        '.toggle-switch input { opacity: 0; width: 0; height: 0; } ' +
        '.toggle-slider { position: absolute; cursor: pointer; inset: 0; background-color: var(--border-default); ' +
            'transition: var(--duration-fast) var(--ease-default); border-radius: var(--radius-full); } ' +
        '.toggle-slider::before { position: absolute; content: ""; height: 20px; width: 20px; left: 3px; bottom: 3px; ' +
            'background-color: white; transition: var(--duration-fast) var(--ease-default); border-radius: var(--radius-full); box-shadow: var(--shadow-xs); } ' +
        'input:checked + .toggle-slider { background: linear-gradient(135deg, var(--color-primary-400), var(--color-primary-600)); } ' +
        'input:checked + .toggle-slider::before { transform: translateX(22px); }';
    document.head.appendChild(style);

    // ========== Start ==========
    if (window.TB?.events && window.TB.config?.get('appRootId')) {
        initializeAdminDashboard();
    } else {
        document.addEventListener('tbjs:initialized', initializeAdminDashboard, { once: true });
    }
}
</script>
"""
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
    if not admin_user:
        return Result.default_user_error(info="Permission denied", exec_code=403)

    users_data = []
    seen_ids = set()

    # 1. Clerk-Benutzer aus Datenbank laden (CLERK_USER::*)
    try:
        clerk_users_result = await app.a_run_any(TBEF.DB.GET, query="CLERK_USER::*", get_results=True)
        if not clerk_users_result.is_error():
            clerk_users_raw = clerk_users_result.get()
            if clerk_users_raw:
                if not isinstance(clerk_users_raw, list):
                    clerk_users_raw = [clerk_users_raw]
                for user_bytes in clerk_users_raw:
                    try:
                        user_str = user_bytes.decode() if isinstance(user_bytes, bytes) else str(user_bytes)
                        user_dict = json.loads(user_str) if user_str.startswith('{') else eval(user_str)
                        uid = user_dict.get("clerk_user_id", user_dict.get("uid", "N/A"))
                        if uid not in seen_ids:
                            seen_ids.add(uid)
                            users_data.append({
                                "uid": uid,
                                "name": user_dict.get("username", user_dict.get("name", "N/A")),
                                "email": user_dict.get("email"),
                                "level": user_dict.get("level", 1),
                                "is_persona": user_dict.get("is_persona", False),
                                "settings": user_dict.get("settings", {}),
                                "source": "clerk"
                            })
                    except Exception as e:
                        app.print("Error parsing Clerk user: " + str(e), "WARNING")
    except Exception as e:
        app.print("Error fetching Clerk users: " + str(e), "WARNING")

    # 2. Legacy-Benutzer aus Datenbank laden (USER::*)
    try:
        legacy_users_result = await app.a_run_any(TBEF.DB.GET, query="USER::*", get_results=True)
        if not legacy_users_result.is_error():
            legacy_users_raw = legacy_users_result.get()
            if legacy_users_raw:
                if not isinstance(legacy_users_raw, list):
                    legacy_users_raw = [legacy_users_raw]
                for user_bytes in legacy_users_raw:
                    try:
                        user_str = user_bytes.decode() if isinstance(user_bytes, bytes) else str(user_bytes)
                        user_dict = json.loads(user_str) if user_str.startswith('{') else eval(user_str)
                        uid = user_dict.get("uid", "N/A")
                        if uid not in seen_ids:
                            seen_ids.add(uid)
                            users_data.append({
                                "uid": uid,
                                "name": user_dict.get("name", "N/A"),
                                "email": user_dict.get("email"),
                                "level": user_dict.get("level", 1),
                                "is_persona": user_dict.get("is_persona", False),
                                "settings": user_dict.get("settings", {}),
                                "source": "legacy"
                            })
                    except Exception as e:
                        app.print("Error parsing legacy user: " + str(e), "WARNING")
    except Exception as e:
        app.print("Error fetching legacy users: " + str(e), "WARNING")

    # 3. Versuche auch Clerk API direkt (falls verfügbar)
    try:
        from .AuthClerk import list_users as clerk_list_users
        clerk_api_result = clerk_list_users(app)
        if not clerk_api_result.is_error():
            for user in clerk_api_result.get() or []:
                uid = user.get("id", "N/A")
                if uid not in seen_ids:
                    seen_ids.add(uid)
                    users_data.append({
                        "uid": uid,
                        "name": user.get("username", "N/A"),
                        "email": user.get("email"),
                        "level": 1,  # Default level für Clerk-API Benutzer
                        "is_persona": False,
                        "settings": {},
                        "source": "clerk_api"
                    })
    except Exception as e:
        app.print("Clerk API not available: " + str(e), "DEBUG")

    return Result.json(data=users_data)


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def list_modules_admin(app: App, request: RequestData):
    admin_user = await _is_admin(app, request)
    if not admin_user: return Result.default_user_error(info="Permission denied", exec_code=403)
    modules = list_all_modules(app).get('modules')
    return Result.json(data=modules)


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def update_user_admin(app: App, request: RequestData, data: dict=None, **kwargs):
    admin_user = await _is_admin(app, request)
    if not admin_user:
        return Result.default_user_error(info="Permission denied", exec_code=403)
    if data is None:
        data = kwargs
    uid_to_update = data.get("uid")
    name_to_update = data.get("name")
    if not uid_to_update:
        return Result.default_user_error(info="User UID is required.")

    # Versuche zuerst Clerk-Benutzer zu aktualisieren
    try:
        clerk_user_result = await app.a_run_any(TBEF.DB.GET, query="CLERK_USER::" + str(uid_to_update), get_results=True)
        if not clerk_user_result.is_error() and clerk_user_result.get():
            user_bytes = clerk_user_result.get()
            if isinstance(user_bytes, list):
                user_bytes = user_bytes[0]
            user_str = user_bytes.decode() if isinstance(user_bytes, bytes) else str(user_bytes)
            user_dict = json.loads(user_str) if user_str.startswith('{') else eval(user_str)

            # Update fields
            if "email" in data:
                user_dict["email"] = data["email"]
            if "level" in data:
                try:
                    user_dict["level"] = int(data["level"])
                except ValueError:
                    return Result.default_user_error(info="Invalid level format.")
            if "settings" in data and isinstance(data["settings"], dict):
                if "settings" not in user_dict or user_dict["settings"] is None:
                    user_dict["settings"] = {}
                user_dict["settings"].update(data["settings"])

            # Save Clerk user
            save_result = await app.a_run_any(
                TBEF.DB.SET,
                query="CLERK_USER::" + str(uid_to_update),
                data=user_dict,
                get_results=True
            )
            if save_result.is_error():
                return Result.default_internal_error(info="Failed to save Clerk user: " + str(save_result.info))
            return Result.ok(info="User updated successfully.")
    except Exception as e:
        app.print("Error updating Clerk user, trying legacy: " + str(e), "DEBUG")

    # Fallback: Legacy-Benutzer aktualisieren
    if not name_to_update:
        return Result.default_user_error(info="User Name is required for legacy users.")

    user_res = await app.a_run_any(TBEF.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=name_to_update,
                                   uid=uid_to_update, get_results=True)
    if user_res.is_error() or not user_res.get():
        return Result.default_user_error(info="User " + str(name_to_update) + " (UID: " + str(uid_to_update) + ") not found.")

    user_to_update = user_res.get()
    if "email" in data:
        user_to_update.email = data["email"]
    if "level" in data:
        try:
            user_to_update.level = int(data["level"])
        except ValueError:
            return Result.default_user_error(info="Invalid level format.")
    if "settings" in data and isinstance(data["settings"], dict):
        if user_to_update.settings is None:
            user_to_update.settings = {}
        user_to_update.settings.update(data["settings"])

    save_result = db_helper_save_user(app, asdict(user_to_update))
    if save_result.is_error():
        return Result.default_internal_error(info="Failed to save user: " + str(save_result.info))
    return Result.ok(info="User updated successfully.")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def delete_user_admin(app: App, request: RequestData, data: dict=None, uid=None):
    admin_user = await _is_admin(app, request)
    if not admin_user:
        return Result.default_user_error(info="Permission denied", exec_code=403)

    uid_to_delete = uid or data.get("uid")
    if not uid_to_delete:
        return Result.default_user_error(info="User UID is required.")

    # Check if admin is trying to delete themselves
    admin_uid = getattr(admin_user, 'clerk_user_id', None) or getattr(admin_user, 'uid', None)
    if admin_uid == uid_to_delete:
        return Result.default_user_error(info="Admin cannot delete themselves.")

    deleted = False

    # 1. Versuche Clerk-Benutzer zu löschen
    try:
        clerk_delete_result = await app.a_run_any(
            TBEF.DB.DELETE,
            query="CLERK_USER::" + str(uid_to_delete),
            get_results=True
        )
        if not clerk_delete_result.is_error():
            deleted = True
            app.print("Deleted Clerk user: " + str(uid_to_delete), "INFO")
    except Exception as e:
        app.print("Error deleting Clerk user: " + str(e), "DEBUG")

    # 2. Versuche Legacy-Benutzer zu löschen
    try:
        user_to_delete_res = await app.a_run_any(
            TBEF.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME,
            username='*',
            uid=uid_to_delete,
            get_results=True
        )
        username_to_delete = None
        if not user_to_delete_res.is_error() and user_to_delete_res.get():
            username_to_delete = user_to_delete_res.get().name
        else:
            # Try to find by UID pattern
            all_users_raw_res = await app.a_run_any(
                TBEF.DB.GET,
                query="USER::*::" + str(uid_to_delete),
                get_results=True
            )
            if not all_users_raw_res.is_error() and all_users_raw_res.get():
                try:
                    user_bytes = all_users_raw_res.get()
                    if isinstance(user_bytes, list):
                        user_bytes = user_bytes[0]
                    user_str = user_bytes.decode() if isinstance(user_bytes, bytes) else str(user_bytes)
                    user_dict_raw = json.loads(user_str) if user_str.startswith('{') else eval(user_str)
                    username_to_delete = user_dict_raw.get("name")
                except Exception:
                    pass

        if username_to_delete:
            delete_result = db_helper_delete_user(app, username_to_delete, uid_to_delete, matching=True)
            if not delete_result.is_error():
                deleted = True
                app.print("Deleted legacy user: " + str(username_to_delete), "INFO")
    except Exception as e:
        app.print("Error deleting legacy user: " + str(e), "DEBUG")

    if deleted:
        return Result.ok(info="User deleted successfully.")
    else:
        return Result.default_user_error(info="User with UID " + str(uid_to_delete) + " not found.")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def reload_module_admin(app: App, request: RequestData, data: dict=None, module_name=None):
    admin_user = await _is_admin(app, request)
    if not admin_user: return Result.redirect("/web/core0/index.html")
    module_name = module_name or data.get("module_name")
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
async def remove_from_waiting_list_admin(app: App, request: RequestData, data: dict=None, email=None):
    admin_user = await _is_admin(app, request)
    if not admin_user: return Result.redirect("/web/core0/index.html")
    email_to_remove = email or data.get("email")
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
async def send_invite_to_waiting_list_user_admin(app: App, request: RequestData, data: dict, email=None, username=None):
    admin_user = await _is_admin(app, request)
    if not admin_user: return Result.redirect("/web/core0/index.html")
    email_to_invite = email or data.get("email")
    proposed_username = username or data.get("username")
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
