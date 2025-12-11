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

/* Override main-content constraints for dashboard */
.content-wrapper:has(.dashboard) {
    padding: var(--space-4);
    padding-block-start: var(--space-8);
}

/* Fallback for browsers without :has() support */
.dashboard.main-content {
    max-width: 1200px;
    width: 100%;
    margin: 0 auto;
    padding: var(--space-6) var(--space-5);
    overflow: visible;
    box-sizing: border-box;
}

.dashboard {
    max-width: 1200px;
    margin: 0 auto;
    padding: var(--space-6) var(--space-5);
    width: 100%;
    box-sizing: border-box;
}

/* Ensure content is not clipped */
#dashboard-content {
    overflow: visible;
}

.content-section {
    overflow: visible;
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

/* ========== Responsive - Tablet ========== */
@media screen and (max-width: 1024px) {
    .content-wrapper:has(.dashboard) {
        padding: var(--space-3);
        padding-block-start: var(--space-6);
    }

    .dashboard.main-content {
        max-width: 100%;
        padding: var(--space-5) var(--space-4);
    }
}

/* ========== Responsive - Mobile ========== */
@media screen and (max-width: 767px) {
    .content-wrapper:has(.dashboard) {
        padding: var(--space-2);
        padding-block-start: var(--space-4);
    }

    .dashboard.main-content,
    .dashboard {
        padding: var(--space-4) var(--space-3);
        border-radius: var(--radius-md);
        max-width: 100%;
    }

    .dashboard-header {
        flex-direction: column;
        align-items: stretch;
        gap: var(--space-3);
    }

    .dashboard-title {
        flex-direction: column;
        text-align: center;
        gap: var(--space-2);
    }

    .dashboard-title h1 {
        font-size: var(--text-xl);
    }

    .user-avatar {
        width: 40px;
        height: 40px;
        font-size: var(--text-base);
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
        gap: var(--space-1);
    }

    .tab-btn {
        padding: var(--space-2);
        min-width: 44px;
        justify-content: center;
    }

    .tab-btn span:not(.material-symbols-outlined) {
        display: none;
    }

    .tab-btn .material-symbols-outlined {
        font-size: 18px;
    }

    .stats-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: var(--space-2);
    }

    .stat-card {
        padding: var(--space-3);
    }

    .stat-value {
        font-size: var(--text-2xl);
    }

    .dashboard-card {
        padding: var(--space-4);
        margin-bottom: var(--space-3);
    }

    .dashboard-card h3 {
        font-size: var(--text-base);
    }

    .module-grid {
        grid-template-columns: 1fr;
        gap: var(--space-2);
    }

    .setting-item {
        flex-direction: column;
        align-items: flex-start;
        gap: var(--space-3);
        padding: var(--space-3);
    }

    .toggle-switch {
        align-self: flex-end;
    }

    .quick-actions {
        flex-direction: column;
        gap: var(--space-2);
    }

    .quick-actions .tb-btn {
        width: 100%;
    }

    .theme-grid {
        grid-template-columns: repeat(3, 1fr);
        gap: var(--space-2);
    }

    .theme-option {
        padding: var(--space-3);
    }

    .theme-option .material-symbols-outlined {
        font-size: 24px;
    }

    .info-table td {
        padding: var(--space-2);
        font-size: var(--text-sm);
    }

    .info-table td:first-child {
        width: 35%;
    }

    /* Hide logout text on mobile */
    .logout-text {
        display: none;
    }
}

/* ========== Color Settings ========== */
.color-settings {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
}

.color-control {
    display: flex;
    align-items: center;
    gap: var(--space-3);
}

.color-value {
    min-width: 50px;
    text-align: right;
    font-family: monospace;
    font-size: var(--text-sm);
    color: var(--text-secondary);
}

@media screen and (max-width: 767px) {
    .color-control {
        flex-direction: column;
        align-items: flex-end;
        gap: var(--space-2);
    }

    .color-control input[type="range"] {
        width: 100px !important;
    }

    .color-value {
        min-width: auto;
    }
}

/* ========== File Tree ========== */
.file-tree {
    font-size: var(--text-sm);
    user-select: none;
}

.file-tree-item {
    display: flex;
    align-items: center;
    padding: var(--space-2) var(--space-3);
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: background var(--duration-fast) var(--ease-default);
    gap: var(--space-2);
}

.file-tree-item:hover {
    background: var(--interactive-muted);
}

.file-tree-item.selected {
    background: oklch(from var(--interactive) l c h / 0.15);
}

.file-tree-item .material-symbols-outlined {
    font-size: 18px;
    color: var(--text-muted);
    flex-shrink: 0;
}

.file-tree-item.folder .material-symbols-outlined {
    color: var(--color-warning);
}

.file-tree-item.file .material-symbols-outlined {
    color: var(--interactive);
}

.file-tree-name {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.file-tree-size {
    font-size: var(--text-xs);
    color: var(--text-muted);
    flex-shrink: 0;
}

.file-tree-children {
    margin-left: var(--space-5);
    border-left: 1px solid var(--border-subtle);
    padding-left: var(--space-2);
}

.file-tree-children.collapsed {
    display: none;
}

/* File Actions */
.file-actions {
    display: flex;
    gap: var(--space-2);
    opacity: 0;
    transition: opacity var(--duration-fast) var(--ease-default);
}

.file-tree-item:hover .file-actions {
    opacity: 1;
}

.file-action-btn {
    padding: var(--space-1);
    background: transparent;
    border: none;
    border-radius: var(--radius-sm);
    cursor: pointer;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    justify-content: center;
}

.file-action-btn:hover {
    background: var(--bg-elevated);
    color: var(--text-primary);
}

.file-action-btn .material-symbols-outlined {
    font-size: 16px;
}

/* Upload Zone */
.upload-zone {
    border: 2px dashed var(--border-default);
    border-radius: var(--radius-md);
    padding: var(--space-6);
    text-align: center;
    transition: all var(--duration-fast) var(--ease-default);
    cursor: pointer;
}

.upload-zone:hover,
.upload-zone.dragover {
    border-color: var(--interactive);
    background: oklch(from var(--interactive) l c h / 0.05);
}

.upload-zone .material-symbols-outlined {
    font-size: 48px;
    color: var(--text-muted);
    margin-bottom: var(--space-3);
}

.upload-zone.dragover .material-symbols-outlined {
    color: var(--interactive);
}

/* Data Tabs */
.data-tabs {
    display: flex;
    gap: var(--space-1);
    margin-bottom: var(--space-4);
    border-bottom: var(--border-width) solid var(--border-subtle);
    padding-bottom: var(--space-2);
}

.data-tab {
    padding: var(--space-2) var(--space-4);
    background: transparent;
    border: none;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0;
    cursor: pointer;
    font-size: var(--text-sm);
    color: var(--text-secondary);
    font-family: inherit;
    transition: all var(--duration-fast) var(--ease-default);
}

.data-tab:hover {
    color: var(--text-primary);
    background: var(--interactive-muted);
}

.data-tab.active {
    color: var(--interactive);
    border-bottom: 2px solid var(--interactive);
    margin-bottom: -2px;
}

.data-panel {
    display: none;
}

.data-panel.active {
    display: block;
}

/* Config Display */
.config-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: var(--space-3);
}

.config-item {
    background: var(--bg-elevated);
    border: var(--border-width) solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: var(--space-3);
}

.config-key {
    font-size: var(--text-xs);
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: var(--space-1);
}

.config-value {
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    color: var(--text-primary);
    word-break: break-all;
}

.config-value.boolean-true {
    color: var(--color-success);
}

.config-value.boolean-false {
    color: var(--color-error);
}

@media screen and (max-width: 767px) {
    .file-actions {
        opacity: 1;
    }

    .config-grid {
        grid-template-columns: 1fr;
    }

    .data-tabs {
        overflow-x: auto;
        scrollbar-width: none;
    }

    .data-tab {
        white-space: nowrap;
        flex-shrink: 0;
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

    var currentUser = null;
    var allModules = [];
    var userInstance = null;
    var modDataCache = {};

    // ========== Initialization ==========
    async function initDashboard() {
        console.log("Dashboard wird initialisiert...");
        TB.ui.DarkModeToggle.init();
        setupNavigation();
        setupLogout();

        try {
            var userRes = await TB.api.request('CloudM.UserAccountManager', 'get_current_user', null, 'GET');
            if (userRes.error === TB.ToolBoxError.none && userRes.get()) {
                currentUser = userRes.get();
                updateHeader();

                var modulesRes = await TB.api.request('CloudM.UserDashboard', 'get_all_available_modules', null, 'GET');
                if (modulesRes.error === TB.ToolBoxError.none) {
                    allModules = modulesRes.get() || [];
                }

                var instanceRes = await TB.api.request('CloudM.UserDashboard', 'get_my_active_instances', null, 'GET');
                if (instanceRes.error === TB.ToolBoxError.none && instanceRes.get() && instanceRes.get().length > 0) {
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
        var avatarEl = document.getElementById('user-avatar');
        var welcomeEl = document.getElementById('welcome-text');
        var emailEl = document.getElementById('user-email');

        if (currentUser) {
            var name = currentUser.username || currentUser.name || 'Benutzer';
            var initial = name.charAt(0).toUpperCase();

            if (avatarEl) avatarEl.textContent = initial;
            if (welcomeEl) welcomeEl.textContent = 'Hallo, ' + name + '!';
            if (emailEl) emailEl.textContent = currentUser.email || '';
        }
    }

    function showNotAuthenticated() {
        document.getElementById('dashboard-content').innerHTML = '<div class="empty-state">' +
            '<span class="material-symbols-outlined">login</span>' +
            '<h3 style="margin-top:var(--space-4);">Nicht angemeldet</h3>' +
            '<p class="text-muted">Bitte melden Sie sich an, um fortzufahren.</p>' +
            '<button onclick="TB.router.navigateTo(\\'/web/assets/login.html\\')" class="tb-btn tb-btn-primary mt-4">' +
                '<span class="material-symbols-outlined">login</span>' +
                'Anmelden' +
            '</button>' +
        '</div>';
    }

    function showConnectionError() {
        document.getElementById('dashboard-content').innerHTML = '<div class="empty-state">' +
            '<span class="material-symbols-outlined">cloud_off</span>' +
            '<h3 style="margin-top:var(--space-4);">Verbindungsfehler</h3>' +
            '<p class="text-muted">Die Verbindung zum Server konnte nicht hergestellt werden.</p>' +
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
        document.getElementById('logoutButtonUser').addEventListener('click', async function() {
            TB.ui.Loader.show("Abmelden...");
            await TB.user.logout();
            window.location.href = '/';
        });
    }

    // ========== Section Loading ==========
    async function showSection(sectionId) {
        document.querySelectorAll('.content-section').forEach(function(s) { s.classList.remove('active'); });
        var section = document.getElementById(sectionId + '-section');
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
        var content = document.getElementById('overview-content');
        var loadedModsCount = (userInstance && userInstance.live_modules) ? userInstance.live_modules.length : 0;
        var savedModsCount = (userInstance && userInstance.saved_modules) ? userInstance.saved_modules.length : 0;
        var cliSessions = (userInstance && userInstance.active_cli_sessions) ? userInstance.active_cli_sessions : 0;
        var userLevel = (currentUser && currentUser.level) ? currentUser.level : 1;
        var userName = (currentUser && (currentUser.username || currentUser.name)) ? (currentUser.username || currentUser.name) : '-';
        var userEmail = (currentUser && currentUser.email) ? currentUser.email : 'Nicht angegeben';

        var html = '<div class="stats-grid">' +
            '<div class="stat-card"><div class="stat-value">' + loadedModsCount + '</div><div class="stat-label">Aktive Module</div></div>' +
            '<div class="stat-card"><div class="stat-value">' + savedModsCount + '</div><div class="stat-label">Gespeichert</div></div>' +
            '<div class="stat-card"><div class="stat-value">' + cliSessions + '</div><div class="stat-label">CLI Sitzungen</div></div>' +
            '<div class="stat-card"><div class="stat-value">' + userLevel + '</div><div class="stat-label">Level</div></div>' +
        '</div>' +
        '<div class="dashboard-card">' +
            '<h3><span class="material-symbols-outlined">bolt</span>Schnellzugriff</h3>' +
            '<div class="quick-actions">' +
                '<button class="tb-btn tb-btn-primary" onclick="showSection(\\'my-modules\\')">' +
                    '<span class="material-symbols-outlined">extension</span>Module verwalten</button>' +
                '<button class="tb-btn tb-btn-secondary" onclick="showSection(\\'settings\\')">' +
                    '<span class="material-symbols-outlined">settings</span>Einstellungen</button>' +
                '<button class="tb-btn tb-btn-secondary" onclick="showSection(\\'appearance\\')">' +
                    '<span class="material-symbols-outlined">palette</span>Theme ändern</button>' +
            '</div>' +
        '</div>';

        if (userInstance && userInstance.live_modules && userInstance.live_modules.length > 0) {
            html += '<div class="dashboard-card">' +
                '<h3><span class="material-symbols-outlined">play_circle</span>Aktive Module</h3>' +
                '<div class="module-grid">';
            userInstance.live_modules.forEach(function(mod) {
                html += '<div class="module-card active"><div class="module-header">' +
                    '<span class="module-name">' + TB.utils.escapeHtml(mod.name) + '</span>' +
                    '<span class="module-status loaded">Aktiv</span></div></div>';
            });
            html += '</div></div>';
        }

        html += '<div class="dashboard-card">' +
            '<h3><span class="material-symbols-outlined">account_circle</span>Konto-Info</h3>' +
            '<table class="info-table">' +
                '<tr><td>Benutzername</td><td><strong>' + TB.utils.escapeHtml(userName) + '</strong></td></tr>' +
                '<tr><td>E-Mail</td><td>' + TB.utils.escapeHtml(userEmail) + '</td></tr>' +
                '<tr><td>Level</td><td>' + userLevel + '</td></tr>' +
            '</table></div>';

        content.innerHTML = html;
    }

    // ========== Module ==========
    async function loadModules() {
        var content = document.getElementById('my-modules-content');

        try {
            var instanceRes = await TB.api.request('CloudM.UserDashboard', 'get_my_active_instances', null, 'GET');
            var resData = instanceRes.get();
            if (instanceRes.error === TB.ToolBoxError.none && resData && resData.length > 0) {
                userInstance = resData[0];
            }
        } catch(e) {}

        var liveModNames = [];
        if (userInstance && userInstance.live_modules) {
            userInstance.live_modules.forEach(function(m) { liveModNames.push(m.name); });
        }
        var savedModNames = (userInstance && userInstance.saved_modules) ? userInstance.saved_modules : [];

        var categories = {};
        allModules.forEach(function(mod) {
            var category = mod.split('.')[0] || 'Andere';
            if (!categories[category]) categories[category] = [];
            categories[category].push(mod);
        });

        var html = '<div class="dashboard-card">' +
            '<h3><span class="material-symbols-outlined">info</span>Hinweis</h3>' +
            '<p class="text-sm text-muted" style="margin:0;">Aktivieren oder deaktivieren Sie Module nach Bedarf. Gespeicherte Module werden beim nächsten Login automatisch geladen.</p>' +
        '</div>';

        // Saved modules section
        html += '<div class="dashboard-card">' +
            '<h3><span class="material-symbols-outlined">bookmark</span>Gespeicherte Module (' + savedModNames.length + ')</h3>';

        if (savedModNames.length > 0) {
            html += '<div class="module-grid">';
            savedModNames.forEach(function(modName) {
                var isLive = liveModNames.indexOf(modName) !== -1;
                var escapedName = TB.utils.escapeHtml(modName);
                html += '<div class="module-card ' + (isLive ? 'active' : '') + '">' +
                    '<div class="module-header">' +
                        '<span class="module-name">' + escapedName + '</span>' +
                        '<span class="module-status ' + (isLive ? 'loaded' : 'available') + '">' + (isLive ? 'Aktiv' : 'Gespeichert') + '</span>' +
                    '</div>' +
                    '<div class="module-actions">';
                if (!isLive) {
                    html += '<button class="tb-btn tb-btn-success tb-btn-sm" onclick="loadModule(\\'' + escapedName + '\\')">' +
                        '<span class="material-symbols-outlined">play_arrow</span>Laden</button>';
                } else {
                    html += '<button class="tb-btn tb-btn-secondary tb-btn-sm" onclick="unloadModule(\\'' + escapedName + '\\')">' +
                        '<span class="material-symbols-outlined">stop</span>Entladen</button>';
                }
                html += '<button class="tb-btn tb-btn-danger tb-btn-sm" onclick="removeFromSaved(\\'' + escapedName + '\\')">' +
                    '<span class="material-symbols-outlined">delete</span></button>' +
                    '</div></div>';
            });
            html += '</div>';
        } else {
            html += '<p class="text-muted">Keine Module gespeichert.</p>';
        }
        html += '</div>';

        // Available modules section
        html += '<div class="dashboard-card">' +
            '<h3><span class="material-symbols-outlined">apps</span>Verfügbare Module (' + allModules.length + ')</h3>' +
            '<div class="mb-4"><input type="text" id="module-search" class="tb-input" placeholder="Module durchsuchen..." oninput="filterModules(this.value)"></div>' +
            '<div id="module-categories">';

        Object.keys(categories).forEach(function(cat) {
            var mods = categories[cat];
            var isOpen = cat === 'CloudM' ? ' open' : '';
            html += '<details class="mb-4"' + isOpen + '>' +
                '<summary style="cursor:pointer; font-weight:var(--weight-semibold); padding:var(--space-3) 0; color:var(--text-primary);">' +
                    '<span class="material-symbols-outlined" style="vertical-align:middle; margin-right:var(--space-2);">folder</span>' +
                    TB.utils.escapeHtml(cat) + ' (' + mods.length + ')' +
                '</summary>' +
                '<div class="module-grid" style="margin-top:var(--space-3);">';

            mods.forEach(function(modName) {
                var isLive = liveModNames.indexOf(modName) !== -1;
                var isSaved = savedModNames.indexOf(modName) !== -1;
                var escapedName = TB.utils.escapeHtml(modName);
                var statusHtml = isLive ? '<span class="module-status loaded">Aktiv</span>' :
                                 isSaved ? '<span class="module-status available">Gespeichert</span>' : '';

                html += '<div class="module-card module-item ' + (isLive ? 'active' : '') + '" data-name="' + modName.toLowerCase() + '">' +
                    '<div class="module-header">' +
                        '<span class="module-name">' + escapedName + '</span>' + statusHtml +
                    '</div>' +
                    '<div class="module-actions">';
                if (!isSaved) {
                    html += '<button class="tb-btn tb-btn-primary tb-btn-sm" onclick="addToSaved(\\'' + escapedName + '\\')">' +
                        '<span class="material-symbols-outlined">bookmark_add</span>Speichern</button>';
                }
                if (!isLive) {
                    html += '<button class="tb-btn tb-btn-success tb-btn-sm" onclick="loadModule(\\'' + escapedName + '\\')">' +
                        '<span class="material-symbols-outlined">play_arrow</span>Laden</button>';
                } else {
                    html += '<button class="tb-btn tb-btn-secondary tb-btn-sm" onclick="unloadModule(\\'' + escapedName + '\\')">' +
                        '<span class="material-symbols-outlined">stop</span>Entladen</button>';
                }
                html += '</div></div>';
            });

            html += '</div></details>';
        });

        html += '</div></div>';
        content.innerHTML = html;
    }

    window.filterModules = function(query) {
        var q = query.toLowerCase();
        document.querySelectorAll('.module-item').forEach(function(item) {
            var name = item.dataset.name;
            item.style.display = name.indexOf(q) !== -1 ? '' : 'none';
        });
    };

    window.loadModule = async function(modName) {
        TB.ui.Loader.show('Lade ' + modName + '...');
        try {
            var res = await TB.api.request('CloudM.UserDashboard', 'add_module_to_instance', {module_name: modName}, 'POST');
            TB.ui.Loader.hide();
            if (res.error === TB.ToolBoxError.none) {
                TB.ui.Toast.showSuccess(modName + ' wurde geladen');
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
        TB.ui.Loader.show('Entlade ' + modName + '...');
        try {
            var res = await TB.api.request('CloudM.UserDashboard', 'remove_module_from_instance', {module_name: modName}, 'POST');
            TB.ui.Loader.hide();
            if (res.error === TB.ToolBoxError.none) {
                TB.ui.Toast.showSuccess(modName + ' wurde entladen');
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
            var res = await TB.api.request('CloudM.UserDashboard', 'add_module_to_saved', {module_name: modName}, 'POST');
            TB.ui.Loader.hide();
            if (res.error === TB.ToolBoxError.none) {
                TB.ui.Toast.showSuccess(modName + ' gespeichert');
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
        if (!confirm('Möchten Sie "' + modName + '" wirklich aus den gespeicherten Modulen entfernen?')) return;
        TB.ui.Loader.show('Entferne...');
        try {
            var res = await TB.api.request('CloudM.UserDashboard', 'remove_module_from_saved', {module_name: modName}, 'POST');
            TB.ui.Loader.hide();
            if (res.error === TB.ToolBoxError.none) {
                TB.ui.Toast.showSuccess(modName + ' entfernt');
                await loadModules();
            } else {
                TB.ui.Toast.showError(res.info.help_text || 'Fehler');
            }
        } catch(e) {
            TB.ui.Loader.hide();
            TB.ui.Toast.showError('Netzwerkfehler');
        }
    };

    // ========== Daten-Tab ==========
    var userFilesCache = [];
    var currentDataTab = 'settings';

    async function loadModData() {
        var content = document.getElementById('mod-data-content');

        // Load mod data
        try {
            var res = await TB.api.request('CloudM.UserDashboard', 'get_all_mod_data', null, 'GET');
            if (res.error === TB.ToolBoxError.none) {
                modDataCache = res.get() || {};
            }
        } catch(e) {}

        // Load user files
        try {
            var filesRes = await TB.api.request('CloudM.UserDashboard', 'list_user_files', null, 'GET');
            if (filesRes.error === TB.ToolBoxError.none) {
                userFilesCache = filesRes.get() || [];
            }
        } catch(e) {
            userFilesCache = [];
        }

        var settings = (currentUser && currentUser.settings) ? currentUser.settings : {};
        var modNames = Object.keys(modDataCache);

        // Build HTML with string concatenation
        var html = '<div class="data-tabs">' +
            '<button class="data-tab ' + (currentDataTab === 'settings' ? 'active' : '') + '" onclick="switchDataTab(\\'settings\\')">' +
                '<span class="material-symbols-outlined" style="font-size:16px;vertical-align:middle;margin-right:4px;">settings</span>Einstellungen</button>' +
            '<button class="data-tab ' + (currentDataTab === 'files' ? 'active' : '') + '" onclick="switchDataTab(\\'files\\')">' +
                '<span class="material-symbols-outlined" style="font-size:16px;vertical-align:middle;margin-right:4px;">folder</span>Dateien</button>' +
            '<button class="data-tab ' + (currentDataTab === 'mods' ? 'active' : '') + '" onclick="switchDataTab(\\'mods\\')">' +
                '<span class="material-symbols-outlined" style="font-size:16px;vertical-align:middle;margin-right:4px;">extension</span>Mod-Daten</button>' +
        '</div>';

        // Settings Panel
        html += '<div id="data-panel-settings" class="data-panel ' + (currentDataTab === 'settings' ? 'active' : '') + '">' +
            '<div class="dashboard-card">' +
                '<h3><span class="material-symbols-outlined">tune</span>Gespeicherte Einstellungen</h3>' +
                '<p class="text-sm text-muted mb-4">Ihre persönlichen Einstellungen, die von Modulen gelesen werden können.</p>';

        var settingsKeys = Object.keys(settings);
        if (settingsKeys.length > 0) {
            html += '<div class="config-grid">';
            settingsKeys.forEach(function(key) {
                var value = settings[key];
                var valueClass = typeof value === 'boolean' ? (value ? 'boolean-true' : 'boolean-false') : '';
                var valueStr = typeof value === 'boolean' ? (value ? '✓ Aktiviert' : '✗ Deaktiviert') :
                              typeof value === 'object' ? JSON.stringify(value) : TB.utils.escapeHtml(String(value));
                html += '<div class="config-item">' +
                    '<div class="config-key">' + TB.utils.escapeHtml(key.replace(/_/g, ' ')) + '</div>' +
                    '<div class="config-value ' + valueClass + '">' + valueStr + '</div></div>';
            });
            html += '</div>';
        } else {
            html += '<div class="empty-state" style="padding:var(--space-6);"><span class="material-symbols-outlined">settings_suggest</span>' +
                '<p class="text-muted">Noch keine Einstellungen gespeichert.</p></div>';
        }
        html += '</div></div>';

        // Files Panel
        html += '<div id="data-panel-files" class="data-panel ' + (currentDataTab === 'files' ? 'active' : '') + '">' +
            '<div class="dashboard-card">' +
                '<h3><span class="material-symbols-outlined">cloud_upload</span>Datei hochladen</h3>' +
                '<div class="upload-zone" id="upload-zone" onclick="document.getElementById(\\'file-input\\').click()">' +
                    '<span class="material-symbols-outlined">upload_file</span>' +
                    '<p class="text-muted mb-2">Dateien hierher ziehen oder klicken</p>' +
                    '<p class="text-sm text-muted">Max. 10 MB pro Datei</p>' +
                '</div>' +
                '<input type="file" id="file-input" style="display:none;" multiple onchange="handleFileUpload(this.files)">' +
            '</div>' +
            '<div class="dashboard-card">' +
                '<h3><span class="material-symbols-outlined">folder_open</span>Meine Dateien</h3>' +
                '<div id="file-tree-container">' + renderFileTree(userFilesCache) + '</div>' +
            '</div></div>';

        // Mod Data Panel
        html += '<div id="data-panel-mods" class="data-panel ' + (currentDataTab === 'mods' ? 'active' : '') + '">' +
            '<div class="dashboard-card">' +
                '<h3><span class="material-symbols-outlined">info</span>Was sind Mod-Daten?</h3>' +
                '<p class="text-sm text-muted" style="margin:0;">Jedes Modul kann eigene Daten für Sie speichern. Hier können Sie diese einsehen und bearbeiten.</p>' +
            '</div>';

        if (modNames.length > 0) {
            modNames.forEach(function(modName) {
                var data = modDataCache[modName] || {};
                var entries = Object.entries(data);
                var escapedModName = TB.utils.escapeHtml(modName);

                html += '<div class="mod-data-panel">' +
                    '<div class="mod-data-header" onclick="this.nextElementSibling.classList.toggle(\\'open\\'); this.querySelector(\\'.expand-icon\\').textContent = this.nextElementSibling.classList.contains(\\'open\\') ? \\'expand_less\\' : \\'expand_more\\';">' +
                        '<span class="flex items-center gap-2"><span class="material-symbols-outlined">extension</span>' + escapedModName + '</span>' +
                        '<span class="material-symbols-outlined expand-icon">expand_more</span>' +
                    '</div>' +
                    '<div class="mod-data-content">';

                if (entries.length > 0) {
                    entries.forEach(function(entry) {
                        var key = entry[0];
                        var value = entry[1];
                        var valStr = typeof value === 'boolean'
                            ? '<span class="' + (value ? 'text-success' : 'text-error') + '">' + (value ? 'Ja' : 'Nein') + '</span>'
                            : TB.utils.escapeHtml(String(value).substring(0, 100));
                        html += '<div class="mod-data-item"><span class="mod-data-key">' + TB.utils.escapeHtml(key) + '</span><span class="mod-data-value">' + valStr + '</span></div>';
                    });
                } else {
                    html += '<p class="text-muted text-sm">Keine Daten gespeichert.</p>';
                }

                html += '<div class="mt-4 flex gap-2">' +
                    '<button class="tb-btn tb-btn-secondary tb-btn-sm" onclick="editModData(\\'' + escapedModName + '\\')">' +
                        '<span class="material-symbols-outlined">edit</span>Bearbeiten</button>' +
                    '<button class="tb-btn tb-btn-danger tb-btn-sm" onclick="clearModData(\\'' + escapedModName + '\\')">' +
                        '<span class="material-symbols-outlined">delete</span>Löschen</button>' +
                '</div></div></div>';
            });
        } else {
            html += '<div class="empty-state"><span class="material-symbols-outlined">folder_off</span>' +
                '<p>Noch keine Mod-Daten vorhanden.</p>' +
                '<p class="text-sm text-muted mt-2">Module speichern hier automatisch Ihre Einstellungen.</p></div>';
        }
        html += '</div>';

        content.innerHTML = html;

        // Setup drag & drop
        setupUploadZone();
    }

    function renderFileTree(files) {
        if (!files || files.length === 0) {
            return '<div class="empty-state" style="padding:var(--space-6);"><span class="material-symbols-outlined">folder_off</span><p class="text-muted">Keine Dateien vorhanden.</p></div>';
        }

        // Build tree structure from flat file list
        var tree = {};
        files.forEach(function(file) {
            var parts = file.path.split('/').filter(function(p) { return p; });
            var current = tree;
            parts.forEach(function(part, i) {
                if (!current[part]) {
                    current[part] = i === parts.length - 1 ? { _file: file } : {};
                }
                current = current[part];
            });
        });

        return '<div class="file-tree">' + renderTreeNode(tree, '') + '</div>';
    }

    function renderTreeNode(node, path) {
        var html = '';
        var entries = Object.entries(node).sort(function(a, b) {
            var aIsFile = a[1]._file;
            var bIsFile = b[1]._file;
            if (aIsFile && !bIsFile) return 1;
            if (!aIsFile && bIsFile) return -1;
            return a[0].localeCompare(b[0]);
        });

        for (var i = 0; i < entries.length; i++) {
            var name = entries[i][0];
            var value = entries[i][1];
            if (name === '_file') continue;

            var fullPath = path ? path + '/' + name : name;
            var isFile = value._file;

            if (isFile) {
                var file = value._file;
                var icon = getFileIcon(file.content_type || file.type || '');
                var size = formatFileSize(file.size || 0);
                var escapedPath = TB.utils.escapeHtml(file.path);
                var escapedName = TB.utils.escapeHtml(name);
                var quotedPath = "\\'" + escapedPath.replace(/'/g, "\\\\'") + "\\'";
                html += '<div class="file-tree-item file" data-path="' + escapedPath + '">' +
                    '<span class="material-symbols-outlined">' + icon + '</span>' +
                    '<span class="file-tree-name">' + escapedName + '</span>' +
                    '<span class="file-tree-size">' + size + '</span>' +
                    '<div class="file-actions">' +
                    '<button class="file-action-btn" onclick="event.stopPropagation(); previewFile(' + quotedPath + ')" title="Vorschau"><span class="material-symbols-outlined">visibility</span></button>' +
                    '<button class="file-action-btn" onclick="event.stopPropagation(); downloadFile(' + quotedPath + ')" title="Download"><span class="material-symbols-outlined">download</span></button>' +
                    '<button class="file-action-btn" onclick="event.stopPropagation(); deleteFile(' + quotedPath + ')" title="Löschen"><span class="material-symbols-outlined">delete</span></button>' +
                    '</div></div>';
            } else {
                var childCount = Object.keys(value).filter(function(k) { return k !== '_file'; }).length;
                var escapedName = TB.utils.escapeHtml(name);
                html += '<div class="file-tree-item folder" onclick="this.nextElementSibling.classList.toggle(\\'collapsed\\'); this.querySelector(\\'.folder-icon\\').textContent = this.nextElementSibling.classList.contains(\\'collapsed\\') ? \\'folder\\' : \\'folder_open\\';">' +
                    '<span class="material-symbols-outlined folder-icon">folder_open</span>' +
                    '<span class="file-tree-name">' + escapedName + '</span>' +
                    '<span class="file-tree-size">' + childCount + ' Elemente</span>' +
                    '</div><div class="file-tree-children">' + renderTreeNode(value, fullPath) + '</div>';
            }
        }
        return html;
    }

    function getFileIcon(contentType) {
        if (contentType.startsWith('image/')) return 'image';
        if (contentType.startsWith('video/')) return 'movie';
        if (contentType.startsWith('audio/')) return 'audio_file';
        if (contentType.includes('pdf')) return 'picture_as_pdf';
        if (contentType.includes('text') || contentType.includes('json')) return 'description';
        if (contentType.includes('zip') || contentType.includes('archive')) return 'folder_zip';
        return 'insert_drive_file';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    window.switchDataTab = function(tab) {
        currentDataTab = tab;
        document.querySelectorAll('.data-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.data-panel').forEach(p => p.classList.remove('active'));
        const tabIndex = tab === 'settings' ? 1 : tab === 'files' ? 2 : 3;
        document.querySelector('.data-tab:nth-child(' + tabIndex + ')').classList.add('active');
        document.getElementById('data-panel-' + tab).classList.add('active');
    };

    function setupUploadZone() {
        const zone = document.getElementById('upload-zone');
        if (!zone) return;

        zone.addEventListener('dragover', e => {
            e.preventDefault();
            zone.classList.add('dragover');
        });

        zone.addEventListener('dragleave', e => {
            e.preventDefault();
            zone.classList.remove('dragover');
        });

        zone.addEventListener('drop', e => {
            e.preventDefault();
            zone.classList.remove('dragover');
            handleFileUpload(e.dataTransfer.files);
        });
    }

    window.handleFileUpload = async function(files) {
        if (!files || files.length === 0) return;

        for (var i = 0; i < files.length; i++) {
            var file = files[i];
            if (file.size > 10 * 1024 * 1024) {
                TB.ui.Toast.showError(file.name + ' ist zu groß (max. 10 MB)');
                continue;
            }

            TB.ui.Loader.show('Lade ' + file.name + ' hoch...');

            try {
                // Read file as base64
                var base64 = await new Promise(function(resolve, reject) {
                    var reader = new FileReader();
                    reader.onload = function() { resolve(reader.result.split(',')[1]); };
                    reader.onerror = reject;
                    reader.readAsDataURL(file);
                });

                var res = await TB.api.request('CloudM.UserDashboard', 'upload_user_file', {
                    file: base64,
                    path: file.name,
                    content_type: file.type || 'application/octet-stream'
                }, 'POST');

                if (res.error === TB.ToolBoxError.none) {
                    TB.ui.Toast.showSuccess(file.name + ' hochgeladen');
                } else {
                    TB.ui.Toast.showError('Fehler: ' + ((res.info && res.info.help_text) ? res.info.help_text : 'Upload fehlgeschlagen'));
                }
            } catch(e) {
                console.error('Upload error:', e);
                TB.ui.Toast.showError('Fehler beim Hochladen von ' + file.name);
            }
        }

        TB.ui.Loader.hide();
        await loadModData();
    };

    window.downloadFile = async function(path) {
        TB.ui.Loader.show('Download wird vorbereitet...');
        try {
            var res = await TB.api.request('CloudM.UserDashboard', 'download_user_file', { path: path }, 'GET');
            TB.ui.Loader.hide();

            if (res.error === TB.ToolBoxError.none) {
                var data = res.get();
                // Convert base64 to binary
                var binaryString = atob(data.content);
                var bytes = new Uint8Array(binaryString.length);
                for (var i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                var blob = new Blob([bytes], { type: data.content_type || 'application/octet-stream' });
                var url = URL.createObjectURL(blob);
                var a = document.createElement('a');
                a.href = url;
                a.download = path.split('/').pop();
                a.click();
                URL.revokeObjectURL(url);
            } else {
                TB.ui.Toast.showError('Download fehlgeschlagen');
            }
        } catch(e) {
            TB.ui.Loader.hide();
            TB.ui.Toast.showError('Netzwerkfehler');
        }
    };

    window.previewFile = async function(path) {
        var file = userFilesCache.find(function(f) { return f.path === path; });
        if (!file) return;

        var contentType = file.content_type || file.type || '';
        var ext = path.split('.').pop().toLowerCase();

        // Fallback: Detect image by extension if content-type is generic
        var isImage = contentType.indexOf('image/') === 0 ||
                      (contentType === 'application/octet-stream' &&
                       ['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg', 'bmp', 'ico'].indexOf(ext) !== -1);

        // Correct content-type for images if needed
        if (isImage && contentType === 'application/octet-stream') {
            var typeMap = {
                'png': 'image/png',
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'gif': 'image/gif',
                'webp': 'image/webp',
                'svg': 'image/svg+xml',
                'bmp': 'image/bmp',
                'ico': 'image/x-icon'
            };
            contentType = typeMap[ext] || 'image/png';
        }

        if (isImage) {
            TB.ui.Loader.show('Lade Vorschau...');
            try {
                var res = await TB.api.request('CloudM.UserDashboard', 'download_user_file', { path: path }, 'GET');
                TB.ui.Loader.hide();
                if (res.error === TB.ToolBoxError.none) {
                    var data = res.get();
                    TB.ui.Modal.show({
                        title: path.split('/').pop(),
                        content: '<img src="data:' + contentType + ';base64,' + data.content + '" style="max-width:100%; max-height:70vh; border-radius:var(--radius-md);">',
                        buttons: [{ text: 'Schließen', action: function(m) { m.close(); } }]
                    });
                }
            } catch(e) {
                TB.ui.Loader.hide();
                TB.ui.Toast.showError('Vorschau fehlgeschlagen');
            }
        } else if (contentType.indexOf('text/') === 0 || contentType.indexOf('json') !== -1) {
            TB.ui.Loader.show('Lade Vorschau...');
            try {
                var res = await TB.api.request('CloudM.UserDashboard', 'download_user_file', { path: path }, 'GET');
                TB.ui.Loader.hide();
                if (res.error === TB.ToolBoxError.none) {
                    var data = res.get();
                    var text = atob(data.content);
                    TB.ui.Modal.show({
                        title: path.split('/').pop(),
                        content: '<pre style="max-height:60vh; overflow:auto; padding:var(--space-4); background:var(--bg-sunken); border-radius:var(--radius-md); font-size:var(--text-sm);">' + TB.utils.escapeHtml(text) + '</pre>',
                        buttons: [{ text: 'Schließen', action: function(m) { m.close(); } }]
                    });
                }
            } catch(e) {
                TB.ui.Loader.hide();
                TB.ui.Toast.showError('Vorschau fehlgeschlagen');
            }
        } else {
            TB.ui.Toast.showInfo('Vorschau für diesen Dateityp nicht verfügbar: ' + contentType);
        }
    };

    window.deleteFile = async function(path) {
        if (!confirm('Möchten Sie "' + path.split('/').pop() + '" wirklich löschen?')) return;

        TB.ui.Loader.show('Lösche...');
        try {
            var res = await TB.api.request('CloudM.UserDashboard', 'delete_user_file', { path: path }, 'POST');
            TB.ui.Loader.hide();
            if (res.error === TB.ToolBoxError.none) {
                TB.ui.Toast.showSuccess('Datei gelöscht');
                await loadModData();
            } else {
                TB.ui.Toast.showError('Löschen fehlgeschlagen');
            }
        } catch(e) {
            TB.ui.Loader.hide();
            TB.ui.Toast.showError('Netzwerkfehler');
        }
    };

    window.editModData = async function(modName) {
        var data = modDataCache[modName] || {};
        var json = JSON.stringify(data, null, 2);

        TB.ui.Modal.show({
            title: modName + ' - Daten bearbeiten',
            content: '<p class="text-sm text-muted mb-4">Vorsicht: Änderungen können die Funktionalität des Moduls beeinflussen.</p>' +
                '<textarea id="mod-data-editor" style="width:100%; height:200px; font-family:var(--font-mono); padding:var(--space-3); border:var(--border-width) solid var(--border-default); border-radius:var(--radius-md); background:var(--input-bg); color:var(--text-primary);">' + TB.utils.escapeHtml(json) + '</textarea>',
            buttons: [
                { text: 'Abbrechen', action: function(m) { m.close(); }, variant: 'secondary' },
                {
                    text: 'Speichern',
                    variant: 'primary',
                    action: async function(m) {
                        try {
                            var newData = JSON.parse(document.getElementById('mod-data-editor').value);
                            TB.ui.Loader.show('Speichere...');
                            var res = await TB.api.request('CloudM.UserAccountManager', 'update_mod_data', {mod_name: modName, data: newData}, 'POST');
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
        if (!confirm('Möchten Sie wirklich alle Daten von "' + modName + '" löschen?')) return;
        TB.ui.Loader.show('Lösche...');
        try {
            var res = await TB.api.request('CloudM.UserAccountManager', 'update_mod_data', {mod_name: modName, data: {}}, 'POST');
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
        var content = document.getElementById('settings-content');
        var settings = (currentUser && currentUser.settings) ? currentUser.settings : {};

        var html = '<div class="dashboard-card">' +
            '<h3><span class="material-symbols-outlined">tune</span>Allgemeine Einstellungen</h3>' +
            '<div class="settings-section">' +
                '<div class="setting-item"><div class="setting-info">' +
                    '<div class="setting-label">Experimentelle Funktionen</div>' +
                    '<div class="setting-description">Aktiviert neue Funktionen in der Testphase</div></div>' +
                    '<label class="toggle-switch"><input type="checkbox" ' + (settings.experimental_features ? 'checked' : '') +
                    ' onchange="updateSetting(\\'experimental_features\\', this.checked)"><span class="toggle-slider"></span></label></div>' +
                '<div class="setting-item"><div class="setting-info">' +
                    '<div class="setting-label">Benachrichtigungen</div>' +
                    '<div class="setting-description">Benachrichtigungen über wichtige Ereignisse</div></div>' +
                    '<label class="toggle-switch"><input type="checkbox" ' + (settings.notifications !== false ? 'checked' : '') +
                    ' onchange="updateSetting(\\'notifications\\', this.checked)"><span class="toggle-slider"></span></label></div>' +
                '<div class="setting-item"><div class="setting-info">' +
                    '<div class="setting-label">Auto-Laden von Modulen</div>' +
                    '<div class="setting-description">Gespeicherte Module beim Login automatisch laden</div></div>' +
                    '<label class="toggle-switch"><input type="checkbox" ' + (settings.auto_load_modules !== false ? 'checked' : '') +
                    ' onchange="updateSetting(\\'auto_load_modules\\', this.checked)"><span class="toggle-slider"></span></label></div>' +
                '<div class="setting-item"><div class="setting-info">' +
                    '<div class="setting-label">Detaillierte Protokolle</div>' +
                    '<div class="setting-description">Ausführliche Protokollierung für Fehlerbehebung</div></div>' +
                    '<label class="toggle-switch"><input type="checkbox" ' + (settings.verbose_logging ? 'checked' : '') +
                    ' onchange="updateSetting(\\'verbose_logging\\', this.checked)"><span class="toggle-slider"></span></label></div>' +
            '</div></div>';

        html += '<div class="dashboard-card">' +
            '<h3><span class="material-symbols-outlined">language</span>Sprache & Region</h3>' +
            '<div class="setting-item"><div class="setting-info">' +
                '<div class="setting-label">Sprache</div>' +
                '<div class="setting-description">Bevorzugte Sprache</div></div>' +
                '<select class="tb-input" style="width:auto; margin-bottom:0;" onchange="updateSetting(\\'language\\', this.value)">' +
                    '<option value="de" ' + (settings.language === 'de' || !settings.language ? 'selected' : '') + '>Deutsch</option>' +
                    '<option value="en" ' + (settings.language === 'en' ? 'selected' : '') + '>English</option>' +
                '</select></div></div>';

        html += '<div class="dashboard-card">' +
            '<h3><span class="material-symbols-outlined">security</span>Datenschutz</h3>' +
            '<div class="setting-item"><div class="setting-info">' +
                '<div class="setting-label">Nutzungsstatistiken</div>' +
                '<div class="setting-description">Anonyme Statistiken zur Verbesserung senden</div></div>' +
                '<label class="toggle-switch"><input type="checkbox" ' + (settings.analytics !== false ? 'checked' : '') +
                ' onchange="updateSetting(\\'analytics\\', this.checked)"><span class="toggle-slider"></span></label></div></div>';

        content.innerHTML = html;
    }

    window.updateSetting = async function(key, value) {
        try {
            var res = await TB.api.request('CloudM.UserAccountManager', 'update_setting', {
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
        var content = document.getElementById('appearance-content');
        var themePreference = (TB.ui.theme && TB.ui.theme.getPreference) ? TB.ui.theme.getPreference() : 'system';

        // Get current CSS variable values
        var rootStyles = getComputedStyle(document.documentElement);
        var currentHue = (currentUser && currentUser.settings && currentUser.settings.hue_primary) ? currentUser.settings.hue_primary : (parseInt(rootStyles.getPropertyValue('--hue-primary')) || 230);
        var currentChroma = (currentUser && currentUser.settings && currentUser.settings.chroma_primary) ? currentUser.settings.chroma_primary : (parseFloat(rootStyles.getPropertyValue('--chroma-primary')) || 0.18);
        var currentBgSun = (currentUser && currentUser.settings && currentUser.settings.theme_bg_sun) ? currentUser.settings.theme_bg_sun : '#ffffff';
        var currentBgLight = (currentUser && currentUser.settings && currentUser.settings.theme_bg_light) ? currentUser.settings.theme_bg_light : '#537FE7';

        var html = '<div class="dashboard-card">' +
            '<h3><span class="material-symbols-outlined">contrast</span>Farbschema</h3>' +
            '<p class="text-sm text-muted mb-4">Wählen Sie Ihr bevorzugtes Farbschema.</p>' +
            '<div class="theme-grid">' +
                '<button class="theme-option ' + (themePreference === 'light' ? 'active' : '') + '" onclick="setTheme(\\'light\\')">' +
                    '<span class="material-symbols-outlined">light_mode</span><span>Hell</span></button>' +
                '<button class="theme-option ' + (themePreference === 'dark' ? 'active' : '') + '" onclick="setTheme(\\'dark\\')">' +
                    '<span class="material-symbols-outlined">dark_mode</span><span>Dunkel</span></button>' +
                '<button class="theme-option ' + (themePreference === 'system' ? 'active' : '') + '" onclick="setTheme(\\'system\\')">' +
                    '<span class="material-symbols-outlined">computer</span><span>System</span></button>' +
            '</div></div>';

        html += '<div class="dashboard-card">' +
            '<h3><span class="material-symbols-outlined">palette</span>Primärfarbe</h3>' +
            '<p class="text-sm text-muted mb-4">Passen Sie die Hauptfarbe des Designs an.</p>' +
            '<div class="color-settings">' +
                '<div class="setting-item"><div class="setting-info">' +
                    '<div class="setting-label">Farbton (Hue)</div>' +
                    '<div class="setting-description">0° = Rot, 120° = Grün, 230° = Blau</div></div>' +
                    '<div class="color-control">' +
                        '<input type="range" id="hue-slider" min="0" max="360" value="' + currentHue + '" ' +
                            'style="width:120px; accent-color:oklch(65% 0.2 ' + currentHue + ');" oninput="updateHue(this.value)">' +
                        '<span id="hue-value" class="color-value">' + currentHue + '°</span></div></div>' +
                '<div class="setting-item"><div class="setting-info">' +
                    '<div class="setting-label">Sättigung (Chroma)</div>' +
                    '<div class="setting-description">0 = Grau, 0.18 = Normal, 0.3 = Kräftig</div></div>' +
                    '<div class="color-control">' +
                        '<input type="range" id="chroma-slider" min="0" max="30" value="' + Math.round(currentChroma * 100) + '" ' +
                            'style="width:120px; accent-color:var(--interactive);" oninput="updateChroma(this.value / 100)">' +
                        '<span id="chroma-value" class="color-value">' + currentChroma.toFixed(2) + '</span></div></div>' +
                '<div class="color-preview" id="color-preview" style="height:60px; border-radius:var(--radius-md); ' +
                    'background:linear-gradient(135deg, oklch(65% ' + currentChroma + ' ' + currentHue + '), oklch(50% ' + currentChroma + ' ' + currentHue + ')); ' +
                    'margin-top:var(--space-4); display:flex; align-items:center; justify-content:center; color:white; ' +
                    'font-weight:var(--weight-semibold); text-shadow:0 1px 2px rgba(0,0,0,0.3);">Vorschau</div>' +
            '</div></div>';

        html += '<div class="dashboard-card">' +
            '<h3><span class="material-symbols-outlined">wallpaper</span>Hintergrundfarben</h3>' +
            '<p class="text-sm text-muted mb-4">Passen Sie die Hintergrundfarben an.</p>' +
            '<div class="color-settings">' +
                '<div class="setting-item"><div class="setting-info">' +
                    '<div class="setting-label">Heller Hintergrund</div>' +
                    '<div class="setting-description">Haupthintergrund im hellen Modus</div></div>' +
                    '<input type="color" id="bg-sun-picker" value="' + currentBgSun + '" ' +
                        'style="width:50px; height:36px; border:none; cursor:pointer; border-radius:var(--radius-sm);" ' +
                        'onchange="updateBgColor(\\'theme_bg_sun\\', this.value)"></div>' +
                '<div class="setting-item"><div class="setting-info">' +
                    '<div class="setting-label">Akzent-Hintergrund</div>' +
                    '<div class="setting-description">Sekundärer Hintergrund / Akzent</div></div>' +
                    '<input type="color" id="bg-light-picker" value="' + currentBgLight + '" ' +
                        'style="width:50px; height:36px; border:none; cursor:pointer; border-radius:var(--radius-sm);" ' +
                        'onchange="updateBgColor(\\'theme_bg_light\\', this.value)"></div>' +
            '</div></div>';

        var fontScale = (currentUser && currentUser.settings && currentUser.settings.font_scale) ? currentUser.settings.font_scale : 100;
        html += '<div class="dashboard-card">' +
            '<h3><span class="material-symbols-outlined">format_size</span>Schriftgröße</h3>' +
            '<p class="text-sm text-muted mb-4">Passen Sie die Schriftgröße an.</p>' +
            '<div class="flex items-center gap-4">' +
                '<span class="text-sm">A</span>' +
                '<input type="range" min="80" max="120" value="' + fontScale + '" ' +
                    'style="flex:1; accent-color:var(--interactive);" ' +
                    'onchange="updateSetting(\\'font_scale\\', this.value); document.documentElement.style.fontSize = this.value + \\'%\\';">' +
                '<span style="font-size:1.25em;">A</span>' +
            '</div></div>';

        html += '<div class="dashboard-card">' +
            '<h3><span class="material-symbols-outlined">restart_alt</span>Zurücksetzen</h3>' +
            '<p class="text-sm text-muted mb-4">Alle Theme-Einstellungen auf Standard zurücksetzen.</p>' +
            '<button class="tb-btn tb-btn-secondary" onclick="resetThemeSettings()">' +
                '<span class="material-symbols-outlined">refresh</span>Auf Standard zurücksetzen</button></div>';

        content.innerHTML = html;
    }

    window.updateHue = function(value) {
        var hue = parseInt(value);
        document.documentElement.style.setProperty('--hue-primary', hue);
        document.getElementById('hue-value').textContent = hue + '°';

        // Update preview
        var chroma = parseFloat(document.getElementById('chroma-slider').value) / 100;
        document.getElementById('color-preview').style.background =
            'linear-gradient(135deg, oklch(65% ' + chroma + ' ' + hue + '), oklch(50% ' + chroma + ' ' + hue + '))';

        // Update slider accent color
        document.getElementById('hue-slider').style.accentColor = 'oklch(65% 0.2 ' + hue + ')';

        // Save setting
        updateSetting('hue_primary', hue);

        // Refresh Clerk theme if available
        if (TB.user && TB.user.refreshClerkTheme) TB.user.refreshClerkTheme();
    };

    window.updateChroma = function(value) {
        var chroma = parseFloat(value).toFixed(2);
        document.documentElement.style.setProperty('--chroma-primary', chroma);
        document.getElementById('chroma-value').textContent = chroma;

        // Update preview
        var hue = parseInt(document.getElementById('hue-slider').value);
        document.getElementById('color-preview').style.background =
            'linear-gradient(135deg, oklch(65% ' + chroma + ' ' + hue + '), oklch(50% ' + chroma + ' ' + hue + '))';

        // Save setting
        updateSetting('chroma_primary', chroma);

        // Refresh Clerk theme if available
        if (TB.user && TB.user.refreshClerkTheme) TB.user.refreshClerkTheme();
    };

    window.updateBgColor = function(key, value) {
        if (key === 'theme_bg_sun') {
            document.documentElement.style.setProperty('--theme-bg-sun', value);
        } else if (key === 'theme_bg_light') {
            document.documentElement.style.setProperty('--theme-bg-light', value);
        }
        updateSetting(key, value);
    };

    window.resetThemeSettings = async function() {
        // Reset to defaults
        var defaults = {
            hue_primary: 230,
            chroma_primary: 0.18,
            theme_bg_sun: '#ffffff',
            theme_bg_light: '#537FE7',
            font_scale: 100
        };

        // Apply defaults
        document.documentElement.style.setProperty('--hue-primary', defaults.hue_primary);
        document.documentElement.style.setProperty('--chroma-primary', defaults.chroma_primary);
        document.documentElement.style.setProperty('--theme-bg-sun', defaults.theme_bg_sun);
        document.documentElement.style.setProperty('--theme-bg-light', defaults.theme_bg_light);
        document.documentElement.style.fontSize = defaults.font_scale + '%';

        // Save all defaults
        var keys = Object.keys(defaults);
        for (var i = 0; i < keys.length; i++) {
            await updateSetting(keys[i], defaults[keys[i]]);
        }

        // Refresh Clerk theme if available
        if (TB.user && TB.user.refreshClerkTheme) TB.user.refreshClerkTheme();

        // Reload appearance section
        loadAppearance();
        TB.ui.Toast.showSuccess('Theme-Einstellungen zurückgesetzt');
    };

    window.setTheme = function(theme) {
        if (TB.ui.theme && TB.ui.theme.setPreference) {
            TB.ui.theme.setPreference(theme);
            var themeName = theme === 'system' ? 'System' : (theme === 'dark' ? 'Dunkel' : 'Hell');
            TB.ui.Toast.showSuccess('Theme: ' + themeName);

            // Refresh Clerk theme if available
            if (TB.user && TB.user.refreshClerkTheme) TB.user.refreshClerkTheme();

            loadAppearance();
        }
    };

    // ========== Profil ==========
    async function loadProfile() {
        var content = document.getElementById('profile-content');
        var userName = (currentUser && (currentUser.username || currentUser.name)) ? (currentUser.username || currentUser.name) : '-';
        var userEmail = (currentUser && currentUser.email) ? currentUser.email : 'Nicht angegeben';
        var userLevel = (currentUser && currentUser.level) ? currentUser.level : 1;
        var deviceType = navigator.userAgent.indexOf('Mobile') !== -1 ? 'Mobiles Gerät' : 'Desktop-Browser';

        var html = '<div class="dashboard-card">' +
            '<h3><span class="material-symbols-outlined">account_circle</span>Profil-Informationen</h3>' +
            '<div class="settings-section">' +
                '<div class="setting-item"><div class="setting-info">' +
                    '<div class="setting-label">Benutzername</div>' +
                    '<div class="setting-description">' + TB.utils.escapeHtml(userName) + '</div></div></div>' +
                '<div class="setting-item"><div class="setting-info">' +
                    '<div class="setting-label">E-Mail-Adresse</div>' +
                    '<div class="setting-description">' + TB.utils.escapeHtml(userEmail) + '</div></div>' +
                    '<button class="tb-btn tb-btn-secondary tb-btn-sm" onclick="openClerkProfile()">' +
                        '<span class="material-symbols-outlined">edit</span>Ändern</button></div>' +
                '<div class="setting-item"><div class="setting-info">' +
                    '<div class="setting-label">Benutzer-Level</div>' +
                    '<div class="setting-description">Level ' + userLevel + '</div></div></div>' +
            '</div></div>';

        html += '<div class="dashboard-card">' +
            '<h3><span class="material-symbols-outlined">key</span>Sicherheit</h3>' +
            '<div class="quick-actions">' +
                '<button class="tb-btn tb-btn-secondary" onclick="requestMagicLink()">' +
                    '<span class="material-symbols-outlined">link</span>Magic Link anfordern</button>' +
                '<button class="tb-btn tb-btn-secondary" onclick="openClerkProfile()">' +
                    '<span class="material-symbols-outlined">security</span>Sicherheitseinstellungen</button>' +
            '</div></div>';

        html += '<div class="dashboard-card">' +
            '<h3><span class="material-symbols-outlined">devices</span>Aktive Sitzungen</h3>' +
            '<p class="text-sm text-muted mb-4">Ihre aktuell angemeldeten Geräte.</p>' +
            '<div id="sessions-list">' +
                '<div class="setting-item"><div class="setting-info">' +
                    '<div class="setting-label">Diese Sitzung</div>' +
                    '<div class="setting-description">' + deviceType + '</div></div>' +
                    '<span class="module-status loaded">Aktiv</span></div>';

        if (userInstance && userInstance.cli_sessions && userInstance.cli_sessions.length > 0) {
            userInstance.cli_sessions.forEach(function(s) {
                var createdAt = new Date(s.created_at * 1000).toLocaleString();
                html += '<div class="setting-item"><div class="setting-info">' +
                    '<div class="setting-label">CLI Sitzung</div>' +
                    '<div class="setting-description">Gestartet: ' + createdAt + '</div></div>' +
                    '<button class="tb-btn tb-btn-danger tb-btn-sm" onclick="closeCLISession(\\'' + s.cli_session_id + '\\')">' +
                        '<span class="material-symbols-outlined">close</span></button></div>';
            });
        }
        html += '</div></div>';

        html += '<div class="dashboard-card" style="border-color:var(--color-error);">' +
            '<h3 style="color:var(--color-error);"><span class="material-symbols-outlined">warning</span>Gefahrenzone</h3>' +
            '<button class="tb-btn tb-btn-danger" onclick="TB.user.logout().then(function() { window.location.href = \\'/\\'; })">' +
                '<span class="material-symbols-outlined">logout</span>Abmelden</button></div>';

        content.innerHTML = html;
    }

    window.openClerkProfile = function() {
        if (TB.user && TB.user.getClerkInstance) {
            var clerk = TB.user.getClerkInstance();
            if (clerk && clerk.openUserProfile) {
                clerk.openUserProfile();
                return;
            }
        }
        TB.ui.Toast.showInfo('Profil wird geladen...');
    };

    window.requestMagicLink = async function() {
        TB.ui.Loader.show('Magic Link wird angefordert...');
        try {
            var res = await TB.api.request('CloudM.UserDashboard', 'request_my_magic_link', null, 'POST');
            TB.ui.Loader.hide();
            if (res.error === TB.ToolBoxError.none) {
                TB.ui.Toast.showSuccess('Magic Link wurde an Ihre E-Mail gesendet');
            } else {
                TB.ui.Toast.showError((res.info && res.info.help_text) ? res.info.help_text : 'Fehler beim Anfordern');
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
            var res = await TB.api.request('CloudM.UserDashboard', 'close_cli_session', {cli_session_id: sessionId}, 'POST');
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
async def add_module_to_instance(app: App, request: RequestData, data: dict=None, module_name=Name):
    """Modul zur Benutzer-Instanz hinzufügen und laden"""
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)
    if data is None:
        data = {}
    module_name = module_name or data.get("module_name")
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
async def remove_module_from_instance(app: App, request: RequestData, data: dict=None, module_name=Name):
    """Modul aus Benutzer-Instanz entladen"""
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)
    if data is None:
        data = {}
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
async def add_module_to_saved(app: App, request: RequestData, data: dict=None, module_name=Name):
    """Modul zu den gespeicherten Modulen hinzufügen"""
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)
    if data is None:
        data = {}
    module_name = module_name or data.get("module_name")
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


# =================== User File Management ===================

def _get_user_storage(uid: str, username: str = ""):
    """Erstellt ScopedBlobStorage für einen Benutzer"""
    from toolboxv2.utils.extras.db.scoped_storage import ScopedBlobStorage, UserContext
    import os
    from pathlib import Path

    user_context = UserContext(
        user_id=uid,
        username=username or uid,
        is_authenticated=True
    )

    # Prüfe ob MinIO secure sein soll (Standard: False für lokale Entwicklung)
    minio_secure = os.getenv("MINIO_SECURE", "false").lower() in ("true", "1", "yes")

    # Lokale DB für USER_PRIVATE Dateien
    local_db_path = os.getenv("TB_USER_FILES_DB", str(Path.home() / ".tb_user_files" / f"{uid}.db"))

    return ScopedBlobStorage(
        user_context,
        minio_secure=minio_secure,
        local_db_path=local_db_path
    )


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['GET'])
async def list_user_files(app: App, request: RequestData, data: dict = None):
    """Liste alle Dateien des Benutzers auf"""
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    uid = getattr(current_user, 'uid', None) or getattr(current_user, 'clerk_user_id', None)
    if not uid:
        return Result.default_user_error(info="Benutzer-ID nicht gefunden")

    try:
        from toolboxv2.utils.extras.db.scoped_storage import Scope

        storage = _get_user_storage(uid, getattr(current_user, 'username', ''))
        blobs = storage.list(prefix="", scope=Scope.USER_PRIVATE, recursive=True)

        # User-Prefix der aus dem Pfad entfernt werden muss
        # build_path erstellt: {uid}/{path} - wir müssen {uid}/ entfernen
        user_prefix = f"{uid}/"

        files = []
        for blob in blobs:
            # Entferne User-Prefix aus dem Pfad für das Frontend
            relative_path = blob.path
            if relative_path.startswith(user_prefix):
                relative_path = relative_path[len(user_prefix):]

            files.append({
                "path": relative_path,
                "size": blob.size,
                "content_type": blob.content_type,
                "created_at": blob.created_at if isinstance(blob.created_at, str) else None,
                "updated_at": blob.updated_at if isinstance(blob.updated_at, str) else None,
            })

        return Result.ok(data=files)
    except ImportError:
        # Fallback: Use simple file storage
        return Result.ok(data=[])
    except Exception as e:
        return Result.default_internal_error(f"Fehler beim Auflisten: {e}")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def upload_user_file(app: App, request: RequestData, data: dict = None, **kwargs):
    """Datei für Benutzer hochladen"""
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    uid = getattr(current_user, 'uid', None) or getattr(current_user, 'clerk_user_id', None)
    if not uid:
        return Result.default_user_error(info="Benutzer-ID nicht gefunden")

    if data is None:
        data = kwargs
    # Get file data from request
    file_data = data.get("file") if data else None
    file_path = data.get("path", "uploaded_file") if data else "uploaded_file"
    content_type = data.get("content_type", "application/octet-stream") if data else "application/octet-stream"

    if not file_data:
        return Result.default_user_error(info="Keine Datei angegeben")

    try:
        from toolboxv2.utils.extras.db.scoped_storage import Scope
        import base64

        # Decode base64 if needed
        if isinstance(file_data, str):
            file_bytes = base64.b64decode(file_data)
        else:
            file_bytes = file_data

        # Check file size (max 10 MB)
        if len(file_bytes) > 10 * 1024 * 1024:
            return Result.default_user_error(info="Datei zu groß (max. 10 MB)")

        storage = _get_user_storage(uid, getattr(current_user, 'username', ''))
        blob = storage.write(
            path=file_path,
            data=file_bytes,
            scope=Scope.USER_PRIVATE,
            content_type=content_type
        )

        return Result.ok(info="Datei hochgeladen", data={
            "path": blob.path,
            "size": blob.size,
            "content_type": blob.content_type
        })
    except ImportError:
        return Result.default_internal_error("Speichersystem nicht verfügbar")
    except Exception as e:
        return Result.default_internal_error(f"Fehler beim Hochladen: {e}")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['GET'])
async def download_user_file(app: App, request: RequestData, data: dict = None, **kwargs):
    """Datei für Benutzer herunterladen"""
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)
    if data is None:
        data = kwargs

    uid = getattr(current_user, 'uid', None) or getattr(current_user, 'clerk_user_id', None)
    if not uid:
        return Result.default_user_error(info="Benutzer-ID nicht gefunden")

    file_path = data.get("path") if data else None
    if not file_path:
        return Result.default_user_error(info="Dateipfad erforderlich")

    try:
        from toolboxv2.utils.extras.db.scoped_storage import Scope
        import base64

        storage = _get_user_storage(uid, getattr(current_user, 'username', ''))
        # read() gibt bytes zurück, nicht ein Objekt
        file_bytes = storage.read(path=file_path, scope=Scope.USER_PRIVATE)

        if file_bytes is None:
            return Result.default_user_error(info="Datei nicht gefunden", exec_code=404)

        # Return base64 encoded content
        content_b64 = base64.b64encode(file_bytes).decode('utf-8')

        # Versuche content_type aus der Liste zu bekommen
        # Liste alle Dateien und finde die passende
        blobs = storage.list(prefix="", scope=Scope.USER_PRIVATE, recursive=True)
        content_type = "application/octet-stream"
        for blob in blobs:
            # blob.path enthält den vollen Pfad (z.B. users/{uid}/private/filename.png)
            # file_path ist nur der relative Pfad (z.B. filename.png)
            if blob.path.endswith("/" + file_path) or blob.path.endswith(file_path):
                content_type = blob.content_type or "application/octet-stream"
                break

        # Fallback: Bestimme content_type aus Dateiendung
        if content_type == "application/octet-stream":
            ext = file_path.rsplit('.', 1)[-1].lower() if '.' in file_path else ''
            mime_map = {
                'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                'gif': 'image/gif', 'webp': 'image/webp', 'svg': 'image/svg+xml',
                'bmp': 'image/bmp', 'ico': 'image/x-icon',
                'pdf': 'application/pdf', 'json': 'application/json',
                'txt': 'text/plain', 'html': 'text/html', 'css': 'text/css',
                'js': 'text/javascript', 'xml': 'text/xml', 'csv': 'text/csv',
                'mp3': 'audio/mpeg', 'wav': 'audio/wav', 'ogg': 'audio/ogg',
                'mp4': 'video/mp4', 'webm': 'video/webm', 'avi': 'video/x-msvideo',
                'zip': 'application/zip', 'tar': 'application/x-tar',
                'gz': 'application/gzip', '7z': 'application/x-7z-compressed',
            }
            content_type = mime_map.get(ext, 'application/octet-stream')

        return Result.ok(data={
            "path": file_path,
            "content": content_b64,
            "content_type": content_type,
            "size": len(file_bytes)
        })
    except ImportError:
        return Result.default_internal_error("Speichersystem nicht verfügbar")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Result.default_internal_error(f"Fehler beim Herunterladen: {e}")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=['POST'])
async def delete_user_file(app: App, request: RequestData, data: dict = None, path=None):
    """Datei für Benutzer löschen"""
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    uid = getattr(current_user, 'uid', None) or getattr(current_user, 'clerk_user_id', None)
    if not uid:
        return Result.default_user_error(info="Benutzer-ID nicht gefunden")

    file_path = path or data.get("path") if data else path
    if not file_path:
        return Result.default_user_error(info="Dateipfad erforderlich")

    try:
        from toolboxv2.utils.extras.db.scoped_storage import Scope

        storage = _get_user_storage(uid, getattr(current_user, 'username', ''))
        success = storage.delete(path=file_path, scope=Scope.USER_PRIVATE)

        if success:
            return Result.ok(info="Datei gelöscht")
        else:
            return Result.default_user_error(info="Datei nicht gefunden oder konnte nicht gelöscht werden")
    except ImportError:
        return Result.default_internal_error("Speichersystem nicht verfügbar")
    except Exception as e:
        return Result.default_internal_error(f"Fehler beim Löschen: {e}")
