/**
 * Widget Area Component
 * =====================
 * 
 * Renders widgets received from ToolBoxV2 backend
 * Used in HUD mode to display mod-registered widgets
 * 
 * Usage:
 * ```javascript
 * import { WidgetArea } from './hud/WidgetArea.js';
 * 
 * const area = new WidgetArea({
 *     container: document.getElementById('widget-area'),
 *     onWidgetClick: (widgetId) => console.log('Clicked:', widgetId)
 * });
 * 
 * area.setWidgets([
 *     { id: 'widget1', title: 'System', type: 'CORE', content: '<div>Running</div>' }
 * ]);
 * ```
 */

export class WidgetArea {
    /**
     * Create a widget area
     * @param {Object} options - Configuration options
     * @param {HTMLElement} options.container - Container element
     * @param {Function} options.onWidgetClick - Callback when widget is clicked
     */
    constructor(options = {}) {
        this.container = options.container || null;
        this.onWidgetClick = options.onWidgetClick || (() => {});
        this.widgets = [];
        this.systemWidget = null;
    }
    
    /**
     * Set the container element
     * @param {HTMLElement} container - Container element
     */
    setContainer(container) {
        this.container = container;
    }
    
    /**
     * Set widgets to display
     * @param {Array} widgets - Array of widget objects
     */
    setWidgets(widgets) {
        this.widgets = widgets || [];
        this.render();
    }
    
    /**
     * Add a widget
     * @param {Object} widget - Widget object
     */
    addWidget(widget) {
        const existingIdx = this.widgets.findIndex(w => 
            (w.widget_id || w.id) === (widget.widget_id || widget.id)
        );
        
        if (existingIdx >= 0) {
            this.widgets[existingIdx] = widget;
        } else {
            this.widgets.push(widget);
        }
        
        this.render();
    }
    
    /**
     * Remove a widget by ID
     * @param {string} widgetId - Widget ID to remove
     */
    removeWidget(widgetId) {
        this.widgets = this.widgets.filter(w => 
            (w.widget_id || w.id) !== widgetId
        );
        this.render();
    }
    
    /**
     * Update a widget
     * @param {string} widgetId - Widget ID to update
     * @param {Object} updates - Properties to update
     */
    updateWidget(widgetId, updates) {
        const idx = this.widgets.findIndex(w => 
            (w.widget_id || w.id) === widgetId
        );
        
        if (idx >= 0) {
            this.widgets[idx] = { ...this.widgets[idx], ...updates };
            this.render();
        }
    }
    
    /**
     * Update or create system widget
     * @param {Object} data - System status data
     */
    updateSystemWidget(data) {
        this.systemWidget = {
            id: 'system',
            title: 'System Status',
            type: 'CORE',
            data: data
        };
        this.render();
    }
    
    /**
     * Clear all widgets
     */
    clear() {
        this.widgets = [];
        this.systemWidget = null;
        this.render();
    }
    
    /**
     * Render widgets to container
     */
    render() {
        if (!this.container) return;
        
        // Clear container
        this.container.innerHTML = '';
        
        // Render system widget first if present
        if (this.systemWidget) {
            const systemEl = this.renderSystemWidget(this.systemWidget);
            this.container.appendChild(systemEl);
        }
        
        // Check if we have any widgets
        if (this.widgets.length === 0 && !this.systemWidget) {
            this.container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">📦</div>
                    <div>No widgets registered</div>
                </div>
            `;
            return;
        }
        
        // Render mod widgets
        for (const widget of this.widgets) {
            const el = this.renderWidget(widget);
            this.container.appendChild(el);
        }
    }
    
    /**
     * Render a single widget
     * @param {Object} widget - Widget data
     * @returns {HTMLElement} Widget element
     */
    renderWidget(widget) {
        const el = document.createElement('div');
        el.className = 'widget';
        el.id = `widget-${widget.widget_id || widget.id || this.generateId()}`;
        
        const title = widget.title || widget.name || 'Widget';
        const type = widget.type || widget.mod_name || 'MOD';
        const content = widget.html || widget.content || '<span style="opacity:0.5">Loading...</span>';
        
        el.innerHTML = `
            <div class="widget-header">
                <span>${this.escapeHtml(title)}</span>
                <span class="widget-type">${this.escapeHtml(type.toUpperCase())}</span>
            </div>
            <div class="widget-content">
                ${content}
            </div>
        `;
        
        // Click handler
        el.addEventListener('click', () => {
            this.onWidgetClick(widget.widget_id || widget.id);
        });
        
        return el;
    }
    
    /**
     * Render system status widget
     * @param {Object} widget - System widget data
     * @returns {HTMLElement} Widget element
     */
    renderSystemWidget(widget) {
        const el = document.createElement('div');
        el.className = 'widget';
        el.id = 'widget-system';
        
        const data = widget.data || {};
        const workerRunning = data.worker_running ?? data.running ?? data.healthy ?? true;
        const isRemote = data.is_remote || data.mode === 'remote';
        const uptime = this.formatUptime(data.uptime_seconds || data.uptime || 0);
        
        el.innerHTML = `
            <div class="widget-header">
                <span>System Status</span>
                <span class="widget-type">CORE</span>
            </div>
            <div class="widget-content">
                <div class="widget-row">
                    <span class="widget-label">Mode</span>
                    <span class="widget-value">${isRemote ? 'Remote' : 'Local'}</span>
                </div>
                <div class="widget-row">
                    <span class="widget-label">Worker</span>
                    <span class="widget-value ${workerRunning ? 'success' : 'error'}">
                        ${workerRunning ? 'Running' : 'Stopped'}
                    </span>
                </div>
                <div class="widget-row">
                    <span class="widget-label">Uptime</span>
                    <span class="widget-value">${uptime}</span>
                </div>
            </div>
        `;
        
        return el;
    }
    
    /**
     * Format uptime seconds to human readable
     * @param {number} seconds - Uptime in seconds
     * @returns {string} Formatted uptime
     */
    formatUptime(seconds) {
        if (!seconds || seconds < 0) return '0s';
        if (seconds < 60) return `${Math.floor(seconds)}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
        if (seconds < 86400) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            return `${h}h ${m}m`;
        }
        return `${Math.floor(seconds / 86400)}d`;
    }
    
    /**
     * Escape HTML special characters
     * @param {string} str - String to escape
     * @returns {string} Escaped string
     */
    escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
    
    /**
     * Generate a random ID
     * @returns {string} Random ID
     */
    generateId() {
        return Math.random().toString(36).slice(2, 10);
    }
    
    /**
     * Show loading state
     */
    showLoading() {
        if (!this.container) return;
        this.container.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">⏳</div>
                <div>Loading widgets...</div>
            </div>
        `;
    }
    
    /**
     * Show error state
     * @param {string} message - Error message
     */
    showError(message) {
        if (!this.container) return;
        this.container.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">⚠️</div>
                <div>${this.escapeHtml(message)}</div>
            </div>
        `;
    }
    
    /**
     * Show empty state
     * @param {string} message - Optional custom message
     */
    showEmpty(message = 'No widgets registered') {
        if (!this.container) return;
        this.container.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">📦</div>
                <div>${this.escapeHtml(message)}</div>
            </div>
        `;
    }
}

export default WidgetArea;
