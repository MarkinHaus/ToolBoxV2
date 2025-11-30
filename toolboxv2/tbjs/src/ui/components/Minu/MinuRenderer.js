// tbjs/ui/components/Minu/MinuRenderer.js
// ==========================================
// Frontend renderer for Minu UI Framework
// Converts JSON component definitions to DOM elements

import TB from '../../../index.js';

/**
 * MinuRenderer - Renders Minu UI JSON to DOM
 *
 * Features:
 * - Component rendering from JSON
 * - Event binding to Python handlers
 * - State bindings with two-way sync
 * - Efficient DOM patching
 */
class MinuRenderer {
    constructor(options = {}) {
        this.container = null;
        this.session = null;
        this.views = new Map(); // viewId -> { element, component, bindings }
        this.eventHandlers = new Map(); // handlerId -> callback
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;

        // Component renderers registry
        this.renderers = this._initRenderers();

        // Options
        this.wsUrl = options.wsUrl || this._buildWsUrl();
        this.onConnect = options.onConnect || (() => {});
        this.onDisconnect = options.onDisconnect || (() => {});
        this.onError = options.onError || ((e) => TB.logger.error('[Minu] Error:', e));
    }

    /**
     * Initialize connection and mount to container
     */
    async mount(containerSelector, viewName, props = {}) {
        this.container = typeof containerSelector === 'string'
            ? document.querySelector(containerSelector)
            : containerSelector;

        if (!this.container) {
            throw new Error(`[Minu] Container not found: ${containerSelector}`);
        }

        // Connect WebSocket
        await this._connectWs();

        // Subscribe to view
        this._send({
            type: 'subscribe',
            viewName: viewName,
            props: props
        });

        return this;
    }

    /**
     * Disconnect and cleanup
     */
    unmount() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        this.views.clear();
        this.eventHandlers.clear();

        if (this.container) {
            this.container.innerHTML = '';
        }
    }

    // =========================================================================
    // WEBSOCKET HANDLING
    // =========================================================================

    _buildWsUrl() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${protocol}//${window.location.host}/ws/Minu/ui`;
    }

    async _connectWs() {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.wsUrl);

            this.ws.onopen = () => {
                TB.logger.info('[Minu] WebSocket connected');
                this.reconnectAttempts = 0;
                this.onConnect();
                resolve();
            };

            this.ws.onmessage = (event) => {
                this._handleMessage(JSON.parse(event.data));
            };

            this.ws.onerror = (error) => {
                TB.logger.error('[Minu] WebSocket error:', error);
                this.onError(error);
                reject(error);
            };

            this.ws.onclose = () => {
                TB.logger.warn('[Minu] WebSocket closed');
                this.onDisconnect();
                this._attemptReconnect();
            };
        });
    }

    _attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
            TB.logger.info(`[Minu] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
            setTimeout(() => this._connectWs().catch(() => {}), delay);
        }
    }

    _send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        } else {
            TB.logger.warn('[Minu] WebSocket not connected, queuing message');
        }
    }

    _handleMessage(data) {
        TB.logger.debug('[Minu] Received:', data);

        switch (data.type) {
            case 'connected':
                this.session = data.sessionId;
                break;

            case 'render':
                this._renderView(data.view);
                break;

            case 'patches':
                this._applyPatches(data.patches);
                break;

            case 'event_result':
                this._handleEventResult(data);
                break;

            case 'error':
                this.onError(new Error(data.message));
                break;
        }
    }

    // =========================================================================
    // RENDERING
    // =========================================================================

    _renderView(viewData) {
        const { viewId, component, state, handlers } = viewData;

        // Create element
        const element = this._renderComponent(component, viewId);

        // Store view data
        this.views.set(viewId, {
            element,
            component,
            state: state || {},
            handlers: handlers || []
        });

        // Mount to container
        this.container.innerHTML = '';
        this.container.appendChild(element);

        // Apply initial state bindings
        this._applyBindings(viewId);

        // Trigger HTMX processing if available
        if (window.htmx) {
            window.htmx.process(this.container);
        }

        TB.logger.info(`[Minu] View rendered: ${viewId}`);
    }

    _renderComponent(comp, viewId) {
        if (typeof comp === 'string') {
            return document.createTextNode(comp);
        }

        const renderer = this.renderers[comp.type];
        if (!renderer) {
            TB.logger.warn(`[Minu] Unknown component type: ${comp.type}`);
            return document.createElement('div');
        }

        const element = renderer.call(this, comp, viewId);

        // Apply common attributes
        if (comp.id) {
            element.id = comp.id;
            element.dataset.minuId = comp.id;
        }

        if (comp.className) {
            element.className = comp.className;
        }

        if (comp.style) {
            Object.assign(element.style, comp.style);
        }

        // Bind events
        if (comp.events) {
            for (const [eventName, handlerName] of Object.entries(comp.events)) {
                if (handlerName) {
                    this._bindEvent(element, eventName, handlerName, viewId);
                }
            }
        }

        // Store bindings for later updates
        if (comp.bindings) {
            element.dataset.minuBindings = JSON.stringify(comp.bindings);
        }

        // Render children
        if (comp.children && Array.isArray(comp.children)) {
            for (const child of comp.children) {
                element.appendChild(this._renderComponent(child, viewId));
            }
        }

        return element;
    }

    _initRenderers() {
        return {
            // Layout Components
            card: (comp) => {
                const el = document.createElement('div');
                el.className = comp.className || 'card';
                return el;
            },

            row: (comp) => {
                const el = document.createElement('div');
                el.className = comp.className || 'flex gap-4 items-center';
                return el;
            },

            column: (comp) => {
                const el = document.createElement('div');
                el.className = comp.className || 'flex flex-col gap-4';
                return el;
            },

            grid: (comp) => {
                const el = document.createElement('div');
                el.className = comp.className || 'grid';
                return el;
            },

            spacer: (comp) => {
                const el = document.createElement('div');
                el.className = comp.className || 'h-4';
                return el;
            },

            divider: (comp) => {
                const el = document.createElement('hr');
                el.className = comp.className || 'border-t border-neutral-200 my-4';
                return el;
            },

            // Content Components
            text: (comp) => {
                const el = document.createElement('span');
                el.textContent = comp.props?.text || '';
                return el;
            },

            heading: (comp) => {
                const level = comp.props?.level || 1;
                const el = document.createElement(`h${Math.min(Math.max(level, 1), 6)}`);
                el.textContent = comp.props?.text || '';
                return el;
            },

            paragraph: (comp) => {
                const el = document.createElement('p');
                el.textContent = comp.props?.text || '';
                return el;
            },

            icon: (comp) => {
                const el = document.createElement('span');
                el.className = comp.className || 'material-symbols-outlined';
                el.textContent = comp.props?.name || '';
                if (comp.props?.size) {
                    el.style.fontSize = `${comp.props.size}px`;
                }
                return el;
            },

            image: (comp) => {
                const el = document.createElement('img');
                el.src = comp.props?.src || '';
                el.alt = comp.props?.alt || '';
                if (comp.props?.width) el.style.width = comp.props.width;
                if (comp.props?.height) el.style.height = comp.props.height;
                return el;
            },

            badge: (comp) => {
                const el = document.createElement('span');
                el.className = comp.className || `badge badge-${comp.props?.variant || 'default'}`;
                el.textContent = comp.props?.text || '';
                return el;
            },

            // Input Components
            button: (comp) => {
                const el = document.createElement('button');
                el.className = comp.className || 'btn btn-primary';
                el.textContent = comp.props?.label || '';
                if (comp.props?.disabled) {
                    el.disabled = true;
                }
                return el;
            },

            input: (comp) => {
                const el = document.createElement('input');
                el.type = comp.props?.inputType || 'text';
                el.placeholder = comp.props?.placeholder || '';
                el.value = comp.props?.value || '';
                if (comp.props?.disabled) el.disabled = true;
                if (comp.props?.readonly) el.readOnly = true;
                return el;
            },

            textarea: (comp) => {
                const el = document.createElement('textarea');
                el.placeholder = comp.props?.placeholder || '';
                el.value = comp.props?.value || '';
                if (comp.props?.rows) el.rows = comp.props.rows;
                return el;
            },

            select: (comp) => {
                const el = document.createElement('select');

                // Add placeholder option
                if (comp.props?.placeholder) {
                    const placeholder = document.createElement('option');
                    placeholder.value = '';
                    placeholder.textContent = comp.props.placeholder;
                    placeholder.disabled = true;
                    placeholder.selected = !comp.props?.value;
                    el.appendChild(placeholder);
                }

                // Add options
                if (comp.props?.options) {
                    for (const opt of comp.props.options) {
                        const option = document.createElement('option');
                        option.value = opt.value;
                        option.textContent = opt.label;
                        if (opt.value === comp.props?.value) {
                            option.selected = true;
                        }
                        el.appendChild(option);
                    }
                }

                return el;
            },

            checkbox: (comp) => {
                const wrapper = document.createElement('label');
                wrapper.className = 'flex items-center gap-2 cursor-pointer';

                const input = document.createElement('input');
                input.type = 'checkbox';
                input.checked = comp.props?.checked || false;

                const label = document.createElement('span');
                label.textContent = comp.props?.label || '';

                wrapper.appendChild(input);
                wrapper.appendChild(label);

                return wrapper;
            },

            switch: (comp) => {
                const wrapper = document.createElement('label');
                wrapper.className = 'flex items-center gap-2 cursor-pointer';

                const toggle = document.createElement('div');
                toggle.className = 'relative w-10 h-6 bg-neutral-300 rounded-full transition-colors';

                const input = document.createElement('input');
                input.type = 'checkbox';
                input.className = 'sr-only';
                input.checked = comp.props?.checked || false;

                const slider = document.createElement('div');
                slider.className = 'absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform';
                if (input.checked) {
                    toggle.classList.add('bg-primary-500');
                    slider.style.transform = 'translateX(16px)';
                }

                toggle.appendChild(slider);
                wrapper.appendChild(input);
                wrapper.appendChild(toggle);

                if (comp.props?.label) {
                    const label = document.createElement('span');
                    label.textContent = comp.props.label;
                    wrapper.appendChild(label);
                }

                return wrapper;
            },

            // Feedback Components
            alert: (comp) => {
                const el = document.createElement('div');
                el.className = comp.className || `alert alert-${comp.props?.variant || 'info'}`;
                el.setAttribute('role', 'alert');

                if (comp.props?.title) {
                    const title = document.createElement('strong');
                    title.className = 'font-semibold';
                    title.textContent = comp.props.title;
                    el.appendChild(title);
                }

                const message = document.createElement('p');
                message.textContent = comp.props?.message || '';
                el.appendChild(message);

                if (comp.props?.dismissible) {
                    const closeBtn = document.createElement('button');
                    closeBtn.className = 'absolute top-2 right-2 opacity-60 hover:opacity-100';
                    closeBtn.innerHTML = '×';
                    closeBtn.onclick = () => el.remove();
                    el.appendChild(closeBtn);
                }

                return el;
            },

            progress: (comp) => {
                const wrapper = document.createElement('div');
                wrapper.className = 'w-full';

                if (comp.props?.label) {
                    const label = document.createElement('div');
                    label.className = 'flex justify-between text-sm mb-1';
                    label.innerHTML = `<span>${comp.props.label}</span><span>${comp.props?.value || 0}%</span>`;
                    wrapper.appendChild(label);
                }

                const track = document.createElement('div');
                track.className = 'w-full h-2 bg-neutral-200 rounded-full overflow-hidden';

                const bar = document.createElement('div');
                bar.className = 'h-full bg-primary-500 transition-all duration-300';
                bar.style.width = `${Math.min(100, Math.max(0, comp.props?.value || 0))}%`;
                bar.dataset.minuProgressBar = 'true';

                track.appendChild(bar);
                wrapper.appendChild(track);

                return wrapper;
            },

            spinner: (comp) => {
                const el = document.createElement('div');
                const size = { sm: '16', md: '24', lg: '32' }[comp.props?.size] || '24';
                el.className = comp.className || 'animate-spin';
                el.innerHTML = `
                    <svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10" stroke-opacity="0.25"/>
                        <path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round"/>
                    </svg>
                `;
                return el;
            },

            // Data Components
            table: (comp, viewId) => {
                const wrapper = document.createElement('div');
                wrapper.className = 'overflow-x-auto';

                const table = document.createElement('table');
                table.className = 'w-full';

                // Header
                if (comp.props?.columns) {
                    const thead = document.createElement('thead');
                    const headerRow = document.createElement('tr');

                    for (const col of comp.props.columns) {
                        const th = document.createElement('th');
                        th.className = 'text-left p-2 border-b font-semibold';
                        th.textContent = col.label || col.key;
                        headerRow.appendChild(th);
                    }

                    thead.appendChild(headerRow);
                    table.appendChild(thead);
                }

                // Body
                const tbody = document.createElement('tbody');
                if (comp.props?.data) {
                    for (const row of comp.props.data) {
                        const tr = document.createElement('tr');
                        tr.className = 'hover:bg-neutral-50 transition-colors';

                        if (comp.events?.rowClick) {
                            tr.style.cursor = 'pointer';
                            tr.onclick = () => this._triggerEvent(viewId, comp.events.rowClick, row);
                        }

                        for (const col of (comp.props.columns || [])) {
                            const td = document.createElement('td');
                            td.className = 'p-2 border-b';
                            td.textContent = row[col.key] ?? '';
                            tr.appendChild(td);
                        }

                        tbody.appendChild(tr);
                    }
                }

                table.appendChild(tbody);
                wrapper.appendChild(table);

                return wrapper;
            },

            list: (comp) => {
                const el = document.createElement(comp.props?.ordered ? 'ol' : 'ul');
                el.className = comp.className || 'space-y-2';
                return el;
            },

            listitem: (comp) => {
                const el = document.createElement('li');
                el.className = comp.className || 'p-2';
                return el;
            },

            // Special Components
            modal: (comp, viewId) => {
                // Create overlay
                const overlay = document.createElement('div');
                overlay.className = 'overlay' + (comp.props?.open ? ' is-active' : '');
                overlay.onclick = (e) => {
                    if (e.target === overlay && comp.events?.close) {
                        this._triggerEvent(viewId, comp.events.close, {});
                    }
                };

                // Create modal
                const modal = document.createElement('div');
                modal.className = 'modal' + (comp.props?.open ? ' is-active' : '');

                // Header
                if (comp.props?.title) {
                    const header = document.createElement('div');
                    header.className = 'flex justify-between items-center p-4 border-b';
                    header.innerHTML = `
                        <h3 class="text-lg font-semibold">${comp.props.title}</h3>
                        <button class="btn-ghost p-1" data-close>×</button>
                    `;
                    header.querySelector('[data-close]').onclick = () => {
                        if (comp.events?.close) {
                            this._triggerEvent(viewId, comp.events.close, {});
                        }
                    };
                    modal.appendChild(header);
                }

                // Content wrapper for children
                const content = document.createElement('div');
                content.className = 'p-4';
                modal.appendChild(content);

                overlay.appendChild(modal);

                // Store content element for child rendering
                overlay._minuContentTarget = content;

                return overlay;
            },

            widget: (comp) => {
                const el = document.createElement('div');
                el.className = comp.className || 'widget';

                if (comp.props?.title) {
                    const header = document.createElement('div');
                    header.className = 'widget-header';
                    header.innerHTML = `<span class="widget-title">${comp.props.title}</span>`;
                    el.appendChild(header);
                }

                const content = document.createElement('div');
                content.className = 'widget-content';
                el.appendChild(content);
                el._minuContentTarget = content;

                return el;
            },

            form: (comp, viewId) => {
                const el = document.createElement('form');
                el.onsubmit = (e) => {
                    e.preventDefault();
                    if (comp.events?.submit) {
                        const formData = new FormData(el);
                        const data = Object.fromEntries(formData.entries());
                        this._triggerEvent(viewId, comp.events.submit, data);
                    }
                };
                return el;
            },

            tabs: (comp, viewId) => {
                const wrapper = document.createElement('div');

                // Tab buttons
                const tabList = document.createElement('div');
                tabList.className = 'flex border-b';
                tabList.setAttribute('role', 'tablist');

                const contentArea = document.createElement('div');
                contentArea.className = 'p-4';

                if (comp.props?.tabs) {
                    comp.props.tabs.forEach((tab, index) => {
                        const button = document.createElement('button');
                        button.className = 'px-4 py-2 border-b-2 transition-colors ' +
                            (index === (comp.props.active || 0)
                                ? 'border-primary-500 text-primary-500'
                                : 'border-transparent hover:border-neutral-300');
                        button.textContent = tab.label;
                        button.onclick = () => {
                            if (comp.events?.change) {
                                this._triggerEvent(viewId, comp.events.change, { index });
                            }
                        };
                        tabList.appendChild(button);
                    });

                    // Render active tab content
                    const activeIndex = comp.props.active || 0;
                    if (comp.props.tabs[activeIndex]?.content) {
                        const content = this._renderComponent(comp.props.tabs[activeIndex].content, viewId);
                        contentArea.appendChild(content);
                    }
                }

                wrapper.appendChild(tabList);
                wrapper.appendChild(contentArea);

                return wrapper;
            },

            custom: (comp) => {
                const el = document.createElement('div');
                if (comp.props?.html) {
                    el.innerHTML = comp.props.html;
                }
                return el;
            }
        };
    }

    // =========================================================================
    // EVENT HANDLING
    // =========================================================================

    _bindEvent(element, eventName, handlerName, viewId) {
        const eventMap = {
            'click': 'click',
            'change': 'change',
            'input': 'input',
            'submit': 'submit',
            'focus': 'focus',
            'blur': 'blur'
        };

        const domEvent = eventMap[eventName] || eventName;

        element.addEventListener(domEvent, (e) => {
            // Collect relevant event data
            const payload = {
                type: domEvent,
                value: e.target?.value,
                checked: e.target?.checked,
                timestamp: Date.now()
            };

            // For form elements, include all form data
            if (e.target?.form) {
                const formData = new FormData(e.target.form);
                payload.formData = Object.fromEntries(formData.entries());
            }

            this._triggerEvent(viewId, handlerName, payload);
        });
    }

    _triggerEvent(viewId, handlerName, payload) {
        TB.logger.debug(`[Minu] Event: ${handlerName}`, payload);

        this._send({
            type: 'event',
            viewId: viewId,
            handler: handlerName,
            payload: payload
        });
    }

    _handleEventResult(data) {
        if (data.result?.error) {
            this.onError(new Error(data.result.error));
        }

        TB.events.emit('minu:eventResult', data);
    }

    // =========================================================================
    // STATE & BINDINGS
    // =========================================================================

    _applyBindings(viewId) {
        const view = this.views.get(viewId);
        if (!view) return;

        // Find all elements with bindings
        const elements = this.container.querySelectorAll('[data-minu-bindings]');

        elements.forEach(el => {
            const bindings = JSON.parse(el.dataset.minuBindings);

            for (const [prop, path] of Object.entries(bindings)) {
                // Get value from state
                const value = this._getStateValue(view.state, path);

                // Apply value based on property
                this._applyBinding(el, prop, value);

                // Set up two-way binding for inputs
                if (['value', 'checked'].includes(prop)) {
                    this._setupTwoWayBinding(el, prop, path, viewId);
                }
            }
        });
    }

    _getStateValue(state, path) {
        const parts = path.split('.');
        let value = state;

        for (const part of parts) {
            if (value && typeof value === 'object') {
                // Handle viewId.stateName format
                const cleanPart = part.replace(/^view-[a-f0-9]+\./, '');
                value = value[cleanPart] ?? value[part];
            } else {
                return undefined;
            }
        }

        return value;
    }

    _applyBinding(element, prop, value) {
        switch (prop) {
            case 'text':
                element.textContent = value ?? '';
                break;
            case 'value':
                element.value = value ?? '';
                break;
            case 'checked':
                element.checked = !!value;
                break;
            case 'disabled':
                element.disabled = !!value;
                break;
            case 'open':
                element.classList.toggle('is-active', !!value);
                break;
            case 'data':
                // Re-render data-bound components (tables, lists)
                // This is handled by patches
                break;
            default:
                TB.logger.warn(`[Minu] Unknown binding property: ${prop}`);
        }
    }

    _setupTwoWayBinding(element, prop, path, viewId) {
        const eventName = element.type === 'checkbox' ? 'change' : 'input';

        element.addEventListener(eventName, (e) => {
            const value = prop === 'checked' ? e.target.checked : e.target.value;

            // Update local state
            const view = this.views.get(viewId);
            if (view) {
                this._setStateValue(view.state, path, value);
            }

            // Notify server
            this._send({
                type: 'state_update',
                viewId: viewId,
                path: path,
                value: value
            });
        });
    }

    _setStateValue(state, path, value) {
        const parts = path.split('.');
        let current = state;

        for (let i = 0; i < parts.length - 1; i++) {
            const cleanPart = parts[i].replace(/^view-[a-f0-9]+\./, '');
            if (!current[cleanPart]) {
                current[cleanPart] = {};
            }
            current = current[cleanPart];
        }

        const lastPart = parts[parts.length - 1];
        current[lastPart] = value;
    }

    // =========================================================================
    // PATCHING
    // =========================================================================

    _applyPatches(patches) {
        for (const patch of patches) {
            this._applyPatch(patch);
        }
    }

    _applyPatch(patch) {
        switch (patch.type) {
            case 'state_update':
                this._patchState(patch);
                break;
            case 'component_update':
                this._patchComponent(patch);
                break;
            case 'remove':
                this._patchRemove(patch);
                break;
            default:
                TB.logger.warn(`[Minu] Unknown patch type: ${patch.type}`);
        }
    }

    _patchState(patch) {
        const { viewId, path, value } = patch;

        // Update local state
        const view = this.views.get(viewId);
        if (view) {
            this._setStateValue(view.state, path, value);
        }

        // Find and update bound elements
        const elements = this.container.querySelectorAll('[data-minu-bindings]');

        elements.forEach(el => {
            const bindings = JSON.parse(el.dataset.minuBindings);

            for (const [prop, bindPath] of Object.entries(bindings)) {
                if (bindPath === path || path.endsWith(bindPath)) {
                    this._applyBinding(el, prop, value);
                }
            }
        });

        TB.logger.debug(`[Minu] State patched: ${path} = ${value}`);
    }

    _patchComponent(patch) {
        const { componentId, component } = patch;

        const existing = this.container.querySelector(`[data-minu-id="${componentId}"]`);
        if (existing) {
            const newElement = this._renderComponent(component, patch.viewId);
            existing.replaceWith(newElement);
        }
    }

    _patchRemove(patch) {
        const { componentId } = patch;

        const existing = this.container.querySelector(`[data-minu-id="${componentId}"]`);
        if (existing) {
            existing.remove();
        }
    }
}

// ============================================================================
// TBJS INTEGRATION
// ============================================================================

/**
 * Create a Minu renderer instance
 */
function createMinuRenderer(options = {}) {
    return new MinuRenderer(options);
}

/**
 * Mount a Minu view to a container
 */
async function mountMinuView(container, viewName, props = {}, options = {}) {
    const renderer = new MinuRenderer(options);
    await renderer.mount(container, viewName, props);
    return renderer;
}

// Export for TBJS
export { MinuRenderer, createMinuRenderer, mountMinuView };
export default MinuRenderer;
