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

        this._injectStyles();
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
        // Bind events - Skip für wrapper-komponenten (checkbox, switch, select, input mit label)
        const skipEventBinding = ['checkbox', 'switch'].includes(comp.type) ||
            (['select', 'input'].includes(comp.type) && comp.props?.label);

        if (comp.events && !skipEventBinding) {
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
    // Kein inline-style - nutze tbjs-main.css .card Klasse
    return el;
},

            row: (comp) => {
                const el = document.createElement('div');
                const gap = comp.props?.gap || '4';
                const justify = comp.props?.justify || 'start';
                const align = comp.props?.align || 'center';

                el.className = comp.className || '';
                el.style.display = 'flex';
                el.style.flexDirection = 'row';
                el.style.flexWrap = 'wrap';
                el.style.gap = `var(--space-${gap})`;
                el.style.alignItems = align === 'center' ? 'center' :
                                      align === 'start' ? 'flex-start' : 'flex-end';
                el.style.justifyContent = justify === 'between' ? 'space-between' :
                                           justify === 'end' ? 'flex-end' :
                                           justify === 'center' ? 'center' : 'flex-start';
                return el;
            },

            column: (comp) => {
                const el = document.createElement('div');
                const gap = comp.props?.gap || '4';

                el.className = comp.className || '';
                el.style.display = 'flex';
                el.style.flexDirection = 'column';
                el.style.gap = `var(--space-${gap})`;
                el.style.width = '100%';
                return el;
            },

            grid: (comp) => {
                const el = document.createElement('div');
                el.className = comp.className || 'grid';
                return el;
            },

            spacer: (comp) => {
                const el = document.createElement('div');
                const size = comp.props?.size || '4';
                el.style.height = `var(--space-${size})`;
                el.style.width = '100%';
                el.style.flexShrink = '0';
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
            button: (comp, viewId) => {
                const el = document.createElement('button');
                el.type = comp.props?.type || 'button'; // Default: button, nicht submit

                const variant = comp.props?.variant || 'primary';
                el.className = comp.className || `btn btn-${variant}`;

                // Icon support
                if (comp.props?.icon) {
                    const icon = document.createElement('span');
                    icon.className = 'material-symbols-outlined btn-icon';
                    icon.textContent = comp.props.icon;
                    el.appendChild(icon);
                }

                if (comp.props?.label) {
                    const label = document.createTextNode(comp.props.label);
                    el.appendChild(label);
                }

                if (comp.props?.disabled) {
                    el.disabled = true;
                }

                return el;
            },

            input: (comp) => {
    const wrapper = document.createElement('div');
    wrapper.style.display = 'flex';
    wrapper.style.flexDirection = 'column';
    wrapper.style.gap = 'var(--space-1)';
    wrapper.style.width = '100%';

    // Label
    if (comp.props?.label) {
        const label = document.createElement('label');
        label.textContent = comp.props.label;
        wrapper.appendChild(label);
    }

    const el = document.createElement('input');
    el.type = comp.props?.inputType || comp.props?.input_type || 'text';
    el.placeholder = comp.props?.placeholder || '';
    el.value = comp.props?.value || '';
    el.style.marginBottom = '0'; // Override
    if (comp.props?.disabled) el.disabled = true;
    if (comp.props?.readonly) el.readOnly = true;

    wrapper.appendChild(el);
    return wrapper;
},

            textarea: (comp) => {
                const el = document.createElement('textarea');
                el.placeholder = comp.props?.placeholder || '';
                el.value = comp.props?.value || '';
                if (comp.props?.rows) el.rows = comp.props.rows;
                return el;
            },

            select: (comp) => {
    const wrapper = document.createElement('div');
    wrapper.style.display = 'flex';
    wrapper.style.flexDirection = 'column';
    wrapper.style.gap = 'var(--space-1)';
    wrapper.style.width = '100%';

    // Label
    if (comp.props?.label) {
        const label = document.createElement('label');
        label.textContent = comp.props.label;
        label.style.fontSize = 'var(--text-sm)';
        label.style.fontWeight = 'var(--weight-medium)';
        label.style.color = 'var(--text-secondary)';
        wrapper.appendChild(label);
    }

    const el = document.createElement('select');
    el.style.marginBottom = '0'; // Override default margin

    // Placeholder
    if (comp.props?.placeholder) {
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = comp.props.placeholder;
        placeholder.disabled = true;
        placeholder.selected = !comp.props?.value;
        el.appendChild(placeholder);
    }

    // Options
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

    wrapper.appendChild(el);
    return wrapper;
},

            checkbox: (comp, viewId) => {
    const wrapper = document.createElement('label');
    wrapper.style.display = 'flex';
    wrapper.style.alignItems = 'flex-start';
    wrapper.style.gap = 'var(--space-3)';
    wrapper.style.cursor = 'pointer';
    wrapper.style.padding = 'var(--space-2) 0';
    wrapper.style.userSelect = 'none';

    // Custom Checkbox Container
    const checkboxContainer = document.createElement('div');
    checkboxContainer.style.position = 'relative';
    checkboxContainer.style.width = '20px';
    checkboxContainer.style.height = '20px';
    checkboxContainer.style.flexShrink = '0';

    // Hidden native input
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = comp.props?.checked || false;
    input.style.position = 'absolute';
    input.style.opacity = '0';
    input.style.width = '100%';
    input.style.height = '100%';
    input.style.margin = '0';
    input.style.cursor = 'pointer';

    // Visual checkbox
    const visual = document.createElement('div');
    visual.style.width = '20px';
    visual.style.height = '20px';
    visual.style.borderRadius = 'var(--radius-sm)';
    visual.style.border = '2px solid var(--border-strong)';
    visual.style.backgroundColor = input.checked ? 'var(--interactive)' : 'var(--bg-surface)';
    visual.style.borderColor = input.checked ? 'var(--interactive)' : 'var(--border-strong)';
    visual.style.transition = 'all var(--duration-fast) var(--ease-default)';
    visual.style.display = 'flex';
    visual.style.alignItems = 'center';
    visual.style.justifyContent = 'center';
    visual.style.pointerEvents = 'none';

    // Checkmark SVG
    const checkmark = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    checkmark.setAttribute('viewBox', '0 0 24 24');
    checkmark.setAttribute('fill', 'none');
    checkmark.setAttribute('stroke', 'white');
    checkmark.setAttribute('stroke-width', '3');
    checkmark.setAttribute('stroke-linecap', 'round');
    checkmark.setAttribute('stroke-linejoin', 'round');
    checkmark.style.width = '14px';
    checkmark.style.height = '14px';
    checkmark.style.opacity = input.checked ? '1' : '0';
    checkmark.style.transform = input.checked ? 'scale(1)' : 'scale(0.5)';
    checkmark.style.transition = 'all var(--duration-fast) var(--ease-default)';
    checkmark.innerHTML = '<polyline points="20 6 9 17 4 12"></polyline>';

    visual.appendChild(checkmark);

    // Update visual on change
    const updateVisual = () => {
        const isChecked = input.checked;
        visual.style.backgroundColor = isChecked ? 'var(--interactive)' : 'var(--bg-surface)';
        visual.style.borderColor = isChecked ? 'var(--interactive)' : 'var(--border-strong)';
        checkmark.style.opacity = isChecked ? '1' : '0';
        checkmark.style.transform = isChecked ? 'scale(1)' : 'scale(0.5)';
    };

    // Event listener für visuelles Update
    input.addEventListener('change', updateVisual);

    // Focus styling
    input.addEventListener('focus', () => {
        visual.style.boxShadow = '0 0 0 3px oklch(from var(--interactive) l c h / 0.2)';
    });
    input.addEventListener('blur', () => {
        visual.style.boxShadow = 'none';
    });

    // Hover
    wrapper.addEventListener('mouseenter', () => {
        if (!input.checked) {
            visual.style.borderColor = 'var(--interactive)';
        }
    });
    wrapper.addEventListener('mouseleave', () => {
        if (!input.checked) {
            visual.style.borderColor = 'var(--border-strong)';
        }
    });

    // Events an Server binden
    if (comp.events) {
        for (const [eventName, handlerName] of Object.entries(comp.events)) {
            if (handlerName) {
                this._bindEvent(input, eventName, handlerName, viewId);
            }
        }
    }

    // ID und Bindings
    if (comp.id) {
        input.dataset.minuId = comp.id;
        if (comp.bindings) {
            input.dataset.minuBindings = JSON.stringify(comp.bindings);
        }
    }

    checkboxContainer.appendChild(input);
    checkboxContainer.appendChild(visual);

    // Label Text
    const labelText = document.createElement('span');
    labelText.style.color = 'var(--text-primary)';
    labelText.style.fontSize = 'var(--text-sm)';
    labelText.style.lineHeight = '1.5';
    labelText.style.paddingTop = '1px';
    labelText.textContent = comp.props?.label || '';

    wrapper.appendChild(checkboxContainer);
    wrapper.appendChild(labelText);

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
    wrapper.className = 'minu-tabs';

    // Tab Header
    const tabList = document.createElement('div');
    tabList.className = 'minu-tabs-nav';
    tabList.style.display = 'flex';
    tabList.style.gap = 'var(--space-1)';
    tabList.style.borderBottom = 'var(--border-width) solid var(--border-default)';
    tabList.style.marginBottom = 'var(--space-4)';
    tabList.style.overflowX = 'auto';
    tabList.style.scrollbarWidth = 'none';
    tabList.setAttribute('role', 'tablist');

    // Content Area
    const contentArea = document.createElement('div');
    contentArea.className = 'minu-tabs-panel';

    const activeIndex = comp.props?.active ?? 0;

    if (comp.props?.tabs) {
        comp.props.tabs.forEach((tab, index) => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.setAttribute('role', 'tab');
            btn.setAttribute('aria-selected', index === activeIndex);
            btn.dataset.tabIndex = index;

            // Styling
            btn.style.display = 'inline-flex';
            btn.style.alignItems = 'center';
            btn.style.gap = 'var(--space-2)';
            btn.style.padding = 'var(--space-3) var(--space-4)';
            btn.style.border = 'none';
            btn.style.background = 'transparent';
            btn.style.cursor = 'pointer';
            btn.style.fontSize = 'var(--text-sm)';
            btn.style.fontWeight = 'var(--weight-medium)';
            btn.style.whiteSpace = 'nowrap';
            btn.style.borderBottom = '3px solid transparent';
            btn.style.marginBottom = '-1px';
            btn.style.transition = 'all var(--duration-fast) var(--ease-default)';
            btn.style.color = index === activeIndex
                ? 'var(--interactive)'
                : 'var(--text-secondary)';
            btn.style.borderBottomColor = index === activeIndex
                ? 'var(--interactive)'
                : 'transparent';

            btn.textContent = tab.label || `Tab ${index + 1}`;

            // Hover
            btn.addEventListener('mouseenter', () => {
                if (!btn.classList.contains('active')) {
                    btn.style.color = 'var(--text-primary)';
                    btn.style.background = 'var(--interactive-muted)';
                }
            });
            btn.addEventListener('mouseleave', () => {
                if (!btn.classList.contains('active')) {
                    btn.style.color = 'var(--text-secondary)';
                    btn.style.background = 'transparent';
                }
            });

            // Click Handler
            btn.addEventListener('click', (e) => {
                e.preventDefault();

                // Update alle Buttons
                tabList.querySelectorAll('button').forEach((b, i) => {
                    const isActive = i === index;
                    b.style.color = isActive ? 'var(--interactive)' : 'var(--text-secondary)';
                    b.style.borderBottomColor = isActive ? 'var(--interactive)' : 'transparent';
                    b.style.background = 'transparent';
                    b.setAttribute('aria-selected', isActive);
                    if (isActive) b.classList.add('active');
                    else b.classList.remove('active');
                });

                // Content wechseln
                contentArea.innerHTML = '';
                if (comp.props.tabs[index]?.content) {
                    const content = this._renderComponent(comp.props.tabs[index].content, viewId);
                    contentArea.appendChild(content);
                }

                // Server notify
                if (comp.events?.change) {
                    this._triggerEvent(viewId, comp.events.change, { index, tab: tab.label });
                }
            });

            if (index === activeIndex) btn.classList.add('active');
            tabList.appendChild(btn);
        });

        // Initial Content
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

    _injectStyles() {
    if (document.getElementById('minu-core-styles')) return;

    const style = document.createElement('style');
    style.id = 'minu-core-styles';
    style.textContent = `
        /* Tabs */
        .minu-tabs { width: 100%; }
        .minu-tabs-header {
            display: flex;
            gap: 0.25rem;
            border-bottom: 2px solid #e5e7eb;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }
        .minu-tab-btn {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem 1rem;
            border: none;
            background: none;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            margin-bottom: -2px;
            white-space: nowrap;
            transition: all 0.2s;
            font-size: 0.875rem;
        }
        .minu-tab-btn:hover { background: #f9fafb; }
        .minu-tab-btn.active {
            border-bottom-color: #3b82f6;
            color: #3b82f6;
            font-weight: 600;
        }
        .minu-tabs-content { padding: 1rem 0; }

        /* Checkbox */
        .minu-checkbox {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            cursor: pointer;
            user-select: none;
        }
        .minu-checkbox-input {
            width: 1.125rem;
            height: 1.125rem;
            accent-color: #3b82f6;
        }

        /* Buttons */
        .btn {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            border: none;
            cursor: pointer;
            font-size: 0.875rem;
            font-weight: 500;
            transition: all 0.15s;
        }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-primary { background: #3b82f6; color: white; }
        .btn-primary:hover:not(:disabled) { background: #2563eb; }
        .btn-secondary { background: #e5e7eb; color: #374151; }
        .btn-secondary:hover:not(:disabled) { background: #d1d5db; }
        .btn-ghost { background: transparent; }
        .btn-ghost:hover:not(:disabled) { background: #f3f4f6; }
        .btn-icon { font-size: 1.125rem; }

        /* Mobile responsiveness */
        @media (max-width: 640px) {
            .minu-tabs-header { gap: 0; }
            .minu-tab-btn { padding: 0.625rem 0.75rem; font-size: 0.8125rem; }
            .btn { padding: 0.625rem 0.875rem; }
        }
    `;
    document.head.appendChild(style);
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
