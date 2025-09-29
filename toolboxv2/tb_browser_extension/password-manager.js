// ToolBox Browser Extension - Password Manager UI
// Advanced password management with import, 2FA, and auto-fill capabilities

class TBPasswordManager {
    constructor() {
        this.passwords = new Map();
        this.currentEntry = null;
        this.importFormats = [
            'chrome', 'firefox', 'lastpass', 'bitwarden', '1password', 'csv', 'json'
        ];
        this.init();
    }

    async init() {
        try {
            await this.loadPasswords();
            this.setupEventListeners();
            TBUtils.info('PasswordManager', 'Password manager initialized');
        } catch (error) {
            TBUtils.handleError('PasswordManager', error);
        }
    }

    async loadPasswords() {
        try {
            const response = await chrome.runtime.sendMessage({
                type: 'TB_API_REQUEST',
                module: 'PasswordManager',
                function: 'list_passwords',
                args: { limit: 1000 }
            });

            if (response.success) {
                this.passwords.clear();
                response.data.forEach(entry => {
                    this.passwords.set(entry.id, entry);
                });
            }
        } catch (error) {
            TBUtils.error('PasswordManager', 'Failed to load passwords', error);
        }
    }

    setupEventListeners() {
        // Listen for password manager requests
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            if (message.type === 'TB_PASSWORD_MANAGER') {
                this.handlePasswordRequest(message, sendResponse);
                return true;
            }
        });

        // Auto-fill detection
        this.setupAutoFillDetection();
    }

    setupAutoFillDetection() {
        // Detect login forms
        const forms = document.querySelectorAll('form');
        forms.forEach(form => {
            const passwordField = form.querySelector('input[type="password"]');
            const usernameField = form.querySelector('input[type="email"], input[type="text"]');

            if (passwordField && usernameField) {
                this.addAutoFillButton(form, usernameField, passwordField);
            }
        });

        // Watch for dynamically added forms
        const observer = new MutationObserver(mutations => {
            mutations.forEach(mutation => {
                mutation.addedNodes.forEach(node => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        const forms = node.querySelectorAll ? node.querySelectorAll('form') : [];
                        forms.forEach(form => {
                            const passwordField = form.querySelector('input[type="password"]');
                            const usernameField = form.querySelector('input[type="email"], input[type="text"]');

                            if (passwordField && usernameField) {
                                this.addAutoFillButton(form, usernameField, passwordField);
                            }
                        });
                    }
                });
            });
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    addAutoFillButton(form, usernameField, passwordField) {
        // Check if button already exists
        if (form.querySelector('.tb-autofill-btn')) return;

        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'tb-autofill-btn';
        button.innerHTML = '🔐';
        button.title = 'ToolBox Auto-fill';
        button.style.cssText = `
            position: absolute;
            right: 5px;
            top: 50%;
            transform: translateY(-50%);
            background: #667eea;
            color: white;
            border: none;
            border-radius: 3px;
            width: 24px;
            height: 24px;
            cursor: pointer;
            font-size: 12px;
            z-index: 10000;
        `;

        // Position relative to username field
        const rect = usernameField.getBoundingClientRect();
        usernameField.style.position = 'relative';
        usernameField.parentNode.style.position = 'relative';
        usernameField.parentNode.appendChild(button);

        button.addEventListener('click', async (e) => {
            e.preventDefault();
            await this.showAutoFillOptions(usernameField, passwordField);
        });
    }

    async showAutoFillOptions(usernameField, passwordField) {
        try {
            const currentUrl = window.location.href;
            const response = await chrome.runtime.sendMessage({
                type: 'TB_API_REQUEST',
                module: 'PasswordManager',
                function: 'get_password_for_autofill',
                args: { url: currentUrl }
            });

            if (response.success && response.data.entry) {
                const entry = response.data.entry;
                const totpCode = response.data.totp_code;

                // Fill the form
                usernameField.value = entry.username;
                passwordField.value = entry.password;

                // Trigger change events
                usernameField.dispatchEvent(new Event('input', { bubbles: true }));
                passwordField.dispatchEvent(new Event('input', { bubbles: true }));

                // Show TOTP code if available
                if (totpCode) {
                    this.showTOTPCode(totpCode, entry.totp_issuer || entry.title);
                }

                // Show alternatives if available
                if (response.data.alternatives && response.data.alternatives.length > 0) {
                    this.showAlternatives(response.data.alternatives, usernameField, passwordField);
                }

                TBUtils.info('PasswordManager', 'Auto-fill completed');
            } else {
                this.showPasswordSearch(usernameField, passwordField);
            }
        } catch (error) {
            TBUtils.error('PasswordManager', 'Auto-fill failed', error);
        }
    }

    showTOTPCode(code, issuer) {
        // Remove existing TOTP display
        const existing = document.querySelector('.tb-totp-display');
        if (existing) existing.remove();

        const totpDiv = document.createElement('div');
        totpDiv.className = 'tb-totp-display';
        totpDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #4CAF50;
            color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 10001;
            font-family: monospace;
            font-size: 16px;
            min-width: 200px;
            text-align: center;
        `;

        totpDiv.innerHTML = `
            <div style="font-size: 12px; margin-bottom: 5px;">${issuer}</div>
            <div style="font-size: 24px; font-weight: bold; letter-spacing: 2px;">${code}</div>
            <div style="font-size: 10px; margin-top: 5px;">Click to copy</div>
        `;

        totpDiv.addEventListener('click', () => {
            navigator.clipboard.writeText(code);
            totpDiv.style.background = '#2196F3';
            totpDiv.querySelector('div:last-child').textContent = 'Copied!';
            setTimeout(() => totpDiv.remove(), 2000);
        });

        document.body.appendChild(totpDiv);

        // Auto-remove after 30 seconds
        setTimeout(() => {
            if (totpDiv.parentNode) totpDiv.remove();
        }, 30000);
    }

    showAlternatives(alternatives, usernameField, passwordField) {
        // Remove existing alternatives
        const existing = document.querySelector('.tb-alternatives');
        if (existing) existing.remove();

        const altDiv = document.createElement('div');
        altDiv.className = 'tb-alternatives';
        altDiv.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            z-index: 10001;
            max-width: 300px;
            max-height: 400px;
            overflow-y: auto;
        `;

        const header = document.createElement('div');
        header.style.cssText = `
            padding: 10px;
            background: #f5f5f5;
            border-bottom: 1px solid #ddd;
            font-weight: bold;
            font-size: 14px;
        `;
        header.textContent = 'Alternative Accounts';
        altDiv.appendChild(header);

        alternatives.forEach(entry => {
            const item = document.createElement('div');
            item.style.cssText = `
                padding: 10px;
                border-bottom: 1px solid #eee;
                cursor: pointer;
                font-size: 13px;
            `;
            item.innerHTML = `
                <div style="font-weight: bold;">${entry.title}</div>
                <div style="color: #666;">${entry.username}</div>
            `;

            item.addEventListener('click', () => {
                usernameField.value = entry.username;
                passwordField.value = entry.password;
                usernameField.dispatchEvent(new Event('input', { bubbles: true }));
                passwordField.dispatchEvent(new Event('input', { bubbles: true }));
                altDiv.remove();
            });

            altDiv.appendChild(item);
        });

        // Close button
        const closeBtn = document.createElement('div');
        closeBtn.style.cssText = `
            position: absolute;
            top: 5px;
            right: 10px;
            cursor: pointer;
            font-size: 18px;
            color: #999;
        `;
        closeBtn.textContent = '×';
        closeBtn.addEventListener('click', () => altDiv.remove());
        altDiv.appendChild(closeBtn);

        document.body.appendChild(altDiv);

        // Auto-remove after 30 seconds
        setTimeout(() => {
            if (altDiv.parentNode) altDiv.remove();
        }, 30000);
    }

    showPasswordSearch(usernameField, passwordField) {
        // Create search modal
        const modal = this.createModal('Search Passwords');

        const searchInput = document.createElement('input');
        searchInput.type = 'text';
        searchInput.placeholder = 'Search passwords...';
        searchInput.style.cssText = `
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-bottom: 15px;
            font-size: 14px;
        `;

        const resultsList = document.createElement('div');
        resultsList.style.cssText = `
            max-height: 300px;
            overflow-y: auto;
        `;

        modal.content.appendChild(searchInput);
        modal.content.appendChild(resultsList);

        let searchTimeout;
        searchInput.addEventListener('input', () => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(async () => {
                await this.performPasswordSearch(searchInput.value, resultsList, usernameField, passwordField, modal);
            }, 300);
        });

        // Initial search with current domain
        const domain = window.location.hostname;
        searchInput.value = domain;
        this.performPasswordSearch(domain, resultsList, usernameField, passwordField, modal);
    }

    async performPasswordSearch(query, resultsList, usernameField, passwordField, modal) {
        try {
            const response = await chrome.runtime.sendMessage({
                type: 'TB_API_REQUEST',
                module: 'PasswordManager',
                function: 'search_passwords',
                args: { query, limit: 20 }
            });

            resultsList.innerHTML = '';

            if (response.success && response.data.length > 0) {
                response.data.forEach(entry => {
                    const item = document.createElement('div');
                    item.style.cssText = `
                        padding: 12px;
                        border-bottom: 1px solid #eee;
                        cursor: pointer;
                        transition: background-color 0.2s;
                    `;
                    item.innerHTML = `
                        <div style="font-weight: bold; margin-bottom: 4px;">${entry.title}</div>
                        <div style="color: #666; font-size: 12px;">${entry.username} • ${entry.url}</div>
                    `;

                    item.addEventListener('mouseenter', () => {
                        item.style.backgroundColor = '#f5f5f5';
                    });
                    item.addEventListener('mouseleave', () => {
                        item.style.backgroundColor = '';
                    });

                    item.addEventListener('click', async () => {
                        // Get full entry with password
                        const fullResponse = await chrome.runtime.sendMessage({
                            type: 'TB_API_REQUEST',
                            module: 'PasswordManager',
                            function: 'get_password',
                            args: { entry_id: entry.id }
                        });

                        if (fullResponse.success) {
                            const fullEntry = fullResponse.data;
                            usernameField.value = fullEntry.username;
                            passwordField.value = fullEntry.password;
                            usernameField.dispatchEvent(new Event('input', { bubbles: true }));
                            passwordField.dispatchEvent(new Event('input', { bubbles: true }));

                            // Generate TOTP if available
                            if (fullEntry.totp_secret) {
                                const totpResponse = await chrome.runtime.sendMessage({
                                    type: 'TB_API_REQUEST',
                                    module: 'PasswordManager',
                                    function: 'generate_totp_code',
                                    args: { entry_id: entry.id }
                                });

                                if (totpResponse.success) {
                                    this.showTOTPCode(totpResponse.data.code, totpResponse.data.issuer);
                                }
                            }
                        }

                        modal.remove();
                    });

                    resultsList.appendChild(item);
                });
            } else {
                resultsList.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">No passwords found</div>';
            }
        } catch (error) {
            resultsList.innerHTML = '<div style="padding: 20px; text-align: center; color: #f44336;">Search failed</div>';
            TBUtils.error('PasswordManager', 'Password search failed', error);
        }
    }

    createModal(title) {
        const overlay = document.createElement('div');
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 10002;
            display: flex;
            align-items: center;
            justify-content: center;
        `;

        const modal = document.createElement('div');
        modal.style.cssText = `
            background: white;
            border-radius: 8px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            max-width: 500px;
            width: 90%;
            max-height: 80%;
            overflow: hidden;
        `;

        const header = document.createElement('div');
        header.style.cssText = `
            padding: 15px 20px;
            background: #667eea;
            color: white;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        `;
        header.innerHTML = `
            <span>${title}</span>
            <span style="cursor: pointer; font-size: 20px;">×</span>
        `;

        const content = document.createElement('div');
        content.style.cssText = `
            padding: 20px;
        `;

        modal.appendChild(header);
        modal.appendChild(content);
        overlay.appendChild(modal);

        // Close handlers
        header.querySelector('span:last-child').addEventListener('click', () => overlay.remove());
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) overlay.remove();
        });

        document.body.appendChild(overlay);

        return { modal: overlay, content, remove: () => overlay.remove() };
    }

    showImportDialog() {
        const modal = this.createModal('Import Passwords');

        modal.content.innerHTML = `
            <div style="margin-bottom: 20px;">
                <label style="display: block; margin-bottom: 8px; font-weight: bold;">Import Format:</label>
                <select id="tb-import-format" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                    <option value="chrome">Chrome CSV</option>
                    <option value="firefox">Firefox CSV</option>
                    <option value="lastpass">LastPass CSV</option>
                    <option value="bitwarden">Bitwarden JSON</option>
                    <option value="1password">1Password CSV</option>
                    <option value="csv">Generic CSV</option>
                    <option value="json">Generic JSON</option>
                </select>
            </div>

            <div style="margin-bottom: 20px;">
                <label style="display: block; margin-bottom: 8px; font-weight: bold;">Folder Name:</label>
                <input type="text" id="tb-import-folder" value="Imported"
                       style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
            </div>

            <div style="margin-bottom: 20px;">
                <label style="display: block; margin-bottom: 8px; font-weight: bold;">Select File:</label>
                <input type="file" id="tb-import-file" accept=".csv,.json,.txt"
                       style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
            </div>

            <div style="margin-bottom: 20px;">
                <label style="display: block; margin-bottom: 8px; font-weight: bold;">Or Paste Content:</label>
                <textarea id="tb-import-content" rows="8" placeholder="Paste your exported password data here..."
                          style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; resize: vertical;"></textarea>
            </div>

            <div style="display: flex; gap: 10px; justify-content: flex-end;">
                <button id="tb-import-cancel" style="padding: 10px 20px; border: 1px solid #ddd; background: white; border-radius: 4px; cursor: pointer;">
                    Cancel
                </button>
                <button id="tb-import-start" style="padding: 10px 20px; border: none; background: #667eea; color: white; border-radius: 4px; cursor: pointer;">
                    Import Passwords
                </button>
            </div>

            <div id="tb-import-progress" style="margin-top: 20px; display: none;">
                <div style="background: #f5f5f5; border-radius: 4px; padding: 15px;">
                    <div style="font-weight: bold; margin-bottom: 10px;">Import Progress</div>
                    <div id="tb-import-status">Starting import...</div>
                    <div style="margin-top: 10px;">
                        <div style="background: #ddd; height: 6px; border-radius: 3px;">
                            <div id="tb-import-bar" style="background: #4CAF50; height: 100%; width: 0%; border-radius: 3px; transition: width 0.3s;"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // File input handler
        const fileInput = modal.content.querySelector('#tb-import-file');
        const contentTextarea = modal.content.querySelector('#tb-import-content');

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    contentTextarea.value = e.target.result;
                };
                reader.readAsText(file);
            }
        });

        // Cancel button
        modal.content.querySelector('#tb-import-cancel').addEventListener('click', () => {
            modal.remove();
        });

        // Import button
        modal.content.querySelector('#tb-import-start').addEventListener('click', async () => {
            await this.performImport(modal);
        });
    }

    async performImport(modal) {
        const format = modal.content.querySelector('#tb-import-format').value;
        const folder = modal.content.querySelector('#tb-import-folder').value || 'Imported';
        const content = modal.content.querySelector('#tb-import-content').value;

        if (!content.trim()) {
            alert('Please select a file or paste content to import');
            return;
        }

        const progressDiv = modal.content.querySelector('#tb-import-progress');
        const statusDiv = modal.content.querySelector('#tb-import-status');
        const progressBar = modal.content.querySelector('#tb-import-bar');

        progressDiv.style.display = 'block';
        statusDiv.textContent = 'Starting import...';
        progressBar.style.width = '10%';

        try {
            const response = await chrome.runtime.sendMessage({
                type: 'TB_API_REQUEST',
                module: 'PasswordManager',
                function: 'import_passwords',
                args: {
                    file_content: content,
                    file_format: format,
                    folder: folder
                }
            });

            progressBar.style.width = '100%';

            if (response.success) {
                const result = response.data;
                statusDiv.innerHTML = `
                    <div style="color: #4CAF50; font-weight: bold;">Import Completed!</div>
                    <div style="margin-top: 8px; font-size: 13px;">
                        ✅ ${result.imported_count} passwords imported<br>
                        ⚠️ ${result.skipped_count} entries skipped<br>
                        ❌ ${result.error_count} errors
                    </div>
                `;

                if (result.errors && result.errors.length > 0) {
                    const errorDetails = document.createElement('details');
                    errorDetails.style.marginTop = '10px';
                    errorDetails.innerHTML = `
                        <summary style="cursor: pointer; color: #f44336;">View Errors (${result.errors.length})</summary>
                        <div style="margin-top: 8px; font-size: 12px; max-height: 100px; overflow-y: auto;">
                            ${result.errors.map(err => `<div>• ${err}</div>`).join('')}
                        </div>
                    `;
                    statusDiv.appendChild(errorDetails);
                }

                // Reload passwords
                await this.loadPasswords();

                // Auto-close after 5 seconds
                setTimeout(() => modal.remove(), 5000);
            } else {
                statusDiv.innerHTML = `
                    <div style="color: #f44336; font-weight: bold;">Import Failed</div>
                    <div style="margin-top: 8px; font-size: 13px;">${response.error || 'Unknown error'}</div>
                `;
            }
        } catch (error) {
            progressBar.style.width = '100%';
            progressBar.style.background = '#f44336';
            statusDiv.innerHTML = `
                <div style="color: #f44336; font-weight: bold;">Import Error</div>
                <div style="margin-top: 8px; font-size: 13px;">${error.message}</div>
            `;
        }
    }

    showPasswordList() {
        const modal = this.createModal('Password Manager');
        modal.modal.style.maxWidth = '800px';

        modal.content.innerHTML = `
            <div style="display: flex; gap: 15px; margin-bottom: 20px;">
                <input type="text" id="tb-password-search" placeholder="Search passwords..."
                       style="flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 4px;">
                <button id="tb-add-password" style="padding: 10px 15px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    Add Password
                </button>
                <button id="tb-import-passwords" style="padding: 10px 15px; background: #2196F3; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    Import
                </button>
            </div>

            <div id="tb-password-list" style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px;">
                <div style="padding: 20px; text-align: center; color: #999;">Loading passwords...</div>
            </div>
        `;

        // Search functionality
        const searchInput = modal.content.querySelector('#tb-password-search');
        let searchTimeout;
        searchInput.addEventListener('input', () => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                this.filterPasswordList(searchInput.value, modal.content.querySelector('#tb-password-list'));
            }, 300);
        });

        // Button handlers
        modal.content.querySelector('#tb-add-password').addEventListener('click', () => {
            this.showAddPasswordDialog();
        });

        modal.content.querySelector('#tb-import-passwords').addEventListener('click', () => {
            modal.remove();
            this.showImportDialog();
        });

        // Load and display passwords
        this.displayPasswordList(modal.content.querySelector('#tb-password-list'));
    }

    async displayPasswordList(container, filter = '') {
        try {
            const response = await chrome.runtime.sendMessage({
                type: 'TB_API_REQUEST',
                module: 'PasswordManager',
                function: 'list_passwords',
                args: { limit: 1000 }
            });

            if (response.success) {
                const passwords = response.data.filter(entry => {
                    if (!filter) return true;
                    const searchText = `${entry.title} ${entry.username} ${entry.url}`.toLowerCase();
                    return searchText.includes(filter.toLowerCase());
                });

                if (passwords.length === 0) {
                    container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">No passwords found</div>';
                    return;
                }

                container.innerHTML = passwords.map(entry => `
                    <div class="tb-password-item" data-id="${entry.id}" style="
                        padding: 15px;
                        border-bottom: 1px solid #eee;
                        cursor: pointer;
                        transition: background-color 0.2s;
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="flex: 1;">
                                <div style="font-weight: bold; margin-bottom: 4px;">${entry.title}</div>
                                <div style="color: #666; font-size: 13px;">${entry.username} • ${entry.url}</div>
                                <div style="color: #999; font-size: 11px; margin-top: 2px;">
                                    ${entry.folder} • Updated: ${new Date(entry.updated_at * 1000).toLocaleDateString()}
                                </div>
                            </div>
                            <div style="display: flex; gap: 8px;">
                                ${entry.totp_secret ? '<span title="2FA Available" style="color: #4CAF50;">🔐</span>' : ''}
                                <button class="tb-copy-password" data-id="${entry.id}" style="
                                    padding: 4px 8px;
                                    background: #667eea;
                                    color: white;
                                    border: none;
                                    border-radius: 3px;
                                    cursor: pointer;
                                    font-size: 11px;
                                ">Copy</button>
                                <button class="tb-edit-password" data-id="${entry.id}" style="
                                    padding: 4px 8px;
                                    background: #FF9800;
                                    color: white;
                                    border: none;
                                    border-radius: 3px;
                                    cursor: pointer;
                                    font-size: 11px;
                                ">Edit</button>
                            </div>
                        </div>
                    </div>
                `).join('');

                // Add event listeners
                container.querySelectorAll('.tb-password-item').forEach(item => {
                    item.addEventListener('mouseenter', () => {
                        item.style.backgroundColor = '#f5f5f5';
                    });
                    item.addEventListener('mouseleave', () => {
                        item.style.backgroundColor = '';
                    });
                });

                container.querySelectorAll('.tb-copy-password').forEach(btn => {
                    btn.addEventListener('click', async (e) => {
                        e.stopPropagation();
                        await this.copyPassword(btn.dataset.id);
                    });
                });

                container.querySelectorAll('.tb-edit-password').forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        this.showEditPasswordDialog(btn.dataset.id);
                    });
                });
            }
        } catch (error) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #f44336;">Failed to load passwords</div>';
            TBUtils.error('PasswordManager', 'Failed to display password list', error);
        }
    }

    async copyPassword(entryId) {
        try {
            const response = await chrome.runtime.sendMessage({
                type: 'TB_API_REQUEST',
                module: 'PasswordManager',
                function: 'get_password',
                args: { entry_id: entryId }
            });

            if (response.success) {
                await navigator.clipboard.writeText(response.data.password);
                TBUtils.info('PasswordManager', 'Password copied to clipboard');

                // Show temporary feedback
                const btn = document.querySelector(`[data-id="${entryId}"].tb-copy-password`);
                if (btn) {
                    const originalText = btn.textContent;
                    btn.textContent = 'Copied!';
                    btn.style.background = '#4CAF50';
                    setTimeout(() => {
                        btn.textContent = originalText;
                        btn.style.background = '#667eea';
                    }, 2000);
                }
            }
        } catch (error) {
            TBUtils.error('PasswordManager', 'Failed to copy password', error);
        }
    }

    filterPasswordList(filter, container) {
        this.displayPasswordList(container, filter);
    }

    async handlePasswordRequest(message, sendResponse) {
        try {
            switch (message.action) {
                case 'show_import_dialog':
                    this.showImportDialog();
                    break;
                case 'show_password_list':
                    this.showPasswordList();
                    break;
                case 'generate_password':
                    await this.showPasswordGenerator();
                    break;
                default:
                    TBUtils.warn('PasswordManager', `Unknown action: ${message.action}`);
            }
            sendResponse({ success: true });
        } catch (error) {
            TBUtils.error('PasswordManager', 'Request handling failed', error);
            sendResponse({ success: false, error: error.message });
        }
    }
}

// Initialize password manager
const tbPasswordManager = new TBPasswordManager();
