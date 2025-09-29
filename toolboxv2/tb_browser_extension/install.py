#!/usr/bin/env python3
"""
ToolBox Browser Extension Auto-Installer
Automatically installs and configures the extension
"""

import os
import sys
import json
import shutil
import subprocess
import webbrowser
from pathlib import Path
import tempfile
import zipfile


def cah_save(text:str):
    text = text.encode(sys.stdout.encoding or 'utf-8', 'replace').decode(sys.stdout.encoding or 'utf-8')
    return text

class TBExtensionInstaller:
    def __init__(self):
        self.extension_dir = Path(__file__).parent
        self.build_dir = self.extension_dir / "build"
        self.cli_auth_server = None

    def build_extension(self):
        """Build the extension for distribution"""
        print("🔨 Building ToolBox Browser Extension...")

        # Create build directory
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        self.build_dir.mkdir()

        # Copy extension files
        files_to_copy = [
            'manifest.json',
            'background.js',
            'content.js',
            'popup.html',
            'popup.js',
            'styles.css',
            "settings.js",
            "settings.html",
            'password-manager.js'
        ]

        for file in files_to_copy:
            src = self.extension_dir / file
            if src.exists():
                shutil.copy2(src, self.build_dir / file)
                print(f"  ✅ Copied {file}")
            else:
                print(f"  ⚠️  Missing {file}")

        # Create additional directories
        self.create_directories()

        # Generate icons
        self.generate_icons()

        # Create plugin interface files
        self.create_plugin_interface()

        # Update manifest with correct paths
        self.update_manifest()

        print("✅ Extension built successfully!")
        return self.build_dir

    def create_directories(self):
        """Create necessary directories"""
        directories = ['icons', 'plugins', 'assets', 'core']
        for dir_name in directories:
            (self.build_dir / dir_name).mkdir(exist_ok=True)

        # Copy core files
        core_src = self.extension_dir / 'core'
        core_dst = self.build_dir / 'core'
        if core_src.exists():
            shutil.copytree(core_src, core_dst, dirs_exist_ok=True)
            print("  ✅ Copied core directory")

    def generate_icons(self):
        """Generate extension icons"""
        print("  🎨 Generating icons...")

        # Create SVG icon template
        svg_template = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}">
            <defs>
                <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
                </linearGradient>
            </defs>
            <rect width="{size}" height="{size}" fill="url(#grad)" rx="{radius}"/>
            <text x="{center}" y="{text_y}" font-family="Arial" font-size="{font_size}"
                  fill="white" text-anchor="middle" dominant-baseline="middle">🧰</text>
        </svg>'''

        sizes = [16, 32, 48, 128]
        for size in sizes:
            # coppy FAVI to
            from toolboxv2 import tb_root_dir
            PATH_TO_FAVISVG = tb_root_dir /"simple-core"/"src-tauri"/"icons"/"svg"/f"favicon{size}.svg"

            shutil.copy2(PATH_TO_FAVISVG, self.build_dir / "icons" / f"tb{size}.svg")

            PATH_TO_FAVPNG = tb_root_dir /"simple-core"/"src-tauri"/"icons"/f"{size}x{size}.png"

            shutil.copy2(PATH_TO_FAVPNG, self.build_dir / "icons" / f"tb{size}.png")


        print("  ✅ Generated icons")

    def create_plugin_interface(self):
        """Create plugin interface system"""
        print("  🔌 Creating plugin interface...")

        plugin_manager = r'''
// ToolBox Plugin Manager
class TBPluginManager {
    constructor() {
        this.plugins = new Map();
        this.hooks = new Map();
    }

    async loadPlugin(pluginId, pluginCode) {
        try {
            const plugin = new Function('TB', 'chrome', pluginCode);
            const instance = plugin(window.TB, chrome);

            this.plugins.set(pluginId, instance);

            // Register plugin hooks
            if (instance.hooks) {
                Object.entries(instance.hooks).forEach(([hook, handler]) => {
                    if (!this.hooks.has(hook)) {
                        this.hooks.set(hook, []);
                    }
                    this.hooks.get(hook).push(handler);
                });
            }

            console.log(`✅ Plugin ${pluginId} loaded`);
            return true;
        } catch (error) {
            console.error(`❌ Failed to load plugin ${pluginId}:`, error);
            return false;
        }
    }

    async executeHook(hookName, data) {
        const handlers = this.hooks.get(hookName) || [];
        const results = [];

        for (const handler of handlers) {
            try {
                const result = await handler(data);
                results.push(result);
            } catch (error) {
                console.error(`Hook ${hookName} failed:`, error);
            }
        }

        return results;
    }

    getPlugin(pluginId) {
        return this.plugins.get(pluginId);
    }

    listPlugins() {
        return Array.from(this.plugins.keys());
    }
}

// Initialize plugin manager
window.TBPluginManager = new TBPluginManager();
        '''

        (self.build_dir / "plugins" / "plugin-manager.js").write_text(cah_save(plugin_manager), encoding=sys.stdout.encoding or 'utf-8')

        # Create sample plugins
        self.create_sample_plugins()

        print("  ✅ Plugin interface created")

    def create_sample_plugins(self):
        """Create sample plugins"""

        # AI Plugin
        ai_plugin = r'''
// AI Analysis Plugin
(function(TB, chrome) {
    return {
        name: 'AI Analyzer',
        version: '1.0.0',
        description: 'Advanced AI page analysis',

        hooks: {
            'page_loaded': async (data) => {
                // Auto-analyze page if enabled
                const settings = await chrome.storage.sync.get('ai_auto_analyze');
                if (settings.ai_auto_analyze) {
                    return await this.analyzePage(data);
                }
            },

            'context_menu': (data) => {
                return {
                    id: 'ai-deep-analysis',
                    title: '🤖 Deep AI Analysis',
                    action: () => this.deepAnalysis(data)
                };
            }
        },

        async analyzePage(pageData) {
            try {
                const response = await TB.api.request('AIAnalyzer', 'analyze_page', {
                    content: pageData.content,
                    url: pageData.url,
                    deep_analysis: true
                });

                return response.data;
            } catch (error) {
                console.error('AI analysis failed:', error);
                return null;
            }
        },

        async deepAnalysis(data) {
            const analysis = await this.analyzePage(data);
            if (analysis) {
                TB.ui.showModal('AI Deep Analysis', this.formatAnalysis(analysis));
            }
        },

        formatAnalysis(analysis) {
            return `
                <div class="ai-analysis">
                    <h4>📊 Content Analysis</h4>
                    <p><strong>Sentiment:</strong> ${analysis.sentiment}</p>
                    <p><strong>Topics:</strong> ${analysis.topics.join(', ')}</p>
                    <p><strong>Readability:</strong> ${analysis.readability_score}/100</p>

                    <h4>🎯 SEO Analysis</h4>
                    <p><strong>Title Score:</strong> ${analysis.seo.title_score}/100</p>
                    <p><strong>Meta Description:</strong> ${analysis.seo.meta_score}/100</p>

                    <h4>💡 Recommendations</h4>
                    <ul>
                        ${analysis.recommendations.map(r => `<li>${r}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
    };
});
        '''

        (self.build_dir / "plugins" / "ai-analyzer.js").write_text(cah_save(ai_plugin), encoding=sys.stdout.encoding or 'utf-8')

        # Business Plugin
        business_plugin = r'''
// Business Tools Plugin
(function(TB, chrome) {
    return {
        name: 'Business Tools',
        version: '1.0.0',
        description: 'Business automation and analytics',

        hooks: {
            'page_loaded': async (data) => {
                // Auto-detect business opportunities
                return await this.detectBusinessOpportunities(data);
            },

            'context_menu': (data) => {
                return [
                    {
                        id: 'extract-contacts',
                        title: '📇 Extract Contacts',
                        action: () => this.extractContacts(data)
                    },
                    {
                        id: 'competitor-analysis',
                        title: '📈 Competitor Analysis',
                        action: () => this.analyzeCompetitor(data)
                    }
                ];
            }
        },

        async detectBusinessOpportunities(pageData) {
            const opportunities = [];

            // Check for contact information
            if (this.hasContactInfo(pageData.content)) {
                opportunities.push('contact_extraction');
            }

            // Check for pricing information
            if (this.hasPricingInfo(pageData.content)) {
                opportunities.push('price_tracking');
            }

            return opportunities;
        },

        async extractContacts(data) {
            try {
                const response = await TB.api.request('BusinessTools', 'extract_contacts', {
                    content: data.content,
                    url: data.url
                });

                if (response.data.contacts.length > 0) {
                    TB.ui.showModal('Extracted Contacts', this.formatContacts(response.data.contacts));
                } else {
                    TB.ui.showNotification('No contacts found on this page');
                }
            } catch (error) {
                TB.ui.showNotification('❌ Contact extraction failed');
            }
        },

        formatContacts(contacts) {
            return `
                <div class="contacts-list">
                    ${contacts.map(contact => `
                        <div class="contact-item">
                            <h5>${contact.name || 'Unknown'}</h5>
                            <p>📧 ${contact.email || 'N/A'}</p>
                            <p>📞 ${contact.phone || 'N/A'}</p>
                            <p>🏢 ${contact.company || 'N/A'}</p>
                            <button onclick="TB.business.saveContact(${JSON.stringify(contact)})">
                                Save to CRM
                            </button>
                        </div>
                    `).join('')}
                </div>
            `;
        },

        hasContactInfo(content) {
            const emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/;
            const phoneRegex = /(\+\d{1,3}[- ]?)?\d{10}/;
            return emailRegex.test(content) || phoneRegex.test(content);
        },

        hasPricingInfo(content) {
            const priceRegex = /\$\d+|\€\d+|£\d+/;
            return priceRegex.test(content);
        }
    };
});
        '''

        (self.build_dir / "plugins" / "business-tools.js").write_text(cah_save(business_plugin), encoding=sys.stdout.encoding or 'utf-8')

        # Password Manager Plugin
        password_manager_plugin = r'''
// Password Manager Plugin
(function(TB, chrome) {
    return {
        name: 'Password Manager',
        version: '1.0.0',
        description: 'Secure password management with 2FA support',

        hooks: {
            'page_loaded': async (data) => {
                // Auto-detect login forms
                return await this.detectLoginForms(data);
            },

            'context_menu': (data) => {
                return [
                    {
                        id: 'password-autofill',
                        title: '🔐 Auto-fill Password',
                        action: () => this.autoFillPassword(data)
                    },
                    {
                        id: 'password-generate',
                        title: '🔑 Generate Password',
                        action: () => this.generatePassword(data)
                    },
                    {
                        id: 'password-save',
                        title: '💾 Save Password',
                        action: () => this.savePassword(data)
                    },
                    {
                        id: 'password-manager',
                        title: '🔒 Password Manager',
                        action: () => this.openPasswordManager()
                    }
                ];
            },

            'form_submit': async (data) => {
                // Auto-save passwords on form submission
                if (this.isLoginForm(data.form)) {
                    return await this.promptSavePassword(data);
                }
            }
        },

        async detectLoginForms(pageData) {
            const forms = document.querySelectorAll('form');
            const loginForms = [];

            forms.forEach(form => {
                const passwordField = form.querySelector('input[type="password"]');
                const usernameField = form.querySelector('input[type="email"], input[type="text"]');

                if (passwordField && usernameField) {
                    loginForms.push({
                        form: form,
                        usernameField: usernameField,
                        passwordField: passwordField,
                        url: window.location.href
                    });

                    // Add auto-fill button
                    this.addAutoFillButton(form, usernameField, passwordField);
                }
            });

            return { loginForms: loginForms.length };
        },

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
            usernameField.style.position = 'relative';
            usernameField.parentNode.style.position = 'relative';
            usernameField.parentNode.appendChild(button);

            button.addEventListener('click', async (e) => {
                e.preventDefault();
                await this.autoFillPassword({ usernameField, passwordField });
            });
        },

        async autoFillPassword(data) {
            try {
                const response = await TB.api.request('PasswordManager', 'get_password_for_autofill', {
                    url: window.location.href
                });

                const result = response.get();
                if (result && result.entry) {
                    const entry = result.entry;

                    if (data.usernameField) data.usernameField.value = entry.username;
                    if (data.passwordField) data.passwordField.value = entry.password;

                    // Trigger change events
                    if (data.usernameField) data.usernameField.dispatchEvent(new Event('input', { bubbles: true }));
                    if (data.passwordField) data.passwordField.dispatchEvent(new Event('input', { bubbles: true }));

                    // Show TOTP if available
                    if (result.totp_code) {
                        this.showTOTPCode(result.totp_code, entry.totp_issuer || entry.title);
                    }

                    TB.ui.showNotification('✅ Password auto-filled');
                } else {
                    TB.ui.showNotification('❌ No matching password found');
                }
            } catch (error) {
                TB.ui.showNotification('❌ Auto-fill failed');
                console.error('Auto-fill error:', error);
            }
        },

        async generatePassword(data) {
            try {
                const response = await TB.api.request('PasswordManager', 'generate_password', {
                    length: 16,
                    include_symbols: true,
                    include_numbers: true,
                    include_uppercase: true,
                    include_lowercase: true
                });

                const result = response.get();
                if (result && result.password) {
                    const password = result.password;

                    // Find password field and fill it
                    const passwordField = document.querySelector('input[type="password"]');
                    if (passwordField) {
                        passwordField.value = password;
                        passwordField.dispatchEvent(new Event('input', { bubbles: true }));
                    }

                    // Copy to clipboard
                    await navigator.clipboard.writeText(password);
                    TB.ui.showNotification('🔑 Password generated and copied');
                }
            } catch (error) {
                TB.ui.showNotification('❌ Password generation failed');
                console.error('Password generation error:', error);
            }
        },

        async savePassword(data) {
            const usernameField = document.querySelector('input[type="email"], input[type="text"]');
            const passwordField = document.querySelector('input[type="password"]');

            if (!usernameField || !passwordField || !usernameField.value || !passwordField.value) {
                TB.ui.showNotification('❌ No login credentials found');
                return;
            }

            try {
                const response = await TB.api.request('PasswordManager', 'add_password', {
                    url: window.location.href,
                    username: usernameField.value,
                    password: passwordField.value,
                    title: document.title || window.location.hostname
                });

                const result = response.get();
                if (result) {
                    TB.ui.showNotification('✅ Password saved successfully');
                } else {
                    TB.ui.showNotification('❌ Failed to save password');
                }
            } catch (error) {
                TB.ui.showNotification('❌ Password save failed');
                console.error('Password save error:', error);
            }
        },

        openPasswordManager() {
            // Send message to content script to open password manager
            chrome.runtime.sendMessage({
                type: 'TB_PASSWORD_MANAGER',
                action: 'show_password_list'
            });
        },

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
                cursor: pointer;
            `;

            totpDiv.innerHTML = `
                <div style="font-size: 12px; margin-bottom: 5px;">${issuer}</div>
                <div style="font-size: 24px; font-weight: bold; letter-spacing: 2px;">${code}</div>
                <div style="font-size: 10px; margin-top: 5px;">Click to copy</div>
            `;

            totpDiv.addEventListener('click', async () => {
                await navigator.clipboard.writeText(code);
                totpDiv.style.background = '#2196F3';
                totpDiv.querySelector('div:last-child').textContent = 'Copied!';
                setTimeout(() => totpDiv.remove(), 2000);
            });

            document.body.appendChild(totpDiv);

            // Auto-remove after 30 seconds
            setTimeout(() => {
                if (totpDiv.parentNode) totpDiv.remove();
            }, 30000);
        },

        isLoginForm(form) {
            const passwordField = form.querySelector('input[type="password"]');
            const usernameField = form.querySelector('input[type="email"], input[type="text"]');
            return passwordField && usernameField;
        },

        async promptSavePassword(data) {
            const form = data.form;
            const usernameField = form.querySelector('input[type="email"], input[type="text"]');
            const passwordField = form.querySelector('input[type="password"]');

            if (usernameField && passwordField && usernameField.value && passwordField.value) {
                // Check if password already exists
                const existing = await TB.api.request('PasswordManager', 'get_password_for_autofill', {
                    url: window.location.href,
                    username: usernameField.value
                });

                const existingResult = existing.get();
                if (!existingResult || !existingResult.entry) {
                    // Show save prompt
                    const shouldSave = confirm(`Save password for ${usernameField.value} on ${window.location.hostname}?`);
                    if (shouldSave) {
                        await this.savePassword(data);
                    }
                }
            }
        }
    };
});
        '''

        (self.build_dir / "plugins" / "password-manager.js").write_text(cah_save(password_manager_plugin), encoding=sys.stdout.encoding or 'utf-8')

    def update_manifest(self):
        """Update manifest.json with build-specific settings"""
        manifest_path = self.build_dir / "manifest.json"

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Add development settings if needed
        if os.getenv('TB_DEV_MODE'):
            manifest['permissions'].append('http://localhost:8080/*')

        # Add settings page
        manifest['options_page'] = 'settings.html'

        # Add web accessible resources for plugins
        manifest['web_accessible_resources'][0]['resources'].extend([
            'plugins/*.js',
            'settings.html',
            'settings.js'
        ])

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    def install_chrome(self):
        """Install extension in Chrome (developer mode)"""
        print("🌐 Installing in Chrome...")

        try:
            # Open Chrome with extension loading
            if sys.platform == "win32":
                chrome_cmd = "C:\Program Files\Google\Chrome\Application\chrome.exe"
            elif sys.platform == "darwin":
                chrome_cmd = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
            else:
                chrome_cmd = "google-chrome"

            res = subprocess.run([
                chrome_cmd,
                f"--load-extension={self.build_dir}",
                "--no-first-run"
            ], check=True)

            print(res)
            if res.returncode != 0:
                raise Exception("Chrome installation failed")

            print("✅ Chrome extension loaded!")
            print("   Enable Developer Mode in chrome://extensions/ if needed")

        except Exception as e:
            print(f"❌ Chrome installation failed: {e}")
            print("   Manual installation:")
            print(f"   1. Open chrome://extensions/")
            print(f"   2. Enable Developer Mode")
            print(f"   3. Click 'Load unpacked' and select: {self.build_dir}")

    def install_firefox(self):
        """Install extension in Firefox (temporary)"""
        print("🦊 Installing in Firefox...")

        try:
            # Create web-ext config
            webext_config = {
                "sourceDir": str(self.build_dir),
                "artifactsDir": str(self.extension_dir / "artifacts"),
                "build": {
                    "overwriteDest": True
                }
            }

            config_path = self.build_dir / "web-ext-config.json"
            with open(config_path, 'w') as f:
                json.dump(webext_config, f, indent=2)

            # Try to use web-ext if available
            try:
                subprocess.run([
                    "web-ext", "run",
                    "--source-dir", str(self.build_dir),
                    "--firefox-profile", "dev-edition-default"
                ], check=True)
                print("✅ Firefox extension loaded!")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("❌ web-ext not found. Manual installation required:")
                print("   1. Open Firefox")
                print("   2. Go to about:debugging")
                print("   3. Click 'This Firefox'")
                print("   4. Click 'Load Temporary Add-on'")
                print(f"   5. Select manifest.json from: {self.build_dir}")

                # Open Firefox debugging page
                webbrowser.open("about:debugging#/runtime/this-firefox")

        except Exception as e:
            print(f"❌ Firefox installation failed: {e}")

    def install_edge(self):
        """Install extension in Edge"""
        print("🌊 Installing in Edge...")

        try:
            # Open Edge with extension loading
            if sys.platform == "win32":
                edge_cmd = "msedge.exe"
            elif sys.platform == "darwin":
                edge_cmd = "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
            else:
                edge_cmd = "microsoft-edge"

            subprocess.run([
                edge_cmd,
                f"--load-extension={self.build_dir}",
                "--no-first-run"
            ], check=False)

            print("✅ Edge extension loaded!")
            print("   Enable Developer Mode in edge://extensions/ if needed")

        except Exception as e:
            print(f"❌ Edge installation failed: {e}")
            print("   Manual installation:")
            print(f"   1. Open edge://extensions/")
            print(f"   2. Enable Developer Mode")
            print(f"   3. Click 'Load unpacked' and select: {self.build_dir}")

    def create_package(self):
        """Create distributable package"""
        print("📦 Creating distribution package...")

        package_dir = self.extension_dir / "dist"
        package_dir.mkdir(exist_ok=True)

        # Create ZIP package
        zip_path = package_dir / "toolbox-extension.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in self.build_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.build_dir)
                    zipf.write(file_path, arcname)

        print(f"✅ Package created: {zip_path}")
        return zip_path

    def detect_mobile_browser(self):
        """Detect if running on Android with Termux and check installed browsers"""
        try:
            apps = subprocess.check_output(["pm", "list", "packages"]).decode()
            if "com.kiwibrowser.browser" in apps:
                return "kiwi"
            elif "com.yandex.browser" in apps:
                return "yandex"
            elif "org.mozilla.firefox" in apps:
                return "firefox_android"
            else:
                return None
        except Exception as e:
            print(f"❌ Could not detect mobile browser: {e}")
            return None

    def install_kiwi(self):
        """Install extension in Kiwi Browser (Android)"""
        try:
            target_dir = Path.home() / "storage/shared/Kiwi/extensions/toolbox"
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.build_dir, target_dir, dirs_exist_ok=True)
            print(f"✅ Extension installed in Kiwi: {target_dir}")
        except Exception as e:
            print(f"❌ Kiwi installation failed: {e}")

    def install_yandex(self):
        """Install extension in Yandex Browser (Android, limited support)"""
        try:
            target_dir = Path.home() / "storage/shared/Yandex/extensions/toolbox"
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.build_dir, target_dir, dirs_exist_ok=True)
            print(f"✅ Extension installed in Yandex: {target_dir}")
        except Exception as e:
            print(f"❌ Yandex installation failed: {e}")

    def install_firefox_android(self):
        """Install extension in Firefox for Android"""
        try:
            target_dir = Path.home() / "storage/shared/Firefox/extensions/toolbox"
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.build_dir, target_dir, dirs_exist_ok=True)
            print(f"✅ Extension installed in Firefox (Android): {target_dir}")
        except Exception as e:
            print(f"❌ Firefox Android installation failed: {e}")

    def auto_install(self):
        """Auto-detect and install to the correct browser"""
        browser = self.detect_mobile_browser()
        if not browser:
            print("⚠️  No supported mobile browser detected, falling back to desktop flow...")
            return self.run_installer()

        self.build_extension()

        if browser == "kiwi":
            return self.install_kiwi()
        elif browser == "yandex":
            return self.install_yandex()
        elif browser == "firefox_android":
            return self.install_firefox_android()
        else:
            print("⚠️  Unsupported mobile browser detected, running desktop installer...")
            return self.run_installer()


    def setup_cli_auth(self):
        """Setup CLI authentication server"""
        print("🔐 Setting up CLI authentication...")

        try:
            from toolboxv2 import App
            app = App()

            # Register extension authentication endpoint
            auth_config = {
                'extension_id': 'toolbox-browser-extension',
                'allowed_origins': ['chrome-extension://*', 'moz-extension://*'],
                'permissions': ['read', 'write', 'execute']
            }

            # Save auth config
            config_path = self.extension_dir / "auth_config.json"
            with open(config_path, 'w') as f:
                json.dump(auth_config, f, indent=2)

            print("✅ CLI authentication configured")

        except ImportError:
            print("⚠️  ToolBox not found. CLI auth will be configured on first run.")

    def run_installer(self):
        """Run the complete installation process"""
        print("🚀 ToolBox Browser Extension Installer")
        print("=" * 50)

        # Build extension
        build_dir = self.build_extension()

        # Setup CLI authentication
        self.setup_cli_auth()

        # Ask user which browsers to install to
        print("\n📋 Select browsers to install to:")
        print("1. Chrome")
        print("2. Firefox")
        print("3. Edge")
        print("4. Mobile browsers (auto-detect)")
        print("5. All browsers")
        print("6. Create package only")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == "1" or choice == "5":
            self.install_chrome()

        if choice == "2" or choice == "5":
            self.install_firefox()

        if choice == "3" or choice == "5":
            self.install_edge()

        if choice == "4" or choice == "5":
            self.auto_install()

        if choice == "6":
            package_path = self.create_package()
            print(f"\n📦 Package ready for manual installation: {package_path}")

        print("\n✅ Installation complete!")
        print("\n🎯 Next steps:")
        print("1. Enable Developer Mode in your browser's extension settings")
        print("2. Load the extension from the build directory")
        print("3. Configure settings via the extension popup")
        print("4. Start using ToolBox on any website!")

        return build_dir


def main():
    """Main installer function"""
    installer = TBExtensionInstaller()


    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "build":
            installer.build_extension()
        elif command == "chrome":
            installer.build_extension()
            installer.install_chrome()
        elif command == "firefox":
            installer.build_extension()
            installer.install_firefox()
        elif command == "edge":
            installer.build_extension()
            installer.install_edge()
        elif command == "package":
            installer.build_extension()
            installer.create_package()
        elif command == "kiwi":
            installer.build_extension()
            installer.install_kiwi()
        elif command == "yandex":
            installer.build_extension()
            installer.install_yandex()
        elif command == "firefox-android":
            installer.build_extension()
            installer.install_firefox_android()
        elif command == "auto-mobile":
            installer.build_extension()
            installer.auto_install()
        elif command == "all":
            installer.run_installer()
        else:
            print("Usage: python install.py [build|chrome|firefox|edge|package|auto-mobile|kiwi|yandex|firefox-android|all]")
    else:
        installer.run_installer()


if __name__ == "__main__":
    main()
