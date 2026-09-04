// installer.js - ToolBoxV2 Universal Installer
// Updated to support all build targets from the new CI/CD pipeline

setTimeout(() => {
    const osSelection = document.getElementById('os-selection');
    const userAgent = navigator.userAgent;
    const platform = navigator.platform;
    const osInfo = document.getElementById('os-info');
    const autoDownloadOptions = document.getElementById('auto-download-options');
    let selectedOsValue = "Automatic";

    const GITHUB_REPO_OWNER = 'MarkinHaus';
    const GITHUB_REPO_NAME = 'ToolBoxV2';
    const GITHUB_RAW_BASE = `https://raw.githubusercontent.com/${GITHUB_REPO_OWNER}/${GITHUB_REPO_NAME}/master`;

    // All available installation options
    const OS_OPTIONS = [
        { value: 'Automatic', label: '🔍 Automatische Erkennung', group: 'auto' },
        { value: 'divider-1', label: '─── Desktop Apps (Tauri) ───', disabled: true },
        { value: 'Windows', label: '🪟 Windows (64-bit)', group: 'desktop' },
        { value: 'MacOS-ARM', label: '🍎 macOS (Apple Silicon)', group: 'desktop' },
        { value: 'MacOS-Intel', label: '🍎 macOS (Intel)', group: 'desktop' },
        { value: 'Linux-AppImage', label: '🐧 Linux (AppImage)', group: 'desktop' },
        { value: 'Linux-Deb', label: '🐧 Linux (Debian/Ubuntu)', group: 'desktop' },
        { value: 'divider-2', label: '─── Mobile ───', disabled: true },
        { value: 'Android', label: '🤖 Android (APK)', group: 'mobile' },
        { value: 'Android-Termux', label: '🤖 Android (Termux/Python)', group: 'mobile' },
        { value: 'iOS', label: '📱 iOS (Web App)', group: 'mobile' },
        { value: 'divider-3', label: '─── CLI & Server ───', disabled: true },
        { value: 'CLI-Python', label: '🐍 Python CLI (pip/uv)', group: 'cli' },
        { value: 'CLI-Nuitka-Win', label: '⚡ Native CLI - Windows', group: 'cli' },
        { value: 'CLI-Nuitka-Mac', label: '⚡ Native CLI - macOS', group: 'cli' },
        { value: 'CLI-Nuitka-Linux', label: '⚡ Native CLI - Linux', group: 'cli' },
        { value: 'Server-Rust', label: '🦀 Rust Server (simple-core)', group: 'server' },
        { value: 'divider-4', label: '─── ToolBoxV2 BlobDB ───', disabled: true },
        { value: 'BlobDB', label: '🧬 BlobDB (Windows)', group: 'blobdb' },
        { value: 'BlobDB-Mac', label: '🧬 BlobDB (macOS)', group: 'blobdb' },
        { value: 'BlobDB-Linux', label: '🧬 BlobDB (Linux)', group: 'blobdb' },

        { value: 'divider-4', label: '─── Entwickler ───', disabled: true },
        { value: 'TBLang', label: '📜 TBLang Compiler', group: 'dev' },
        { value: 'Browser-Extension', label: '🌐 Browser Extension', group: 'dev' },
        { value: 'Web-App', label: '🌐 Web App (PWA)', group: 'web' },
        { value: 'Source', label: '📦 Source Code', group: 'dev' }
    ];

    // Asset patterns for matching GitHub release assets
    const osAssetPatterns = {
        // Tauri Desktop Apps
        'Windows': [
            /simple-core_.*_x64-setup\.exe$/,
            /simple-core_.*_x64\.msi\.zip$/,
            /simple-core_.*\.msi$/,
            /simple-core_.*\.exe$/
        ],
        'MacOS-ARM': [
            /simple-core_.*_aarch64\.dmg$/,
            /simple-core_.*arm64\.dmg$/,
            /simple-core_.*_universal\.dmg$/
        ],
        'MacOS-Intel': [
            /simple-core_.*_x64\.dmg$/,
            /simple-core_.*_x86_64\.dmg$/,
            /simple-core_.*intel\.dmg$/
        ],
        'Linux-AppImage': [
            /simple-core_.*\.AppImage$/,
            /simple-core_.*_amd64\.AppImage$/
        ],
        'Linux-Deb': [
            /simple-core_.*_amd64\.deb$/,
            /simple-core_.*\.deb$/
        ],

        // Mobile
        'Android': [
            /simple-core_.*\.apk$/,
            /app-release.*\.apk$/,
            /toolbox.*\.apk$/i,
            /\.apk$/
        ],

        // Nuitka Native CLI
        'CLI-Nuitka-Win': [
            /tb-toolbox-windows.*\.exe$/,
            /toolbox-windows.*\.exe$/,
            /nuitka.*windows.*\.exe$/i
        ],
        'CLI-Nuitka-Mac': [
            /tb-toolbox-macos.*$/,
            /toolbox-macos.*\.tar\.gz$/,
            /nuitka.*macos/i
        ],
        'CLI-Nuitka-Linux': [
            /tb-toolbox-linux.*$/,
            /toolbox-linux.*$/,
            /nuitka.*linux/i
        ],

        // Rust Server
        'Server-Rust': [
            /simple-core-server-linux-x64$/,
            /simple-core-server-windows.*\.exe$/,
            /simple-core-server-macos.*$/,
            /simple-core-server.*/
        ],

        // TBLang Compiler
        'TBLang': [
            /tblang-linux.*$/,
            /tblang-windows.*\.exe$/,
            /tblang-macos.*$/,
            /tblang.*/
        ],

        // Browser Extension
        'Browser-Extension': [
            /tb-browser.*\.zip$/,
            /browser-extension.*\.zip$/,
            /extension.*\.zip$/
        ],

        // Termux Installer
        'Android-Termux': [
            /termux.*installer.*\.tar\.gz$/,
            /termux.*\.sh$/,
            /install-toolboxv2\.sh$/
        ],

        // Source
        'Source': [
            /ToolBoxV2.*\.tar\.gz$/,
            /ToolBoxV2.*\.whl$/,
            /source.*\.zip$/
        ]
    };

    // Populate select with styled options
    osSelection.innerHTML = OS_OPTIONS.map(opt => {
        if (opt.disabled) {
            return `<option value="${opt.value}" disabled style="font-weight:bold;color:#666;">${opt.label}</option>`;
        }
        return `<option value="${opt.value}">${opt.label}</option>`;
    }).join('');

    function displayLoading() {
        autoDownloadOptions.innerHTML = `
            <div style="text-align:center;padding:20px;">
                <p>🔄 Lade Release-Informationen...</p>
            </div>`;
    }

    function displayError(message) {
        autoDownloadOptions.innerHTML = `
            <div style="padding:15px;border:1px solid #f5c6cb;border-radius:8px;">
                <p>⚠️ <strong>Fehler:</strong> ${message}</p>
                <p>Bitte prüfe die <a href="https://github.com/${GITHUB_REPO_OWNER}/${GITHUB_REPO_NAME}/releases" target="_blank">GitHub Releases Seite</a> manuell.</p>
            </div>`;
        if (window.TB?.ui?.processDynamicContent) {
            window.TB.ui.processDynamicContent(autoDownloadOptions);
        }
    }

    function createDownloadCard(title, description, links, icon = '📦') {
        const linksHtml = links.map(link => {
            if (link.onclick) {
                const needsTBf = /TBf/.test(link.onclick);
                const tbfReady = !needsTBf || (typeof window.TBf !== 'undefined' && typeof window.TBf.registerServiceWorker === 'function');
                if (!tbfReady) {
                    return `<button disabled title="Nur innerhalb der ToolBox-App verfügbar (PWA-Registrierung)" style="margin:5px;padding:8px 16px;cursor:not-allowed;border-radius:4px;border:1px solid var(--border-subtle,#999);background:var(--bg-elevated,#eee);color:var(--text-muted,#888);opacity:.6;">${link.text}</button>`;
                }
                return `<button onclick="${link.onclick}" style="margin:5px;padding:8px 16px;cursor:pointer;border-radius:4px;border:1px solid var(--primary,#007bff);background:var(--primary,#007bff);color:#000;transition:var(--duration-fast,.15s) var(--ease-default,ease);">${link.text}</button>`;
            }
            return `<a href="${link.url}" ${link.download ? 'download' : ''} target="_blank" style="display:inline-block;margin:5px;padding:8px 16px;background:var(--primary,#28a745);color:#000;text-decoration:none;border-radius:4px;transition:var(--duration-fast,.15s) var(--ease-default,ease);">${link.text}</a>`;
        }).join('');

        return `
            <div style="padding:15px;margin:10px 0">
                <h4 style="margin:0 0 10px 0;">${icon} ${title}</h4>
                <p style="margin:0 0 10px 0;color:#666;font-size:0.9em;">${description}</p>
                <div>${linksHtml}</div>
            </div>`;
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1024 / 1024).toFixed(2) + ' MB';
    }

    async function fetchAndDisplayLinks(selectedOSKey = null) {
        displayLoading();

        try {
            // Fetch all releases
            const response = await fetch(`https://api.github.com/repos/${GITHUB_REPO_OWNER}/${GITHUB_REPO_NAME}/releases`);
            if (!response.ok) {
                throw new Error(`GitHub API: ${response.status} ${response.statusText}`);
            }

            const allReleases = await response.json();
            const releases = (allReleases || []).filter(r => !r.draft && !/demo/i.test(r.tag_name || ''));
            if (!releases || releases.length === 0) {
                throw new Error('Keine Releases gefunden.');
            }

            // Find the latest "App" release for GUI apps, or latest overall
            let latestAppRelease = releases.find(r =>
                r.tag_name.includes('-App') ||
                r.name?.toLowerCase().includes('app v')
            );

            // Also get the absolute latest for CLI/Server builds
            const latestRelease = releases[0];

            if (!latestAppRelease) {
                latestAppRelease = latestRelease;
            }

            const assets = latestAppRelease.assets;
            const allAssets = latestRelease.assets;

            let htmlOutput = '';

            // Handle special cases first
            switch (selectedOSKey) {
                case 'CLI-Python':
                    htmlOutput = createDownloadCard(
                        'Python CLI Installation',
                        'Installiert ToolBoxV2 über pip oder den Universal Installer Script.',
                        [
                            { url: `${GITHUB_RAW_BASE}/installer.sh`, text: '📜 Installer Script', download: true },
                            { url: 'https://pypi.org/project/ToolBoxV2/', text: '📦 PyPI Seite' }
                        ],
                        '🐍'
                    );
                    htmlOutput += `
                        <div style="padding:15px;border-radius:8px;margin-top:10px;">
                            <h5>Installation:</h5>
                            <pre style="background:#1e1e1e;color:#d4d4d4;padding:10px;border-radius:4px;overflow-x:auto;">
# Option 1: Universal Installer (empfohlen)
curl -sSL ${GITHUB_RAW_BASE}/installer.sh | bash

# Option 2: pip
pip install ToolBoxV2

# Option 3: uv (schneller)
uv pip install ToolBoxV2

# Mit ISAA Extras
pip install "ToolBoxV2[isaa]"</pre>
                        </div>`;
                    break;

                case 'Android-Termux':
                    htmlOutput = createDownloadCard(
                        'Termux Installation (Android)',
                        'Installiert die Python-basierte CLI in Termux auf Android.',
                        [
                            { url: `${GITHUB_RAW_BASE}/termux-install.sh`, text: '📜 Termux Installer', download: true }
                        ],
                        '🤖'
                    );
                    htmlOutput += `
                        <div style="padding:15px;border-radius:8px;margin-top:10px;">
                            <h5>Voraussetzungen:</h5>
                            <ol style="margin:10px 0;padding-left:20px;">
                                <li>Installiere <a href="https://f-droid.org/packages/com.termux/" target="_blank">Termux von F-Droid</a> (nicht Play Store!)</li>
                                <li>Öffne Termux und führe aus:</li>
                            </ol>
                            <pre style="background:#1e1e1e;color:#d4d4d4;padding:10px;border-radius:4px;overflow-x:auto;">
# Schnell-Installation
curl -sSL ${GITHUB_RAW_BASE}/termux-install.sh | bash

# Mit allen Extras (Nuitka, Rust Server, Dev Tools)
curl -sSL ${GITHUB_RAW_BASE}/termux-install.sh | bash -s -- --full --server --dev</pre>
                        </div>`;
                    break;

                case 'Web-App':
                    htmlOutput = createDownloadCard(
                        'Web App (PWA)',
                        'Füge die ToolBoxV2 Web App zu deinem Gerät hinzu.',
                        [
                            { onclick: "window.TBf?.registerServiceWorker?.()", text: '➕ Als App installieren' }
                        ],
                        '🌐'
                    );
                    htmlOutput += `
                        <div style="padding:15px;border-radius:8px;margin-top:10px;">
                            <p><strong>Hinweis:</strong> Die PWA funktioniert auf allen modernen Browsern und kann wie eine native App verwendet werden.</p>
                        </div>`;
                    break;

                case 'iOS':
                    htmlOutput = createDownloadCard(
                        'iOS Installation',
                        'Für iOS ist die Web App (PWA) die beste Option.',
                        [
                            { onclick: "window.TBf?.registerServiceWorker?.()", text: '➕ Als App installieren' }
                        ],
                        '📱'
                    );
                    htmlOutput += `
                        <div style="padding:15px;border-radius:8px;margin-top:10px;">
                            <h5>Alternative für Power User:</h5>
                            <p>Mit <a href="https://ish.app/" target="_blank">iSH Shell</a> kannst du die Python CLI installieren:</p>
                            <pre style="background:#1e1e1e;color:#d4d4d4;padding:10px;border-radius:4px;">apk add python3 py3-pip git
pip install ToolBoxV2</pre>
                        </div>`;
                    break;

                case 'Server-Rust':
                    // Find server binaries
                    const serverAssets = findMatchingAssets(allAssets, 'Server-Rust');
                    const serverLinks = serverAssets.length > 0
                        ? serverAssets.map(a => ({ url: a.browser_download_url, text: a.name, download: true }))
                        : [{ url: `https://github.com/${GITHUB_REPO_OWNER}/${GITHUB_REPO_NAME}#3-server-only-deployment-rust-actix-server`, text: '📖 Build Anleitung' }];

                    htmlOutput = createDownloadCard(
                        'Rust Server (simple-core-server)',
                        'Der hochperformante API Server geschrieben in Rust/Actix.',
                        serverLinks,
                        '🦀'
                    );

                    if (serverAssets.length === 0) {
                        htmlOutput += `
                            <div style="padding:15px;border-radius:8px;margin-top:10px;">
                                <h5>Manueller Build:</h5>
                                <pre style="background:#1e1e1e;color:#d4d4d4;padding:10px;border-radius:4px;">git clone https://github.com/${GITHUB_REPO_OWNER}/${GITHUB_REPO_NAME}.git
cd ToolBoxV2/toolboxv2/src-core
cargo build --release</pre>
                            </div>`;
                    }
                    break;

                case 'TBLang':
                    const tblangAssets = findMatchingAssets(allAssets, 'TBLang');
                    const tblangLinks = tblangAssets.length > 0
                        ? tblangAssets.map(a => ({ url: a.browser_download_url, text: a.name, download: true }))
                        : [{ url: `https://github.com/${GITHUB_REPO_OWNER}/${GITHUB_REPO_NAME}/tree/master/toolboxv2/tb-exc`, text: '📖 Source Code' }];

                    htmlOutput = createDownloadCard(
                        'TBLang Compiler',
                        'Der ToolBox Language Compiler.',
                        tblangLinks,
                        '📜'
                    );
                    break;

                case 'Browser-Extension':
                    const extAssets = findMatchingAssets(allAssets, 'Browser-Extension');
                    const extLinks = extAssets.length > 0
                        ? extAssets.map(a => ({ url: a.browser_download_url, text: a.name, download: true }))
                        : [{ url: `https://github.com/${GITHUB_REPO_OWNER}/${GITHUB_REPO_NAME}/tree/master/toolboxv2/tb_browser`, text: '📖 Source Code' }];

                    htmlOutput = createDownloadCard(
                        'Browser Extension',
                        'Chrome/Firefox Extension für ToolBoxV2 Integration.',
                        extLinks,
                        '🌐'
                    );
                    break;

                case 'Source':
                    htmlOutput = createDownloadCard(
                        'Source Code',
                        'Kompletter Quellcode und Python Distribution.',
                        [
                            { url: latestRelease.zipball_url, text: '📦 ZIP Download' },
                            { url: latestRelease.tarball_url, text: '📦 TAR.GZ Download' },
                            { url: `https://github.com/${GITHUB_REPO_OWNER}/${GITHUB_REPO_NAME}`, text: '🔗 GitHub Repository' }
                        ],
                        '📦'
                    );

                    // Add PyPI packages if available
                    const pypiAssets = allAssets.filter(a => a.name.endsWith('.whl') || a.name.endsWith('.tar.gz'));
                    if (pypiAssets.length > 0) {
                        htmlOutput += '<h5>Python Packages:</h5><ul>';
                        pypiAssets.forEach(a => {
                            htmlOutput += `<li><a href="${a.browser_download_url}" download>${a.name}</a> (${formatFileSize(a.size)})</li>`;
                        });
                        htmlOutput += '</ul>';
                    }
                    break;

                default:
                    // Standard asset matching for desktop/mobile
                    if (selectedOSKey && osAssetPatterns[selectedOSKey]) {
                        const matchedAssets = findMatchingAssets(assets, selectedOSKey);

                        if (matchedAssets.length > 0) {
                            const primaryAsset = matchedAssets[0];
                            const icon = getOSIcon(selectedOSKey);

                            htmlOutput = createDownloadCard(
                                `${selectedOSKey} Download`,
                                `Release: ${latestAppRelease.name || latestAppRelease.tag_name}`,
                                [{ url: primaryAsset.browser_download_url, text: `⬇️ ${primaryAsset.name} (${formatFileSize(primaryAsset.size)})`, download: true }],
                                icon
                            );

                            // Show alternative downloads if multiple found
                            if (matchedAssets.length > 1) {
                                htmlOutput += '<details style="margin-top:10px;"><summary style="cursor:pointer;">Weitere Downloads</summary><ul>';
                                matchedAssets.slice(1).forEach(a => {
                                    htmlOutput += `<li><a href="${a.browser_download_url}" download>${a.name}</a> (${formatFileSize(a.size)})</li>`;
                                });
                                htmlOutput += '</ul></details>';
                            }
                        } else {
                            htmlOutput = `<p>⚠️ Kein passender Download für "${selectedOSKey}" in diesem Release gefunden.</p>`;
                            htmlOutput += showAllAssets(assets, latestAppRelease);
                        }
                    } else {
                        // Automatic or unknown - show all
                        htmlOutput = showAllAssets(assets, latestAppRelease);
                    }
            }

            // Add release info footer
            htmlOutput += `
                <div style="margin-top:20px;padding-top:15px;border-top:1px solid #ddd;font-size:0.85em;color:#666;">
                    <p>📋 <strong>Release:</strong> <a href="${latestAppRelease.html_url}" target="_blank">${latestAppRelease.name || latestAppRelease.tag_name}</a></p>
                    <p>📅 Veröffentlicht: ${new Date(latestAppRelease.published_at).toLocaleDateString('de-DE')}</p>
                    <p><a href="https://github.com/${GITHUB_REPO_OWNER}/${GITHUB_REPO_NAME}/releases" target="_blank">Alle Releases anzeigen →</a></p>
                </div>`;

            autoDownloadOptions.innerHTML = htmlOutput;

        } catch (error) {
            console.error('Error fetching release info:', error);
            displayError(error.message);
        }

        if (window.TB?.ui?.processDynamicContent) {
            window.TB.ui.processDynamicContent(autoDownloadOptions);
        }
    }

    function findMatchingAssets(assets, osKey) {
        const patterns = osAssetPatterns[osKey];
        if (!patterns) return [];

        const matches = [];
        for (const pattern of patterns) {
            const found = assets.filter(a => pattern.test(a.name));
            matches.push(...found);
        }

        // Remove duplicates
        return [...new Map(matches.map(a => [a.id, a])).values()];
    }

    function getOSIcon(osKey) {
        const icons = {
            'Windows': '🪟',
            'MacOS-ARM': '🍎',
            'MacOS-Intel': '🍎',
            'Linux-AppImage': '🐧',
            'Linux-Deb': '🐧',
            'Android': '🤖',
            'iOS': '📱'
        };
        return icons[osKey] || '📦';
    }

    function showAllAssets(assets, release) {
        if (!assets || assets.length === 0) {
            return '<p>Keine Download-Assets in diesem Release verfügbar.</p>';
        }

        let html = `<h5>Verfügbare Downloads (${release.name || release.tag_name}):</h5>`;

        // Group assets by type
        const groups = {
            'Desktop Apps': [],
            'Mobile': [],
            'CLI & Server': [],
            'Andere': []
        };

        assets.forEach(asset => {
            const name = asset.name.toLowerCase();
            if (name.includes('.dmg') || name.includes('.exe') || name.includes('.msi') ||
                name.includes('.appimage') || name.includes('.deb')) {
                groups['Desktop Apps'].push(asset);
            } else if (name.includes('.apk')) {
                groups['Mobile'].push(asset);
            } else if (name.includes('server') || name.includes('tblang') || name.includes('nuitka') ||
                       name.includes('toolbox-') || name.includes('.sh')) {
                groups['CLI & Server'].push(asset);
            } else {
                groups['Andere'].push(asset);
            }
        });

        for (const [groupName, groupAssets] of Object.entries(groups)) {
            if (groupAssets.length > 0) {
                html += `<h6 style="margin-top:15px;">${groupName}</h6><ul style="list-style:none;padding:0;">`;
                groupAssets.forEach(asset => {
                    html += `<li style="margin:5px 0;"><a href="${asset.browser_download_url}" download style="display:inline-flex;align-items:center;gap:5px;">📥 ${asset.name}</a> <small style="color:#666;">(${formatFileSize(asset.size)})</small></li>`;
                });
                html += '</ul>';
            }
        }

        return html;
    }

    function determineOSAndDisplay() {
        let detectedOSKey = null;
        let detectedOSName = 'Betriebssystem nicht eindeutig erkannt';

        const ua = userAgent.toLowerCase();
        const plat = platform.toLowerCase();

        if (/windows|win32|win64|wow64/.test(ua)) {
            detectedOSKey = 'Windows';
            detectedOSName = '🪟 Windows erkannt';
        } else if (/macintosh|macintel|macppc/.test(ua) || plat.includes('mac')) {
            // Check for Apple Silicon
            if (plat.includes('arm') || ua.includes('arm64') || ua.includes('aarch64')) {
                detectedOSKey = 'MacOS-ARM';
                detectedOSName = '🍎 macOS (Apple Silicon) erkannt';
            } else {
                // Could also try to detect via GPU/WebGL for more accuracy
                detectedOSKey = 'MacOS-Intel';
                detectedOSName = '🍎 macOS (Intel) erkannt';

                // Try to detect Apple Silicon through WebGL
                try {
                    const canvas = document.createElement('canvas');
                    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
                    if (gl) {
                        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
                        if (debugInfo) {
                            const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
                            if (renderer && renderer.toLowerCase().includes('apple')) {
                                detectedOSKey = 'MacOS-ARM';
                                detectedOSName = '🍎 macOS (Apple Silicon) erkannt';
                            }
                        }
                    }
                } catch (e) { /* ignore */ }
            }
        } else if (/android/.test(ua)) {
            detectedOSKey = 'Android';
            detectedOSName = '🤖 Android erkannt';
        } else if (/iphone|ipad|ipod/.test(ua)) {
            detectedOSKey = 'iOS';
            detectedOSName = '📱 iOS erkannt';
        } else if (/linux/.test(plat) || /linux/.test(ua)) {
            detectedOSKey = 'Linux-AppImage';
            detectedOSName = '🐧 Linux erkannt';
        } else if (/cros/.test(ua)) {
            detectedOSKey = 'Web-App';
            detectedOSName = '💻 ChromeOS erkannt (Web App empfohlen)';
        }

        osInfo.textContent = detectedOSName;
        fetchAndDisplayLinks(detectedOSKey);
    }

    // Event listener for manual selection
    osSelection.addEventListener('change', function() {
        selectedOsValue = osSelection.value;

        // Skip dividers
        if (selectedOsValue.startsWith('divider')) {
            return;
        }

        if (selectedOsValue === 'Automatic') {
            osInfo.textContent = '🔍 Automatische Erkennung...';
            determineOSAndDisplay();
        } else {
            const selectedOption = OS_OPTIONS.find(o => o.value === selectedOsValue);
            osInfo.textContent = `✅ Ausgewählt: ${selectedOption?.label || selectedOsValue}`;
            fetchAndDisplayLinks(selectedOsValue);
        }
    });

    // Initial automatic detection
    determineOSAndDisplay();

}, 200);
