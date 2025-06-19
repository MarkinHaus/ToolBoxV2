// installer.js
setTimeout(() => {
    var osSelection = document.getElementById('os-selection');
    var userAgent = navigator.userAgent;
    var platform = navigator.platform; // More reliable for desktop OS
    var osInfo = document.getElementById('os-info');
    var autoDownloadOptions = document.getElementById('auto-download-options');
    var selectedOsValue = "Automatic"; // To store the selection

    const GITHUB_REPO_OWNER = 'MarkinHaus';
    const GITHUB_REPO_NAME = 'ToolBoxV2';

    // Prioritized suffixes for better matching
    const osAssetPatterns = {
        'Windows': [/simple-core_.*_x64-setup\.exe$/, /simple-core_.*\.msi\.zip$/, /simple-core_.*\.exe$/],
        'MacOS-ARM': [/simple-core_.*_aarch64\.dmg$/],
        'MacOS-Intel': [/simple-core_.*_x86_64\.dmg$/],
        'Linux': [/simple-core_.*\.AppImage$/, /simple-core_.*amd64\.deb$/], // .deb is more specific
        'Android': [/simple-core_.*\.apk$/, /\.apk$/] // Broader match for APKs
    };

    function displayLoading() {
        autoDownloadOptions.innerHTML = '<p>Fetching latest release information...</p>';
    }

    function displayError(message) {
        autoDownloadOptions.innerHTML = `<p>Error: ${message}. Please check the <a href="https://github.com/${GITHUB_REPO_OWNER}/${GITHUB_REPO_NAME}/releases" target="_blank">GitHub Releases page</a> manually.</p>`;
        window.TB.ui.processDynamicContent(autoDownloadOptions);
    }

    async function fetchAndDisplayLinks(selectedOSKey = null) {
        displayLoading();
        try {
            // Fetch all releases to find the latest "App" release
            const response = await fetch(`https://api.github.com/repos/${GITHUB_REPO_OWNER}/${GITHUB_REPO_NAME}/releases`);
            if (!response.ok) {
                throw new Error(`GitHub API request failed: ${response.status}`);
            }
            const releases = await response.json();
            if (!releases || releases.length === 0) {
                throw new Error('No releases found.');
            }

            // Find the latest release that contains "App" in its tag_name or name,
            // or fall back to the absolute latest if no "App" release is found.
            let latestAppRelease = releases.find(r => r.tag_name.includes('-App') || r.name.toLowerCase().includes('app v'));
            if (!latestAppRelease) {
                latestAppRelease = releases[0]; // Fallback to the absolute latest release
                osInfo.innerHTML += ' <small>(Could not find a specific "App" release, showing latest overall release. May not contain GUI installers.)</small>';
            }


            const assets = latestAppRelease.assets;
            let htmlOutput = `<p><strong>Release:</strong> <a href="${latestAppRelease.html_url}" target="_blank">${latestAppRelease.name || latestAppRelease.tag_name}</a></p>`;
            let foundAsset = false;

            if (selectedOSKey === 'CLI (Python)') {
                htmlOutput += `<p><a href="https://raw.githubusercontent.com/${GITHUB_REPO_OWNER}/${GITHUB_REPO_NAME}/master/installer.sh" target="_blank" title="Downloads installer.sh">Install Core Engine (via Universal Script)</a></p>
                               <p><small>This script helps install Python and the ToolBoxV2 Core CLI.</small></p>`;
                foundAsset = true;
            } else if (selectedOSKey === 'Web-App') {
                 htmlOutput += '<p><a onclick="window.TBf.registerServiceWorker()" style="cursor:pointer;">Add Web App to Device (PWA)</a></p>';
                 foundAsset = true;
            } else if (selectedOSKey === 'Server (Rust)') {
                 htmlOutput += `<p>For the server, please <a href="https://github.com/${GITHUB_REPO_OWNER}/${GITHUB_REPO_NAME}#3-server-only-deployment-rust-actix-server" target="_blank">see build instructions</a> in the README.</p>
                                <p><small>The Rust server binary is not directly attached to releases currently.</small></p>`;
                foundAsset = true;
            } else if (selectedOSKey) { // Manual selection from dropdown (Windows, MacOS-ARM, etc.)
                const patterns = osAssetPatterns[selectedOSKey];
                if (patterns) {
                    for (const pattern of patterns) {
                        const asset = assets.find(a => pattern.test(a.name));
                        if (asset) {
                            htmlOutput += `<p><a href="${asset.browser_download_url}" download>${asset.name}</a></p>`;
                            foundAsset = true;
                            break; // Found best match for this OS key
                        }
                    }
                }
            }

            if (!foundAsset && selectedOSKey && selectedOSKey !== 'CLI (Python)' && selectedOSKey !== 'Web-App'  && selectedOSKey !== 'Server (Rust)') {
                htmlOutput += `<p>No specific installer found for "${selectedOSKey}" in this release. You can check all assets:</p>`;
            }

            if (!selectedOSKey || !foundAsset && (selectedOSKey !== 'CLI (Python)' && selectedOSKey !== 'Web-App' && selectedOSKey !== 'Server (Rust)')) { // Show all assets if no specific OS or no asset found for specific OS
                 if (assets.length > 0) {
                    if (selectedOSKey && !foundAsset) { // Already have a message if specific OS was selected but not found
                        // No additional message needed here
                    } else if (!selectedOSKey) { // If it's "Automatic" and couldn't auto-detect a specific one
                         htmlOutput += `<p>Could not automatically determine a specific installer. Available assets for release <strong>${latestAppRelease.name || latestAppRelease.tag_name}</strong>:</p>`;
                    }
                    htmlOutput += '<ul>';
                    assets.forEach(asset => {
                        htmlOutput += `<li><a href="${asset.browser_download_url}" download>${asset.name}</a> (${(asset.size / 1024 / 1024).toFixed(2)} MB)</li>`;
                    });
                    htmlOutput += '</ul>';
                } else {
                    htmlOutput += '<p>No downloadable assets found in this release.</p>';
                }
            }


            autoDownloadOptions.innerHTML = htmlOutput;

        } catch (error) {
            console.error('Error fetching release info:', error);
            displayError(error.message);
        }
        window.TB.ui.processDynamicContent(autoDownloadOptions);
    }


    function determineOSAndDisplay() {
        let detectedOSKey = null;
        let detectedOSName = 'Betriebssystem nicht eindeutig erkannt';

        if (/windows nt|win32|win64|wow64/i.test(userAgent)) { // More specific Windows check
            detectedOSKey = 'Windows';
            detectedOSName = 'Windows erkannt';
        } else if (/macintosh|macintel|macppc|mac_powerpc/i.test(userAgent)) { // More specific Mac check
            detectedOSName = 'macOS erkannt';
            if (platform.toLowerCase().includes('arm') || userAgent.toLowerCase().includes('arm64') || userAgent.toLowerCase().includes('aarch64')) {
                detectedOSKey = 'MacOS-ARM';
                detectedOSName = 'macOS (Apple Silicon) erkannt';
            } else {
                detectedOSKey = 'MacOS-Intel';
                detectedOSName = 'macOS (Intel) erkannt';
            }
        } else if (/linux/i.test(platform)) { // platform is better for Linux
            detectedOSKey = 'Linux';
            detectedOSName = 'Linux erkannt';
        } else if (/android/i.test(userAgent)) {
            detectedOSKey = 'Android';
            detectedOSName = 'Android erkannt';
        } else if (/iphone|ipad|ipod/i.test(userAgent)) {
            detectedOSName = 'iOS erkannt';
            // iOS native apps are not typically distributed this way.
            // Offer Web App or Python Core.
            autoDownloadOptions.innerHTML = `<p>For iOS, you can use the <a onclick="window.TBf.registerServiceWorker()" style="cursor:pointer;">Web App (PWA)</a> or install the <a href="https://raw.githubusercontent.com/${GITHUB_REPO_OWNER}/${GITHUB_REPO_NAME}/master/installer.sh" target="_blank">Core Engine</a> if you have a compatible environment (e.g., iSH Shell).</p>
                                            <p><small>Native iOS app requires App Store distribution.</small></p>`;
            osInfo.textContent = detectedOSName;
            window.TB.ui.processDynamicContent(autoDownloadOptions);
            return; // Early exit for iOS specific message
        }

        osInfo.textContent = detectedOSName;
        fetchAndDisplayLinks(detectedOSKey); // Try to fetch specific asset for detected OS
    }

    osSelection.addEventListener('change', function() {
        selectedOsValue = osSelection.value;
        osInfo.textContent = 'Manuelle Auswahl: ' + (selectedOsValue || 'Automatische Erkennung');
        if (selectedOsValue === "" || selectedOsValue === "Automatic") { // "Automatic" or empty value
            determineOSAndDisplay();
        } else {
            fetchAndDisplayLinks(selectedOsValue);
        }
    });

    // Initial automatic detection
    if (selectedOsValue === "Automatic") {
        determineOSAndDisplay();
    } else { // If a value was pre-selected or changed by user
        fetchAndDisplayLinks(selectedOsValue);
    }

}, 200);
