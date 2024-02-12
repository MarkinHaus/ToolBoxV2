document.addEventListener('DOMContentLoaded', function() {
    var osSelection = document.getElementById('os-selection');
    var manualInstallLink = document.getElementById('manual-install-link');
    var userAgent = navigator.userAgent;
    var platform = navigator.platform;
    var osInfo = document.getElementById('os-info');
    var autoDownloadOptions = document.getElementById('auto-download-options');

    // Funktion zur Anzeige des Installationslinks
    function displayInstallLink(os) {
        var baseUrl = "https://github.com/MarkinHaus/ToolBoxV2";
        var linkText = "Installer für " + os;
        return '<p><a href="' + baseUrl + '">' + linkText + '</a></p>';
    }

    // Event-Listener für die manuelle Auswahl
    osSelection.addEventListener('change', function() {
        var selectedOS = osSelection.value;
        switch (selectedOS) {
            case 'Desktop_cli':
                manualInstallLink.innerHTML = displayInstallLink('Desktop CLI (Python)');
                break;
            case 'Desktop_web':
                manualInstallLink.innerHTML = displayInstallLink('Desktop (Web-App)');
                break;
            case 'Mobile_web':
                manualInstallLink.innerHTML = displayInstallLink('Mobile (Web-App)');
                break;
            case 'Desktop_rust':
                manualInstallLink.innerHTML = displayInstallLink('Desktop (Rust)');
                break;
            default:
                manualInstallLink.innerHTML = '';
        }
    });

    // Automatische Erkennung des Betriebssystems
    if (/win/i.test(platform)) {
        osInfo.textContent = 'Windows erkannt';
        autoDownloadOptions.innerHTML = displayInstallLink('Windows');
    } else if (/Mac/.test(platform)) {
        osInfo.textContent = 'MacOS erkannt';
        autoDownloadOptions.innerHTML = displayInstallLink('MacOS');
    } else if (/Linux/.test(platform)) {
        osInfo.textContent = 'Linux erkannt';
        autoDownloadOptions.innerHTML = displayInstallLink('Linux');
    } else if (/iPhone|iPad|iPod/.test(userAgent)) {
        osInfo.textContent = 'iOS erkannt';
        autoDownloadOptions.innerHTML = '<p>In Entwicklung</p>';
    } else if (/Android/.test(userAgent)) {
        osInfo.textContent = 'Android erkannt';
        autoDownloadOptions.innerHTML = '<p>Auf der Roadmap</p>';
    } else {
        osInfo.textContent = 'Betriebssystem nicht erkannt';
        autoDownloadOptions.innerHTML = '<p>Bitte wählen Sie Ihr Betriebssystem manuell aus.</p>';
    }
});
