document.addEventListener('DOMContentLoaded', function() {
    var osSelection = document.getElementById('os-selection');
    var userAgent = navigator.userAgent;
    var platform = navigator.platform;
    var osInfo = document.getElementById('os-info');
    var autoDownloadOptions = document.getElementById('auto-download-options');

    // Funktion zur Anzeige des Installationslinks
    function displayInstallLink(os) {
        var baseUrl = "/web/core0/initialistaller/init.html";
        var devApp = "/installer/data/widgets_dev8000.exe";
        var linkText = "Installer für " + os;
        if (os.includes("Desktop (Rust)")){
            if (osInfo.textContent.startsWith("Windows ")) {
                return '<p><a href="' + devApp + '" target="_blank">' + linkText + '</a></p>';
            }else{
                return '<p>Currently, not suportet / build it ur self withe cargo from the native folder <a href="https://github.com/MarkinHaus/ToolBoxV2" target="_blank">Source</a></p>';
            }
        }
        if (os.includes("Web")){
            return '<a onclick="registerServiceWorker()"> Add to Device </a>'
        }
        return '<p><a href="' + baseUrl + '">' + linkText + '</a></p>';
    }

    // Event-Listener für die manuelle Auswahl
    osSelection.addEventListener('change', function() {
        var selectedOS = osSelection.value;
        switch (selectedOS) {
            case 'Desktop_cli':
                autoDownloadOptions.innerHTML = displayInstallLink('Desktop CLI (Python)');
                break;
            case 'Desktop_web':
                autoDownloadOptions.innerHTML = displayInstallLink('Desktop (Web-App)');
                break;
            case 'Mobile_web':
                autoDownloadOptions.innerHTML = displayInstallLink('Mobile (Web-App)');
                break;
            case 'Desktop_rust':
                autoDownloadOptions.innerHTML = displayInstallLink('Desktop (Rust)');
                break;
            default:
                autoDisplay();
        }
    });
    function autoDisplay() {
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
            osInfo.textContent = 'iOS erkannt nur Web';
            autoDownloadOptions.innerHTML = '<p>In Entwicklung</p>';
        } else if (/Android/.test(userAgent)) {
            osInfo.textContent = 'Android erkannt';
            autoDownloadOptions.innerHTML = '<p>Auf der Roadmap</p>';
        } else {
            osInfo.textContent = 'Betriebssystem nicht erkannt';
            autoDownloadOptions.innerHTML = '<p>Bitte wählen Sie Ihr Betriebssystem manuell aus.</p>';
        }
    }
    autoDisplay()
});
