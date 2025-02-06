// Python Lib unix und not unix | exe, apk
setTimeout(()=> {
    var osSelection = document.getElementById('os-selection');
    var userAgent = navigator.userAgent;
    var platform = navigator.platform;
    var osInfo = document.getElementById('os-info');
    var autoDownloadOptions = document.getElementById('auto-download-options');

    // Funktion zur Anzeige des Installationslinks
    function displayInstallLink(os) {
        var baseUrl = "/web/core0/initialistaller/init.html?os="+os;
        var devApp = "/installer/data/widgets_dev8000.exe";
        var apkApp = "/installer/data/simple.apk";
        var exeApp = "/installer/data/simple.exe";
        var linkText = "Installer für " + os;
        if (os.includes("Web")){
            return '<a onclick="window.TBf.registerServiceWorker()"> Add ' + linkText + 'to Device</a>'
        }
        if (os.includes("exe")){
             return '<p><a href="' + exeApp + '" target="_blank">' + linkText + '</a></p>';
        }
        if (os.includes("apk")){
             return '<p><a href="' + apkApp + '" target="_blank">' + linkText + '</a></p>';
        }
        if (os.includes("dmg") || os.includes("iOS")){
             return '<p>Currently, not suportet / build it ur self withe cargo from the native folder <a href="https://github.com/MarkinHaus/ToolBoxV2" target="_blank">Source</a></p>';
        }
        return '<p><a href="' + baseUrl + '">' + linkText + '</a></p>';
    }

    // Event-Listener für die manuelle Auswahl
    osSelection.addEventListener('change', function() {
        var selectedOS = osSelection.value;
        switch (selectedOS) {
            case 'Python Runtime':
                autoDownloadOptions.innerHTML = displayInstallLink('CLI (Python)');
                window.TBf.processRow(autoDownloadOptions)
                break;
            case 'exe':
                autoDownloadOptions.innerHTML = displayInstallLink('Desktop (exe)');
                window.TBf.processRow(autoDownloadOptions)
                break;
            case 'dmg':
                autoDownloadOptions.innerHTML = displayInstallLink('Desktop (dmg)');
                window.TBf.processRow(autoDownloadOptions)
                break;
            case 'apk':
                autoDownloadOptions.innerHTML = displayInstallLink('Mobile (apk)');
                window.TBf.processRow(autoDownloadOptions)
                break;
            case 'iOS-IPA':
                autoDownloadOptions.innerHTML = displayInstallLink('Mobile (iOS-App)');
                window.TBf.processRow(autoDownloadOptions)
                break;
            case 'Web':
                autoDownloadOptions.innerHTML = displayInstallLink('(Web-App)');
                window.TBf.processRow(autoDownloadOptions)
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
            autoDownloadOptions.innerHTML = '<p>give 500$ and i start</p>';
        } else if (/Android/.test(userAgent)) {
            osInfo.textContent = 'Android erkannt';
            autoDownloadOptions.innerHTML = displayInstallLink('Android');
        } else {
            osInfo.textContent = 'Betriebssystem nicht erkannt';
            autoDownloadOptions.innerHTML = '<p>Bitte wählen Sie Ihr Betriebssystem manuell aus.</p>';
        }
        window.TBf.processRow(autoDownloadOptions)
    }
    autoDisplay()
}, 200);
