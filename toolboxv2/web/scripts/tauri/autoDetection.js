setTimeout(() => {
    console.log("AUTO:DETECTION :: ", window.__TAURI__)
    if (!window.location.href.endsWith("/web/dashboard")) {

    }else if (!window.__TAURI__) {
        console.log(window.history.state.preUrl )
        //if(!window.history.state.preUrl.toString().includes("user_dashboard.html")){
            window.TBf.router('/web/dashboards/user_dashboard.html')
        //)}
    } else {

        const { spawn } = window.__TAURI__.process;

// Funktion zum Ausführen von Toolbox-Befehlen
        function runToolboxCommand(command) {
            return new Promise((resolve, reject) => {
                const child = spawn('ToolBoxV2', [ ...command]);
                let output = '';

                child.stdout.on('data', (data) => {
                    output += data.toString();
                });

                child.stderr.on('data', (data) => {
                    reject(data.toString());
                });

                child.on('close', (code) => {
                    if (code === 0) {
                        resolve(output);
                    } else {
                        reject(`Child process exited with code ${code}`);
                    }
                });
            });
        }

// Funktion zum Hinzufügen von Toolbox-Befehlen zum window-Objekt
        async function addToolboxCommandsToWindow() {
            try {
                const commands = ['init', 'get-version', 'mm', 'sm', 'lm', 'modi', 'kill', 'remote', 'background-application', 'bgr', 'fg', 'docker', 'install', 'remove', 'update', 'mod-version-name', 'name', 'port', 'host', 'load-all-mod-in-files', 'save-function-enums-in-file', 'hot-reload', 'debug', 'delete-config-all', 'delete-data-all', 'delete-config', 'delete-data', 'test', 'profiler'];

                for (const command of commands) {
                    window[`runToolbox${command.charAt(0).toUpperCase() + command.slice(1)}`] = async (...args) => {
                        try {
                            return await runToolboxCommand([`-${command}`, ...args]);
                        } catch (error) {
                            console.error(`Error running Toolbox command ${command}:`, error);
                            throw error;
                        }
                    };
                }
            } catch (error) {
                console.error('Error adding Toolbox commands to window:', error);
            }
        }

// Initialisierungsfunktion
        async function initialize() {
            try {
                await addToolboxCommandsToWindow();
                console.log('Toolbox commands added to window object.');
            } catch (error) {
                console.error('Initialization error:', error);
            }
        }

        initialize().then(r =>window.TBf.router('/web/dashboards/widgetbord.html') );


    }

}, 600);
