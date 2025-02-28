import {rendererPipeline} from "/web/scripts/WorkerSocketRenderer.js";
import {AuthHttpPostData, httpPostData, httpPostUrl, ToolBoxError, wrapInResult} from "/web/scripts/httpSender.js";
import {addRenderer,EndBgInteract, Set_animation_xyz,Set_zoom, toggleDarkMode} from "/web/scripts/scripts.js";
import {autocomplete} from "/web/scripts/util.js";
import htmx from "./web/node_modules/htmx.org/dist/htmx.esm.js";
window.htmx = htmx;

const rpIdUrl_f = ()=> {
    if (window.location.href.match("localhost")) {
        return "http://localhost:" + window.location.port
    } else {
        return window.location.origin
    }
}

const rpIdUrl_fs = ()=> {
    if (window.location.href.match("localhost")) {
        return "ws://localhost:" + window.location.port
    } else {
        return "wss://" + window.location.host
    }
}


let init_d = false
let DOME;
let isHtmxAfterRequestListenerAdded = false;
let scriptSto = [];

const state = {
    TBf: {
        router,
        renderer,
        getState,
        addState,
        autocomplete,
        unRegisterServiceWorker: ()=>{
            navigator.serviceWorker.getRegistrations().then(function (registrations) {
                registrations.forEach(function (registration) {
                    registration.unregister().then(function (success) {
                        if (success) {
                            console.log('ServiceWorker mit dem Namen', registration.scope, 'wurde erfolgreich entfernt.');
                        } else {
                            console.log('Fehler beim Entfernen des ServiceWorker mit dem Namen', registration.scope);
                        }
                    });
                });
            });
        },
        registerServiceWorker: async () => {
            if ("serviceWorker" in navigator) {
                try {
                    const registration = await navigator.serviceWorker.register("./web/sw.js", {
                        scope: "/web/",
                    });
                    if (registration.installing) {
                        console.log("Service worker installing");
                    } else if (registration.waiting) {
                        console.log("Service worker installed");
                    } else if (registration.active) {
                        console.log("Service worker active");
                    }
                } catch (error) {
                    console.error(`Registration failed with ${error}`);
                }
            }else {
                console.log("Service worker not found")}
        },
        httpPostData,
        httpPostUrl,
        AuthHttpPostData,
        animator,
        loadHtmlFile,
        processRow:updateDome,
        initVar: (v_name, v_value)=>{if(!state.TBc[v_name]){state.TBc[v_name] = v_value}},
        delVar: (v_name)=>{delete state.TBc[v_name]},
        getVar: (v_name)=>{return state.TBc[v_name]},
        setVar: (v_name, v_value)=>{state.TBc[v_name] = v_value},
        setM: (v_name, v_value)=>{state.TBm[v_name] = v_value},
        getM: (v_name)=>{return state.TBm[v_name]},
        base: rpIdUrl_f(),
        ws_base: rpIdUrl_fs(),
    },
    TBv:{
        base: rpIdUrl_f(),
        ws_base: rpIdUrl_fs(),
        user: null,
        session: null,
    },
    TBc:{

    },
    TBm:{

    }
}

window.TBf = state.TBf
window.TBm = state.TBm
console.log("TB Online :", window.TBf)

if (document.getElementById("MainContent")){
    //
    updateDome(document.body)
    if (window.history.state && window.history.state.url){
        router(window.history.state.url)
    }else{
        router(window.location.href)
    }
    DOME = document.getElementById("MainContent")
    console.log("f DOME:", DOME)
    const baseElement = document.createElement("base");
    baseElement.href = state.TBv.base + '/'
    document.head.appendChild(baseElement);
    initDome()
    updateDome(DOME)
    linksInit()
}else{
    let stoUrl = window.location.href;

    // if (stoUrl.includes("/web/assets/m_log_in.html?")){
    //     return
    // }
    const firstDiv = document.querySelector('div');
    let helper_dome;
    if (firstDiv) {
        // Do something with the firstDiv
        helper_dome = document.createElement("div");
    } else {
        // <div> element not found
        helper_dome = document.body
    }


    router("/index.html", false, "root", helper_dome, ()=>{ setTimeout(()=>{
          if (firstDiv){
           renderer({content:helper_dome.innerHTML, insert:true, Dome:document.body})
         }
        // document.body.innerHTML = helper_dome.innerHTML;
         DOME = document.getElementById("MainContent")
        function helper_init() {
            window.TBf.initVar('c', 0)
            if (!DOME && window.TBf.getVar('c') < 3){
                window.location = stoUrl
                window.TBf.setVar(window.TBf.getVar('c')+1)
            }
            if (firstDiv){
                DOME.appendChild(firstDiv)
            }
            updateDome(DOME)
            initDome()
            addRenderer()
            const s = document.getElementsByClassName('loaderCenter')
            if (s){if(s[0]){s[0].classList.add("none")}}

            const baseElement = document.createElement("base");
            baseElement.href = state.TBv.base + '/'
            document.head.appendChild(baseElement);

            if(!init_d && stoUrl.includes("/web/dashboard")){
                router("/web/dashboard")
            }

            linksInit()

            window.history.pushState({ url: stoUrl, TB: state.TBv, TBc: state.TBc }, "", stoUrl);
            console.log("saved:", stoUrl)}

         if (!DOME) {
            function delay(time) {
                return new Promise(resolve => setTimeout(resolve, time));
            }

            delay(4000).then(() => helper_init());
         }else{
            helper_init()
         }

    }, 250)})

    // setTimeout(, 350)


}
try{
    document.body.removeEventListener('htmx:afterRequest', handleHtmxAfterRequest);
}catch (e){
    console.log("Fist init handleHtmxAfterRequest")
}
if (!isHtmxAfterRequestListenerAdded) {
    document.body.addEventListener('htmx:afterRequest', handleHtmxAfterRequest);
    isHtmxAfterRequestListenerAdded = true;
}


// Add this near the top of your script, after the state initialization
window.addEventListener('popstate', function(event) {
    // Get URL from state, fallback to current location if state is missing
    const url = event.state?.url || window.location.href;

    // Call router with the URL from history state
    router(url);
});
// Renderer

document.addEventListener("DOMContentLoaded", () => {
    console.log("NOW DOMContentLoaded", document.getElementById('Nav-Controls'))
    forceRerender(document.getElementById('Nav-Controls'))
});

function linkEffect() {
        let transition = document.getElementById('overlay');
        if (transition!==null){
            transition.style.width = '100vw';
            transition.style.height = '100vh';
            transition.style.top = '0';
            transition.style.left = '0';
        }
        setTimeout(function() {
            setTimeout(function() {
                if (transition!==null){
                    transition.style.width = '0';
                    transition.style.height = '0';
                    transition.style.top = '50%';
                    transition.style.left = '50%';
                }
            }, 45);
        }, 160);
    }

function updateDome(dome, add_script=true, linkExtra=null){

    function addSrc(script){
        let scriptContent = script.textContent;

        // Create a new Blob containing the script content
        let blob = new Blob([scriptContent], { type: "application/javascript" });

        // Create a URL for the Blob
        return  URL.createObjectURL(blob);

    }

    dome.querySelectorAll("a").forEach(function(link) {
        console.log("Links: ",link.href.toString(), state.TBv.base, link.href.toString().startsWith(state.TBv.base))
        if (link.href.toString().startsWith(state.TBv.base)) {
            let route = "/" + new URL(link.href).pathname.split('/').slice(1).join('/');
            // console.log("REGISTERED:", route)
            // && !link.href.toString().includes("/gui/")
            link.addEventListener("click", function(e) {
                e.preventDefault();
                linkEffect()
                if(linkExtra){linkExtra()}
                console.log("REGISTERED: [route]", route)
                router(route); // Use link.href to get the URL of the clicked link
                // linkEffect()
            });
        }else {
            // console.log("Scip external link to:", link.href)
        }

    });

    if (add_script){
    dome.querySelectorAll("script").forEach(function(script) {
        let attributes = script.attributes;
        let src = script.src
        const js = document.createElement("script");

        js.type = script.type ? script.type : "application/javascript";
        for (let i = 0; i < attributes.length; i++) {
            js[attributes[i].name] = attributes[i].value;
            if(attributes[i].name === "unsave" && attributes[i].value){
                src = addSrc(script)
                console.log("UNSAVE", src)
                // js.type = "application/javascript"
            }
        }
        js.src = src;
        if (js.src!==script.src){
            console.log("[TEST SRC]",js.src,"|",script.src )
        }
        // console.log("[TEST SRC]",js.src===script.src )
        // console.log(js, src !== state.TBv.base+'/index.js')
        if (!scriptSto.includes(js.src) && !document.querySelector('script[src="'+js.src+'"]') && src.slice(0, state.TBv.base.length+'/index.js'.length) !== state.TBv.base+'/index.js' && !script.src.endsWith("/@vite/client" )&& !script.src.includes("scripts/scripts.js")&& !script.src.includes("main.js")){
            console.log("Adding ", script.src);
            dome.appendChild(js);
            scriptSto.push(js.src);
        }else{
            //console.log("Src ", js.src, "not addet")
            //console.log(!scriptSto.includes(js.src) , !document.querySelector('script[src="'+js.src+'"]') , src.slice(0, state.TBv.base.length+'/index.js'.length) !== state.TBv.base+'/index.js' , !script.src.endsWith("/@vite/client" ) , !script.src.includes("scripts/scripts.js"))
        }
    });
    }
    htmx.process(dome);
}

function initDome(){
    const darkModeToggle = document.getElementById('darkModeToggle');

    if(!darkModeToggle){
        console.error("No toggle found")
        return
    }
    darkModeToggle.addEventListener('change', (event) => {

        const labelToggel = document.getElementById('toggleLabel')
        if(!labelToggel){
            throw "NO Dark mode found"
        }
        labelToggel.style.transition = "transform 0.5s ease";

        if (event.target.checked) {
            labelToggel.innerHTML = `<span class="material-symbols-outlined">
dark_mode
</span>`;
            labelToggel.style.transform = 'rotate(0deg)'

        } else {
            labelToggel.innerHTML = `<span class="material-symbols-outlined">
light_mode
</span>`;
            labelToggel.style.transform = 'rotate(360deg)'
        }
        toggleDarkMode()
    });


    const stored_mode = sessionStorage.getItem('darkModeStatus')? sessionStorage.getItem('darkModeStatus') : "dark";
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches && stored_mode==="dark") {
        // The user prefers dark color scheme
        toggleDarkMode(true, "dark")
        document.getElementById('toggleLabel').innerHTML = `<span class="material-symbols-outlined">
light_mode
</span>`
        const labelToggel = document.getElementById('toggleLabel')
        if(labelToggel){
            labelToggel.style.transform = 'rotate(360deg)'
        }
    } else {
        // The user prefers light color scheme
        toggleDarkMode(true, "light")
        document.getElementById('toggleLabel').innerHTML = `<span class="material-symbols-outlined">
dark_mode
</span>`
    }

}

function forceRerender(element) {
    if (!element){
        console.log("NO ELEMENT TO forceRerender")
        return
    }
    element.style.display = "none";
    setTimeout(() => {
        element.style.display = "";
        console.log("forceRerender DONE for", element)
    }, 0); // Small delay ensures reflow
}

function renderer({content="", extend = false, id = "main", Dome = DOME, add_script = true, helper = undefined, insert=false}) {

    if (!extend && !insert && Dome){
        Dome.innerHTML = ''
        if (Dome.hasChildNodes()){
            Dome.outerHTML = '<main id="MainContent"></main>'
        }
        Dome.innerHTML = content
        updateDome(Dome, add_script)
    } else if (Dome) {
        if (!helper){
            helper = document.createElement("div");
            helper.innerHTML = content
        }
        helper.id = id
        helper.classList.add("Mcontent")
        updateDome(helper, add_script)
        if (insert){
            Dome.insertBefore(helper, Dome.firstChild);
        }else{
            Dome.appendChild(helper)
        }


    }else {
        console.error("No Dome found", Dome, DOME)
    }

}

function dashboard_init() {

    init_d = true
    const helper = document.createElement("div");
    helper.id = "D-Provider"
    const helper_dome = document.getElementById("MainContent");
    helper_dome.classList.add("autoMarkdownVisible");
    if (helper_dome && window.history.state.preUrl){
        helper_dome.innerHTML = `<div class="inv-main" id="main" style="width: 90vw;text-align: center">
            <h1>Dashboard Provider</h1>
            <a href="/web/dashboards/user_dashboard.html"> user </a>
            <hr/>
            <a href="/web/dashboards/widgetbord.html"> desktop </a>
            <hr/>
            <a href="/web/dashboards/dashboard_builder.html"> Dev </a>
            </div>`
            updateDome(helper_dome)
        }
    const controls = document.getElementById("Nav-Main")
    if (controls){
        controls.appendChild(helper)
    }else{
        throw "No Nave found"
    }

    return helper

}

// Funktion zum Erstellen der Ladeansicht mit zufälligem Bild von Picsum
async function createLoadingView() {
    // Zufällige ID zwischen 1 und 1000 für verschiedene Bilder
    const randomId = Math.floor(Math.random() * 1000) + 1;
    const imageUrl = `https://picsum.photos/id/${randomId}/800/600`;

    // Alternative APIs, falls Picsum nicht funktioniert
    const fallbackApis = [
        'https://source.unsplash.com/random/800x600',
        'https://api.lorem.space/image/movie?w=800&h=600',
    ];

    return `
        <div class="loading-container">
            <div class="loading-image-container">
                <img src="${imageUrl}"
                     alt="Loading"
                     class="loading-image"
                     onerror="this.onerror=null; this.src='${fallbackApis[0]}'">
                <div class="loading-placeholder">
                    <div class="loading-spinner"></div>
                </div>
            </div>
            <div class="loading-text">
                <h2>Wird vorbereitet...</h2>
                <div class="loading-progress"></div>
            </div>
        </div>
    `;
}

// Verbesserte CSS-Styles
const loadingStyles = `
    <style>
        .loading-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background: rgba(0, 0, 0, 0.9);
            z-index: 9999;
            transition: opacity 0.5s ease;
        }

        .loading-image-container {
            position: relative;
            width: 80vw;
            max-width: 800px;
            height: 60vh;
            max-height: 600px;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 0 30px rgba(0, 0, 0, 0.5);
        }

        .loading-image {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: opacity 0.3s ease;
        }

        .loading-placeholder {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.2);
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .loading-text {
            margin-top: 20px;
            color: white;
            text-align: center;
            font-family: Arial, sans-serif;
        }

        .loading-text h2 {
            margin: 0;
            font-size: 24px;
            font-weight: 300;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
        }

        .loading-spinner {
            width: 40px;
            height: 40px;
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-top: 4px solid #ffffff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        .loading-progress {
            margin-top: 10px;
            width: 200px;
            height: 4px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 2px;
            overflow: hidden;
        }

        .loading-progress::after {
            content: '';
            display: block;
            width: 40%;
            height: 100%;
            background: white;
            animation: progress 2s ease-in-out infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @keyframes progress {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(350%); }
        }
    </style>
`;

function createLoadingManager() {
    const events = new EventTarget();
    let resourcesLoaded = 0;
    let totalResources = 0;

    return {
        // Signalisiert, dass eine neue Ressource geladen wird
        addResource() {
            totalResources++;
        },
        // Signalisiert, dass eine Ressource fertig geladen wurde
        resourceLoaded() {
            resourcesLoaded++;
            const progress = (resourcesLoaded / totalResources) * 100;
            events.dispatchEvent(new CustomEvent('loadingProgress', {
                detail: { progress }
            }));

            if (resourcesLoaded >= totalResources) {
                events.dispatchEvent(new CustomEvent('loadingComplete'));
            }
        },
        // Wartet auf das Laden aller Ressourcen
        waitForLoading() {
            return new Promise((resolve) => {
                if (resourcesLoaded >= totalResources) {
                    resolve();
                } else {
                    events.addEventListener('loadingComplete', () => resolve());
                }
            });
        },
        // Event-Listener für den Ladefortschritt
        onProgress(callback) {
            events.addEventListener('loadingProgress', (e) => callback(e.detail.progress));
        },
        // Setzt den Loading-Manager zurück
        reset() {
            resourcesLoaded = 0;
            totalResources = 0;
        }
    };
}

// Erstelle eine Instanz des Loading-Managers



// Router
function router(url, extend = false, id = "main", Dome = DOME, callback=null) {
    const isFirstLoad = !sessionStorage.getItem('appInitialized');
    let loadingManager =  {resourceLoaded() {},}
    if (isFirstLoad) {
        const loadingElement = document.createElement('div');
        createLoadingView().then(loadingView => {
            loadingElement.innerHTML = loadingStyles + loadingView;
            document.body.appendChild(loadingElement);

            const progressBar = loadingElement.querySelector('.loading-progress');

            // Überwache den Ladefortschritt
            loadingManager.onProgress((progress) => {
                progressBar.style.setProperty('--progress', `${progress}%`);
            });

            // Definiere minimale und maximale Anzeigezeit
            const MIN_DISPLAY_TIME = 750; // .75 Sekunde minimum
            const MAX_DISPLAY_TIME = 25000; // 25 Sekunden maximum

            // Starte Timer für minimale Anzeigezeit
            const minTimePromise = new Promise(resolve =>
                setTimeout(resolve, MIN_DISPLAY_TIME)
            );

            // Timer für maximale Anzeigezeit
            const maxTimePromise = new Promise(resolve =>
                setTimeout(resolve, MAX_DISPLAY_TIME)
            );

            // Warte auf das Laden der App
            const loadingPromise = loadingManager.waitForLoading();

            // Warte auf minimale Zeit UND Laden oder maximale Zeit
            Promise.race([
                Promise.all([minTimePromise, loadingPromise]),
                maxTimePromise
            ]).then(() => {
                // Sanft ausblenden
                loadingElement.style.opacity = '0';
                setTimeout(() => {
                    loadingElement.remove();
                    sessionStorage.setItem('appInitialized', 'true');
                }, 500);
            });
        });
            loadingManager = createLoadingManager();
            loadingManager.addResource();
            loadingManager.addResource();
            loadingManager.addResource();
            loadingManager.addResource();
            loadingManager.addResource();
            loadingManager.addResource();
    }

    // console.log("[url]:", url.toString())
    if (url.startsWith(state.TBv.base)) {
        url = url.replace(state.TBv.base, "");
    }

    url = url.startsWith("/") ? url : "/" + url

    if (url === "/" || url === "" || url === "/web" || url === "/web/") {
        url = "/web/core0/index.html"
    }/*
    if (url === "/web" || url === "/web/") {
        url = "/web/mainContent.html"
    }
    if (url === "/index.html" && Dome === DOME){
        url = "/web/mainContent.html"
    }*/

    let uri= url
    let is_d = false

    if (!init_d && uri.endsWith("/web/dashboard") && DOME){
        Dome = dashboard_init()
        is_d = true
    }else if (!init_d && uri.includes("/web/dashboard")){
        router("/web/dashboard")
    } else if (init_d && uri.endsWith("/web/dashboard") && Dome === DOME){
        uri = "/web/dashboards/user_dashboard.html"
    }

    async function fetchFromLocal(uri) {
        const localUrl = `${window.location.origin}${uri}`;
        try {
            const response = await fetch(localUrl);
            if (response.ok) {
                return await response.text(); // File found locally
            } else {
                console.log(`Local file not found: ${localUrl}, status: ${response.status}`);
                return null;
            }
        } catch (error) {
            console.log(`Error fetching from local: ${error}`);
            return null;
        }
    }

    // Function to fetch the file from the backend API running on port 5000
    async function fetchFromBackend(uri) {
        const backendUrl = `http://localhost:5000${uri}`;
        try {
            const response = await fetch(backendUrl);
            if (response.ok) {
                return await response.text(); // Fetched from backend
            } else {
                console.log(`Backend file not found: ${backendUrl}, status: ${response.status}`);
                return null;
            }
        } catch (error) {
            console.log(`Error fetching from backend: ${error}`);
            return null;
        }
    }

    async function fetchFromRBackend(uri) {
        const backendUrl = `{window.location.origin}/${uri}`;
        try {
            const response = await fetch(backendUrl);
            if (response.ok) {
                return await response.text(); // Fetched from backend
            } else {
                console.log(`Backend file not found: ${backendUrl}, status: ${response.status}`);
                return null;
            }
        } catch (error) {
            console.log(`Error fetching from backend: ${error}`);
            return null;
        }
    }


    console.log("[uri]:", uri.toString(), window.location.origin)

    let content = null
    const isProduction = window.location.origin.includes('3001')

    setTimeout(async ()=>{

        if (!isProduction){
            content = await fetchFromLocal(uri);
        }
        loadingManager.resourceLoaded();
        // If the file wasn't found locally, fetch it from the backend API
        if (!content) {
            content = await fetchFromBackend(uri);
        }
        loadingManager.resourceLoaded();
        if (!content) {
            content = await loadHtmlFile(uri);
        }
        loadingManager.resourceLoaded();
        if (!content) {
            content = await fetchFromRBackend(uri);
        }
        loadingManager.resourceLoaded();
        console.log("[content]:", content.toString().length, content)

        if (content.toString().startsWith("HTTP error!") && content.toString().includes("404")){
            router("/web/assets/404.html", extend, id, DOME)
            if (is_d){init_d = false}
        }else if (content.toString().startsWith("HTTP error!") && content.toString().includes("401")){
            router("/web/assets/401.html", extend, id, DOME)
            if (is_d){init_d = false}
        }else{
            if (!content.toString()){
                console.error("could not load html file", content)
                content = "<h1>Error "+content.toString()+"</h1> <br/> <a href='/'>Home</a>"
            }

            renderer({content, extend, id, Dome})
            Dome.scrollIntoView({ behavior: "instant" });
            loadingManager.resourceLoaded();
        }

    }, 0)

    const preUrl = window.history.state?window.history.state.url:undefined

    if (!extend && !window.__TAURI__){
        window.history.pushState({ url: cleanUrl(url), preUrl, TB: state.TBv, TBc: state.TBc }, "", cleanUrl(url));
    }

    if (callback){
        callback();
    }
    loadingManager.resourceLoaded();
}

function cleanUrl(url) {
    // Use a regular expression to match all occurrences of 'http://' or 'https://'
    const regex = /(https?:\/\/)/g;

    // Split the URL by the regex and filter out empty strings
    const parts = url.split(regex).filter(part => part !== '');

    // If there are multiple parts, keep only the last one
    if (parts.length > 1) {
        return `${parts[parts.length - 1]}`;
    }

    // If there's only one part, return it
    return url;
}

export async function loadHtmlFile(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            return `HTTP error! status: ${response.status}`;
        }
        return await response.text();
    } catch (error) {
        console.error("Could not load the HTML file:", error);
        return error
    }
}

export async function loadHtmlFile_(url) {
    const localUrl = `${window.location.origin}/${url}`; // Local server
    const localBackendUrl = `http://localhost:5000/${url}`; // Local backend at different port
    const localFolderPath = url; // Local folder (copied assets)
    const remoteUrl = `${process.env.TOOLBOXV2_REMOTE_BASE}/${url}`; // Remote backend

    // Try to fetch from local URL
    try {
        const localResponse = await fetch(localUrl);
        if (localResponse.ok) {
            return await localResponse.text();
        }
        console.log(`Local fetch failed: ${localResponse.status}`);
    } catch (error) {
        console.log("Error fetching from local URL:", error);
    }

    // Try to fetch from local backend running on a different port
    try {
        const backendResponse = await fetch(localBackendUrl);
        if (backendResponse.ok) {
            return await backendResponse.text();
        }
        console.log(`Local backend fetch failed: ${backendResponse.status}`);
    } catch (error) {
        console.log("Error fetching from local backend:", error);
    }

    // Try to fetch from local folder (copied assets)
    try {
        const localFileResponse = await fetch(localFolderPath);
        if (localFileResponse.ok) {
            return await localFileResponse.text();
        }
        console.log(`Local file fetch failed: ${localFileResponse.status}`);
    } catch (error) {
        console.log("Error fetching from local folder:", error);
    }

    // Fallback to remote backend
    try {
        const remoteResponse = await fetch(remoteUrl);
        if (remoteResponse.ok) {
            return await remoteResponse.text();
        }
        return `HTTP error! status: ${remoteResponse.status}`;
    } catch (error) {
        console.error("Could not load the HTML file from remote backend:", error);
        return error;
    }
}


// State

function initState({remote= false, local=true}){
    let TBc = null
    if (local){
        TBc = window.localStorage.getItem("TBc")
    }

    if (TBc){
        state.TBc = JSON.parse(TBc)
    }
    if (!state.TBc){
        console.error("Could not load the TBc Data:", TBc)
        state.TBc = []
    }
}

function persistState({remote=false, local=true}){
    if (local){
        window.localStorage.setItem("TBc", JSON.stringify(state.TBc))
    }
    if (remote) {
        httpPostData("CloudM", "SaveWebUserData", state.TBc,
            (e)=>{e.log()},
            (s)=>{s.log()})
    }
}

function addState(key, value){
    state.TBc[key] = value
}

function getState(key){
    return state.TBc[key]
}

// lazy load

function getWidgetUtility(){

}

function getOverlayUtility(){

}


// Htmx

async function handleHtmxAfterRequest(event) {
    let xhr = event.detail.xhr; // Der XMLHttpRequest
    let response = xhr.response;

    let json = {}

    if (response.toString().startsWith("{") && response.toString().endsWith("}")){

        json = JSON.parse(response);

    }else {

        json = response.toString()
    }

    try {
        // Versuchen Sie, die Antwort als JSON zu parsen
        if (json.toString().startsWith('<')){
            if (event.detail && event.detail.target) {
                // console.log("event.detail.target", event.detail.target)
                let target = event.detail.target
                target.innerHTML = json;
                updateDome(target)
                return "successCallback(result);"
            }
        }
        const result = wrapInResult(json, true)
        result.log()

        //console.log("result:", result.origin.at(2) === 'REMOTE')
        //console.log(result.get().toString().startsWith('<'), result.origin.at(2))

        if (result.error !== ToolBoxError.none) {
            // Handle error case
            return "errorCallback(result);"
        } else if (result.get().toString().startsWith('<')) {
            // Handle success case
            if (event.detail && event.detail.target) {
                // console.log("event.detail.target", event.detail.target)
                let target = event.detail.target
                target.innerHTML = result.get();
                updateDome(target)
            }
            return "successCallback(result);"
        }else if (result.origin.at(2) === 'REMOTE') {
            await rendererPipeline(result.get()).then(r => {
                // console.log("Rendering Don")
                return "successCallback(result);"
            })
        }

    } catch (e) {
        console.log("Result Parsing error", e)
    }
}

// alnimation

function parseInput(input) {
    // Regular expression to match the input pattern
    const regex = /^([RPYZ])(\d+)([+-])(\d+)(\d+)$/;

    // Use regex to extract animation parameters
    const match = input.match(regex);

    if (match) {
        // Extract animation parameters
        const animationType = match[1];
        const repeat = parseInt(match[2]);
        const direction = match[3] === "+" ? 1 : -1;
        const speed = parseInt(match[4]);
        const complex = parseInt(match[5])+1+repeat*1000;

        // Return an object with extracted parameters
        return {
            animationType,
            repeat,
            direction,
            speed,
            complex
        };
    } else {
        // Return null if input does not match the pattern
        return null;
    }
}

function animator(input, after=null, f=0.02, s= 12) {
    const animationParams = parseInput(input);

    console.log("Animation:", animationParams)

    function repeatZoom(direction, speed, repeat, smo=1) {
        if (repeat > 0) {
            // Call Set_zoom after 1000 milliseconds
            setTimeout(() => {
                Set_zoom(direction * speed / 10);
                // Call repeatZoom recursively with repeat - 1
                repeatZoom(direction, speed, repeat - 1, smo);
            }, 1000/smo);
        }
    }

    if (animationParams) {
        const { animationType, repeat, direction, speed, complex } = animationParams;

        // Determine animation based on animationType
        switch(animationType) {
            case "Y":
                Set_animation_xyz(0, f * direction * speed, 0, s * speed)
                break;
            case "R":
                Set_animation_xyz(f * direction * speed, 0, 0, s * speed)
                break;
            case "P":
                Set_animation_xyz(0, 0, f * direction * speed, s * speed)
                break;
            case "Z":
                let smo = s*3
                repeatZoom(direction, speed, repeat*smo, smo);
                break;
            default:
                console.log("Invalid animation type.");
        }

        setTimeout(()=> {
            if (after && typeof after === "string") {
                if (after.includes(":")){
                    after = after.split(":");
                    animator(after.pop(), after.join(":"));
                }else{
                    animator(after, null)
                }
            }else if (after){
                after()
            }else if (after === null){
                EndBgInteract()
                console.log("END ALIM")
            }
        }, complex)
    } else {
        console.log("Invalid input format.");
    }

}

// Service worker

state.TBf.unRegisterServiceWorker()

// nav


window.TBf.initVar("linkToggle", [false])

function linksInit(){

    const linksButton = document.querySelector('#links');
    if (!linksButton){
        return;
    }
    linksButton.style.transition = "transform 0.5s ease";
    let linksContent =  `<div class="links-form" >
            <ul>
                <li><a href="/">Home</a></li>
                <li><a href="/web/mainContent.html">Apps</a></li>
                <hr style="margin: -0.25vh 0"/>
                <li><a href="/web/assets/login.html">Login</a></li>
                <li><a href="/web/assets/signup.html">Sign Up</a></li>
                <li><a href="/web/core0/Poffer/PublicDashboard.html">Offer</a></li>
                <hr style="margin: -0.25vh 0"/>
                <li><a href="/web/assets/terms.html">Terms and Conditions</a></li>
                <li><a href="/web/core0/kontakt.html">Contact</a></li>
            </ul>
        </div>`

    let linksIcon = document.getElementById("linkIcon");
    let linksAcker = document.getElementById("overlay");

    async function openLinksOverlay(){


        linksButton.style.transform = 'rotate(360deg)  scale(1.26)';
        // linksButton.style.scale = 1.26
        linksButton.style.zIndex = '9999';

        linksAcker.insertAdjacentHTML('afterend', linksContent);
        updateDome(linksAcker.nextElementSibling, false, onCloseLinksOverlay)
        if (linksIcon){
            linksIcon.outerHTML = `<span id="linkIcon" class="plus material-symbols-outlined">close</span>`
            linksIcon = document.getElementById("linkIcon");
        }
        window.TBf.getVar("linkToggle")[0] = true
    }

    function onCloseLinksOverlay(){
        linksButton.style.transform = 'rotate(0deg) scale(1)';
        linksButton.style.zIndex = '2'
        const insertedContent = linksAcker.nextElementSibling;
        if (insertedContent) {
            insertedContent.classList.add("fadeOutToLeft")
            insertedContent.remove();
        }
        if (linksIcon){
            linksIcon.outerHTML = `<span id="linkIcon" class="plus material-symbols-outlined">menu</span>`
            linksIcon = document.getElementById("linkIcon");
        }
        window.TBf.getVar("linkToggle")[0] = false
    }

    linksButton.addEventListener('click', () => {

        if (!window.TBf.getVar("linkToggle")[0]){
            openLinksOverlay().then(r => console.log("Links opened"))
        }else{
            onCloseLinksOverlay()
        }

    });

}


setTimeout((function() {
  const STORAGE_KEY = 'cookieConsent';
  const savedSettings = localStorage.getItem(STORAGE_KEY);

  if (savedSettings) {
    const { analytics, mode } = JSON.parse(savedSettings);
    if (analytics) initPosthog(mode);
    return;
  }

  // Create minimalist banner
  const banner = document.createElement('div');
  banner.id = 'cookie-banner';
  banner.innerHTML = `
    <button id="close-banner" title="Accept">&times;</button>
    <h3>We value your privacy.</h3>
    </p> Data is keep privat!</p>
    <a href="/web/assets/terms.html">Terms and Conditions (8)</a>
    <br>
    <button id="accept-minimal" style="padding: 12px">Continue</button><label for="accept-minimal">Recommended</label>
     <h4>Cookies Configuration</h4>
    <button id="show-complex" style="margin-left:8px; padding: 6px; text-decoration:underline">
      Options
    </button><label for="show-complex">Advanced</label>
  `;
  document.body.appendChild(banner);

  // Complex modal structure
  const modal = document.createElement('div');
  modal.id = 'cookie-modal';
  modal.innerHTML = `
    <div class="modal-screen active" data-step="1">
      <h3>Enhanced Privacy Controls</h3>
      <div class="complex-option" data-action="accept">
        <h4>Optimal Experience (Recommended)</h4>
        <p>Allow all features for full functionality</p>
      </div>
      <div class="complex-option" data-action="customize">
        <h4>Custom Configuration</h4>
        <p>Disable Profile Analytics</p>
        <p>Disable Analytics</p>
      </div>
    </div>

    <div class="modal-screen" data-step="2">
      <h3>Technical Preferences</h3>
      <p>Profile Analytics ar disabled press 'Continue'</p>

      <div class="complex-option" onclick="this.classList.toggle('selected')">
        <h4>Performance Analytics</h4>
        <p>Disabling may impact site optimization</p>
        <p>click to enable disable</p>
      </div>
      <input type="checkbox" class="hidden-toggle">

      <p>Continue without Profile Analytics</p>

      <div style="margin-top:2rem">
        <button id="showScreen-1" style="padding: 8px">Back</button>
        <button id="finalize-settings" style="padding: 6px">Continue</button>
      </div>
    </div>

    <div class="modal-screen" data-step="3">
      <h3>Confirm Your Selection</h3>
      <p>By disabling enhanced features, you may experience:</p>
      <ul>
        <li>Reduced personalization</li>
        <li>Limited functionality</li>
        <li>No Profile Analytics</li>
        <li>No Analytics</li>
      </ul>
      <button  id="showScreen-2" style="padding: 6px">Modify Choices</button>
      <button id="confirm-complex" style="padding: 8px">Accept Limitations</button>
    </div>
  `;
  document.body.appendChild(modal);

  // Interaction handlers
  document.getElementById('accept-minimal').addEventListener('click', () => {
    saveSettings(true, 'always');
    banner.remove();
  });

  // Interaction handlers
  document.getElementById('close-banner').addEventListener('click', () => {
    saveSettings(true, 'always');
    banner.remove();
  });

  document.getElementById('show-complex').addEventListener('click', () => {
    modal.style.display = 'block';
    showScreen(1);
  });

  modal.addEventListener('click', e => {
    const option = e.target.closest('.complex-option');
    if (!option) return;

    const action = option.dataset.action;
    if (action === 'accept') {
      saveSettings(true, 'always');
      modal.style.display = 'none';
      banner.remove();
    }
    if (action === 'customize') showScreen(2);
  });

  document.getElementById('finalize-settings').addEventListener('click', () => {
    const analyticsEnabled = !!modal.querySelector('.selected');
    if (analyticsEnabled) showScreen(3);
    else {
      saveSettings(true, 'identified_only');
      modal.style.display = 'none';
      banner.remove();
    }
  });

  document.getElementById('showScreen-1').addEventListener('click', () => {
   showScreen(1)
  });
  document.getElementById('showScreen-2').addEventListener('click', () => {
   showScreen(2)
  });

  document.getElementById('confirm-complex').addEventListener('click', () => {
    saveSettings(false, 'none');
    modal.style.display = 'none';
    banner.remove();
  });

  function showScreen(step) {
    modal.querySelectorAll('.modal-screen').forEach(screen => {
      screen.classList.toggle('active', screen.dataset.step == step);
    });
  }

  function saveSettings(analytics, mode) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ analytics, mode }));
    if (analytics) initPosthog(mode);
  }

  function initPosthog(mode) {
    !function(t,e){var o,n,p,r;e.__SV||(window.posthog=e,e._i=[],e.init=function(i,s,a){function g(t,e){var o=e.split(".");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}(p=t.createElement("script")).type="text/javascript",p.crossOrigin="anonymous",p.async=!0,p.src=s.api_host.replace(".i.posthog.com","-assets.i.posthog.com")+"/static/array.js",(r=t.getElementsByTagName("script")[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a="posthog",u.people=u.people||[],u.toString=function(t){var e="posthog";return"posthog"!==a&&(e+="."+a),t||(e+=" (stub)"),e},u.people.toString=function(){return u.toString(1)+".people (stub)"},o="init capture register register_once register_for_session unregister unregister_for_session getFeatureFlag getFeatureFlagPayload isFeatureEnabled reloadFeatureFlags updateEarlyAccessFeatureEnrollment getEarlyAccessFeatures on onFeatureFlags onSessionId getSurveys getActiveMatchingSurveys renderSurvey canRenderSurvey getNextSurveyStep identify setPersonProperties group resetGroups setPersonPropertiesForFlags resetPersonPropertiesForFlags setGroupPropertiesForFlags resetGroupPropertiesForFlags reset get_distinct_id getGroups get_session_id get_session_replay_url alias set_config startSessionRecording stopSessionRecording sessionRecordingStarted captureException loadToolbar get_property getSessionProperty createPersonProfile opt_in_capturing opt_out_capturing has_opted_in_capturing has_opted_out_capturing clear_opt_in_out_capturing debug getPageViewId".split(" "),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])},e.__SV=1)}(document,window.posthog||[]);
    posthog.init('phc_zsEwhB79hF41y7DjaAkGSExrJNuPffyOUKlU1bM0r3V', {api_host: 'https://eu.i.posthog.com', person_profiles: mode});
  }
}), 2100)
