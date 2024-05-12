import {rendererPipeline} from "./web/scripts/WorkerSocketRenderer";
import {AuthHttpPostData, httpPostData, httpPostUrl, ToolBoxError} from "./web/scripts/httpSender";
import "./web/node_modules/htmx.org";
import {addRenderer, EndBgInteract, Set_animation_xyz, StartBgInteract, toggleDarkMode} from "./web/scripts/scripts.js";

const rpIdUrl_f = ()=> {
    if (window.location.href.match("localhost")) {
        return "http://localhost:" + window.location.port
    } else {
        return "https//simplecore.app"
    }
}

let DOME;
let isHtmxAfterRequestListenerAdded = false;

const state = {
    TBf: {
        router,
        renderer,
        getState,
        addState,
        getWidgetUtility,
        getOverlayUtility,
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
        getModule,
        animator,
    },
    TBv:{
        base: rpIdUrl_f(),
        user: null,
        session: null,
    },
    TBc:{

    },
    TBm:{

    }
}

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
}else{

    router("/index.html", false, "root", document.body)
    updateDome(document.body)

    let stoUrl = window.location.href
    window.history.pushState({ url: stoUrl, TB: state.TBv, TBc: state.TBc }, "", stoUrl);
    console.log("saved:", stoUrl, "/")
    // window.location.href = "/"
     setTimeout(()=>{
     DOME = document.getElementById("MainContent")
     console.log("c DOME:", DOME)
     const baseElement = document.createElement("base");
     baseElement.href = state.TBv.base + '/'
     document.head.appendChild(baseElement);
     initDome()
     updateDome(DOME)

     addRenderer()
     // if (stoUrl && stoUrl !== "/index.html" && stoUrl !== "index.html" && stoUrl !== ""){stoUrl="/web/core0/index.html"}
     // router(stoUrl)
     }, 150)
    // router("/index.html", false, "root", document.body)
    // setTimeout(()=>{
    //     DOME = document.getElementById("MainContent")
    //     console.log("c DOME:", DOME)
    //     initDome()
    //     if (stoUrl && stoUrl !== "/index.html" && stoUrl !== "index.html" && stoUrl !== ""){stoUrl="/web/core0/index.html"}
    //     router(stoUrl)
    // }, 50)

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

// Renderer

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

function updateDome(dome){

    function addSrc(script){
        let scriptContent = script.textContent;

        // Create a new Blob containing the script content
        let blob = new Blob([scriptContent], { type: "application/javascript" });

        // Create a URL for the Blob
        return  URL.createObjectURL(blob);

    }

    dome.querySelectorAll("a").forEach(function(link) {

        if (link.href.toString().startsWith(state.TBv.base)) {
            let route = "/" + new URL(link.href).pathname.split('/').slice(1).join('/');
            // console.log("REGISTERED:", route)

            link.addEventListener("click", function(e) {
                e.preventDefault();
                linkEffect()
                console.log("REGISTERED: [route]", route)
                router(route); // Use link.href to get the URL of the clicked link
                // linkEffect()
            });
        }else {
            // console.log("Scip external link to:", link.href)
        }

    });

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
        // console.log("Adding ", script.src);
        js.src = src;
        console.log(js, src !== state.TBv.base+'/index.js')
        if (src.slice(0, state.TBv.base.length+'/index.js'.length) !== state.TBv.base+'/index.js'){
            dome.appendChild(js);
        }
    });

    htmx.process(dome);
}

function initDome(){
    const darkModeToggle = document.getElementById('darkModeToggle');
    if(!darkModeToggle){
        console.error("No toggle found")
        return
    }
    darkModeToggle.addEventListener('change', (event) => {

        if (event.target.checked) {
            document.getElementById('toggleLabel').innerHTML = `<span class="material-symbols-outlined">
light_mode
</span>`;
        } else {
            document.getElementById('toggleLabel').innerHTML = `<span class="material-symbols-outlined">
dark_mode
</span>`;
        }
        toggleDarkMode()
    });

    document.body.addEventListener('mousedown', () => {StartBgInteract()});
    document.body.addEventListener('mouseup', () => {EndBgInteract()});

    document.body.addEventListener("touchstart", () => {StartBgInteract()});
    document.body.addEventListener("touchend", () => {EndBgInteract()});


}

function renderer(content, extend=false, id="main", Dome = DOME) {

    if (!extend && Dome){
        Dome.innerHTML = content
        updateDome(Dome)
    } else if (Dome) {
        const helper = document.createElement("div");
        helper.innerHTML = content
        helper.id = id
        updateDome(helper)
        Dome.appendChild(helper)
    }else {
        console.error("No Dome found", Dome, DOME)
    }

}

// Router
function router(url, extend=false, id="main", Dome=DOME){

    console.log("[url]:", url.toString())
    if (url.startsWith(state.TBv.base)){
        url = url.replace(state.TBv.base, "");
    }

    url = url.startsWith("/") ? url : "/" + url

    if (url === "/" || url === ""){
        url = "/web/core0/index.html"
    }
    if (url === "/web" || url === "/web/"){
        url = "/web/mainContent.html"
    }
    if (url === "/index.html" && Dome === DOME){
        url = "/web/mainContent.html"
    }

    let uri= url

    // try{
    //     uri = new URL(url)
    // }catch (e){
//
    //     uri = new URL(window.location.origin + url)
    // }

    console.log("[uri]:", uri.toString())

    let content = null
    setTimeout(async ()=>{
        content = await loadHtmlFile(uri)

        // console.log("[content]:", content.toString(), typeof content)

        if (content.toString().startsWith("HTTP error!") && content.toString().includes("404")){
            router("/web/assets/404.html", extend, id, Dome)
        }else if (content.toString().startsWith("HTTP error!") && content.toString().includes("401")){
            router("/web/assets/401.html", extend, id, Dome)
        }else{
            if (!content.toString()){
                console.error("could not load html file", content)
                content = "<h1>Error "+content.toString()+"</h1> <br/> <a href='/'>Home</a>"
            }
            renderer(content, extend, id, Dome)
        }

    }, 1)

    window.history.pushState({ url: url, TB: state.TBv, TBc: state.TBc }, "", url);
    window.TBf = state.TBf
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

async function getModule(name){
    if (!name in state.TBm){
        try{
            state.TBm[name] =  await import(`./web/scrips/modules/${name}.js`)
        }catch(e){
            state.TBm[name] = await import(`/api/${name}/scriptJs`)
        }

    }
    return state.TBm[name]
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

    try {
        // Versuchen Sie, die Antwort als JSON zu parsen
        json = JSON.parse(response);

    } catch (e) {
        // Wenn ein Fehler auftritt, handelt es sich wahrscheinlich nicht um eine JSON-Antwort
        // oder das JSON-Format entspricht nicht der Erwartung
        console.log("Invalid JSON error", e)
        return "Error"
    }

    try {
        // Versuchen Sie, die Antwort als JSON zu parsen
        const result = wrapInResult(json, true)
        result.log()

        console.log("result:", result.origin.at(2) === 'REMOTE')
        console.log(result.get().toString().startsWith('<'), result.origin.at(2))

        if (result.error !== ToolBoxError.none) {
            // Handle error case
            return "errorCallback(result);"
        } else if (result.get().toString().startsWith('<')) {
            // Handle success case
            if (event.detail && event.detail.target) {
                console.log("event.detail.target", event.detail.target)
                let target = event.detail.target
                target.innerHTML = result.get();
            }
            return "successCallback(result);"
        }else if (result.origin.at(2) === 'REMOTE') {
            await rendererPipeline(result.get()).then(r => {
                console.log("Rendering Don")
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
    const regex = /^([RPY])(\d+)([+-])(\d+)(\d+)$/;

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

function animator(input, after=null) {
    const animationParams = parseInput(input);

    console.log("Animation:", animationParams)

    if (animationParams) {
        const { animationType, repeat, direction, speed, complex } = animationParams;

        // Determine animation based on animationType
        switch(animationType) {
            case "Y":
                Set_animation_xyz(0, 0.02 * direction * speed, 0, 8 * speed)
                break;
            case "R":
                Set_animation_xyz(0.02 * direction * speed, 0, 0, 8 * speed)
                break;
            case "P":
                Set_animation_xyz(0, 0, 0.02 * direction * speed, 8 * speed)
                break;
            default:
                console.log("Invalid animation type.");
        }

        setTimeout(()=> {
            console.log("END ALIM", complex)
            EndBgInteract()
            if (after && typeof after === "string") {
                animator(after)
            }else if (after){
                after()
            }
        }, complex)
    } else {
        console.log("Invalid input format.");
    }

}

// Service worker

state.TBf.unRegisterServiceWorker()


