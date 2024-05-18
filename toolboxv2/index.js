import {rendererPipeline} from "./web/scripts/WorkerSocketRenderer";
import {AuthHttpPostData, httpPostData, httpPostUrl, ToolBoxError, wrapInResult} from "./web/scripts/httpSender";
import "./web/node_modules/htmx.org";
import {addRenderer, EndBgInteract, Set_animation_xyz, StartBgInteract, toggleDarkMode} from "./web/scripts/scripts.js";

const rpIdUrl_f = ()=> {
    if (window.location.href.match("localhost")) {
        return "http://localhost:" + window.location.port
    } else {
        return "https//simplecore.app"
    }
}

let init_d = false
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
        loadHtmlFile,
        processRow:updateDome,
        initVar: (v_name, v_value)=>{if(!state.TBc[v_name]){state.TBc[v_name] = v_value}},
        delVar: (v_name)=>{delete state.TBc[v_name]},
        getVar: (v_name)=>{return state.TBc[v_name]},
        setVar: (v_name, v_value)=>{state.TBc[v_name] = v_value},
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

window.TBf = state.TBf

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

    // document.body.innerHTML = '';

    router("/index.html", false, "root", helper_dome)

    // updateDome(helper_dome)

    window.history.pushState({ url: stoUrl, TB: state.TBv, TBc: state.TBc }, "", stoUrl);
    console.log("saved:", stoUrl)
    // window.location.href = "/"
     //setTimeout(()=>{
    // const bodyClone = document.body.cloneNode(true);


    // renderer(document.body.innerHTML, true, "main", helper_dome, false)

    setTimeout(()=>{
        if (firstDiv){
            renderer({content:helper_dome.innerHTML, insert:true, Dome:document.body})
        }
        // document.body.innerHTML = helper_dome.innerHTML;
         DOME = document.getElementById("MainContent")
         console.log("c DOME:", DOME)
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

        // renderer({extend:true, helper:bodyClone})
    }, 350)
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

function updateDome(dome, add_script=true){

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
            console.log("[TEST SRC]",js.src,script.src )
        }
        // console.log("[TEST SRC]",js.src===script.src )
        // console.log(js, src !== state.TBv.base+'/index.js')
        if (!document.querySelector('script[src="'+js.src+'"]') && src.slice(0, state.TBv.base.length+'/index.js'.length) !== state.TBv.base+'/index.js' && !script.src.endsWith("/@vite/client" )&& !script.src.includes("scripts/scripts.js")){
            console.log("Adding ", script.src);
            dome.appendChild(js);
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

    document.getElementById('toggleLabel').innerHTML = `<span class="material-symbols-outlined">
dark_mode
</span>`

    document.body.addEventListener('mousedown', () => {StartBgInteract()});
    document.body.addEventListener('mouseup', () => {EndBgInteract()});

    document.body.addEventListener("touchstart", () => {StartBgInteract()});
    document.body.addEventListener("touchend", () => {EndBgInteract()});

}

function renderer({content="", extend = false, id = "main", Dome = DOME, add_script = true, helper = undefined, insert=false}) {

    if (!extend && !insert && Dome){
        document.createElement('div').hasChildNodes()
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
        if (insert){
            Dome.insertBefore(helper, Dome.firstChild);
        }else{
            Dome.appendChild(helper)
        }
        updateDome(helper, add_script)

    }else {
        console.error("No Dome found", Dome, DOME)
    }

}

function dashboard_init() {

    init_d = true
    const helper = document.createElement("div");
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
    const controls = document.getElementById("Nav-Controls")
    if (controls){
        document.getElementById("Nav-Main").classList.add("none")
        controls.appendChild(helper)
    }else{
        throw "No Nave found"
    }

    return helper

}

// Router
function router(url, extend = false, id = "main", Dome = DOME) {

    // console.log("[url]:", url.toString())
    if (url.startsWith(state.TBv.base)) {
        url = url.replace(state.TBv.base, "");
    }

    url = url.startsWith("/") ? url : "/" + url

    if (url === "/" || url === "") {
        url = "/web/core0/index.html"
    }
    if (url === "/web" || url === "/web/") {
        url = "/web/mainContent.html"
    }
    if (url === "/index.html" && Dome === DOME){
        url = "/web/mainContent.html"
    }

    let uri= url

    let is_d = false

    if (!init_d && uri.endsWith("/web/dashboard") && DOME){
        Dome = dashboard_init()
        is_d = true
    }else if (!init_d && uri.includes("/web/dashboard")){
        router("/web/dashboard")
    }

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

        console.log("[content]:", content.toString().length)

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

        }

    }, 1)

    const preUrl = window.history.state?window.history.state.url:undefined

    window.history.pushState({ url: url, preUrl, TB: state.TBv, TBc: state.TBc }, "", url);
    window.TBf = state.TBf
    window.TBm = state.TBm
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
            state.TBm[name] =  await import(`/api/${name}/scriptJs`)
        }catch(e){
            state.TBm[name] = null
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

    if (response.toString().startsWith("{") && response.toString().endsWith("}")){

        json = JSON.parse(response);

    }else {

        json = response.toString()
    }

    try {
        // Versuchen Sie, die Antwort als JSON zu parsen
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
                Set_animation_xyz(0, 0.02 * direction * speed, 0, 12 * speed)
                break;
            case "R":
                Set_animation_xyz(0.02 * direction * speed, 0, 0, 12 * speed)
                break;
            case "P":
                Set_animation_xyz(0, 0, 0.02 * direction * speed, 12 * speed)
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
                    animator(after)
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


