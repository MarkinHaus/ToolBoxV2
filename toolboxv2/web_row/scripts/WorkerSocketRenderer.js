
let WorkerSocketResEvents = []
let WorkerSocketOneTimeEvents = []
let WorkerSocketOneTimeEventsDelIndexes = []

const rpIdUrl_f = ()=> {
    if (window.location.href.match("localhost")) {
        return "localhost"
    } else {
        return "simplecore.app"
    }
}
const rpIdUrl = rpIdUrl_f()
function WSInit() {

    const btn = document.getElementById("InitMainButton");
    if (btn){
        btn.remove()
    }

    const ws_id = localStorage.getItem("WsID");
    //var ws_id = "app-live-test-DESKTOP-CI57V1L1";
    let local_ws;
    if (WS === undefined){
        local_ws = new WebSocket("ws://"+rpIdUrl+":5000/ws/" + ws_id);
        WS = local_ws;
    }else {
        local_ws = WS;
    }

    console.log("INIT")

    local_ws.onmessage = async function(event) {

        const data = JSON.parse(event.data);

        console.log("Receive data", data)
        await rendererPipeline(data)

        if (data.hasOwnProperty("res")){
            if (WorkerSocketResEvents.length){
                WorkerSocketResEvents.forEach(((callbacks)=>{
                    callbacks(data)
                }))
            }
            const infoTextVar = document.getElementById('infoText');
            if (infoTextVar){
                infoTextVar.innerText = data.res;
            }
        }

        console.log("[WorkerSocketOneTimeEvents.length] = ", WorkerSocketOneTimeEvents.length)
        if (WorkerSocketOneTimeEvents.length){
            WorkerSocketOneTimeEvents.forEach((ob, index)=>{
console.log("[WorkerSocketOneTimeEvents.key] = ", ob.onKey)
console.log("[data.hasOwnProperty(key)] = ", data.hasOwnProperty(ob.onKey), data.hasOwnProperty('render'))
console.log("[data] = ", data)
console.log("[data] = ", ob.onKey in data)
console.log("[data] = ", data[ob.onKey])
                if (data.hasOwnProperty(ob.onKey)){
                    console.log("[WorkerSocketOneTimeEvents] = Running")
                    ob.func(data)
                    console.log("[WorkerSocketOneTimeEvents] = Don")
                }
            })
        }
        if (WorkerSocketOneTimeEventsDelIndexes.length){
            WorkerSocketOneTimeEventsDelIndexes.forEach((index)=>{
                WorkerSocketOneTimeEvents.slice(index, 1)
            })
        }

    };

    local_ws.onopen = async function(event) {
        const init_do = localStorage.getItem("local_ws.onopen:installMod-welcome")
        if (init_do){
            if (init_do === 'true'){
                await local_ws.send(JSON.stringify({"ServerAction":"installMod", "name": "welcome"}));
            }else {
                await getsInit();
                const mainElement = document.getElementById('main');
                if (mainElement){
                     mainElement.classList.remove('main-content')
                }
            }
        }else {
            alert("Pleas Log or Sing In to Visit the DasBord")
        }
    };

    return local_ws
}

export async function rendererPipeline(data){
    if (typeof data === "string"){
        data = JSON.parse(data)
    }
    if (data.hasOwnProperty('render')) {
        console.log("is renderer")
        const renderData = data.render;
        const { content, place, id, externals, placeholderContent } = renderData;
        // Display the placeholder content
        displayPlaceholderContent(placeholderContent, place, id);
        // Download and store the content on the page
        storeContent(content, place, id);
        if (externals.length > 0){

            console.log("Start downloadExternalFiles")
            // Download external files using a background worker and custom event
            downloadExternalFiles(externals);
            console.log("done downloadExternalFiles")
        }
        // Update the page with the downloaded content and external files
        updatePage(content, place, id);
        if (id==="infoText"){
            const infoPopup = document.getElementById('infoPopup')
            if (infoPopup){
                infoPopup.style.display = 'block';
            }

        }

    }
}

function callbackOncOn(func, key){
    const index = WorkerSocketOneTimeEvents.length
    const wrapper = (data)=>{
        console.log("[:EVENT:] on '", key, "' Triggert with ", data)
        try{
            func(data)
        }catch (e) {
            console.log(e)
        }
        WorkerSocketOneTimeEventsDelIndexes.push(index)
    }
    WorkerSocketOneTimeEvents.push({'onKey':key, 'func':wrapper})
}


let WS = undefined


function displayPlaceholderContent(placeholderContent, place, id) {
    const targetElement = id ? document.getElementById(id) : document.querySelector(place);
    if (targetElement){
        targetElement.innerHTML = placeholderContent;
    }
}

function storeContent(content, place, id) {
    const targetElement = id ? document.getElementById(id) : document.querySelector(place);
    if (targetElement){
        targetElement.dataset.content = content;
    }
}

function downloadExternalFiles(externals) {

    for (const url of externals) {
        console.log("Testing url", url)
        if (url.endsWith("js")){
            let existingScript = document.querySelector(`script[src="${url}"]`);
            if (existingScript) {
                existingScript.remove();
            }
            console.log("Adding js", url)
            const js = document.createElement("script");
            js.type = url.toString().includes("module") ? "module" : "text/javascript";
            js.src = url;
            document.body.appendChild(js);
        }else{
            console.log("Need to add to sw for saving coming soon")
        }
    }
}

function updatePage(content, place, id) {
    const targetElement = id ? document.getElementById(id) : document.querySelector(place);
    if (targetElement){
        targetElement.innerHTML = targetElement.dataset.content;
        htmx.process(targetElement);
    }
}

function sendMessageONmessageText(event) {
    var input = document.getElementById("messageText")
    if (!input){
        return
    }
    WS.send(input.value)
    input.value = ''
    event.preventDefault()
}


async function getsInit(){
    console.log("Init")
    await getContent("getsMSG");
    await getContent("getWidgetNave");
    await getContent("getDrag");
}

function createElementFromDict(dict) {
    // Create element
    let element = document.createElement(dict.tag);

    // Set attributes
    for (let attr in dict.attributes) {
        element.setAttribute(attr, dict.attributes[attr]);
    }

    // Set content
    element.innerHTML = dict.content;

    // Set event listeners
    for (let event in dict.events) {
        if (element.addEventListener) {
            element.addEventListener(event, dict.events[event]);
        } else if (element.attachEvent) {
            element.attachEvent('on' + event, dict.events[event]);
        }
    }

    // Return the created element
    return element;
}


