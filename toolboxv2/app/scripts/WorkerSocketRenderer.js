let WorkerSocketResEvents = []

function MainInit() {

    document.getElementById("InitMainButton").remove()

    const ws_id = localStorage.getItem("WsID");
    //var ws_id = "app-live-test-DESKTOP-CI57V1L1";
    let local_ws;
    if (WS === undefined){
        local_ws = new WebSocket("ws://localhost:5000/ws/" + ws_id);
        WS = local_ws;
    }else {
        local_ws = WS;
    }

    console.log("INIT")

    local_ws.onmessage = async function(event) {

        const data = JSON.parse(event.data);

        console.log("Receive data", data)

        if (data.hasOwnProperty('render')) {
            console.log("is renderer")
            const renderData = data.render;
            const { content, place, id, externals, placeholderContent } = renderData;

            console.log("Data", renderData)

            // Display the placeholder content
            displayPlaceholderContent(placeholderContent, place, id);

            console.log("displayPlaceholderContent")

            // Download and store the content on the page
            storeContent(content, place, id);

            console.log("storeContent")

            if (externals.length > 0){

                console.log("Start downloadExternalFiles")
                // Download external files using a background worker and custom event
                await downloadExternalFiles(externals);

                console.log("done downloadExternalFiles")
            }else {
                console.log("No file to download")
            }

            // Update the page with the downloaded content and external files
            updatePage(content, place, id);

            if (id==="infoText"){
                document.getElementById('infoPopup').style.display = 'block';
            }

            console.log("updatePage Done")

        }

        if (data.hasOwnProperty("res")){
            if (WorkerSocketResEvents.length){
                WorkerSocketResEvents.forEach(((callbacks)=>{
                    callbacks(data)
                }))
            }else{
                const infoTextVar = document.getElementById('infoText');
                if (infoTextVar){
                    infoTextVar.innerText = data.res;
                }
            }
        }
    };

    local_ws.onopen = async function(event) {
        console.log("local_ws.onopen:installMod-welcome")
        const init_do = localStorage.getItem("local_ws.onopen:installMod-welcome")
        if (init_do){
            if (init_do === 'true'){
                await local_ws.send(JSON.stringify({"ServerAction":"installMod", "name": "welcome"}));
            }else {
                await getsInit();
                document.getElementById('main').classList.remove('main-content')
            }
        }else {
            alert("Pleas Log or Sing In to Visit the DasBord")
        }
    };

    return local_ws
}

let WS = undefined


function displayPlaceholderContent(placeholderContent, place, id) {
    const targetElement = id ? document.getElementById(id) : document.querySelector(place);
    targetElement.innerHTML = placeholderContent;
}

function storeContent(content, place, id) {
    const targetElement = id ? document.getElementById(id) : document.querySelector(place);
    targetElement.dataset.content = content;
}

function downloadExternalFiles(externals) {

    for (const url of externals) {
        console.log("Testing url", url)
        if (url.endsWith("js")){
            console.log("Adding js", url)
            const js = document.createElement("script");
            js.type = "text/javascript";
            js.src = url;
            document.body.appendChild(js);
        }else{
            console.log("Need to add to sw for saving coming soon")
        }
    }
}

function updatePage(content, place, id) {
    const targetElement = id ? document.getElementById(id) : document.querySelector(place);
    targetElement.innerHTML = targetElement.dataset.content;
}

function sendMessage(event) {
    var input = document.getElementById("messageText")
    console.log("sendMessage:input", input.value)
    WS.send(input.value)
    input.value = ''
    event.preventDefault()
}


async function getsInit(){
    console.log("Init")
    await WS.send(JSON.stringify({"ServerAction":"getsMSG"}));
    await WS.send(JSON.stringify({"ServerAction":"getWidgetNave"}));
    await WS.send(JSON.stringify({"ServerAction":"getDrag"}));
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
