function init(ws_id) {

    //var ws_id = "app-live-test-DESKTOP-CI57V1L1";
    const ws = new WebSocket("ws://localhost:5000/ws/" + ws_id);

    console.log("INIT")

    ws.onmessage = async function(event) {

        const data = JSON.parse(event.data);

        console.log("Recevf data", data)

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
                console.log("No file downloadExternalFiles")
            }

            // Update the page with the downloaded content and external files
            updatePage(content, place, id);

            console.log("updatePage Done")
        }
    };

    ws.onopen = function(event) {
        ws.send("{'web-app':'online'}");
    };
}


function displayPlaceholderContent(placeholderContent, place, id) {
    const targetElement = id ? document.getElementById(id) : document.querySelector(place);
    targetElement.innerHTML = placeholderContent;
}

function storeContent(content, place, id) {
    const targetElement = id ? document.getElementById(id) : document.querySelector(place);
    targetElement.dataset.content = content;
}

function downloadExternalFiles(externals) {
    return new Promise((resolve) => {
        const worker = new Worker('externalFilesWorker.js');
        worker.postMessage({ externals });

        // Custom event to handle the completion of external files download
        document.addEventListener('externalFilesDownloaded', () => {
            resolve();
            worker.terminate();
        });

        worker.onmessage = function(event) {
            if (event.data.status === 'complete') {
                // Dispatch the custom event when external files are downloaded
                const externalFilesDownloadedEvent = new CustomEvent('externalFilesDownloaded');
                document.dispatchEvent(externalFilesDownloadedEvent);
            }
        };
    });
}

function updatePage(content, place, id) {
    const targetElement = id ? document.getElementById(id) : document.querySelector(place);
    targetElement.innerHTML = targetElement.dataset.content;
}
function sendMessage(event) {
    var input = document.getElementById("messageText")
    ws.send(input.value)
    input.value = ''
    event.preventDefault()
}
