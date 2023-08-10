let initChatOnce = false;
let selectedTask = {"name":"Welcome","task":{"use":"agent","name":"self","args":"Please introduce yourself to the user. Tell the user abut yur goals and capabilities. add 2 ideas for the user to start using you","return":"$ret"}};
function initChat() {
    console.log("initChat 0")
    if (initChatOnce){
        console.log("already inited true")
        return
    }

    console.log("initChat 1")
    initChatOnce = true;
    const docChaForm = document.getElementById("chat-form");
    const taskSelector = document.getElementById("taskSelector");
    if (!docChaForm){
        console.log("NO Chat Form found")
    }

    const messageInput = document.getElementById("message-input");
    WorkerSocketResEvents.push((data)=>{
        if (data.res){
            console.log(data.res)
            displayLoadingSpinner(false);
            displayMessage(data.res, "bot-message");
            messageInput.value = "";
        }
    })
    docChaForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const messageText = messageInput.value.trim();
        if (messageText) {
            displayMessage(messageText, "user-message");
            displayLoadingSpinner(true);
            await sendMessage(messageText);
        }
        loadTasks()
    });

    taskSelector.addEventListener('change', (event)=> {
        const tasks = JSON.parse(localStorage.getItem('tasks')) || [];
        selectedTask = tasks.find(t => t.name === event.target.value);
        console.log("selectedTask:", event.target.value, selectedTask)
    });
    loadTasks();

    function loadTasks() {
        const tasks = JSON.parse(localStorage.getItem('tasks')) || [];
        while (taskSelector.firstChild) {
            taskSelector.removeChild(taskSelector.lastChild);
        }
        tasks.forEach(task => {
            const option = document.createElement('option');
            option.value = task['name'];
            option.textContent = task['name'];
            taskSelector.appendChild(option);
        });
    }



}


async function sendMessage(text) {
    console.log("sendMessage", initChatOnce)
    console.log("sendMessage:runMod-isaa")
    if (text===""){
        return
    }
    console.log("[selectedTask]:",selectedTask)
    const inputElement = document.getElementById('ChatActionSection').querySelector('button');
    const inputId = inputElement ? inputElement.id : 'Input not found';
    const IChain = selectedTask['task'];
    const dataToChair = {"widgetID": inputId, "task":text, "IChain": IChain};
    const sendMessage_message_d = JSON.stringify({"ChairData":true, "data":dataToChair});
  console.log("[IChain]", IChain)
  console.log("[dataToChair]", dataToChair)
  console.log("[sendMessage_message_d]", sendMessage_message_d)
    // const sendMessage_message = JSON.stringify({"ServerAction":"runMod", "name":"isaa","function":"run", "command":inputId, "data":subject
    //         {"token": "**SelfAuth**", "data":{
    //                 "name": "self",
    //                 "text": text
    //             }}});
    await WS.send(sendMessage_message_d);

}

function displayMessage(text, className) {
    const chatMessages = document.getElementById("chat-messages");
    const messageElement = document.createElement("div");
    messageElement.classList.add("message", className, "markdown");
    messageElement.textContent = text;
    chatMessages.prepend(messageElement);
}

function displayLoadingSpinner(show) {
    const loadingSpinner = document.getElementById("loading-spinner");
    loadingSpinner.style.display = show ? "block" : "none";
}

setTimeout(()=>{
    initChat()
    WS.send(JSON.stringify({"ServerAction":"runMod", "name":"isaa","function":"initIsaa", "command":"", "data":
            {"token": "**SelfAuth**", "data":{
                    "modis": [""],
                }}}));
}, 55)

