const settingsWidgetTemplate = `
<style>
#editor {
    margin: 0 auto;
    padding: 20px;
    border-radius: 5px;
}

#editor select {
    width: 100%;
    padding: 5px;
    margin-bottom: 10px;
    border: 1px solid #ccc;
    border-radius: 3px;
}


#editor .key-value-group {
    border: 1px solid var(--text-color);
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 3px;
}



#editorFrame {
    width: 300px;
    margin: 0 auto;
    padding: 20px;
    border: 1px solid #ccc;
    border-radius: 5px;
}


.key-value-pair {
    display: flex;
    justify-content: space-between;
    margin-top: 10px;
}

</style>
    <div class="widget-inner widget draggable" id="settingsWidget-widget">
    <div class="widget-from" id="settingsWidget-widget-from">Controls</div>
        <span class="widget-close-button" id="settingsWidget-widget-close-button">X</span>
        <label for="search-input-mods"></label>
        <div style="    padding: 20px;
    border: 1px solid #ccc;
    border-radius: 5px;">
        <input type="text" id="search-input-mods" placeholder="Search..." list="mods-list">
        <datalist id="mods-list" class="mods-list">
            <option>Welcome</option>
            <option>Isaa</option>
        </datalist>
        <button id="install-mod">Install Selected Mod</button>
        <hr/>
            <div id="Mods" class="mod-Aria">

            </div>
            <hr/>
        </div>
        <hr/>
        <div id="editorFrame">
            <h1>Task Editor</h1>
            <div id="editor">
                <input type="text" id="task-input" placeholder="Enter Name" list="task-list">
                <datalist id="task-list">
                </datalist>
                <div id="task-form">
                    <!-- Task key-value pairs will be added here -->
                </div>
                <select id="key-select">
                    <option value=disabled selected>Select a key</option>
                    <option value="name" title="self|think|summary|todolist|search|execution">name</option>
                    <option value="args" title="$user-input">args</option>
                    <option value="return" title="$return">return</option>
                    <option value="mode" title="text|free|tools|conversation|planning|execution|generate">Mode</option>
                    <option value="completion-mode" title="text|chat|edit">Completion-Mode</option>
                    <option value="infos" title="$Infos">infos</option>
                    <option value="short-mem" title="full|summary|clear">Memory</option>
                    <option value="text-splitter" title=10000>text-splitter</option>
                    <!-- Add more options here -->
                </select>
                <button id=save-task>Save Task</button>
                <button id=new-task>New Task</button>
            </div>
        </div>
        <div class="widget-items" id="settingsWidget-widget-items"></div>
    </div>

`;
let workingTask = {}
let itemsContainerSettingsWidget = null
let MODLIST = []

function sendKeyValue(mod_name){
    const key_element = document.getElementById("mod-card-key-"+mod_name)
    const value_element = document.getElementById("mod-card-value-"+mod_name)
    let key = null
    let value = null
    if (key_element){
        key = key_element.value
        key_element.value = ''
    }
    if (value_element){
        value = value_element.value
        value_element.value = ''
    }
    if (key && value){
        setTimeout(async ()=>{
            await WS.send(JSON.stringify({"ServerAction":"addConfig", "key":key, "value":value}));
        }, 5)
    }
}

function createWidget(json, targetElement) {
    let data = typeof json === 'string' ? JSON.parse(json) : json;

    const widget = document.createElement('div');
    widget.innerHTML = settingsWidgetTemplate;

    const itemsContainer = widget.querySelector('.widget-items');
    const mod_content = widget.querySelector('.mods-list');
    const modsArea = widget.querySelector('.mod-Aria');
    const installBtn = widget.querySelector('#install-mod');
    const searchInputMods = widget.querySelector('#search-input-mods');

    itemsContainerSettingsWidget = itemsContainer;

    function fillModContentServer(ModsList){
        let mod_list = typeof ModsList === 'string' ? JSON.parse(ModsList) : ModsList;
        if (!mod_list){
            console.log("No Dta")
            return
        }
        if (!mod_list.modlist.length){
            console.log("No Dta")
            return
        }
        mod_list.modlist.forEach((mod_name)=>{
            const valueOpt = document.createElement('option');
            valueOpt.value = mod_name;
            valueOpt.innerHTML = mod_name;
            mod_content.appendChild(valueOpt);
        })
        MODLIST = mod_list.modlist
    }
    function fillModContentUser(ModsList){
        let mod_list = typeof ModsList === 'string' ? JSON.parse(ModsList) : ModsList;
        if (!mod_list){
            console.log("No Dta")
            return
        }
        if (!mod_list.modlist.length){
            console.log("No Dta")
            return
        }
        mod_list.modlist.forEach((mod_name)=>{

            const mod_card = document.createElement('div'); //## defolt just the installed onec and minimisabel
            mod_card.innerHTML = `<h2>`+mod_name+`</h2><p>Text What the Mod Dos</p>
            <input type="text" id="mod-card-key-`+mod_name+`" placeholder="Key">
            <input type="text" id="mod-card-value-`+mod_name+`" placeholder="Value">
            <button onclick='sendKeyValue(`+mod_name+`)'>Set</button>`;
            mod_card.classList.add("mod")
            modsArea.appendChild(mod_card); //## on enter Load mod Card for Installation (user) (server) (local) (cloud)
        })
    }

    callbackOncOn(fillModContentServer, "modlistA")
    callbackOncOn(fillModContentUser, "modlistI")
    setTimeout(async ()=>{
    await WS.send(JSON.stringify({"ServerAction":"getModListAll"}));
    }, 5)
    setTimeout(async ()=>{
    await WS.send(JSON.stringify({"ServerAction":"getModListInstalled"}));
    }, 5)

    if (data.items.length){
        data.items.forEach(item => {
            const element = createElementFromDict(item);
            itemsContainer.appendChild(element);
        });
    }

    const closeButton = widget.querySelector('.widget-close-button');
    closeButton.addEventListener('click', closeWidget);

    installBtn.addEventListener('click', ()=>{
        if (searchInputMods.value in MODLIST){
                    setTimeout(async ()=>{
            await WS.send(JSON.stringify({"ServerAction":"installMod", "name":searchInputMods.value}));
        }, 5)
        }
    })

    function closeWidget() {
        widget.style.animation = 'text-widget-fadeOut 0.5s';
        setTimeout(() => {
            widget.style.display = 'none';
            targetElement.removeChild(widget);
        }, 500);
    }

    return widget;
}

function crateSettingsWidget(json, containerId) {
    const container = document.getElementById(containerId)
    const widget = createWidget(json, container);
    container.appendChild(widget);
    initTaskEditor();
    return widget
}

function addSettingsWidget(data, widget) {
    const element = createElementFromDict(data);
    if (itemsContainerSettingsWidget){
        itemsContainerSettingsWidget.appendChild(element);
    }
    itemsContainerSettingsWidget.appendChild(element);
}


let tasks = JSON.parse(localStorage.getItem('tasks')) || [];
function initTaskEditor() {
    const taskNameInput = document.getElementById('task-input');
    const taskForm = document.getElementById('task-form');
    const keySelect = document.getElementById('key-select');
    const saveTaskBtn = document.getElementById('save-task');
    const newTaskBtn = document.getElementById('new-task');
    const editorFrame = document.getElementById('editorFrame');
    const editorInner = document.getElementById('editor');
    const taskListPrw = document.getElementById('task-list');

    const switchIcon = document.createElement('span');

    let keySelectStore = []

    let kevValuePairs = []

    taskNameInput.value = 'Task-' + tasks.length;
    tasks = JSON.parse(localStorage.getItem('tasks')) || [];
    switchIcon.className = 'text-widget-close-button';
    let switchIconBool = true;
    switchIcon.innerHTML = '▲▽';

    for (let i = 0; i < tasks.length; i++) {
        let name = tasks[i].name
        const valueOpt = document.createElement('option');
        valueOpt.value = name;
        valueOpt.innerHTML = name;
        taskListPrw.appendChild(valueOpt)
    }

    function openEditorFrame() {
        editorInner.classList.remove('none');
        switchIcon.innerHTML = '△▼';
    }

    function closeEditorFrame() {
        editorInner.classList.add('none');
        switchIcon.innerHTML = '▲▽';
    }

    switchIcon.onclick = function () {
        if (switchIconBool) {
            openEditorFrame();
        } else {
            closeEditorFrame();
        }
        switchIconBool = !switchIconBool;
    };

    editorFrame.appendChild(switchIcon);

    function createKeyValuePair(key = '', value = '' || []) {
        const group = document.createElement('div');
        group.className = 'key-value-group';
        workingTask[key] = value;

        const keyInput = document.createElement('p');
        keyInput.innerText = key;

        //const valueInput = document.createElement('input');
        //valueInput.placeholder = 'Value';
        //valueInput.value = value;

        let optt = null

        if (value.startsWith("$")){
            optt = document.createElement('input');
            optt.id = 'input-settings-task-editor-'+key
            optt.value = value
            optt.placeholder = value
            if (key==="args"){
                optt.addEventListener("keyup", ({key}) => {
                    if (key === "Enter") {
                        const selectedKey = 'return'
                        const optionSelect = keySelect.querySelector(`option[value=${selectedKey}]`)
                        createKeyValuePair(selectedKey, optionSelect.title);
                        keySelectStore.push(optionSelect)
                        optionSelect.style.display = 'none';
                    }
                });
            }
        }else{
            optt = document.createElement('select');
            optt.id = "task-editor-task-select-"+key
            let values = null
            console.log("[typeof value]:", typeof value)
            if (typeof value === "string"){
                values = value.split('|');
            }else {
                values = value
            }

            for(let j = 0; j < values.length; j++) {
                const valueOpt = document.createElement('option');
                valueOpt.value = values[j];
                valueOpt.innerHTML = values[j];
                optt.appendChild(valueOpt);
            }
            if (key==="use"){
                optt.addEventListener('change', (ev) => {

                    const selectedKey = 'name'
                    createKeyValuePair(selectedKey, "");
                    const optionSelect = keySelect.querySelector(`option[value=${selectedKey}]`)
                    keySelectStore.push(optionSelect)
                    optionSelect.style.display = 'none';

                    function fillNameContent(nameList){
                        let name_list = typeof nameList === 'string' ? JSON.parse(nameList) : nameList;
                        const name_select = document.getElementById("task-editor-task-select-name")
                        name_select.innerHTML = ''
                        if (!name_select){
                            console.log("Error Lading Task Editor Names")
                            return
                        }
                        const names = name_list.isaaUseResponse
                        if (!names){
                            console.log("Error Lading Task Editor Names")
                            return
                        }
                        if (!names.length){
                            console.log("No Dta")
                            return
                        }
                        names.forEach((name)=>{
                            const valueOpt = document.createElement('option');
                            valueOpt.value = name;
                            valueOpt.innerHTML = name;
                            name_select.appendChild(valueOpt);
                        })
                    }

                    callbackOncOn(fillNameContent, "isaaUseResponse")

                    setTimeout(async ()=>{
                    await WS.send(JSON.stringify({"ServerAction":"runMod", "name":"isaa","function":"get_use", "command":"", "data":
            {"token": "**SelfAuth**", "data":{
                    "use": optt.value,
                }}}));
                    }, 5)

                })
            }
            else if (key==="name"){
                optt.addEventListener('change', (ev) => {
                    const selectedKey = 'args'
                    const optionSelect = keySelect.querySelector(`option[value=${selectedKey}]`)
                    createKeyValuePair(selectedKey, optionSelect.title);
                    keySelectStore.push(optionSelect)
                    optionSelect.style.display = 'none';
                });
            }
        }

        if (key!=="use") {
            const deleteIcon = document.createElement('span');
            deleteIcon.className = 'text-widget-close-button';
            deleteIcon.innerHTML = 'X';
            deleteIcon.onclick = function () {
                taskForm.removeChild(group);
            };
            group.appendChild(deleteIcon);
        }

        group.appendChild(keyInput);
        if (optt){
            group.appendChild(optt);
        }
        taskForm.appendChild(group);

        kevValuePairs.push({key, group})
    }

    keySelect.addEventListener('change', () => {
        const selectedKey = keySelect.value;
        const optionSelect = keySelect.querySelector(`option[value=${selectedKey}]`)

        console.log("[keySelect.querySelector(selectedKey):",optionSelect.title)
        if (selectedKey) {
            createKeyValuePair(selectedKey, optionSelect.title);
            keySelectStore.push(optionSelect)
            optionSelect.style.display = 'none';
            keySelect.value = '';
        }
    });

    saveTaskBtn.addEventListener('click', () => {
        const taskName = taskNameInput.value;
        if (!taskName) {
            alert('Please enter a task name.');
            return;
        }

        tasks.push({ name: taskName, data:JSON.parse(JSON.stringify(workingTask))});
        workingTask = {}
        localStorage.setItem('tasks', JSON.stringify(tasks));
        const valueOpt = document.createElement('option');
        valueOpt.value = taskName;
        valueOpt.innerHTML = taskName;
        taskListPrw.appendChild(valueOpt)
    });

    newTaskBtn.addEventListener('click', () => {

        taskForm.innerHTML = '';
        for (let i = 0; i < keySelectStore.length; i++) {
            keySelectStore[i].style.display = '';
        }

        kevValuePairs = []
        keySelectStore = []

        createKeyValuePair('use', 'select|agent|tool|function|chain');
    });

    taskNameInput.addEventListener('keyup', (e) => {
        if (e.key !== "Enter"){
            return
        }
        const taskName = taskNameInput.value;
        if (!taskName) {
            alert('Please enter a task name.');
            return;
        }

        tasks = JSON.parse(localStorage.getItem('tasks')) || [];
        const taskObj = tasks.find(t => t.name === taskName);
        if (taskObj) {
            const task = taskObj.task;
            taskForm.innerHTML = '';
            for (const key in task) {
                createKeyValuePair(key, task[key]);
            }
        }
    });


    // Add default key-value pairs
    createKeyValuePair('use', 'select|agent|tool|function|chain');
    closeEditorFrame()
}


crateSettingsWidget({'items':[{tag: 'div', attributes: {id: 'main-settings', class: ''}, content: '<p> ...</p>',}]}
    ,"MainContent");



