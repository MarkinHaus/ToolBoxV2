window.TBf.initVar("boards",{});


class WidgetUtility {
    constructor() {
        console.log("[WidgetUtility Online]")
        this.widgetStore = {};
        this.WidgetIDStore = [];
        this.maxZIndex = 2;
        this.storage = {}
        this.ackey = null
    }

    // Erstellt ein neues Widget und fügt es dem DOM hinzu

    async fetchTemplate(identifier, titel = '') {
        const extra = titel? '&Wid=': ''
        try {
            const template = await window.TBf.httpPostUrl("WidgetsProvider", "open_widget", "name=" + identifier + extra +titel,
                (e) => {
                    console.log(e)
                    return `<h2> Error ` + identifier + ` ` + titel + `</h2>`
                }, (result) => {
                    console.log("Susses", result)
                    return result.get()
                }, true
            )
            return template
        } catch (error) {
            console.error('Error fetching template:', error);
            return `<h2>Fatal Error ${identifier} ${titel}</h2>`;
        }
    }

    async fetchBoard(name) {
        return await window.TBf.httpPostUrl("WidgetsProvider", "get_sto", "sto_name=" + this.ackey,
            (e) => {
                console.log(e)
                window.TBf.getM("addBalloon")("MainContent", 0, ["Error Receiving Board Data for " + name, e.html()], []);
                return null
            }, (result) => {
                console.log("Susses", result)
                return JSON.parse(result.get().slice(2, result.get().length-1))
            }, true
        )
    }

    async addBoard(name) {
        await window.TBf.httpPostUrl("WidgetsProvider", "add_sto", "sto_name=" + name,
            (e) => {
                console.log(e)
                window.TBf.getM("addBalloon")("MainContent", 0, ["Error Crating Board " + name, e.html()], [["retry", () => {
                    this.addBoard(name)
                }]]);
            }, (result) => {
                console.log("Susses", result.get())
                window.TBf.getM("addBalloon")("MainContent", 0, ["Crating Board " + name], [])
            }, true
        )
    }

    async removeBoard(name) {
        await window.TBf.httpPostUrl("WidgetsProvider", "delete_sto", "sto_name=" + name,
            (e) => {
                console.log(e)
                window.TBf.getM("addBalloon")("MainContent", 0, ["Error Removing Board " + name, e.html()], [["retry", () => {
                    this.removeBoard(name)
                }]]);
            }, (result) => {
                console.log("Susses", result.get())
                window.TBf.getM("addBalloon")("MainContent", 0, ["Removed Board :" + name], [])
            }, true
        )
    }

    async getAllBordNames() {
        const data = await window.TBf.httpPostUrl("WidgetsProvider", "get_names", "",
            (e) => {
                console.log(e)
                window.TBf.getM("addBalloon")("MainContent", 0, ["Error Removing Board :" + name, e.html()], []);
                return null
            }, (result) => {
                console.log("Susses", result)
                return result.get()
            }, true
        )
        if (!data) {
            return ["Main"]
        }
        return data
    }

    async createWidget({id = undefined, titel, identifier = "", mount_to_id = "MainContent", max = false}) {

        if (!this.ackey) {
            window.TBf.getM("addBalloon")("MainContent", 0, ["Error Adding Widget no Board open!", "open 'Main' Board and retry"], [["retry", () => {
                this.ackey = "main";
                this.createWidget({id, titel, identifier, mount_to_id, max})
            }]]);
            return
        }

        if (!identifier && titel) {
            identifier = titel
        }

        console.log("[WidgetUtility Crating]", identifier, titel)

        if (identifier==="") {
            throw "invalid identifier"
        }

        const template = await this.fetchTemplate(identifier, titel)

        if (id === undefined) {
            id = ""
            const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
            const charactersLength = characters.length;
            for (let i = 0; i < 16; i++) {
                id += characters.charAt(Math.floor(Math.random() * charactersLength));
            }
        }

        const widgetContainer = document.createElement('div');

        widgetContainer.classList.add("widget")
        widgetContainer.classList.add("widget-fadeIn")
        widgetContainer.classList.add("draggable")

        if (max) {
            widgetContainer.style.width = window.innerWidth - 50 + 'px'
            widgetContainer.style.height = window.innerHeight - 50 + 'px'
        }

        widgetContainer.id = id
        widgetContainer.innerHTML = widgetUtility.getTemplate(id, titel, template);

        const targetElement = document.getElementById(mount_to_id);

        targetElement.appendChild(widgetContainer);

        widgetUtility.attachEventHandlers(widgetContainer, targetElement, titel, template);
        widgetUtility.addWidget2Manager(widgetContainer)


        this.storage[id] = {id, identifier, titel, mount_to_id}

        this.saveStorage()

        try {
            window.TBf.processRow(widgetContainer);
        } catch (e) {
            console.log("Error Adding HTMX to widget !!!")
        }

        return widgetContainer;
    }


    saveStorage(){
            if (!this.ackey) {
                console.log("No Bord ac")
                return
            }
            console.log("Saving Widgets Board id:", this.ackey)
            if (Object.values(this.storage).length) {
                setTimeout(async () => {
                    await window.TBf.httpPostData("WidgetsProvider", "set_sto?sto_name=" + this.ackey, this.storage, (e) => {
                        window.TBf.getM("addBalloon")("MainContent", 0, ["Error Saving Board Data: " + this.ackey, e.html()], [["retry", this.saveStorage]]);
                    }, (s) => {
                        console.log("Saved", this.ackey)
                    })
                }, 200)
            }
        }

    closeAll(name = null, anker = null, del = false) {
            if (!anker) {
                anker = document.getElementById("MainContent")
            }
            Object.keys(this.widgetStore).forEach(board => {
                if (!name || name === board) {
                    this.widgetStore[board].forEach(widget => {
                        try {
                            widget.style.animation = 'widget-fadeOut 0.5s';
                            setTimeout(() => {
                                if (del) {
                                    this.closeWidget(widget, anker)
                                } else {
                                    anker.removeChild(widget);
                                }
                            }, 500);
                        } catch (e) {
                            console.log(e)
                        }

                    });
                }

            })
        }

    save_close(){
            if (!this.ackey) {
                console.log("No Bord ac")
                return
            }
            this.saveStorage()
            this.closeAll(this.ackey)

            this.widgetStore[this.ackey] = [];
            this.WidgetIDStore = [];
            this.maxZIndex = 2;
            this.storage = {}
            this.ackey = null
        }

    async get_storage(){
            if (Object.keys(this.storage).length !== 0) {
                console.log("Save Storage, before Open New ")
            }
            let saved_widgets_storage;
            try {
                saved_widgets_storage = await this.fetchBoard(this.ackey)
                if (!saved_widgets_storage) {
                    console.log("Invalid Storage data or key =" + this.ackey + 'Data: ' + saved_widgets_storage);
                    return
                }
            } catch (e) {
                console.error("Error parsing storage data:", e);
                console.log("Invalid Storage data or key =" + this.ackey);
                return
            }

            if (!saved_widgets_storage) {
                return
            }
            console.log("Opening Widgets Board id:", this.ackey, saved_widgets_storage.length, saved_widgets_storage)
            Object.values(saved_widgets_storage).forEach(widgetData => {
                this.createWidget({
                    id: widgetData.id,
                    titel: widgetData.titel,
                    identifier: widgetData.identifier,
                    mount_to_id: widgetData.mount_to_id
                })
            });
        }

        // Liefert das Template basierend auf dem Widget-Typ
    getTemplate(id, context, content){
            return `
            <div class="widget-from">${context}</div>
            <span class="widget-close-button">X</span>
            <span class="widget-action-button material-symbols-outlined">component_exchange</span>
            <div id="widget-conten-${id}">${content}</div>
            <div class="widget-resize-handle"></div>
        `;
        }

    addWidget2Manager(widget){
            widgetUtility.WidgetIDStore.push(widget.id)
            if (!widgetUtility.widgetStore[this.ackey]) {
                widgetUtility.widgetStore[this.ackey] = []
            }
            widgetUtility.widgetStore[this.ackey].push(widget);
            try {
                console.log("makeDraggable")
                makeDraggable(widget)
            } catch (e) {
                console.log(e)
            }
            //autoZIndex
            widget.addEventListener('click', function (e) {
                // Erhöhe den z-index, wenn er kleiner als 100 ist
                if (widgetUtility.maxZIndex < 100) {
                    widgetUtility.maxZIndex++;
                }
                // Setze den z-index des angeklickten divs auf den maximalen Wert
                widget.style.zIndex = widgetUtility.maxZIndex;
                let currentMax = 2;
                if (!widgetUtility.widgetStore[this.ackey]) {
                    return
                }
                for (let j = 0; j < widgetUtility.widgetStore[this.ackey].length; j++) {
                    if (widgetUtility.widgetStore[this.ackey][j] !== widget && widgetUtility.widgetStore[this.ackey][j].style.zIndex > 2) {
                        currentMax = widgetUtility.widgetStore[this.ackey][j].style.zIndex > currentMax ? widgetUtility.widgetStore[this.ackey][j].style.zIndex : currentMax;
                        widgetUtility.widgetStore[this.ackey][j].style.zIndex--;
                    }
                }
                widgetUtility.maxZIndex = currentMax
            });
        }

        // Fügt Event-Handler zu einem Widget hinzu
    attachEventHandlers(widgetElement, target, titel, template) {
            const closeButton = widgetElement.querySelector('.widget-close-button');
            closeButton.addEventListener('click', () => this.closeWidget(widgetElement, target));
            const actionButton = widgetElement.querySelector('.widget-action-button');
            actionButton.addEventListener('click', () => overlayUtility.createOverlay({
                content: `<h2>Action called on : ` + titel + `</h2><hr>` + template,
                buttons: [
                    {text: "Abbrechen", action: () => overlayUtility.closeOverlay()},
                    {
                        text: "Bestätigen", action: () => {
                            console.log("Aktion bestätigt");
                            overlayUtility.closeOverlay();
                        }
                    }
                ]
            }));
            const resizeHandle = widgetElement.querySelector('.widget-resize-handle');
            resizeHandle.addEventListener('click', () => {
                if (resizeHandle.classList.contains('toggle-on')) {
                    widgetElement.style.overflow = 'visible'
                    resizeHandle.classList.remove('toggle-on')
                } else {
                    widgetElement.style.overflow = 'auto'
                    resizeHandle.classList.add('toggle-on')
                }
            });
        }

        // Schließt und entfernt ein Widget
    closeWidget(widgetElement, targetElement = document.getElementById("MainContent")) {
            widgetElement.style.animation = 'widget-fadeOut 0.5s';
            delete this.storage[widgetElement.id]
            if (this.ackey) {
                this.saveStorage()

            }
            setTimeout(() => {
                targetElement.removeChild(widgetElement);
                widgetUtility.widgetStore[this.ackey] = widgetUtility.widgetStore[this.ackey].filter(widget => widget !== widgetElement);
            }, 500);
            window.localStorage.removeItem("WIDGET-" + widgetElement.id)
    }
}
window.widgetUtility = new WidgetUtility();

// Funktion zum Erstellen eines neuen Boards
function createNewBoard(boardName) {
    window.widgetUtility.addBoard(boardName).then(()=>{
        window.widgetUtility.save_close()
        window.widgetUtility.ackey = boardName;
    })
    window.TBf.getVar("boards")[boardName] = { name: boardName, editing: false, visible: true, widgets: []};
    editBoard(boardName);
}

// Funktion zum Anzeigen eines Boards
function showBoard(name) {
    window.widgetUtility.get_storage(name).then(()=>{
        window.TBf.getVar("boards")[name].visible = true;
    })
    return renderBoards();
}

// Funktion zum Ausblenden eines Boards
function hideBoard(name) {
    window.TBf.getVar("boards")[name].visible = false;
    window.widgetUtility.closeAll(name)
    return renderBoards();
}

// Funktion zum Anzeigen des Bearbeitungsmodus für ein Board
function closeAllBoards() {
    for (const boardName in window.TBf.getVar("boards")) {
        hideBoard(boardName)
    }
    window.widgetUtility.ackey = "main"
    return renderBoards();
}

function editBoard(name) {
    for (const boardName in window.TBf.getVar("boards")) {
        window.TBf.getVar("boards")[boardName].editing = (boardName === name);
    }
    window.TBf.getVar("boards")[name].visible = true;
    window.widgetUtility.ackey = name
    return renderBoards();
}

// Funktion zum Rendern der Boards in der Benutzeroberfläche
function renderBoards() {
    let boardTableBody = document.getElementById("boardTableBody");
    let currentEditBoardSpan = document.getElementById("currentEditBoard");

    if(!boardTableBody){boardTableBody = document.createElement("div")}
    if(!currentEditBoardSpan){currentEditBoardSpan = document.createElement("span")}

    boardTableBody.innerHTML = "";

    for (const name in window.TBf.getVar("boards")) {
        const board = window.TBf.getVar("boards")[name];
        const boardRow = document.createElement("tr");

        const nameCell = document.createElement("td");
        nameCell.textContent = board.name;
        boardRow.appendChild(nameCell);

        const openCloseCell = document.createElement("td");
        const openCloseButton = document.createElement("button");
        openCloseButton.textContent = board.visible ? "Close" : "Open";
        openCloseButton.onclick = () => board.visible ? hideBoard(name) : showBoard(name);
        openCloseCell.appendChild(openCloseButton);
        boardRow.appendChild(openCloseCell);

        const setEditCell = document.createElement("td");
        const setEditButton = document.createElement("button");
        setEditButton.textContent = "Set Edit";
        setEditButton.onclick = () => editBoard(name);
        setEditCell.appendChild(setEditButton);
        boardRow.appendChild(setEditCell);

        const deleteCell = document.createElement("td");
        const deleteButton = document.createElement("button");
        deleteButton.textContent = "Delete";
        deleteButton.onclick = () => deleteBoard(name);
        deleteCell.appendChild(deleteButton);
        boardRow.appendChild(deleteCell);

        if (board.editing) {
            boardRow.style.backgroundColor = "lightblue";
            currentEditBoardSpan.textContent = board.name;
        }

        boardTableBody.appendChild(boardRow);
    }
    return [boardTableBody, currentEditBoardSpan]
}

// Funktion zum Löschen eines Boards
function deleteBoard(name) {
    window.overlayUtility.createOverlay({
        content: `<h1>Are you sure you want to delete the board "` + name + `"?</h1>`,
        closeOnOutsideClick:false,
        buttons:[{text: "Abbrechen"},{text: "Delete", action: () => {
                delete window.TBf.getVar("boards")[name];
                window.widgetUtility.closeAll(name, document.getElementById("MainContent"), true)
                window.widgetUtility.removeBoard(name).then(
                    renderBoards
                )
            }}, ]
    })
}

window.TBf.setM("createNewBoard",createNewBoard)
window.TBf.setM("closeAllBoards",closeAllBoards)
window.TBf.setM("initBoard",  async ()=>{
        window.widgetUtility.getAllBordNames()
            .then(names => {
                names.forEach(name => {
                    window.TBf.getVar("boards")[name] = {
                        name: name,
                        editing: false,
                        visible: true,
                    }
                })
                renderBoards()
            })
        })
    // Instanz der WidgetUtility erstellen

    // setTimeout(()=>{
    //     widgetUtility.createWidget({
    //         id: "TextWidget-1",
    //         titel: "Mein Text-Widget",
    //         callbacks: (e)=>{
    //             console.log(e)},
    //         template: `<h1>test</h1>`,
    //     });
    //     widgetUtility.createWidget({
    //         id: "SimpleIframe",
    //         titel: "Simple",
    //         template: `
    //     <iframe src="/" style="width: 600px; height: 400px; border: 1px;" title="Simple"></iframe>
    // `,
    //         callbacks: (widgetElement) => {
    //             console.log("Google Iframe Widget created", widgetElement);
    //         }
    //     });
    // }, 1000)



