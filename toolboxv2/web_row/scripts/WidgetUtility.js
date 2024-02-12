
function getRandomId(length) {
    let result = '';
    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    const charactersLength = characters.length;
    for (let i = 0; i < length; i++) {
        result += characters.charAt(Math.floor(Math.random() * charactersLength));
    }
    return result;
}


class WidgetUtility {
    constructor() {
        console.log("[WidgetUtility Online]")
        this.widgetStore = [];
        this.WidgetIDStore = [];
        this.maxZIndex = 2;
        this.storage = {}
        this.ackey = null
    }

    // Erstellt ein neues Widget und fügt es dem DOM hinzu
    createWidget({id = undefined, titel , template = "", mount_to_id = "MainContent"}) {
        console.log("[WidgetUtility Crating]", titel, id, template)

        if (id === undefined){
            id = getRandomId(16)
        }
        const widgetContainer = document.createElement('div');
        widgetContainer.classList.add("widget")
        widgetContainer.classList.add("widget-fadeIn")
        widgetContainer.classList.add("draggable")
        widgetContainer.id = id
        widgetContainer.innerHTML = widgetUtility.getTemplate(id, titel, template);
        const targetElement = document.getElementById(mount_to_id);

        targetElement.appendChild(widgetContainer);
        widgetUtility.attachEventHandlers(widgetContainer, targetElement, titel, template);
        widgetUtility.addWidget2Manager(widgetContainer)

        this.storage[id] = {id, titel , template, mount_to_id}

        if (this.ackey){
            this.saveStorage()
        }

        return widgetContainer;
    }

    saveStorage(){
        if (!this.ackey){
            console.log("No Bord ac")
            return
        }
        console.log("Saving Widgets Board id:", this.ackey, this.storage)
        if (Object.values(this.storage).length){
            window.localStorage.setItem("Widget-Storage-Board-"+this.ackey, JSON.stringify(this.storage))
            console.log("")
        }

    }

    save_clos_all(){
        if (!this.ackey){
            console.log("No Bord ac")
            return
        }
        this.saveStorage()
        const anker = document.getElementById("MainContent")
        this.widgetStore.forEach(widget => {
            widget.style.animation = 'widget-fadeOut 0.5s';
            setTimeout(() => {
                anker.removeChild(widget);
            }, 500);
        });

        this.widgetStore = [];
        this.WidgetIDStore = [];
        this.maxZIndex = 2;
        this.storage = {}
        this.ackey = null
    }

    get_storage(){
        if (Object.keys(this.storage).length !== 0){
             console.log("Save Storage, before Open New ")
        }
        let saved_widgets_storage;
        try{
            saved_widgets_storage = JSON.parse(window.localStorage.getItem("Widget-Storage-Board-"+this.ackey))
            if (!saved_widgets_storage) {
                console.log("Invalid Storage data or key =" + this.ackey + 'Data: ' +saved_widgets_storage);
                return
            }
        }catch (e){
            console.error("Error parsing storage data:", e);
            console.log("Invalid Storage data or key =" + this.ackey);
            return
        }

        if(!saved_widgets_storage){
            return
        }
        console.log("Opening Widgets Board id:", this.ackey, saved_widgets_storage.length, saved_widgets_storage)
        Object.values(saved_widgets_storage).forEach(widgetData => {
            this.createWidget({
                id: widgetData.id,
                titel: widgetData.titel,
                template: widgetData.template,
                mount_to_id: widgetData.mount_to_id
            });
        });
    }

    // Liefert das Template basierend auf dem Widget-Typ
    getTemplate(id, context, content) {
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
        widgetUtility.widgetStore.push(widget);
        try{
            console.log("makeDraggable")
            makeDraggable(widget)
        }catch (e) {
            console.log(e)
        }
        //autoZIndex
        widget.addEventListener('click', function(e) {
            // Erhöhe den z-index, wenn er kleiner als 100 ist
            if (widgetUtility.maxZIndex < 100) {
                widgetUtility.maxZIndex++;
            }
            // Setze den z-index des angeklickten divs auf den maximalen Wert
            widget.style.zIndex = widgetUtility.maxZIndex;
            let currentMax = 2;
            for (let j = 0; j < widgetUtility.widgetStore.length; j++) {
                if (widgetUtility.widgetStore[j] !== widget && widgetUtility.widgetStore[j].style.zIndex > 2) {
                    currentMax = widgetUtility.widgetStore[j].style.zIndex  > currentMax ?  widgetUtility.widgetStore[j].style.zIndex : currentMax;
                    widgetUtility.widgetStore[j].style.zIndex--;
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
            content: `<h2>Action called on : `+titel+`</h2><hr>`+template,
            buttons:[
                {text: "Abbrechen", action: () => overlayUtility.closeOverlay()},
                {text: "Bestätigen", action: () => { console.log("Aktion bestätigt"); overlayUtility.closeOverlay(); }}
            ]
        }));
        const resizeHandle = widgetElement.querySelector('.widget-resize-handle');
        resizeHandle.addEventListener('click', () => {
            if(resizeHandle.classList.contains('toggle-on')){
                widgetElement.style.overflow = 'visible'
                resizeHandle.classList.remove('toggle-on')
            }else{
                widgetElement.style.overflow = 'auto'
                resizeHandle.classList.add('toggle-on')
            }
        });
    }

    // Schließt und entfernt ein Widget
    closeWidget(widgetElement, targetElement =  document.getElementById("MainContent") ) {
        widgetElement.style.animation = 'widget-fadeOut 0.5s';
        delete this.storage[widgetElement.id]
        if (this.ackey){
            this.saveStorage()

        }
        setTimeout(() => {
            targetElement.removeChild(widgetElement);
            widgetUtility.widgetStore = widgetUtility.widgetStore.filter(widget => widget !== widgetElement);
        }, 500);
        window.localStorage.removeItem("WIDGET-"+widgetElement.id)
    }
}
const widgetUtility = new WidgetUtility();
document.addEventListener('DOMContentLoaded', () => {
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


});

