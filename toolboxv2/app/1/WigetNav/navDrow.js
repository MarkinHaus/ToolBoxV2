let WidgetIDStore = [];
let WidgetStore = [];
let WidgetInit = false;
let maxZIndex = 2; // Startwert


function addWidget2Manager(widget){
    WidgetStore.push(widget)
    try{
        console.log("makeDraggable")
        makeDraggable(widget)
    }catch (e) {
        console.log(e)
    }
    //autoZIndex
    widget.addEventListener('click', function(e) {
        // Erhöhe den z-index, wenn er kleiner als 100 ist
        if (maxZIndex < 100) {
            maxZIndex++;
        }
        // Setze den z-index des angeklickten divs auf den maximalen Wert
        widget.style.zIndex = maxZIndex;
        for (let j = 0; j < WidgetStore.length; j++) {
            if (WidgetStore[j] !== widget && WidgetStore[j].style.zIndex > 2) {
                WidgetStore[j].style.zIndex--;
            }
        }
    });
}

function dropdownInit(){

    if (WidgetInit){
        return
    }
    WidgetInit = true

const dropdownCircleDropdown = document.querySelector('.circle-dropdown');
const dropdownCircle = document.querySelector('.circle');
const dropdownMenu = document.querySelector('.dropdown-menu');
const dropdownTitle = document.getElementById("dropdown-Title")
const dropdownHelper = document.getElementById("dropdown-helper-text")


dropdownCircleDropdown.addEventListener('mouseenter', () => {
    dropdownCircle.style.transform = 'rotate(180deg)';
    dropdownMenu.style.display = 'block';
});

dropdownMenu.addEventListener('mouseleave', () => {
    dropdownCircle.style.transform = 'rotate(0deg)';
    setTimeout(() => {
        dropdownMenu.style.display = 'none';
    }, 500);
});

dropdownMenu.addEventListener('click', (event) => {
    if (event.target.classList.contains('dropdown-item')) {
        dropdownHelper.style.color = "var(--text-color)"
        dropdownHelper.innerText = "Add Widget"
        if (event.target.id === 'close'){
            dropdownCircle.style.transform = 'rotate(0deg)';
            setTimeout(() => {
                dropdownMenu.style.display = 'none';
            }, 500);
        }

        if (event.target.id.endsWith('-w')){
            if (dropdownTitle.value === ''){
                dropdownHelper.style.color = "var(--error-color)"
                dropdownHelper.innerText = "Pleas Add an Title for context"
                return
            }
        }

        if (event.target.id === 'text-w'){
            try{
                const textWidget = addTextWidget("MainContent", "TextWidget-"+WidgetIDStore.length, dropdownTitle.value)
                WidgetIDStore.push(dropdownTitle.value)
                addWidget2Manager(textWidget)

            }catch (e){
                console.log("getTextWidget", e)

                WS.send(JSON.stringify({"ServerAction":"getTextWidget"}));
                dropdownHelper.style.color = "var(--info-color)"
                dropdownHelper.innerText = "Installing Widget Pleas click agan"
            }
        }

        else if (event.target.id === 'isaa'){
            console.log("Installing starting")
            const message = JSON.stringify({"ServerAction":"runMod", "name":"isaa","function":"start_widget", "command":"", "data":
                    {"token": "**SelfAuth**", "data":{
                            "index": WidgetIDStore.length,
                            "Title": dropdownTitle.value
                        }}});
            console.log("Installing an Isaa widget", message)
            setTimeout(async ()=>{
                await WS.send(message);
            }, 50)
        }

        else if (event.target.id === 'ControlsWidget'){

             try{
                const controlsWidget =crateSettingsWidget({'items':[{tag: 'div', attributes:
                            {id: 'main-settings', class: ''}, content: '<p> ...</p>',}]},"MainContent")
                controlsWidget.id = "TextWidget-"+WidgetIDStore.length
                WidgetIDStore.push(dropdownTitle.value)
                addWidget2Manager(controlsWidget)

            }catch (e){
                console.log("getControlsWidget", e)
                WS.send(JSON.stringify({"ServerAction":"getControls"}));
                dropdownHelper.style.color = "var(--info-color)"
                dropdownHelper.innerText = "Installing Widget"
            }

        }

        else if (event.target.id === 'path'){
            //try{
                const pathWidget = addPathWidget("MainContent", "PathWidget-"+WidgetIDStore.length, dropdownTitle.value)
                WidgetIDStore.push(dropdownTitle.value)
                addWidget2Manager(pathWidget)

            //}catch (e){
            //    console.log("getPathWidget", e)
            //    WS.send(JSON.stringify({"ServerAction":"getPathWidget"}));
            //    dropdownHelper.style.color = "var(--info-color)"
            //    dropdownHelper.innerText = "Installing Widget Pleas click agan"
            //}
        }

        console.log(`Clicked on: ${event.target.textContent} ${event.target.id} `);
    }
});

}

dropdownInit()
WS.send(JSON.stringify({"ServerAction":"getTextWidget"}));
WS.send(JSON.stringify({"ServerAction":"getPathWidget"}));
