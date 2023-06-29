let WidgetIDStore = [];
let WidgetInit = false;
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
                const widgetText = addTextWidget("MainContent", "TextWidget-"+WidgetIDStore.length, dropdownTitle.value)
                WidgetIDStore.push(dropdownTitle.value)
                try{
                    console.log("makeDraggable")
                    makeDraggable(widgetText)
                }catch (e) {
                    console.log(e)
                }
            }catch (e){
                console.log("getTextWidget", e)
                WS.send(JSON.stringify({"ServerAction":"getTextWidget"}));
                dropdownHelper.style.color = "var(--info-color)"
                dropdownHelper.innerText = "Installing Widget Pleas click agan"
            }
        }

        else if (event.target.id === 'isaa-w'){
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

        else if (event.target.id === 'isaa-editor-w'){
            console.log("Installing starting")
            const message = JSON.stringify({"ServerAction":"getEditor"});
            console.log("Installing an Isaa task editor widget", message)
            setTimeout(async ()=>{
                await WS.send(message);
            }, 50)
        }

        else if (event.target.id === 'path-w'){
            //try{
                const pathText = addPathWidget("MainContent", "PathWidget-"+WidgetIDStore.length, dropdownTitle.value)
                WidgetIDStore.push(dropdownTitle.value)
            //    try{
                    console.log("makeDraggable")
                    makeDraggable(pathText)
            //    }catch (e) {
            //        console.log(e)
            //    }
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
