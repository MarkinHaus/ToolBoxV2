if(window.history.state.TB){

window.TBf.initVar("offsetX", {})
window.TBf.initVar("offsetY", {})

function saveState(id, x, y) {
    const state = { x, y };
    localStorage.setItem("WIDGET-"+id, JSON.stringify(state));
}

function loadState(id) {
    const state = localStorage.getItem("WIDGET-"+id);
    return state ? JSON.parse(state) : null;
}

function moveElement(id, x, y) {
    const element = document.getElementById(id);
    element.style.top = y+ 'px';
    element.style.left = x+ 'px';
}

function updateWidgetPosition(element) {
    const rect = element.getBoundingClientRect();
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;

    if (rect.left < 0 || rect.right > windowWidth || rect.top < 0 || rect.bottom > windowHeight){
        moveElement(element.id, windowWidth / 2, windowHeight / 2);
    }

}

function handleDrag(handel, element) {
    let moveElementFunc = function(event) {
        // Kompatibilität mit Touch-Events
        let touchEvent = event.type.includes('touch') ? event.touches[0] : event;

        let x = touchEvent.clientX;
        let y = touchEvent.clientY;
        moveElement(element.id, x, y);
        saveState(element.id, x, y);
    };

    // Start Dragging
    let startDrag = function(event) {
        event.preventDefault(); // Verhindert Standard-Touch-Events (Scrollen, Zoomen)
        element.style.userSelect = "none";

        document.addEventListener('mousemove', moveElementFunc);
        document.addEventListener('touchmove', moveElementFunc, {passive: false}); // 'passive: false' verhindert Scrollen während des Dragging
    };

    // Stop Dragging
    let stopDrag = function() {
        element.style.userSelect = "";
        document.removeEventListener('mousemove', moveElementFunc);
        document.removeEventListener('touchmove', moveElementFunc);
    };

    // Event-Listener für Maus
    handel.addEventListener('mousedown', startDrag);
    document.addEventListener('mouseup', stopDrag);

    // Event-Listener für Touch
    handel.addEventListener('touchstart', startDrag);
    document.addEventListener('touchend', stopDrag);
}

function handleDoubleClickFunction(element){
    element.style.userSelect = "none";

    let moveElementFunc = function(event) {
        let touchEvent = event.type.includes('touch') ? event.touches[0] : event;

        let x = touchEvent.clientX;
        let y = touchEvent.clientY;
        moveElement(element.id, x, y);
        saveState(element.id, x, y);
    };

    document.addEventListener('mousemove', moveElementFunc);
    document.addEventListener('touchmove', moveElementFunc, {passive: false});

    function stopMove() {
        element.style.userSelect = "";
        document.removeEventListener('mousemove', moveElementFunc);
        document.removeEventListener('touchmove', moveElementFunc);
        document.removeEventListener('click', stopMove, true);
    }

    // Verwendet 'true' für die 'useCapture' Option, um sicherzustellen, dass der Event-Listener im Capture-Phase aktiviert wird
    document.addEventListener('click', stopMove, true);
}

function handleDoubleClick(element) {
    element.addEventListener('dblclick', function() {
        handleDoubleClickFunction(element)
    });
}

window.TBf.initVar("lastTouchTime", 0)

function handleDoubleTap(element) {
    const currentTime = new Date().getTime();
    const tapDelay = currentTime - window.TBf.getVar("lastTouchTime");

    window.TBf.setVar("lastTouchTime", currentTime)
    if (tapDelay < 500 && tapDelay > 0) {
        // Doppel-Tipp erkannt
        console.log("Doppel-Tipp erkannt");
        // Führen Sie hier Ihre Logik für das Doppel-Tipp-Ereignis aus
        handleDoubleClickFunction(element)
    }
    window.TBf.setVar("lastTouchTime", currentTime)
}

function makeDraggable(element) {
    console.log("element:", element)
    if (!element){
        return;
    }
    console.log("element:", element.id)
    if (element.id === ""){
        return
    }
    const handle = document.createElement("span");
    handle.style.position = "absolute";
    handle.style.top = "-2.5px";
    handle.style.left = "-2.5px";
    handle.style.width = "10px";
    handle.style.height = "10px";
    handle.style.borderRadius = "25%";
    handle.style.cursor = 'move';
    handle.classList.add('on-hover')
    handle.setAttribute("data-tauri-drag-region", "")

    // Add the handle to the element
    element.appendChild(handle);

    handleDrag(handle, element)
    handleDoubleClick(element)

    element.addEventListener('touchend', ()=>{
        handleDoubleTap(element)
    })


    element.addEventListener('resize', () => {
        window.TBf.getVar("offsetX")[element.id] = element.getBoundingClientRect().left;
        window.TBf.getVar("offsetY")[element.id] = element.getBoundingClientRect().top;
    });



    handle.ondragover = function() {
        console.log("element: ondragover", element.id)
        return false;
    };

    handle.oncopy = function(event) {
        const selection = document.getSelection();
        console.log("element: ondragstart", element.id, selection)
        // event.clipboardData.setData("text/plain", selection.toString().toUpperCase());
        // event.preventDefault();
        return false;
    };

    handle.ondragstart = function() {
        console.log("element: ondragstart", element.id)
        return false;
    };
    handle.ondragend= function() {
        console.log("element: ondragend", element.id)
        return false;
    };
    handle.ondrop= function() {
        console.log("element: ondrop", element.id)
        return false;
    };
}

function StateLoadBtn_drag() {
    const draggableElements = document.querySelectorAll('.draggable');
    draggableElements.forEach(element => {
        const state = loadState(element.id);
        if (state) {
            element.style.transition = "transform 1.55s ease-in";
            moveElement(element.id, state.x, state.y);
            setTimeout(() => {element.style.transition = ""}, 1550)
        }
    });
};

function DragInit(){
    const draggableElements = document.querySelectorAll('.draggable');
    draggableElements.forEach((value)=> {
        makeDraggable(value);
    });


    // document.getElementById('StateInit').style.display = 'none'
    StateLoadBtn_drag()


    window.addEventListener('resize', () => {
        draggableElements.forEach(element => {
            updateWidgetPosition(element);
        });
    });

    console.log("Drag Online")
}


setTimeout(()=>{
    DragInit()
}, 115)
}else{
    console.log("pending on index")
}
