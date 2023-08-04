function saveState(id, x, y) {
    const state = { x, y };
    localStorage.setItem(id, JSON.stringify(state));
}

function loadState(id) {
    const state = localStorage.getItem(id);
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

let offsetX = {}, offsetY = {}

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

    // Add the handle to the element
    element.appendChild(handle);

    handle.onmousedown = function(event) {

        element.style.userSelect = "none";
        document.onmousemove = function(event) {
            let x = event.clientX
            let y = event.clientY
            moveElement(element.id, x, y);
            saveState(element.id, x, y);
        };

        document.onmouseup = function() {
            element.style.userSelect = "";
            document.onmousemove = null;
            document.onmouseup = null;
        };
    };
    element.addEventListener('resize', () => {
        offsetX[element.id] = element.getBoundingClientRect().left;
        offsetY[element.id] = element.getBoundingClientRect().top;
    });
    handle.ondragstart = function() {
        return false;
    };
}



function DragInit(){
    const draggableElements = document.querySelectorAll('.draggable');
    draggableElements.forEach((value)=> {
        makeDraggable(value);
    });

    document.getElementById('StateInit').style.display = 'none'

    document.getElementById('StateLoadBtn').onclick = function() {
        draggableElements.forEach(element => {
            const state = loadState(element.id);
            if (state) {
                element.style.transition = "transform 1.55s ease-in";
                moveElement(element.id, state.x, state.y);
                setTimeout(() => {element.style.transition = ""}, 1550)
            }
        });


    };

    window.addEventListener('resize', () => {
        draggableElements.forEach(element => {
            updateWidgetPosition(element);
        });
    });
}
