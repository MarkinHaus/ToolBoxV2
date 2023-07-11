function addPathWidget(element_id, id, context, path="", vKey="") {
    console.log("ADDING Path Widget ", element_id);
    const targetElement = document.getElementById(element_id);
    let pathBalloon = createPathWidget(id, context, targetElement, path, vKey);
    targetElement.appendChild(pathBalloon);
    pathBalloon.id = id;
    return pathBalloon;
}

function createPathWidget(input_id, context, targetElement, path="", vKey="") {
    const template = document.getElementById('path-widget-template');
    const widget = template.content.cloneNode(true).querySelector('.path-widget');
    const fromElement = widget.querySelector('.path-widget-from');
    fromElement.textContent = context;
    const input = widget.querySelector('#verification-key-input');
    const fileData = widget.querySelector('#path-widget-input-fileData');
    const vKeyInput = widget.querySelector('#verification-key-input');
    const widget_injection = widget.querySelector('#path-widget-injection');
    const userFeedback = widget.querySelector('#user-feedback');
    input.id = input_id+'-Path';
    input.value = path;
    vKeyInput.value = vKey;
    const closeButton = widget.querySelector('.path-widget-close-button');
    closeButton.addEventListener('click', closeWidget);
    widget_injection.addEventListener('click', ()=>{
        console.log("widget_injection:pathWidget")
        const reader = new FileReader();
        reader.onload = function(e) {
            //WS.send(JSON.stringify({"ChairData":true, "data": {"LiveFile": "pathWidgetData", "context": context, "id": input_id, "path": fileData.value,  "vKey": vKeyInput.value, "new": true}}));
            userFeedback.textContent = "Detected Changes";
        };

        WS.send(JSON.stringify({"ChairData":true, "data": {"LiveFile": "pathWidgetData", "context": context, "id": input_id, "path": fileData.value,  "vKey": vKeyInput.value, "new": true}}));
         userFeedback.textContent = "Data sent. Waiting for response...";
        var dropdownContainer = createExplorer(elements);
        widget.querySelector('#explorerContainer').appendChild(dropdownContainer)

    });

    function closeWidget() {
        widget.style.animation = 'path-widget-fadeOut 0.5s';
        setTimeout(() => {
            widget.style.display = 'none';
            targetElement.removeChild(widget);
        }, 500);
    }

    function createExplorer(elements) {
        var explorerContainer = document.createElement('div');
        explorerContainer.classList.add('explorer-container');
        var root = createDirectoryNode('root', true);
        explorerContainer.appendChild(root);
        elements.forEach(function(element){
            var parts = element.split('/');
            var currentNode = root;
            parts.forEach(function(part) {
                var existingNode = findChildNode(currentNode, part);
                if (existingNode) { currentNode = existingNode; }
                else {
                    var isLeafNode = (parts.indexOf(part) === parts.length - 1);
                    var newNode = createDirectoryNode(part, isLeafNode);
                    currentNode.appendChild(newNode); currentNode = newNode;
                }
            });
        });
        return explorerContainer;
    }
    function createDirectoryNode(name, isLeafNode) {
        var directoryNode = document.createElement('div');
        directoryNode.classList.add('directory-node');
        if (!isLeafNode) { directoryNode.classList.add('collapsed'); }
        var directoryName = document.createElement('span'); directoryName.classList.add('directory-name'); directoryName.textContent = name; directoryNode.appendChild(directoryName); directoryName.addEventListener('click', function() { if (directoryNode.classList.contains('collapsed')) { directoryNode.classList.remove('collapsed'); } else { directoryNode.classList.add('collapsed'); } }); return directoryNode; } function findChildNode(parentNode, name) { var childNodes = parentNode.children; for (var i = 0; i < childNodes.length; i++) { var childNode = childNodes[i]; var directoryName = childNode.querySelector('.directory-name'); if (directoryName && directoryName.textContent === name) { return childNode; } } return null; }

    var elements = [
        'path/to/element1',
        'path/to/element2',
        'path/element3',
        'path/to/element4',
        'path/element5',
        'element6'
    ];


    function wsHandelEvent(event) {
        const data = JSON.parse(event.data);
        if (data.type === 'pathWidgetData') {
            const userFeedback = widget.querySelector('#user-feedback');
            if (data.success) {
                userFeedback.textContent = "Data received successfully!";
                userFeedback.style.color = "green";
            } else {
                userFeedback.textContent = "Failed to receive data.";
                userFeedback.style.color = "red";
            }
        }
    };

    return widget;
}


