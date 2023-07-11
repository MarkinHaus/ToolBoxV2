function addTextWidget(element_id, id, context, content="") {
    console.log("ADDING Widget ", element_id);
    const targetElement = document.getElementById(element_id);
    let speechBalloon = createTextWidget(id, context, targetElement, content);
    targetElement.appendChild(speechBalloon);
    speechBalloon.id = id;
    return speechBalloon
}

function createTextWidget(textarea_id, context, targetElement, content="") {
    const template = document.getElementById('text-widget-template');
    const widget = template.content.cloneNode(true).querySelector('.text-widget');
    const fromElement = widget.querySelector('.text-widget-from');
    fromElement.textContent = context;
    const textarea = widget.querySelector('#text-widget-text-input');
    const widget_injection = widget.querySelector('#text-widget-injection');
    textarea.id = textarea_id+'-Text';
    textarea.value = content;

    const closeButton = widget.querySelector('.text-widget-close-button');
    closeButton.addEventListener('click', closeWidget);
    widget_injection.addEventListener('click', ()=>{
        console.log("widget_injection:testWidget")
        WS.send(JSON.stringify({"ChairData":true, "data": {"type":"textWidgetData","context":context,
                "id": textarea_id, "text": textarea.value}}));
    });

    function closeWidget() {
        widget.style.animation = 'text-widget-fadeOut 0.5s';
        setTimeout(() => {
            widget.style.display = 'none';
            targetElement.removeChild(widget);
        }, 500);
    }

    return widget;
}
