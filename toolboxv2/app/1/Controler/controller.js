const settingsWidgetTemplate = `

    <div class="widget-inner" id="settingsWidget-widget">
    <div class="widget-from" id="settingsWidget-widget-from">Settings</div>
        <label class="widget-label" id="settingsWidget-widget-label"></label>
        <span class="widget-close-button" id="settingsWidget-widget-close-button">X</span>
        <span class="widget-tooltip" id="settingsWidget-widget-tooltip"></span>
        <div class="widget-items" id="settingsWidget-widget-items"></div>
    </div>
`;

function createWidget(json, targetElement) {
    let data = typeof json === 'string' ? JSON.parse(json) : json;

    const widget = document.createElement('div');
    widget.innerHTML = settingsWidgetTemplate;

    const label = widget.querySelector('.widget-label');
    label.textContent = data.label;

    const tooltip = widget.querySelector('.widget-tooltip');
    tooltip.textContent = data.tooltip;

    const itemsContainer = widget.querySelector('.widget-items');
    data.items.forEach(item => {
        const element = createElementFromDict(item);
        itemsContainer.appendChild(element);
    });

    const closeButton = widget.querySelector('.widget-close-button');
    closeButton.addEventListener('click', closeWidget);

    function closeWidget() {
        widget.style.animation = 'text-widget-fadeOut 0.5s';
        setTimeout(() => {
            widget.style.display = 'none';
            targetElement.removeChild(widget);
        }, 500);
    }

    return widget;
}

function addSettingsWidget(json, containerId) {
    const container = document.getElementById('main-settings');
    const widget = createWidget(json, container);
    container.appendChild(widget);
    return widget
}

function crateSettingsWidget(containerId) {
    const container = document.getElementById(containerId);
    const widget = createElementFromDict({
    tag: 'div',
    attributes: {
        id: 'main-settings',
        class: 'widget draggable'
    },
    content: '<h1> Hallo wort</h1>',
    },);
    container.appendChild(widget);
    return widget
}
