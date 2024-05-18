// Array zum Speichern der erstellten Widgets
import {openBuildInWidget} from "/web/scripts/tauri/widgetManager.js";

let widgets = [];

function createWidget() {
    const widgetName = document.getElementById('widget-name').value;
    if (widgetName.trim() === '') {
        alert('Bitte geben Sie einen Widget-Namen ein.');
        return;
    }

    const widget = {
        name: widgetName,
        id: Date.now()
    };
    widgets.push(widget);

    if (widgetName.includes("Note#")) {
        const actualName = widgetName.replace("Note#", "");
        setTimeout(async () => {
            var url = "web/dashboards/DynamicWidget.html?widgetName="+actualName+"&widgetType=quickNote.QnWidget" //&widgetID="+widgetName
            await openBuildInWidget(actualName, 0, 0, 350, 470, url);
        }, 1);
        console.log("Opened Note widget");
    }
    if (widgetName.includes("Test#")) {
        const actualName = widgetName.replace("Test#", "");
        setTimeout(async () => {
            var url = "web/dashboards/DynamicWidget.html?widgetName="+actualName+"&widgetType=TestWidget" //&widgetID="+widgetName
            await openBuildInWidget(actualName, 0, 0, 350, 870, url);
        }, 1);
        console.log("Opened Note widget");
    }
    if (widgetName.includes("CMUI#")) {
        const actualName = widgetName.replace("CMUI#", "");
        setTimeout(async () => {
            var url = "web/dashboards/DynamicWidget.html?widgetName="+actualName+"&widgetType=CloudM.UI.widget" //&widgetID="+widgetName
            await openBuildInWidget(actualName, 0, 0, 650, 670, url);
        }, 1);
        console.log("Opened Note widget");
    }

    addWidgetToManager(widget);
    document.getElementById('widget-name').value = '';
}

function addWidgetToManager(widget) {
    const widgetList = document.getElementById('widget-list');
    const widgetElement = document.createElement('div');
    widgetElement.classList.add('widgetM');
    widgetElement.innerHTML = `
        <div class="widget-info">${widget.name}</div>
        <div class="widget-controls">
            <button onclick="openWidget(${widget.id})">NoOs Port</button>
            <button onclick="closeWidget(${widget.id})">Schließen</button>
            <button onclick="deleteWidget(${widget.id})">Löschen</button>
        </div>
    `;
    widgetList.appendChild(widgetElement);
}

function openWidget(widgetId) {
    alert(`Öffne Widget mit ID: ${widgetId}`);
}

function closeWidget(widgetId) {
    alert(`Schließe Widget mit ID: ${widgetId}`);
}

function deleteWidget(widgetId) {
    widgets = widgets.filter(widget => widget.id !== widgetId);
    document.getElementById('widget-list').innerHTML = '';
    widgets.forEach(widget => addWidgetToManager(widget));
}

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("createWidget").addEventListener('click', createWidget);
    widgets.forEach(widget => addWidgetToManager(widget));
});
