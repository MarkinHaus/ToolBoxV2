class TaskWidget {
    constructor(use, mode, name, args, returnVal) {
        this.use = use;
        this.mode = mode;
        this.name = name;
        this.args = args;
        this.returnVal = returnVal;
    }

    createWidget() {
        const template = document.getElementById('task-widget-template');
        const widget = template.content.cloneNode(true).querySelector('.task-widget');

        widget.querySelector('.task-widget-use').textContent = this.use;
        widget.querySelector('.task-widget-mode').value = this.mode;
        widget.querySelector('.task-widget-name').value = this.name;
        widget.querySelector('.task-widget-args').textContent = this.args;
        widget.querySelector('.task-widget-return').textContent = this.returnVal;

        widget.querySelector('.task-widget-mode').addEventListener('change', (event) => {
            this.switchMode(event.target.value, widget);
        });

        return widget;
    }

    switchMode(mode, widget) {
        this.mode = mode;
        const nameInput = widget.querySelector('.task-widget-name');
        if (mode === 'edit') {
            nameInput.disabled = false;
        } else {
            nameInput.disabled = true;
        }
    }
}

// Beispiel für die Verwendung der Klasse
const taskWidget = new TaskWidget('Task 1', 'display', 'My Task', 'arg1, arg2', 'return value');
document.body.appendChild(taskWidget.createWidget());
