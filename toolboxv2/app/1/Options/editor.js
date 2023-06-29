let tasks = [];


function initTaskEditor(){

const taskNameInput = document.getElementById('task-name');
const taskForm = document.getElementById('task-form');
const keySelect = document.getElementById('key-select');
const addPairBtn = document.getElementById('add-pair');
const saveTaskBtn = document.getElementById('save-task');
const loadTaskBtn = document.getElementById('load-task');
const newTaskBtn = document.getElementById('new-task');
const editorFrame = document.getElementById('editorFrame');
const switchIcon = document.createElement('span');

taskNameInput.value =  "Task-"+tasks.length;
tasks = JSON.parse(localStorage.getItem('tasks'))
switchIcon.className = 'text-widget-close-button';
let switchIconBool = false
switchIcon.innerHTML = '▲▽';

function openEditorFrame(){
    editorFrame.classList.remove('none')
    switchIcon.innerHTML = '△▼';
}
function closeEditorFrame(){
    editorFrame.classList.add('none')
    switchIcon.innerHTML = '▲▽';
}

switchIcon.onclick = function () {
    if (switchIconBool){
        openEditorFrame()
    }else {
        closeEditorFrame()
    }
    switchIconBool = !switchIconBool;
};

editorFrame.appendChild(switchIcon)

function createKeyValuePair(key = '', value = '', deletabel=true) {
    const group = document.createElement('div');
    group.className = 'key-value-group';

    const keyInput = document.createElement('h3');
    keyInput.placeholder = 'Key';
    keyInput.innerText = key;
    const valueInput = document.createElement('input');
    valueInput.placeholder = 'Value';
    valueInput.value = value;

    if (deletabel){
        const deleteIcon = document.createElement('span');
        deleteIcon.className = 'text-widget-close-button';
        deleteIcon.innerHTML = 'X';
        deleteIcon.onclick = function () {
            taskForm.removeChild(group);
        };
        group.appendChild(deleteIcon);
    }

    group.appendChild(keyInput);
    group.appendChild(valueInput);
    taskForm.appendChild(group);
}

addPairBtn.addEventListener('click', () => {
    const selectedKey = keySelect.value;
    if (selectedKey) {
        createKeyValuePair(selectedKey);
        keySelect.querySelector(`option[value="${selectedKey}"]`).remove();
        keySelect.value = '';
    }
});

saveTaskBtn.addEventListener('click', () => {
    const taskName = taskNameInput.value;
    if (!taskName) {
        alert('Please enter a task name.');
        return;
    }

    const inputs = taskForm.querySelectorAll('input');
    const values = taskForm.querySelectorAll('h3');
    const task = {};
    for (let i = 0; i < inputs.length; i += 1) {
        task[values[i].innerText] = inputs[i].value;
    }
    tasks.push({ name: taskName, task });
    localStorage.setItem('tasks', JSON.stringify(tasks));
});

loadTaskBtn.addEventListener('click', () => {
    const taskName = taskNameInput.value;
    if (!taskName) {
        alert('Please enter a task name.');
        return;
    }

    tasks = JSON.parse(localStorage.getItem('tasks')) || [];
    const taskObj = tasks.find(t => t.name === taskName);
    if (taskObj) {
        const task = taskObj.task;
        taskForm.innerHTML = '';
        for (const key in task) {
            createKeyValuePair(key, task[key]);
        }
    } else {
        alert('Task not found.');
    }
});

newTaskBtn.addEventListener('click', () => {

    taskForm.innerHTML = '';
    taskNameInput.value =  "Task-"+tasks.length;
    createKeyValuePair('use', 'agent|tool|function|chain', false);
    createKeyValuePair('name', 'self|think|summary|todolist|search|execution', false);
    createKeyValuePair('args', '$user-input', false);
    createKeyValuePair('return', '$return', false);

});


// Add default key-value pairs
createKeyValuePair('use', 'agent|tool|function|chain', false);
createKeyValuePair('name', 'self|think|summary|todolist|search|execution', false);
createKeyValuePair('args', '$user-input', false);
createKeyValuePair('return', '$return', false);


}


initTaskEditor()
