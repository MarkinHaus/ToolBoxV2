let counter = 0;
let mac_val = 1; // Standardwert auf 1 setzen, entsprechend dem Platzhalter im Input

document.getElementById('iCounter').addEventListener('change', function(e) {
    mac_val = parseInt(e.target.value, 10) || 0; // Stellen Sie sicher, dass mac_val eine Zahl ist
});

function updateCounterDisplay() {
    document.getElementById('counterValue').textContent = counter;
}

function addCounter() {
    counter++;
    updateCounterDisplay();
}

function mAddCounter() {
    counter += mac_val;
    updateCounterDisplay();
}

function resetCounter() {
    counter = 0;
    updateCounterDisplay();
}

// Initialer Aufruf, um den Zähler beim Laden der Seite zu setzen
updateCounterDisplay();
