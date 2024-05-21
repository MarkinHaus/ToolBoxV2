function adjustGridForButtons(contentControlsElement, buttonsLength) {
    const maxButtonsPerRow = 3;
    let rows = Math.ceil(buttonsLength / maxButtonsPerRow);
    let buttonsInLastRow = buttonsLength % maxButtonsPerRow || maxButtonsPerRow;

    // Basis-Styles
    contentControlsElement.style.padding = '5px';
    contentControlsElement.style.display = "grid";
    contentControlsElement.style.gap = "10px";

    if (buttonsLength <= maxButtonsPerRow) {
        // Wenn 3 oder weniger Buttons, einfach alle in einer Reihe
        contentControlsElement.style.gridTemplateColumns = `repeat(${buttonsLength}, 1fr)`;
    } else {
        // Mehr als 3 Buttons, Anpassungen für die letzte Zeile
        let columnsTemplate = `repeat(${maxButtonsPerRow}, 1fr)`;
        if (rows > 1) {
            // Berechne, wie die Buttons in der letzten Zeile zentriert werden können
            let emptySpaces = maxButtonsPerRow - buttonsInLastRow;
            let centeringMargin = emptySpaces / 2;
            let lastRowTemplate = `[start] repeat(${centeringMargin}, 1fr [empty-start]) repeat(${buttonsInLastRow}, 1fr [button-start]) repeat(${centeringMargin}, 1fr [empty-end]) [end]`;

            // Setze die Spaltenvorlage für alle Zeilen außer der letzten
            for (let i = 1; i < rows; i++) {
                contentControlsElement.style.gridTemplateRows += `1fr `;
            }
            // Spezielle Vorlage für die letzte Zeile
            contentControlsElement.style.gridTemplateColumns = columnsTemplate;
            contentControlsElement.style.gridTemplateAreas = `"${lastRowTemplate}"`;
        }
    }
}

class OverlayUtility {
    constructor() {
        this.overlayElement = null;
        this.contentElement = null;
        this.closeOnOutsideClick = true;
        this.onClose = null
    }

    // Erstellt und zeigt das Overlay an
    createOverlay({content = "", closeOnOutsideClick = true, buttons = [], onClose = null, afterCrate = null} = {}) {
        if (this.overlayElement) {
            console.log("Ein Overlay ist bereits aktiv.");
            return;
        }

        this.onClose = onClose
        this.closeOnOutsideClick = closeOnOutsideClick;

        // Overlay-Container
        this.overlayElement = document.createElement('div');
        this.overlayElement.id = 'overlayElementOverlay';
        this.overlayElement.style.position = 'fixed';
        this.overlayElement.style.top = '0';
        this.overlayElement.style.left = '0';
        this.overlayElement.style.width = '100vw';
        this.overlayElement.style.height = '100vh';
        this.overlayElement.style.backgroundColor = 'rgba(0, 0, 0, 0.4)';
        this.overlayElement.style.display = 'flex';
        this.overlayElement.style.justifyContent = 'center';
        this.overlayElement.style.alignItems = 'center';
        this.overlayElement.style.zIndex = '9998';

        // Inhalt des Overlays
        this.contentElement = document.createElement('div');
        this.contentElement.id = 'overlayContentElement';
        this.contentElement.classList.add("widget")
        this.contentElement.style.background = 'var(--background-color)';
        this.contentElement.style.padding = '20px';
        this.contentElement.style.borderRadius = '10px';
        this.contentElement.style.minWidth = '300px';
        this.contentElement.style.maxWidth = '90vw';
        this.contentElement.style.borderWidth = '3px';
        this.contentElement.style.textAlign = 'center';
        this.contentElement.innerHTML = content;


        // Schaltflächen hinzufügen
        const contentControlsElement = document.createElement('div');


        buttons.forEach(button => {
            const btnElement = document.createElement('button');
            btnElement.textContent = button.text;
            btnElement.addEventListener('click', (e)=> {
                this.event = e
                if(button.action){button.action()}
                this.closeOverlay()
            })
            // btnElement.onclick =
            contentControlsElement.appendChild(btnElement);
        });

        if(buttons.length !== 0){
            adjustGridForButtons(contentControlsElement, buttons.length)
            this.contentElement.appendChild(document.createElement('hr'))
            this.contentElement.appendChild(contentControlsElement)
        }

        this.overlayElement.appendChild(this.contentElement);

        // Klick außerhalb des Inhalts schließt das Overlay, falls aktiviert
        if (this.closeOnOutsideClick) {
            this.overlayElement.addEventListener('click', (event) => {
                if (event.target === this.overlayElement) {
                    this.closeOverlay();
                }
            });

            this.overlayElement.title = "Klick outside to clos.\n" +
                "no decision necessary"
            this.contentElement.style.borderColor = 'var(--no-decision-color)'
        }else {

            this.overlayElement.title = "decision necessary"
            this.contentElement.style.borderColor = 'var(--decision-color)'
        }
        window.TBf.processRow(this.overlayElement)
        document.body.appendChild(this.overlayElement);

        if (afterCrate){
            afterCrate()
        }

    }

    // Schließt und entfernt das Overlay
    closeOverlay(action = true) {
        if (this.overlayElement) {
            document.body.removeChild(this.overlayElement);
            this.overlayElement = null;
            this.contentElement = null;
        }
        if (action && this.onClose){
            this.onClose()
        }
    }
}

// Beispiel für die Verwendung
window.overlayUtility = new OverlayUtility();
console.log("[OverlayUtility Online]")
// document.addEventListener('DOMContentLoaded', () => {
// document.getElementById('show-overlay-btn').addEventListener('click', () => {
//     overlayUtility.createOverlay({
//         content: "<h2>Bestätigen Sie Ihre Aktion</h2><p>Möchten Sie fortfahren?</p>",
//         closeOnOutsideClick: false,
//         buttons: [
//             {text: "Abbrechen", action: () => overlayUtility.closeOverlay()},
//             {text: "Bestätigen", action: () => { console.log("Aktion bestätigt"); overlayUtility.closeOverlay(); }}
//         ]
//     });
// });
// })
