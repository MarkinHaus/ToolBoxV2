// const { invoke } = window.__TAURI__.tauri;
const {WebviewWindow, PhysicalPosition, PhysicalSize} = window.__TAURI__.window
let greetInputEl;
let greetMsgEl;

async function openStaticWidget(name, payload){

}

async function openStaticDynamic(){}

function open_window(pp=[100, 100], wz=[800, 600], name, url){
    const position = new PhysicalPosition(pp[0], pp[1]); // Beispielposition: x=100, y=100
    const size = new PhysicalSize(wz[0], wz[1]); // Beispielgröße: Breite=800, Höhe=600
    return new WebviewWindow(name, {
        url: url,
        position: position,
        size: size
    });
}
async function greet() {

  const webview = open_window([100, 100], [800, 600], greetInputEl.value, "/src/staticWidget.html")
  webview.once('tauri://created', function () {
    greetMsgEl.textContent =  "Fenster erfolgreich erstellt";
  });
  webview.once('tauri://error', function (e) {
    greetMsgEl.textContent = e;
  });
}

window.addEventListener("DOMContentLoaded", () => {
  greetInputEl = document.querySelector("#greet-input");
  greetMsgEl = document.querySelector("#greet-msg");
  document.querySelector("#greet-form").addEventListener("submit", (e) => {
    e.preventDefault();
    greet();
  });
});
