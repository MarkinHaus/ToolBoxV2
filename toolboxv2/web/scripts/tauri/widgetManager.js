const { WebviewWindow} = window.__TAURI__.window;
// import { WebviewWindow, PhysicalPosition, PhysicalSize} from "/web/node_modules/@tauri-apps/api/window.js";
export async function openBuildInWidget(titel, x,y,width,height, url='web/0/template.html') {

    const webview = new WebviewWindow(
        titel,{
        hiddenTitle:false,
        url: url,
        alwaysOnTop: false,
        focus: true,
        skipTaskbar: false,
        theme:'dark',
        titleBarStyle: 'dark',
        decorations: false,
        transparent: true,
        x, y, width, height
    });

    await webview.once('tauri://created', function () {
        console.log("Fenster erfolgreich erstellt");
    });

    await webview.once('tauri://error', function (e) {
        console.log(e);
    });
    return webview
}
