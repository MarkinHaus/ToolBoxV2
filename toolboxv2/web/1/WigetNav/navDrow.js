if(window.history.state.TB){

    window.TBf.initVar("naturalToggle", [false, false, false])

function PSOverlayInit(){

const profileButton = document.querySelector('#profile');
const logOutButton = document.querySelector('#logOut');
const systemSettingsButton = document.querySelector('#systemSettings');

let profileContent = null;
let systemSettingsContent = null;

async function openProfileSettingsOverlay(){

    window.TBf.getVar("naturalToggle")[0] = true
    profileButton.style.transform = 'rotate(360deg)';
    profileButton.style.scale = 1.26
    profileButton.style.zIndex = '9999';

    if (profileContent === null) {
        profileContent = await window.TBf.loadHtmlFile('/api/WidgetsProvider/open_widget?name=CloudM.UI.widget');
    }
    window.overlayUtility.createOverlay({
        content: profileContent,
        closeOnOutsideClick: true,
        onClose: onCloseProfileSettingsOverlay,
    });
}

function onCloseProfileSettingsOverlay(){
    window.TBf.getVar("naturalToggle")[0] = false
    profileButton.style.transform = 'rotate(0deg)';
    profileButton.style.zIndex = '2'
    profileButton.style.scale = 1
}


async function logOutOverlay(){

    window.TBf.getVar("naturalToggle")[2] = true
    logOutButton.style.transform = 'rotate(360deg)';
    logOutButton.style.scale = 1.26
    logOutButton.style.zIndex = '9999'

    window.overlayUtility.createOverlay({
        content: "<style>.Btn {\n" +
            "  display: flex;\n" +
            "  align-items: center;\n" +
            "  justify-content: flex-start;\n" +
            "  min-width: max-content;\n" +
            "  height: 45px;\n" +
            "  border: none;\n" +
            "  border-radius: 5px;\n" +
            "  cursor: pointer;\n" +
            "  position: relative;\n" +
            "  overflow: hidden;\n" +
            "  transition-duration: .3s;\n" +
            "  box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.199);\n" +
            "  background-color: var(--background-color);\n" +
            "}\n" +
            "\n" +
            "/* plus sign */\n" +
            ".sign {\n" +
            "  width: 100%;\n" +
            "  transition-duration: .3s;\n" +
            "  display: flex;\n" +
            "  align-items: center;\n" +
            "  justify-content: center;\n" +
            "}\n" +
            "\n" +
            ".sign svg {\n" +
            "  width: 17px;\n" +
            "}\n" +
            "\n" +
            ".sign svg path {\n" +
            "  fill: #000000;\n" +
            "}\n" +
            "/* text */\n" +
            ".text {\n" +
            "  position: absolute;\n" +
            "  opacity: 0;\n" +
            "  font-size: 1.2em;\n" +
            "  font-weight: 600;\n" +
            "  transition-duration: .3s;\n" +
            "}\n" +
            "/* hover effect on button width */\n" +
            ".Btn:hover {\n" +
            "  width: 125px;\n" +
            "  border-radius: 5px;\n" +
            "  transition-duration: .3s;\n" +
            "}\n" +
            "\n" +
            ".Btn:hover .sign {\n" +
            "  width: 20%;\n" +
            "  transition-duration: .3s;\n" +
            "  padding-left: 5px;\n" +
            "}\n" +
            "/* hover effect button's text */\n" +
            ".Btn:hover .text {\n" +
            "  opacity: 1;\n" +
            "  width: 100%;\n" +
            "  transition-duration: .3s;\n" +
            "  padding-right: 15px;\n" +
            "}\n" +
            "/* button click effect*/\n" +
            ".Btn:active {\n" +
            "  transform: translate(2px ,2px);\n" +
            "}</style><h1>Logout</h1><button class=\"Btn\" style='scale:1.8' onclick=\"fetch('/web/logoutS', {method:'GET'});window.TBf.router('/web/logout');\n" +
            "document.getElementById('D-Provider').remove();window.overlayUtility.closeOverlay(true)\">\n" +
            "  \n" +
            "  <div class=\"sign\">" +
            "<span class=\"material-symbols-outlined\" style='color: #000000'>\n" +
            "key\n" +
            "</span>" +
            "</div>\n" +
            "  \n" +
            "  <div class=\"text\" style='color: #000000'><span class=\"material-symbols-outlined\">\n" +
            "check_circle\n" +
            "</span></div>\n" +
            "</button>\n<hr/><a href='/'> Home </a>",
        closeOnOutsideClick: true,
        onClose: onCloseLogOutOverlay,
    });
}

function onCloseLogOutOverlay(){
        window.TBf.getVar("naturalToggle")[2] = false
        logOutButton.style.transform = 'rotate(0deg)';
        logOutButton.style.zIndex = '2'
        logOutButton.style.scale = 1
}


async function openSystemSettingsOverlay(){

    window.TBf.getVar("naturalToggle")[1] = true
    systemSettingsButton.style.transform = 'rotate(360deg)';
    systemSettingsButton.style.scale = 1.26
    systemSettingsButton.style.zIndex = '9999'

    if (systemSettingsContent === null) {
        systemSettingsContent = await window.TBf.loadHtmlFile('/api/WidgetsProvider/open_widget?name=WidgetsProvider.BoardWidget');
    }
    window.overlayUtility.createOverlay({
        content: systemSettingsContent,
        closeOnOutsideClick: true,
        onClose: onCloseSystemSettingsOverlay,
        afterCrate: window.TBf.getM("initBoard")
    });
}

function onCloseSystemSettingsOverlay(){
    window.TBf.getVar("naturalToggle")[1] = false
    systemSettingsButton.style.transform = 'rotate(0deg)';
    systemSettingsButton.style.zIndex = '2'
    systemSettingsButton.style.scale = 1
}


profileButton.addEventListener('click', () => {

    if (!window.TBf.getVar("naturalToggle")[0]){
        openProfileSettingsOverlay().then(r => console.log("Profile opened"))
    }else{
        window.overlayUtility.closeOverlay()
    }

});

systemSettingsButton.addEventListener('click', () => {

    if (!window.TBf.getVar("naturalToggle")[1]){
        openSystemSettingsOverlay().then(r => console.log("Profile opened"))
    }else{
        window.overlayUtility.closeOverlay()
    }

});
logOutButton.addEventListener('click', () => {

    if (!window.TBf.getVar("naturalToggle")[2]){
        logOutOverlay().then(r => console.log("logOut opened"))
    }else{
        window.overlayUtility.closeOverlay()
    }

});


}

console.log("Starting NAV")
PSOverlayInit()
}else{
    console.log("pending on index")
}
//WS.send(JSON.stringify({"ServerAction":"getTextWidget"}));
//WS.send(JSON.stringify({"ServerAction":"getPathWidget"}));

