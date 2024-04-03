let naturalToggle = [false, false, false]


function PSOverlayInit(){

const profileButton = document.querySelector('#profile');
const logOutButton = document.querySelector('#logOut');
const systemSettingsButton = document.querySelector('#systemSettings');

let profileContent = null;
let systemSettingsContent = null;

async function openProfileSettingsOverlay(){

    naturalToggle[0] = true
    profileButton.style.transform = 'rotate(360deg)';
    profileButton.style.scale = 1.26
    profileButton.style.zIndex = '9999';

    if (profileContent === null) {
        profileContent = await loadHtmlFile('/web/1/insightsWidget/profileTab.html'); // TODO change wit api call and ad profile specific information in be
    }
    overlayUtility.createOverlay({
        content: profileContent,
        closeOnOutsideClick: true,
        onClose: onCloseProfileSettingsOverlay,
    });
}

function onCloseProfileSettingsOverlay(){
    naturalToggle[0] = false
    profileButton.style.transform = 'rotate(0deg)';
    profileButton.style.zIndex = '2'
    profileButton.style.scale = 1
}


async function logOutOverlay(){

    naturalToggle[2] = true
    logOutButton.style.transform = 'rotate(360deg)';
    logOutButton.style.scale = 1.26
    logOutButton.style.zIndex = '9999'

    overlayUtility.createOverlay({
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
            "  background-color: var(--primary-color);\n" +
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
            "}</style><h1>Logout</h1><button class=\"Btn\" onclick=\"window.location.href = '/web/logoutS' \">\n" +
            "  \n" +
            "  <div class=\"sign\">" +
            "<span class=\"material-symbols-outlined\" style='color: #000000'>\n" +
            "key\n" +
            "</span>" +
            "</div>\n" +
            "  \n" +
            "  <div class=\"text\" style='color: #000000'>lock</div>\n" +
            "</button>\n",
        closeOnOutsideClick: true,
        onClose: onCloseLogOutOverlay,
    });
}


function onCloseLogOutOverlay(){
        naturalToggle[2] = false
        logOutButton.style.transform = 'rotate(0deg)';
        logOutButton.style.zIndex = '2'
        logOutButton.style.scale = 1
    }


async function openSystemSettingsOverlay(){

    naturalToggle[1] = true
    systemSettingsButton.style.transform = 'rotate(360deg)';
    systemSettingsButton.style.scale = 1.26
    systemSettingsButton.style.zIndex = '9999'

    if (systemSettingsContent === null) {
        systemSettingsContent = await loadHtmlFile('/web/1/insightsWidget/SystemTab.html'); // TODO change wit api call and ad profile specific information in be
    }
    overlayUtility.createOverlay({
        content: systemSettingsContent,
        closeOnOutsideClick: true,
        onClose: onCloseSystemSettingsOverlay,
    });
}

function onCloseSystemSettingsOverlay(){
    naturalToggle[1] = false
    systemSettingsButton.style.transform = 'rotate(0deg)';
    systemSettingsButton.style.zIndex = '2'
    systemSettingsButton.style.scale = 1
}


profileButton.addEventListener('click', () => {

    if (!naturalToggle[0]){
        openProfileSettingsOverlay().then(r => console.log("Profile opened"))
    }else{
        overlayUtility.closeOverlay()
    }

});

systemSettingsButton.addEventListener('click', () => {

    if (!naturalToggle[1]){
        openSystemSettingsOverlay().then(r => console.log("Profile opened"))
    }else{
        overlayUtility.closeOverlay()
    }

});
logOutButton.addEventListener('click', () => {

    if (!naturalToggle[2]){
        logOutOverlay().then(r => console.log("logOut opened"))
    }else{
        overlayUtility.closeOverlay()
    }

});


}

console.log("Starting NAV")
PSOverlayInit()
//WS.send(JSON.stringify({"ServerAction":"getTextWidget"}));
//WS.send(JSON.stringify({"ServerAction":"getPathWidget"}));

