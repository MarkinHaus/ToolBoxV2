let naturalToggle = [false, false]


function PSOverlayInit(){

const profileButton = document.querySelector('#profile');
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


}

console.log("Starting NAV")
PSOverlayInit()
//WS.send(JSON.stringify({"ServerAction":"getTextWidget"}));
//WS.send(JSON.stringify({"ServerAction":"getPathWidget"}));

