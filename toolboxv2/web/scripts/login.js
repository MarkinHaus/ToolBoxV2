
import {httpPostData, httpPostUrl, AuthHttpPostData} from "/web/scripts/httpSender.js";
import {authorisazeUser, signMessage, retrievePrivateKey, decryptAsymmetric} from "/web/scripts/cryp.js";

let base64_privat_key = null
let checked = [false]
let caunter = 0
let username

let next_url = "/web/mainContent.html"
getKeyFromURL()

function getKeyFromURL() {
    // Create a URL object based on the current window location
    const url = new URL(window.location.href);
    console.log("URL:", url)
    // Use URLSearchParams to get the 'key' query parameter
    const next_ = url.searchParams.get('next');

    if (next_) {
        next_url = next_;
    }

}


function get_handleLoginError(id) {
    function handleLoginError(data) {
        console.log("[handleCreateUserError data]:",id, data)
        window.TBf.animator("R2+102")
        if (data.info.help_text){
            document.getElementById('infoText').textContent = data.info.help_text;
        }else {
            document.getElementById('infoText').textContent = "Unknown error"
        }
        document.getElementById('infoPopup').style.display = 'block';
        return data
    }
    return handleLoginError
}

function handleLoginSMK(data) {
    window.TBf.animator("R2-203")
    console.log("[handleLoginSMK data]:", data)
    document.getElementById('infoText').textContent = "Bitte schauen sie in ihre Emails";
    document.getElementById('infoPopup').style.display = 'block';
    return data
}

function rr(){
    setTimeout(async () => {
        localStorage.setItem("local_ws.onopen:installMod-welcome", 'false');
        window.TBf.animator("Y2+203")
        await AuthHttpPostData(username, get_handleLoginError("Session Error"), (_)=>{
            window.TBf.router(next_url)
        })
    }, 200);
}

async function handleLoginSuccessVP(data) {
    console.log("[handleLoginSuccessVP data]:", data)
    window.TBf.animator("Y1+101")
    document.getElementById('infoText').textContent = data.info.help_text;
    document.getElementById('infoPopup').style.display = 'block';
    setTimeout(() => {
        if (data.get()){
            localStorage.setItem("local_ws.onopen:installMod-welcome", 'false');
            window.TBf.router(data.get())
        }
    }, 200);

    return data
}

async function handleLoginSuccessVD(data) {
    console.log("[handleLoginSuccessVD data]:", data)
    window.TBf.animator("Y2-101")
    document.getElementById('infoText').textContent = data.info.help_text;
    document.getElementById('infoPopup').style.display = 'block';

    if (data.get().toPrivat){
        try {
            const claim = await decryptAsymmetric(data.get().key, privateKey_base64, true)
            localStorage.setItem('jwt_claim_device', claim);
            rr()
        }catch (e){
            console.log("Error handleLoginSuccessVP", e)
        }
    }else {
        if (base64_privat_key === null){
            document.getElementById('infoText').textContent = "Sie erhalten in kürze eine Email zum einloggen"
            httpPostUrl('CloudM.AuthManager', 'get_magic_link_email', 'username='+username, get_handleLoginError("mk"), handleLoginSMK);
            return data
        }else{
            localStorage.setItem('jwt_claim_device', data.get().key);
            rr()
        }
    }

    return data
}


async function handleLoginSuccess(data) {
    window.TBf.animator("Y2+101")
    console.log("[handleLoginSuccess data]:", data)
    document.getElementById('infoText').textContent = "Login in progress";
    document.getElementById('infoPopup').style.display = 'block';

    console.log("[checked]:", document.getElementById("register-device").classList.contains('none'))

    if (document.getElementById("register-device").classList.contains('none')){
        if(await authorisazeUser(data.get().rowId, data.get().challenge, document.getElementById('username').value, get_handleLoginError("authorisazeUser"), handleLoginSuccessVP)){
            document.getElementById('infoText').textContent = "Validate user successful";
        }
    }else{
        base64_privat_key = await retrievePrivateKey(username)

        if (base64_privat_key === "Invalid user name device not registered"){
            document.getElementById('infoText').textContent = "Invalid user name device not registered on '"+document.getElementById('username').value+"'";
            document.getElementById('infoPopup').style.display = 'block';
            caunter++;
        }else{
            const signature = await signMessage(base64_privat_key, data.get().challenge)
            httpPostData('CloudM.AuthManager', 'validate_device', { username:document.getElementById('username').value, signature},
                get_handleLoginError("signMessage"), handleLoginSuccessVD);
        }

    }

    return data
}

function loginDevice(username) {
    window.TBf.animator("P2-101")
    httpPostUrl('CloudM.AuthManager', 'get_to_sing_data', 'username='+username + '&personal_key='+document.getElementById("register-device").classList.contains('none').toString(), get_handleLoginError("loginDevice"), handleLoginSuccess);
}

document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault();

    username = document.getElementById('username').value;
    loginDevice(username);
});
