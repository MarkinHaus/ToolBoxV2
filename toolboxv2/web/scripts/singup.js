import {httpPostData} from "/web/scripts/httpSender.js";
import {registerUser, generateAsymmetricKeys, storePrivateKey, decryptAsymmetric, signMessage} from "/web/scripts/cryp.js";

let UserData = undefined
let registrationData = undefined
let privateKey_base64 = null
let username

function handleCreateUserError(data) {
    console.log("[handleCreateUserError data]:", data)
    document.getElementById('infoText').textContent = data.info.help_text;
    document.getElementById('infoPopup').style.display = 'block';
    UserData = undefined
    document.getElementById('titel-h2').textContent = "Sing Up"
    document.getElementById('username').classList.remove('none')
    document.getElementById('email').classList.remove('none')
    document.getElementById('initiation').classList.remove('none')
    return data
}

async function handleCreateUserSuccess(data) {
    console.log("[handleCreateUserSuccess data]:", data)
    registrationData = data.get()
    UserData = true
    document.getElementById('submit-button').textContent = "Retry";
    document.getElementById('infoText').textContent = "Now we linke your account with your Persona. A pop-up window should appear. If it does not appear, please click on 'Retry' manually.";
    document.getElementById('infoPopup').style.display = 'block';
    document.getElementById('titel-h2').textContent = "Sing Up for "+document.getElementById('username').value
    document.getElementById('username').classList.add('none')
    document.getElementById('email').classList.add('none')
    document.getElementById('initiation').classList.add('none')

    console.log("[registrationData--]:", registrationData)
    console.log("[registrationData.dSync, keys.privateKey_base64]:", registrationData.dSync, privateKey_base64)
    const deviceID = await decryptAsymmetric(registrationData.dSync, privateKey_base64, true)
    await window.sessionStorage.setItem("SKey", deviceID)
    await storePrivateKey(privateKey_base64, username)
    await registrate_user_personal(await signMessage(privateKey_base64, registrationData.challenge))

    return data
}
function handleCreateUserErrorPersona(data) {
    console.log("[handleCreateUserError data]:", data)
    document.getElementById('infoText').textContent = data.info.help_text;
    document.getElementById('infoPopup').style.display = 'block';
    UserData = undefined
    return data
}

async function handleCreateUserSuccessPersona(data) {
    console.log("[handleCreateUserSuccess data]:", data)
    setTimeout(()=>{
        window.location.href = "/web/login";}, 120)
    return data
}

async function createUser(name, email, invitation, pub_key) {
    return httpPostData('CloudM.AuthManager',
        'create_user',
        {
            name, email ,pub_key ,invitation,
            web_data:true, as_base64:false
        },
        handleCreateUserError, handleCreateUserSuccess);
}

async function registrate_user_personal(sing){
    const done = await registerUser(registrationData, sing, handleCreateUserErrorPersona, handleCreateUserSuccessPersona)
    UserData = !done
    if (done){
        sessionStorage.setItem("local_ws.onopen:installMod-welcome", 'true')
        UserData = false
        document.getElementById('titel-h2').textContent = "Welcome to SimpleCore "+document.getElementById('username').value
        document.getElementById('submit-button').textContent = "Go to dashboard";
        document.getElementById('infoText').textContent = "Verification Done";
        document.getElementById('infoPopup').style.display = 'block';
    }else{
        document.getElementById('infoText').textContent = "Pleas press on 'Retry'";
        document.getElementById('infoPopup').style.display = 'block';
    }
}

document.getElementById('signupForm').addEventListener('submit',async function(event) {
    event.preventDefault();

    if (document.getElementById('submit-button').textContent === "Retry"){
        document.getElementById('skip-persona-button').classList.remove('none')
        document.getElementById('infoText').textContent = "If the web auth interface is not working you can add the personal data letter. To continue press on 'Only Device'";
    }

    if (UserData === false) {
        window.location.href = "/web/login";
        return
    }

    if (UserData === undefined){
        username = document.getElementById('username').value;
        const email = document.getElementById('email').value;
        const invitation = document.getElementById('initiation').value;
        document.getElementById('infoText').textContent = "Validate information's";
        document.getElementById('infoPopup').style.display = 'block';
        await generateAsymmetricKeys().then(async keys => {
            await createUser(username, email, invitation, keys.publicKey)
            privateKey_base64 =  keys.privateKey_base64
        }, 600);
        return
    }
    console.log("[registrationData]:", registrationData)
    try {
        if (UserData === true && registrationData !== {}){
            console.log("[registrationData--]:", registrationData)
            if (privateKey_base64 !== undefined && privateKey_base64 !== null){
                console.log("[registrationData.dSync, keys.privateKey_base64]:", registrationData.dSync, privateKey_base64)
                const deviceID = await decryptAsymmetric(registrationData.dSync, privateKey_base64, true)
                await window.sessionStorage.setItem("SKey", deviceID)
                await storePrivateKey(privateKey_base64, username)
                await window.sessionStorage.removeItem('temp_base64_key')
            }
            await registrate_user_personal(await signMessage(privateKey_base64, registrationData.challenge))
            return
        }
    }
    catch (e) {
        console.log(e)
    }
});
