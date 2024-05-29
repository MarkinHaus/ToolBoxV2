
import
    {
        httpPostData, httpPostUrl, AuthHttpPostData
    }
    from
    "/web/scripts/httpSender.js";
import
    {
        generateAsymmetricKeys, signMessage, decryptAsymmetric, storePrivateKey
    }
    from
    "/web/scripts/cryp.js";

if(window.history.state && window.history.state.TB) {
    let privateKey_base64 = null
    let username
    // Save the key in a variable
    let [invitation, NL] = getKeyFromURL();


    async function startMkLink() {

        if (!username) {
            username = document.getElementById('username').value;
            document.getElementById('username').classList.add('none')
        }
        localStorage.setItem("StartMLogIN", "True")
        await generateAsymmetricKeys().then(async keys => {
            privateKey_base64 = keys.privateKey_base64
            await storePrivateKey(privateKey_base64, username)
            await registerUserDevice(username, keys.publicKey)
        }, 600);

    }

    function show_error(from) {
        console.log("ERROR", from)
        document.getElementById('infoText').textContent = from;
        document.getElementById("Main-content").classList.add("none")
        document.getElementById("error-content").classList.remove("none")
    }

    // Log the key to the console (for daemonstration purposes)

    function getKeyFromURL() {
        // Create a URL object based on the current window location
        const url = new URL(window.location.href);

        // Use URLSearchParams to get the 'key' query parameter
        const key = url.searchParams.get('key');
        const nl = url.searchParams.get('nl');
        const name = url.searchParams.get('name');
        console.log("key:", key)
        console.log("nl:", nl)
        console.log("name:", name)
        console.log("!nl && !name:", !nl && !name, !nl, !name)
        if (!key) {
            show_error("No key in url fund")
        }
        if (name) {
            username = name;
            document.getElementById('username').classList.add("none")
        }
        if (!nl && !name) {
            show_error("No name found in url or name length")
        }
        return [key, nl];
    }

    function get_handleLoginError(id) {
        function handleLoginError(data) {
            console.log("[handleCreateUserError data]:", id, data)
            document.getElementById('infoText').textContent = data.info.help_text;
            document.getElementById('infoPopup').style.display = 'block';
            show_error(id + " handleLoginError")
            return data
        }

        return handleLoginError
    }

    function rr() {

        setTimeout(async () => {
            localStorage.setItem("local_ws.onopen:installMod-welcome", 'false');
            await AuthHttpPostData(username, get_handleLoginError("Session Error"), (_) => {
                window.TBf.router("/web/dashboard");
                localStorage.removeItem("StartMLogIN")
            })

        }, 200);
    }

    async function handleLoginSuccessVD(data) {
        console.log("[handleLoginSuccessVD data]:", data)
        document.getElementById('infoText').textContent = data.info.help_text;
        document.getElementById('infoPopup').style.display = 'block';
        if (data.get().toPrivat) {
            try {
                const claim = await decryptAsymmetric(data.get().key, privateKey_base64, true)
                localStorage.setItem('jwt_claim_device', claim);
                rr()
            } catch (e) {
                console.log("Error handleLoginSuccessVP", e)
                show_error("handleLoginSuccessVD claim saveing")
            }
        } else {
            localStorage.setItem('jwt_claim_device', data.get().key);
            rr()
        }


        return data
    }

    async function handleLoginSuccess(data) {
        document.getElementById('infoText').textContent = "Login in progress for " + username;
        document.getElementById('infoPopup').style.display = 'block';

        const signature = await signMessage(privateKey_base64, data.get().challenge)
        await httpPostData('CloudM.AuthManager', 'validate_device', {username, signature},
            get_handleLoginError("signMessage"), handleLoginSuccessVD);

        return data
    }

    async function registerUserDevice(name, pub_key) {
        return httpPostData('CloudM.AuthManager',
            'add_user_device',
            {
                name: username, pub_key, invitation,
                web_data: true, as_base64: false
            },
            get_handleLoginError("registerUserDevice"), handleCreateUserSuccess);
    }

    async function handleCreateUserSuccess(data) {
        console.log("[handleCreateUserSuccess data]:", data)
        document.getElementById('infoText').textContent = data.info.help_text;
        document.getElementById('infoPopup').style.display = 'block';
        try {
            const deviceID = await decryptAsymmetric(data.get().dSync, privateKey_base64, true)
            await window.sessionStorage.setItem("SKey", deviceID)
            await storePrivateKey(privateKey_base64, username)
            loginDevice(username)
        } catch (e) {
            show_error("Invalid privat key")
        }
        return data
    }


    function loginDevice(username) {
        httpPostUrl('CloudM.AuthManager', 'get_to_sing_data', 'username=' + username + '&personal_key=False', get_handleLoginError("loginDevice"), handleLoginSuccess);
    }

    if (username) {
        startMkLink()
    }

    const userInput = document.getElementById("username")

    if (userInput) {
        userInput.addEventListener("oninput", (e) => {

            if (e.value.length >= NL) {
                startMkLink();
            }

        })
    } else {
        console.log("Invalid no addEventListener")
    }
}
