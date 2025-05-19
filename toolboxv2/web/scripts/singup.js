// /web/scripts/signup.js (Refactored with Animation)

function setupSignup() {
    const signupForm = document.getElementById('signupForm');
    const usernameInput = document.getElementById('username');
    const emailInput = document.getElementById('email');
    const initiationKeyInput = document.getElementById('initiation');
    const skipPersonaButton = document.getElementById('skip-persona-button');
    const infoPopup = document.getElementById('infoPopup');
    const infoText = document.getElementById('infoText');

    function showInfo(message, isError = false, animationSequence = null) {
        infoText.textContent = message;
        infoPopup.style.display = 'block';
        if (isError) {
            TB.ui.Toast.show(message, 'error', 5000);
            if (TB.graphics?.playAnimationSequence) TB.graphics.playAnimationSequence(animationSequence || "R0-31");
        } else {
            TB.ui.Toast.show(message, 'success', 3000);
            if (TB.graphics?.playAnimationSequence) TB.graphics.playAnimationSequence(animationSequence || "P0+21");
        }
    }

    async function handleSignup(registerAsPersona) {
        const username = usernameInput.value.trim();
        const email = emailInput.value.trim();
        const initiationKey = initiationKeyInput.value.trim();

        if (!username || !email) {
            showInfo("Username and Email are required.", true, "Y0-22");
            return;
        }

        showInfo(`Attempting to sign up ${username}...`);
        if (TB.graphics?.playAnimationSequence) TB.graphics.playAnimationSequence("Y1+11:R1-11");


        try {
            const result = await TB.user.signup(username, email, initiationKey, registerAsPersona);

            if (result.success) {
                let successMessage = result.message || "Signup successful!";
                let successAnimation = "Z1+32:Y0+50"; // Default success

                if (registerAsPersona && result.data && result.data.needsWebAuthnRegistration) {
                    successMessage = "Signup successful. Now, register your security key/device (WebAuthn).";
                    // Potentially a different animation for this pending step
                    successAnimation = "P1+21:Y1+21";
                    showInfo(successMessage, false, successAnimation);
                    // const webAuthnRegResult = await TB.user.registerWebAuthnForCurrentUser(username);
                    // showInfo(webAuthnRegResult.message, !webAuthnRegResult.success, webAuthnRegResult.success ? "Z1+42" : "R2-42");
                    // if (webAuthnRegResult.success) setTimeout(() => TB.router.navigateTo('/web/dashboard'), 800);
                } else if (result.data && result.data.token) {
                    showInfo(successMessage, false, successAnimation);
                    setTimeout(() => TB.router.navigateTo('/web/dashboard'), 800);
                } else {
                    showInfo("Signup successful. Please log in.", false, "P0+31");
                    setTimeout(() => TB.router.navigateTo('/web/assets/login.html'), 800);
                }
            } else {
                showInfo(result.message || "Signup failed.", true, "R2-42");
            }
        } catch (error) {
            TB.logger.error('[Signup Page] Signup submission error:', error);
            showInfo(error.message || "An unexpected error occurred.", true, "P2-52:Y2-52");
        }
    }

    if (signupForm) {
        signupForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            await handleSignup(true);
        });
    }
    if (skipPersonaButton) {
        skipPersonaButton.addEventListener('click', async (event) => {
            event.preventDefault();
            await handleSignup(false);
        });
    }
}

if (window.TB?.user?.init) { // Simpler check for TB readiness
    setupSignup();
} else {
    window.addEventListener('tbjs:initialized', setupSignup, { once: true });
}
