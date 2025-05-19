// /web/scripts/login.js (Refactored with Animation)

function setupLogin() {
    const loginForm = document.getElementById('loginForm');
    const usernameInput = document.getElementById('username');
    const registerDeviceCheckbox = document.getElementById('register-device');
    const infoPopup = document.getElementById('infoPopup');
    const infoText = document.getElementById('infoText');

    let next_url = "/web/mainContent.html";
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('next')) next_url = urlParams.get('next');

    function showInfo(message, isError = false, animationSequence = null) {
        infoText.textContent = message;
        infoPopup.style.display = 'block';
        if (isError) {
            TB.ui.Toast.show(message, 'error', 5000);
            if (TB.graphics && TB.graphics.playAnimationSequence) TB.graphics.playAnimationSequence(animationSequence || "R0-31"); // Default error animation
        } else {
            TB.ui.Toast.show(message, 'success', 3000);
            if (TB.graphics && TB.graphics.playAnimationSequence) TB.graphics.playAnimationSequence(animationSequence || "P0+21"); // Default success animation
        }
    }

    if (loginForm) {
        loginForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            const username = usernameInput.value.trim();
            if (!username) {
                showInfo("Username is required.", true, "Y0-22"); // Yaw shake for error
                return;
            }

            const wantsWebAuthn = !registerDeviceCheckbox.classList.contains('none');
            let result;

            // Play a 'working' animation
            if (TB.graphics && TB.graphics.playAnimationSequence) TB.graphics.playAnimationSequence("R1+11:P1-11"); // Gentle alternating rotation

            try {
                if (wantsWebAuthn) {
                    showInfo("Attempting WebAuthn login...");
                    result = await TB.user.loginWithWebAuthn(username);
                } else {
                    showInfo("Attempting device key login...");
                    result = await TB.user.loginWithDeviceKey(username);
                    if (!result.success && result.message && result.message.includes("No device key found")) {
                        showInfo(result.message + " Would you like to request a magic link?", true, "Y1-32");
                        return;
                    }
                }

                if (result.success) {
                    showInfo(result.message || "Login successful!", false, "Z1+32:R0+50"); // Zoom in, fast spin success
                    setTimeout(() => TB.router.navigateTo(next_url), 800); // Give animation time
                } else {
                    showInfo(result.message || "Login failed.", true, "P2-42"); // Sharp pan for failure
                }
            } catch (error) {
                TB.logger.error('[Login Page] Login submission error:', error);
                showInfo(error.message || "An unexpected error occurred.", true, "R2-52:P2-52"); // Tumbling error
            } finally {

                if (TB.graphics && TB.graphics.stopAnimationSequence) TB.graphics.stopAnimationSequence();
            }
        });
    } else {
        TB.logger.warn('[Login Page] Login form not found.');
    }
}

if (window.TB && window.TB.user && window.TB.user.init) {
    setupLogin();
} else {
    window.addEventListener('tbjs:initialized', setupLogin, { once: true });
}
