// /web/scripts/login.js (Refactored with tbjs Framework)

function setupLogin() {
    const loginForm = document.getElementById('loginForm');
    const usernameInput = document.getElementById('username');
    const registerDeviceCheckbox = document.getElementById('register-device');
    const infoPopup = document.getElementById('infoPopup'); // Assuming this is a local UI element for direct messages
    const infoText = document.getElementById('infoText');   // Assuming this is a local UI element

    let next_url = "/web/dashboard"; // Default to dashboard or a main content page
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('next')) {
        next_url = urlParams.get('next');
    }

    function showInfo(message, isError = false, animationSequence = null) {
        if (infoPopup && infoText) { // For local popups
            infoText.textContent = message;
            infoPopup.style.display = 'block';
        }

        if (isError) {
            window.TB.ui.Toast.showError(message);
            if (window.TB.graphics?.playAnimationSequence) {
                window.TB.graphics.playAnimationSequence(animationSequence || "R0-31"); // Default error animation
            }
        } else {
            window.TB.ui.Toast.showSuccess(message);
            if (window.TB.graphics?.playAnimationSequence) {
                window.TB.graphics.playAnimationSequence(animationSequence || "P0+21"); // Default success animation
            }
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

            // Determine login type based on UI state (e.g., checkbox visibility/status)
            // This example assumes 'none' class means the WebAuthn option is not primary.
            // Adjust this logic based on your actual UI for choosing login method.
            const wantsWebAuthn = registerDeviceCheckbox ? !registerDeviceCheckbox.classList.contains('none') : false;
            let result;

            if (window.TB.graphics?.playAnimationSequence) {
                window.TB.graphics.playAnimationSequence("R1+11:P1-11"); // Gentle alternating rotation
            }
            window.TB.ui.Loader.show('Attempting login...');


            try {
                if (wantsWebAuthn) {
                    // For WebAuthn, it's usually better to prompt user if multiple passkeys exist,
                    // or if no passkey is found for the username, guide them.
                    // The TB.user.loginWithWebAuthn(username) itself handles getting the challenge.
                    showInfo("Attempting WebAuthn login..."); // Local info
                    result = await window.TB.user.loginWithWebAuthn(username);
                } else {
                    showInfo("Attempting device key login..."); // Local info
                    result = await window.TB.user.loginWithDeviceKey(username);
                    if (!result.success && result.message && result.message.includes("No device key found")) {
                        // Example: Offer magic link if device key not found
                        // This could be a modal or a different UI flow
                        showInfo(result.message + " Consider registering this device or requesting a magic link.", true, "Y1-32");
                        window.TB.ui.Loader.hide();
                        if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
                        return;
                    }
                }

                if (result.success) {
                    showInfo(result.message || "Login successful!", false, "Z1+32:R0+50"); // Zoom in, fast spin success
                    setTimeout(() => {
                        window.TB.router.navigateTo(next_url);
                        if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
                    }, 800); // Give animation and toast time
                } else {
                    showInfo(result.message || "Login failed.", true, "P2-42"); // Sharp pan for failure
                    if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
                }
            } catch (error) {
                window.TB.logger.error('[Login Page] Login submission error:', error);
                showInfo(error.message || "An unexpected error occurred.", true, "R2-52:P2-52"); // Tumbling error
                if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
            } finally {
                 window.TB.ui.Loader.hide();
                // Stop any generic 'working' animation if not handled by success/error specific sequences
                // This might be redundant if success/error sequences always conclude or are short.
            }
        });
    } else {
        if (window.TB && window.TB.logger) {
            window.TB.logger.warn('[Login Page] Login form not found.');
        } else {
            console.warn('[Login Page] Login form not found, TB.logger not available.');
        }
    }
}

// Wait for tbjs to be initialized
if (window.TB?.events) {
    if (window.TB.config?.get('appRootId')) { // A sign that TB.init might have run
         setupLogin();
    } else {
        window.TB.events.on('tbjs:initialized', setupLogin, { once: true });
    }
} else {
    // Fallback if TB is not even an object yet, very early load
    document.addEventListener('tbjs:initialized', setupLogin, { once: true }); // Custom event dispatch from TB.init
}
