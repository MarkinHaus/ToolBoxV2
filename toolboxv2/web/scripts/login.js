// /web/scripts/login.js - V2 WebAuthn Only

function setupLogin() {
    window.TB.graphics?.playAnimationSequence("Z0+12");
    setTimeout(async () => {
        await setupLogin_();
    }, 100);
}

async function setupLogin_() {
    const loginForm = document.getElementById('loginForm');
    const usernameInput = document.getElementById('username');
    const infoPopup = document.getElementById('infoPopup');
    const infoText = document.getElementById('infoText');

    let next_url = "/web/mainContent.html";
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('next')) {
        next_url = urlParams.get('next');
    }

    function showInfo(message, isError = null, animationSequence = null) {
        if (infoPopup && infoText) {
            infoText.textContent = message;
            infoPopup.style.display = 'block';
        }

        if (isError) {
            window.TB.ui.Toast.showError(message);
            window.TB.graphics?.playAnimationSequence(animationSequence || "R0-31");
        } else if (isError === null) {
            window.TB.ui.Toast.showInfo(message);
            window.TB.graphics?.playAnimationSequence(animationSequence || "R0+31");
        } else {
            window.TB.ui.Toast.showSuccess(message);
            window.TB.graphics?.playAnimationSequence(animationSequence || "P0+21");
        }
    }

    if (loginForm) {
        loginForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            const username = usernameInput.value.trim();
            if (!username) {
                showInfo("Username is required.", true, "Y0-22");
                return;
            }

            window.TB.graphics?.playAnimationSequence("R1+11:P1-11");
            window.TB.ui.Loader.show('Logging in with WebAuthn...');

            try {
                showInfo("Authenticating with passkey...", null);
                const result = await window.TB.user.login(username);

                if (result.success) {
                    showInfo("Login successful! Redirecting...", false, "Z1+32:R0+50");
                    setTimeout(() => {
                        window.TB.router.navigateTo(next_url);
                        window.TB.graphics?.stopAnimationSequence();
                    }, 800);
                } else {
                    showInfo(result.message || "Login failed. Please try again.", true, "P2-42");
                    window.TB.graphics?.stopAnimationSequence();
                }
            } catch (error) {
                window.TB.logger?.error('[Login Page] Error:', error);
                showInfo(error.message || "An unexpected error occurred during login.", true, "R2-52:P2-52");
                window.TB.graphics?.stopAnimationSequence();
            } finally {
                window.TB.ui.Loader.hide();
            }
        });
    } else {
        window.TB.logger?.warn('[Login Page] Login form not found.');
    }
}

// Wait for tbjs to be initialized
if (window.TB?.events) {
    if (window.TB.config?.get('appRootId')) {
        setupLogin();
    } else {
        window.TB.events.on('tbjs:initialized', setupLogin, { once: true });
    }
} else {
    document.addEventListener('tbjs:initialized', setupLogin, { once: true });
}

