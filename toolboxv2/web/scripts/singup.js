// /web/scripts/signup.js - V2 WebAuthn Only

import userV2 from '../../tbjs/src/core/userV2.js';

function setupSignup() {
    const signupForm = document.getElementById('signupForm');
    const usernameInput = document.getElementById('username');
    const emailInput = document.getElementById('email');
    const inviteCodeInput = document.getElementById('inviteCode');
    const deviceLabelInput = document.getElementById('deviceLabel');
    const infoPopup = document.getElementById('infoPopup');
    const infoText = document.getElementById('infoText');

    // Parse URL and prefill fields
    const urlParams = new URLSearchParams(window.location.search);
    const defaultEmail = urlParams.get('email');
    const defaultUsername = urlParams.get('username');
    const defaultInviteCode = urlParams.get('invite');

    if (defaultEmail && emailInput) emailInput.value = decodeURIComponent(defaultEmail);
    if (defaultUsername && usernameInput) usernameInput.value = decodeURIComponent(defaultUsername);
    if (defaultInviteCode && inviteCodeInput) inviteCodeInput.value = decodeURIComponent(defaultInviteCode);

    // Set default device label
    if (deviceLabelInput && !deviceLabelInput.value) {
        deviceLabelInput.value = `${navigator.platform} - ${new Date().toLocaleDateString()}`;
    }

    function showInfo(message, isError = false, animationSequence = null) {
        if (infoPopup && infoText) {
            infoText.textContent = message;
            infoPopup.style.display = 'block';
            if (isError) infoPopup.classList.add('error'); else infoPopup.classList.remove('error');
        }

        if (isError) {
            window.TB.ui.Toast.showError(message);
            if (window.TB.graphics?.playAnimationSequence) {
                window.TB.graphics.playAnimationSequence(animationSequence || "R0-31");
            }
        } else {
            window.TB.ui.Toast.showSuccess(message);
            if (window.TB.graphics?.playAnimationSequence) {
                window.TB.graphics.playAnimationSequence(animationSequence || "P0+21");
            }
        }
    }

    async function handleSignup() {
        const username = usernameInput.value.trim();
        const email = emailInput.value.trim();
        const inviteCode = inviteCodeInput ? inviteCodeInput.value.trim() : null;
        const deviceLabel = deviceLabelInput ? deviceLabelInput.value.trim() : 'My Device';

        if (!username || !email) {
            showInfo("Username and Email are required.", true, "Y0-22");
            return;
        }

        showInfo(`Creating account for ${username}...`, false, "Y1+11:R1-11");
        window.TB.ui.Loader.show('Registering with WebAuthn...');

        try {
            // V2 signup - WebAuthn only
            const result = await userV2.signup(username, email, inviteCode, deviceLabel);

            if (result.success) {
                showInfo("Signup successful! Redirecting to dashboard...", false, "Z1+32:Y0+50");
                setTimeout(() => {
                    window.TB.router.navigateTo('/web/dashboard');
                    if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
                }, 800);
            } else {
                showInfo(result.message || "Signup failed. Please try again.", true, "R2-42");
                if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
            }
        } catch (error) {
            window.TB.logger?.error('[Signup Page] Error:', error);
            showInfo(error.message || "An unexpected error occurred during signup.", true, "P2-52:Y2-52");
            if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
        } finally {
            window.TB.ui.Loader.hide();
        }
    }

    if (signupForm) {
        signupForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            await handleSignup();
        });
    } else {
        window.TB.logger?.warn('[Signup Page] Signup form not found.');
    }
}

// Wait for tbjs to be initialized
if (window.TB?.events) {
     if (window.TB.config?.get('appRootId')) {
         setupSignup();
    } else {
        window.TB.events.on('tbjs:initialized', setupSignup, { once: true });
    }
} else {
    document.addEventListener('tbjs:initialized', setupSignup, { once: true });
}
