// /web/scripts/m_link.js - V2 Magic Link for Device Registration

function setupMagicLinkLogin() {
    const usernameInput = document.getElementById('username');
    const infoPopup = document.getElementById('infoPopup');
    const infoText = document.getElementById('infoText');
    const mainContent = document.getElementById('Main-content');
    const errorContent = document.getElementById('error-content');
    const errorInfoText = document.getElementById('EinfoText');

    function showGeneralInfo(message, isError = false, animationSequence = null) {
        if (infoPopup && infoText) {
            infoText.textContent = message;
            infoPopup.style.display = 'block';
            if (isError) infoPopup.classList.add('error');
            else infoPopup.classList.remove('error');
        }

        if (isError) {
            window.TB.ui.Toast.showError(message);
            window.TB.graphics?.playAnimationSequence(animationSequence || "R0-31");
        } else {
            window.TB.ui.Toast.showSuccess(message);
            window.TB.graphics?.playAnimationSequence(animationSequence || "P0+21");
        }
    }

    function showErrorPage(message, animationSequence = "Y2-52:P2-52") {
        if (mainContent) mainContent.classList.add('none');
        if (errorContent) errorContent.classList.remove('none');
        if (errorInfoText) errorInfoText.textContent = message;

        window.TB.ui.Toast.showError(message, { duration: 0 });
        window.TB.graphics?.playAnimationSequence(animationSequence);
    }

    function getTokenFromURL() {
        const urlParams = new URLSearchParams(window.location.search);
        const token = urlParams.get('token');
        const name = urlParams.get('name');
        return { token, name };
    }

    async function consumeMagicLink(token, username) {
        showGeneralInfo(`Logging in ${username}...`, false, "R1+11:P1-11");
        window.TB.ui.Loader.show('Authenticating with magic link...');

        try {
            window.TB.logger?.info('[Magic Link] Consuming magic link token');

            // Call backend to consume magic link and get tokens
            const result = await window.TB.api.request(
                'CloudM.AuthManagerV2',
                'consume_magic_link',
                { token }
            );

            if (!result.success) {
                throw new Error(result.message || 'Magic link authentication failed');
            }

            const data = result.data;

            // Store tokens in userV2 state
            userV2._updateState({
                isAuthenticated: true,
                user: data.user,
                accessToken: data.access_token,
                refreshToken: data.refresh_token,
                tokenExpiry: Date.now() + (15 * 60 * 1000) // 15 min
            });

            // Schedule token refresh
            userV2._scheduleTokenRefresh();

            showGeneralInfo("Login successful! Redirecting...", false, "Z1+32:R0+50");

            // Redirect to dashboard
            setTimeout(() => {
                window.TB.router.navigateTo('/web/mainContent.html');
                window.TB.graphics?.stopAnimationSequence();
            }, 800);

        } catch (error) {
            window.TB.logger?.error('[Magic Link] Error:', error);
            showErrorPage(error.message || "Magic link authentication failed.", "R2-52:P2-52");
        } finally {
            window.TB.ui.Loader.hide();
        }
    }

    async function processMagicLink() {
        const { token, name: usernameFromUrl } = getTokenFromURL();

        if (!token) {
            showErrorPage("No magic link token found in URL. Please check the link.", "R1-42");
            return;
        }

        if (!usernameFromUrl) {
            showErrorPage("Username not provided with the magic link.", "P1-32");
            return;
        }

        // Hide username input if present
        if (usernameInput) usernameInput.classList.add('none');

        // Consume magic link and auto-login
        await consumeMagicLink(token, usernameFromUrl);
    }

    // Start processing
    processMagicLink();
}

// Wait for tbjs to be initialized
if (window.TB?.events) {
    if (window.TB.config?.get('appRootId')) {
        setupMagicLinkLogin();
    } else {
        window.TB.events.on('tbjs:initialized', setupMagicLinkLogin, { once: true });
    }
} else {
    document.addEventListener('tbjs:initialized', setupMagicLinkLogin, { once: true });
}

