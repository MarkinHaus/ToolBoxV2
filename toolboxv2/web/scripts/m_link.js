// /web/scripts/m_link.js (Refactored with Animation)

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
        }
        if (isError) {
            TB.ui.Toast.show(message, 'error', 5000);
            if (TB.graphics?.playAnimationSequence) TB.graphics.playAnimationSequence(animationSequence || "R0-31");
        } else {
            TB.ui.Toast.show(message, 'success', 3000);
            if (TB.graphics?.playAnimationSequence) TB.graphics.playAnimationSequence(animationSequence || "P0+21");
        }
    }

    function showErrorPage(message, animationSequence = "Y2-52:P2-52") {
        if (mainContent) mainContent.classList.add('none');
        if (errorContent) errorContent.classList.remove('none');
        if (errorInfoText) errorInfoText.textContent = message;
        TB.ui.Toast.show(message, 'error', 0);
        if (TB.graphics?.playAnimationSequence) TB.graphics.playAnimationSequence(animationSequence);
    }

    function getKeyFromURL() {
        const urlParams = new URLSearchParams(window.location.search);
        const key = urlParams.get('key');
        const nl = urlParams.get('nl'); // name length?
        const name = urlParams.get('name');
        return { key, nl, name };
    }

    async function processMagicLink() {
        const { key: invitationKey, name: usernameFromUrl } = getKeyFromURL();
        let username = usernameFromUrl;

        if (!invitationKey) {
            showErrorPage("No invitation key found in URL.", "R1-42");
            return;
        }
        if (!username && usernameInput && usernameInput.offsetParent !== null) {
            showGeneralInfo("Please enter your username.", false, "Y0+10"); // Gentle prompt
            // Add listener to usernameInput to call processMagicLink again or a specific handler
            usernameInput.addEventListener('change', async (e) => { // Or 'blur' or a button click
                username = e.target.value.trim();
                if (username) await processMagicLink(); // Re-trigger with username
            }, {once: true});
            return;
        }
        if (!username) {
            showErrorPage("Username not provided.", "P1-32");
            return;
        }
        if (usernameInput) usernameInput.classList.add('none');

        showGeneralInfo(`Processing magic link for ${username}...`, false, "R1+11:P1-11:Y1+11");
        if (TB.graphics?.playAnimationSequence) TB.graphics.playAnimationSequence("R1+11:P1-11:Y1+11");


        try {
            const result = await TB.user.registerDeviceWithInvitation(username, invitationKey);
            if (result.success) {
                showGeneralInfo(result.message || `Device registered for ${username}!`, false, "Z2+52:R0+50");
                setTimeout(() => TB.router.navigateTo('/web/dashboard'), 1000);
            } else {
                showErrorPage(result.message || "Failed to process magic link.", "R2-42:P1-31");
            }
        } catch (error) {
            TB.logger.error('[MagicLink] Error processing magic link:', error);
            showErrorPage(error.message || "An unexpected error occurred.");
        }
    }

    processMagicLink();
}

if (window.TB?.user?.init) {
    setupMagicLinkLogin();
} else {
    window.addEventListener('tbjs:initialized', setupMagicLinkLogin, { once: true });
}
