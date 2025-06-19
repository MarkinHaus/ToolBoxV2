// /web/scripts/m_link.js (Refactored with tbjs Framework)

function setupMagicLinkLogin() {
    const usernameInput = document.getElementById('username');
    const infoPopup = document.getElementById('infoPopup'); // Local UI
    const infoText = document.getElementById('infoText');   // Local UI
    const mainContent = document.getElementById('Main-content'); // Local UI
    const errorContent = document.getElementById('error-content'); // Local UI
    const errorInfoText = document.getElementById('EinfoText');   // Local UI

    function showGeneralInfo(message, isError = false, animationSequence = null) {
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

    function showErrorPage(message, animationSequence = "Y2-52:P2-52") {
        if (mainContent) mainContent.classList.add('none'); // Assuming 'none' is a utility class like 'd-none' or Tailwind 'hidden'
        if (errorContent) errorContent.classList.remove('none');
        if (errorInfoText) errorInfoText.textContent = message;

        window.TB.ui.Toast.showError(message, { duration: 0 }); // Sticky error
        if (window.TB.graphics?.playAnimationSequence) {
            window.TB.graphics.playAnimationSequence(animationSequence);
        }
    }

    function getKeyFromURL() {
        const urlParams = new URLSearchParams(window.location.search);
        const key = urlParams.get('key');
        // const nl = urlParams.get('nl'); // name length? Not used in TB.user.registerDeviceWithInvitation
        const name = urlParams.get('name');
        return { key, name };
    }

    async function processMagicLink() {
        const { key: invitationKey, name: usernameFromUrl } = getKeyFromURL();
        let username = usernameFromUrl;

        if (!invitationKey) {
            showErrorPage("No invitation key found in URL. Please check the link.", "R1-42");
            return;
        }

        // Check if username is provided in URL or if input field is visible and empty
        if (!username && usernameInput && usernameInput.offsetParent !== null) { // offsetParent check for visibility
            showGeneralInfo("Please enter your username to complete device registration.", false, "Y0+10");
            usernameInput.focus();
            usernameInput.addEventListener('change', async (e) => {
                const enteredUsername = e.target.value.trim();
                if (enteredUsername) {
                    // Hide input and proceed
                    usernameInput.classList.add('none'); // Or disable
                    username = enteredUsername;
                    await processMagicLinkWithUsername(username, invitationKey);
                }
            }, { once: true });
            return; // Wait for user input
        }

        if (!username) {
            showErrorPage("Username not provided with the magic link.", "P1-32");
            return;
        }

        // If username was from URL, and input field exists, maybe hide it
        if (usernameFromUrl && usernameInput) {
            usernameInput.value = usernameFromUrl; // Pre-fill if needed
            usernameInput.classList.add('none'); // Or disable
        }
        await processMagicLinkWithUsername(username, invitationKey);
    }

    async function processMagicLinkWithUsername(username, invitationKey) {
        showGeneralInfo(`Processing magic link for ${username}... This may take a moment.`, false, "R1+11:P1-11:Y1+11");
        window.TB.ui.Loader.show('Registering your device...');


        try {
            const result = await window.TB.user.registerDeviceWithInvitation(username, invitationKey);
            if (result.success) {
                showGeneralInfo(result.message || `Device registered for ${username}! Logging you in...`, false, "Z2+52:R0+50");
                setTimeout(() => {
                    let next_url = "/web/mainContent.html";
                    const urlParams = new URLSearchParams(window.location.search);
                    if (urlParams.get('next')) {
                        next_url = urlParams.get('next');
                    }
                    window.TB.router.navigateTo(next_url); // Or a more appropriate post-registration page
                    if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
                }, 1200);
            } else {
                showErrorPage(result.message || "Failed to process magic link. The link may be invalid or expired.", "R2-42:P1-31");
            }
        } catch (error) {
            window.TB.logger.error('[MagicLink] Error processing magic link:', error);
            showErrorPage(error.message || "An unexpected error occurred during device registration.");
        } finally {
            window.TB.ui.Loader.hide();
            // Stop animation if not handled by success/error specific sequences
        }
    }

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
