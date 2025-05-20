// /web/scripts/signup.js (Refactored with tbjs Framework)

function setupSignup() {
    const signupForm = document.getElementById('signupForm');
    const usernameInput = document.getElementById('username');
    const emailInput = document.getElementById('email');
    const initiationKeyInput = document.getElementById('initiation'); // Assuming this is the ID
    const skipPersonaButton = document.getElementById('skip-persona-button'); // Button to signup without WebAuthn initially
    const infoPopup = document.getElementById('infoPopup'); // Local UI
    const infoText = document.getElementById('infoText');   // Local UI

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

    async function handleSignup(registerAsPersona) {
        const username = usernameInput.value.trim();
        const email = emailInput.value.trim();
        const initiationKey = initiationKeyInput ? initiationKeyInput.value.trim() : ''; // Handle if key is optional

        if (!username || !email) {
            showInfo("Username and Email are required.", true, "Y0-22");
            return;
        }
        // Add more validation as needed (e.g., email format, password strength if applicable)

        showInfo(`Attempting to sign up ${username}...`, false, "Y1+11:R1-11");
        window.TB.ui.Loader.show('Processing signup...');


        try {
            // The `initiationKey` might only be relevant for certain signup flows (e.g., invite-only)
            // `registerAsPersona` determines if WebAuthn should be part of the initial signup.
            const result = await window.TB.user.signup(username, email, initiationKey, registerAsPersona);

            if (result.success) {
                let successMessage = result.message || "Signup successful!";
                let successAnimation = "Z1+32:Y0+50"; // Default success animation

                // Example flow: if signup response indicates further action like WebAuthn registration
                if (registerAsPersona && result.data?.needsWebAuthnRegistration) {
                    successMessage = "Account created! Now, let's secure it with a passkey (WebAuthn).";
                    successAnimation = "P1+21:Y1+21";
                    showInfo(successMessage, false, successAnimation);
                    // This would ideally trigger the WebAuthn registration flow
                    // For now, we might redirect to a page or show instructions.
                    // e.g., await window.TB.user.registerWebAuthnForCurrentUser(username);
                    // For simplicity, let's assume the user module or a subsequent page handles this.
                    // Or, we can navigate to a specific "setup passkey" page.
                    setTimeout(() => window.TB.router.navigateTo('/web/setup-passkey.html?username=' + encodeURIComponent(username)), 1200);

                } else if (result.data?.token) { // Signup resulted in immediate login
                    showInfo(successMessage, false, successAnimation);
                    setTimeout(() => {
                        window.TB.router.navigateTo('/web/dashboard');
                        if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
                    }, 800);
                } else { // Signup successful, but requires separate login
                    showInfo("Signup successful! Please log in.", false, "P0+31");
                    setTimeout(() => {
                        window.TB.router.navigateTo('/web/assets/login.html');
                         if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
                    }, 800);
                }
            } else {
                showInfo(result.message || "Signup failed. Please check your details and try again.", true, "R2-42");
                 if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
            }
        } catch (error) {
            window.TB.logger.error('[Signup Page] Signup submission error:', error);
            showInfo(error.message || "An unexpected error occurred during signup.", true, "P2-52:Y2-52");
            if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
        } finally {
            window.TB.ui.Loader.hide();
        }
    }

    if (signupForm) {
        signupForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            // Default submit action might imply registering with Persona (WebAuthn) if UI supports it
            await handleSignup(true); // Assuming default signup tries to register as Persona
        });
    } else {
         if (window.TB && window.TB.logger) {
            window.TB.logger.warn('[Signup Page] Signup form not found.');
        } else {
            console.warn('[Signup Page] Signup form not found, TB.logger not available.');
        }
    }

    if (skipPersonaButton) {
        skipPersonaButton.addEventListener('click', async (event) => {
            event.preventDefault();
            // This button explicitly skips the initial WebAuthn registration part
            await handleSignup(false);
        });
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
