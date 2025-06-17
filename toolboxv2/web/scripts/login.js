// /web/scripts/login.js (Refactored with tbjs Framework)

function setupLogin(){
    window.TB.graphics.playAnimationSequence("Z0+12")
    setTimeout(async () => {
        await setupLogin_()
    }, 100);
}
async function setupLogin_() {
    const loginForm = document.getElementById('loginForm');
    const usernameInput = document.getElementById('username');
    const registerDeviceCheckbox = document.getElementById('register-device');
    const infoPopup = document.getElementById('infoPopup'); // Assuming this is a local UI element for direct messages
    const infoText = document.getElementById('infoText');   // Assuming this is a local UI element

    let next_url = "/web/mainContent.html";
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('next')) {
        next_url = urlParams.get('next');
    }

    function showInfo(message, isError = null, animationSequence = null) {
        if (infoPopup && infoText) { // For local popups
            infoText.textContent = message;
            infoPopup.style.display = 'block';
        }

        if (isError) {
            window.TB.ui.Toast.showError(message);
            if (window.TB.graphics?.playAnimationSequence) {
                window.TB.graphics.playAnimationSequence(animationSequence || "R0-31"); // Default error animation
            }
        } else if (isError === null) {
            window.TB.ui.Toast.showInfo(message);
            if (window.TB.graphics?.playAnimationSequence) {
                window.TB.graphics.playAnimationSequence(animationSequence || "R0+31"); // Default error animation
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

            const wantsWebAuthn = registerDeviceCheckbox ? registerDeviceCheckbox.checked : false;
            let result;

            if (window.TB.graphics?.playAnimationSequence) {
                window.TB.graphics.playAnimationSequence("R1+11:P1-11"); // Gentle alternating rotation
            }
            window.TB.ui.Loader.show('Attempting login...');

            // Capture username and next_url for potential use in async toast actions
            const usernameForActions = username;
            const capturedNextUrl = next_url;

            try {
                if (wantsWebAuthn) {
                    showInfo("Attempting WebAuthn login..."); // Local info
                    result = await window.TB.user.loginWithWebAuthn(username);
                    console.log("[WebAuthn result]:", result)
                } else {
                    showInfo("Attempting device key login..."); // Local info
                    result = await window.TB.user.loginWithDeviceKey(username);

                    if (!result.success && result.message) {
                        window.TB.ui.Loader.hide(); // Hide loader first
                        if (window.TB.graphics?.stopAnimationSequence) { // Stop the general loading animation
                            window.TB.graphics.stopAnimationSequence();
                        }

                        // Play the specific animation for this error condition ("Y1-32")
                        if (window.TB.graphics?.playAnimationSequence) {
                            window.TB.graphics.playAnimationSequence("Y1-32");
                        }

                        window.TB.ui.Toast.showError(result.message + " Please choose an alternative:", {
                            title: "Device Key Not Found",
                            duration: 0, // Sticky toast
                            closable: true,
                            actions: [
                                {
                                    text: "Try Passkey/WebAuthn",
                                    action: async () => {
                                        window.TB.ui.Loader.show('Attempting WebAuthn login...');
                                        if (window.TB.graphics?.playAnimationSequence) {
                                            window.TB.graphics.playAnimationSequence("R1+11:P1-11"); // Standard processing animation
                                        }
                                        try {
                                            const webAuthnResult = await window.TB.user.loginWithWebAuthn(usernameForActions);
                                            if (webAuthnResult.success) {
                                                window.TB.ui.Toast.showSuccess(webAuthnResult.message || "WebAuthn login successful! Redirecting...", { duration: 3000 });
                                                if (window.TB.graphics?.playAnimationSequence) {
                                                    window.TB.graphics.playAnimationSequence("Z1+32:R0+50"); // Success animation
                                                }
                                                setTimeout(() => {
                                                    window.TB.router.navigateTo(capturedNextUrl);
                                                    if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
                                                }, 800);
                                            } else {
                                                window.TB.ui.Toast.showError(webAuthnResult.message || "WebAuthn login failed.");
                                                if (window.TB.graphics?.playAnimationSequence) {
                                                    window.TB.graphics.playAnimationSequence("P2-42"); // Failure animation
                                                }
                                            }
                                        } catch (e) {
                                            window.TB.logger.error('[Login Page] WebAuthn action error:', e);
                                            window.TB.ui.Toast.showError(e.message || "An unexpected error occurred during WebAuthn login.");
                                            if (window.TB.graphics?.playAnimationSequence) {
                                                 window.TB.graphics.playAnimationSequence("R2-52:P2-52"); // Tumbling error
                                            }
                                        } finally {
                                            window.TB.ui.Loader.hide();
                                            if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
                                        }
                                    }
                                },
                                {
                                    text: "Register with Invitation",
                                    action: async () => {
                                        const invitationKey = prompt("If you have an invitation key to register this device, please enter it:");
                                        if (invitationKey && invitationKey.trim() !== "") {
                                            window.TB.ui.Loader.show('Registering device...');
                                            if (window.TB.graphics?.playAnimationSequence) {
                                               window.TB.graphics.playAnimationSequence("R0+11"); // Generic processing
                                            }
                                            try {
                                                const regResult = await window.TB.user.registerDeviceWithInvitation(usernameForActions, invitationKey.trim());
                                                if (regResult.success) {
                                                    window.TB.ui.Toast.showSuccess(regResult.message || "Device registered and logged in! Redirecting...", { duration: 3000 });
                                                    if (window.TB.graphics?.playAnimationSequence) {
                                                        window.TB.graphics.playAnimationSequence("Z1+32:R0+50");
                                                    }
                                                    setTimeout(() => {
                                                        window.TB.router.navigateTo(capturedNextUrl);
                                                        if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
                                                    }, 800);
                                                } else {
                                                    window.TB.ui.Toast.showError(regResult.message || "Device registration failed.");
                                                     if (window.TB.graphics?.playAnimationSequence) {
                                                        window.TB.graphics.playAnimationSequence("P1-22"); // Another failure indication
                                                    }
                                                }
                                            } catch (e) {
                                                window.TB.logger.error('[Login Page] Device registration action error:', e);
                                                window.TB.ui.Toast.showError(e.message || "An unexpected error occurred during device registration.");
                                                if (window.TB.graphics?.playAnimationSequence) {
                                                    window.TB.graphics.playAnimationSequence("R2-52:P2-52");
                                                }
                                            } finally {
                                                window.TB.ui.Loader.hide();
                                                if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
                                            }
                                        } else if (invitationKey !== null) {
                                            window.TB.ui.Toast.showWarning("Invitation key cannot be empty.", { duration: 3000 });
                                        } else {
                                            window.TB.ui.Toast.showInfo("Device registration cancelled.", { duration: 2000 });
                                        }
                                    }
                                },
                                {
                                    text: "Send Magic Link Email",
                                    action: async () => {
                                        window.TB.ui.Loader.show('Requesting magic link...');
                                         if (window.TB.graphics?.playAnimationSequence) {
                                            window.TB.graphics.playAnimationSequence("R0+11"); // Generic processing
                                        }
                                        try {
                                            const magicLinkRes = await window.TB.user.requestMagicLink(usernameForActions);
                                            if (magicLinkRes.success) {
                                                window.TB.ui.Toast.showSuccess(magicLinkRes.message || "Magic link email requested. Please check your inbox.", { duration: 5000 });
                                            } else {
                                                window.TB.ui.Toast.showError(magicLinkRes.message || "Failed to request magic link.");
                                            }
                                        } catch (e) {
                                             window.TB.logger.error('[Login Page] Magic link request action error:', e);
                                             window.TB.ui.Toast.showError(e.message || "An unexpected error occurred while requesting magic link.");
                                             if (window.TB.graphics?.playAnimationSequence) {
                                                 window.TB.graphics.playAnimationSequence("R2-52:P2-52");
                                             }
                                        } finally {
                                            window.TB.ui.Loader.hide();
                                            if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
                                        }
                                    }
                                }
                            ]
                        });
                        return; // Important: Stop further execution in this submit handler
                    }
                }

                // General success/failure handling for the initial login attempt
                if (result.success) {
                    showInfo(result.message || "Login successful!", false, "Z1+32:R0+50"); // Zoom in, fast spin success
                    setTimeout(() => {
                        window.TB.router.navigateTo(capturedNextUrl);
                        if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
                    }, 800);
                } else {
                    // This 'else' handles failures OTHER than 'No device key found' if it was a device key attempt,
                    // or any failure if it was a WebAuthn attempt.
                    showInfo(result.message || "Login failed.", true, "P2-42"); // Sharp pan for failure
                    if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
                }
            } catch (error) {
                window.TB.logger.error('[Login Page] Login submission error:', error);
                showInfo(error.message || "An unexpected error occurred.", true, "R2-52:P2-52"); // Tumbling error
                if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
            } finally {
                 window.TB.ui.Loader.hide();
                // Ensure any persistent login attempt animation is stopped if not handled by specific branches.
                // This might be redundant if all paths call stopAnimationSequence, but can be a safeguard.
                // if (window.TB.graphics?.stopAnimationSequence) window.TB.graphics.stopAnimationSequence();
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
TB.once(setupLogin);

