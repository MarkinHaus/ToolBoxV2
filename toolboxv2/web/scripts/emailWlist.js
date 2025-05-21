// Assuming TB is globally available or this script is bundled where TB can be imported/accessed.
// For <script type="module">, if TB is set on window by another module, it will be accessible.
// import TB from '/tbjs/index.js'; // This might be needed if not global and using bundler

function initEmailWaitingList() {
    if (!window.TB || !TB.api || !TB.ui || !TB.ui.Toast || !TB.logger || !TB.graphics) {
        console.error('TB module or required submodules (api, ui.Toast, logger, graphics) not available. Waiting list form will not function correctly.');
        // Optionally, disable the form
        const form = document.getElementById('email-subscription-form');
        if (form) {
            form.querySelector('button[type="submit"]').disabled = true;
            form.insertAdjacentHTML('afterend', '<p style="color: var(--color-error);">Registration system is temporarily unavailable.</p>');
        }
        return;
    }

    const emailForm = document.getElementById('email-subscription-form');

    if (emailForm) {
        emailForm.addEventListener('submit', async function(event) {
            event.preventDefault();

            const emailInput = document.getElementById('email');
            const email = emailInput.value;
            const submitButton = emailForm.querySelector('button[type="submit"]');

            if (!email) {
                TB.ui.Toast.showWarning('Please enter your email address.', { title: 'Input Required' });
                emailInput.focus();
                return;
            }

            // Disable button during submission
            submitButton.disabled = true;
            submitButton.textContent = 'Subscribing...';

            // Call your 3D animation functions via TB.graphics
            // Assuming Set_animation_xyz(speedMultiplier, rotX, rotY, factor)
            // Call 1: Set_animation_xyz(0, 0.6, 0, 16)
            if (TB.graphics && typeof TB.graphics.setAnimationSpeed === 'function') {
                TB.graphics.setAnimationSpeed(0.6, 0, 0, 16); // Mapped: x=0.6, y=0, z=0 (implicit), factor=16
            }


            try {
                // Using TB.api.request for more control if needed, or TB.api.httpPostUrl
                // Payload for httpPostUrl was "email=" + email, which is x-www-form-urlencoded
                // TB.api.request usually sends JSON. If backend expects x-www-form-urlencoded for this specific endpoint,
                // the 'api.js' would need modification or a specific helper.
                // For now, assuming backend can take { email: emailValue } as JSON payload.
                // If it MUST be "email=...", then params for httpPostUrl need to be a string.
                // The original httpPostUrl in api.js doesn't fully handle the string params for Tauri.
                // Let's use `request` and assume backend expects JSON or Tauri handles string payload correctly.
                const responseResult = await TB.api.request(
                    "/email_waiting_list/add", // moduleName
                    "email="+email,                // functionName
                    { },       // payload as object
                    'GET'                // method
                );

                // responseResult is an instance of the Result class from api.js
                if (responseResult && responseResult.info) {
                    if (responseResult.info.exec_code >= 0 && responseResult.error === TB.api.ToolBoxError.none) { // Assuming exec_code >= 0 is success
                        TB.logger.info('[EmailWList] Subscription successful:', responseResult.info.help_text, responseResult);
                        TB.ui.Toast.showSuccess(responseResult.info.help_text || 'Thank you for subscribing!', { title: 'Success' });
                        emailInput.value = ''; // Clear the input
                    } else {
                        TB.logger.error('[EmailWList] Subscription error:', responseResult.info.help_text, responseResult);
                        TB.ui.Toast.showError(responseResult.info.help_text || 'Subscription failed. Please try again.', { title: 'Error' });
                    }
                } else {
                    TB.logger.error('[EmailWList] Invalid response from server:', responseResult);
                    TB.ui.Toast.showError('An unexpected error occurred. Please try again later.', { title: 'Server Error' });
                }

            } catch (error) {
                TB.logger.error('[EmailWList] Network or critical error during subscription:', error);
                TB.ui.Toast.showError('Could not connect to the server. Please check your internet connection and try again.', { title: 'Network Error' });
            } finally {
                // Reset animation and re-enable button regardless of outcome
                 // Call 2: Set_animation_xyz(1, 0.02, 0, 6)
                if (TB.graphics && typeof TB.graphics.setAnimationSpeed === 'function') {
                    // Using setTimeout like original, though could be chained with promises if API was async
                    setTimeout(() => {
                        TB.graphics.setAnimationSpeed(0.02, 0, 0, 6); // Mapped: x=0.02, y=0, z=0, factor=6
                    }, 300);
                }

                // Assuming EndBgInteract means reset to a default slow spin or stop specific sequence
                if (TB.graphics && typeof TB.graphics.stopAnimationSequence === 'function') {
                     setTimeout(() => {
                        TB.graphics.stopAnimationSequence(); // Or set to a default gentle spin
                        // TB.graphics.setAnimationSpeed(0.002, 0.002, 0.002, 21); // Example default
                    }, 300); // Original had 300ms timeout
                }

                submitButton.disabled = false;
                submitButton.textContent = 'Subscribe';
            }
        });
    } else {
        TB.logger.warn('[EmailWList] Email subscription form not found on this page.');
    }
};


// Wait for tbjs to be initialized
if (window.TB?.events) {
    if (window.TB.config?.get('appRootId')) { // A sign that TB.init might have run
         initEmailWaitingList();
    } else {
        window.TB.events.on('tbjs:initialized', initEmailWaitingList, { once: true });
    }
} else {
    // Fallback if TB is not even an object yet, very early load
    document.addEventListener('tbjs:initialized', initEmailWaitingList, { once: true }); // Custom event dispatch from TB.init
}
