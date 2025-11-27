// /web/scripts/singup.js
// ToolBox V2 Signup Script with Clerk Integration

function setupSignup() {
    // Play initial animation if graphics available
    if (window.TB?.graphics?.playAnimationSequence) {
        window.TB.graphics.playAnimationSequence("Z0+12");
    }

    setTimeout(async () => {
        await initializeClerkSignup();
    }, 100);
}

async function initializeClerkSignup() {
    const container = document.getElementById('clerk-sign-up') || document.getElementById('signupForm');

    if (!container) {
        console.warn('[Signup] No signup container found');
        return;
    }

    // Check if TB is loaded
    if (!window.TB) {
        console.error('[Signup] TB framework not loaded');
        showError(container, 'Framework not loaded. Please refresh the page.');
        return;
    }

    try {
        // Wait for TB.user to initialize
        if (!window.TB.user) {
            console.warn('[Signup] TB.user not available yet, waiting...');
            await new Promise(resolve => setTimeout(resolve, 500));
        }

        // Check if user is already authenticated
        if (window.TB.user?.isAuthenticated()) {
            const username = window.TB.user.getUsername();
            showSuccess(container, `Already logged in as ${username}. Redirecting...`);

            setTimeout(() => {
                const urlParams = new URLSearchParams(window.location.search);
                const nextUrl = urlParams.get('next') || '/web/mainContent.html';
                window.TB.router?.navigateTo(nextUrl) || (window.location.href = nextUrl);
            }, 1000);
            return;
        }

        // Initialize Clerk sign-up
        await initClerkSignUp(container);

    } catch (error) {
        console.error('[Signup] Initialization error:', error);
        showError(container, error.message || 'Failed to initialize signup');
    }
}

async function initClerkSignUp(container) {
    if (window.TB?.user?.mountSignUp) {
        // Setup hash change listener for Clerk's #/continue route
        setupClerkContinueHandler();

        // Get redirect URL for after signup
        const urlParams = new URLSearchParams(window.location.search);
        const nextUrl = urlParams.get('next') || '/web/mainContent.html';

        await window.TB.user.mountSignUp(container, {
            afterSignUpUrl: nextUrl,
            afterSignInUrl: nextUrl
        });

        // Listen for sign-up events
        window.TB.events?.on('user:signedIn', (data) => {
            showSuccess(container, `Welcome, ${data.username}! Account created successfully.`);

            if (window.TB.graphics?.playAnimationSequence) {
                window.TB.graphics.playAnimationSequence("Z1+32:R0+50");
            }

            // Redirect nach erfolgreicher Registrierung
            setTimeout(() => {
                window.TB?.router?.navigateTo(nextUrl) || (window.location.href = nextUrl);
            }, 1000);
        });

        return;
    }

    console.log('[Signup] TB.user.mountSignUp not available');
}

function setupClerkContinueHandler() {
    // Listen for hash changes
    window.addEventListener('hashchange', checkAndHandleContinueRoute);

    // Bei #/continue: Clerk UI neu mounten, um Username-Eingabe zu ermöglichen
    if (window.location.hash.includes('#/continue')) {
        console.log('[Signup] Continue route detected, ensuring Clerk UI is mounted...');

        // Kurze Verzögerung, dann prüfen ob Clerk UI vorhanden ist
        // setTimeout(() => {
        //     const container = document.getElementById('clerk-sign-up');
        //     if (container && container.children.length <= 1) {
        //         // Clerk UI nicht geladen - neu mounten
        //         console.log('[Signup] Clerk UI not fully loaded, remounting...');
        //         if (window.TB?.user?.mountSignUp) {
        //             container.innerHTML = '';
        //             window.TB.user.mountSignUp(container);
        //         }
        //     }
        // }, 1000);
    }
}

function checkAndHandleContinueRoute() {
    const hash = window.location.hash;

    if (hash.includes('#/continue')) {
        console.log('[Signup] Detected Clerk continue route - allowing username entry...');

        // NICHT weiterleiten! Clerk zeigt das Username-Formular an.
        // Starte Polling um zu prüfen wann User fertig ist
        const checkAuthInterval = setInterval(() => {
            if (window.TB?.user?.isAuthenticated()) {
                const clerkUser = window.TB.user.getClerkUser();
                // Prüfe ob Username gesetzt ist
                if (clerkUser?.username) {
                    console.log('[Signup] User completed signup with username:', clerkUser.username);
                    clearInterval(checkAuthInterval);

                    // Redirect zu mainContent
                    const urlParams = new URLSearchParams(window.location.search);
                    const nextUrl = urlParams.get('next') || '/web/mainContent.html';

                    setTimeout(() => {
                        window.TB?.router?.navigateTo(nextUrl) || (window.location.href = nextUrl);
                    }, 500);
                }
            }
        }, 1000);

        // Timeout nach 5 Minuten
        setTimeout(() => clearInterval(checkAuthInterval), 300000);
    }
}

function showError(container, message) {
    if (window.TB?.ui?.Toast?.showError) {
        window.TB.ui.Toast.showError(message);
    }

    if (window.TB?.graphics?.playAnimationSequence) {
        window.TB.graphics.playAnimationSequence("R0-31");
    }

    const errorEl = document.getElementById('error-message');
    if (errorEl) {
        errorEl.textContent = message;
        errorEl.classList.add('show');
    }

    const infoPopup = document.getElementById('infoPopup');
    const infoText = document.getElementById('infoText');
    if (infoPopup && infoText) {
        infoText.textContent = message;
        infoPopup.style.display = 'block';
    }
}

function showSuccess(container, message) {
    if (window.TB?.ui?.Toast?.showSuccess) {
        window.TB.ui.Toast.showSuccess(message);
    }

    if (window.TB?.graphics?.playAnimationSequence) {
        window.TB.graphics.playAnimationSequence("P0+21");
    }

    const infoPopup = document.getElementById('infoPopup');
    const infoText = document.getElementById('infoText');
    if (infoPopup && infoText) {
        infoText.textContent = message;
        infoText.style.color = '#22c55e';
        infoPopup.style.display = 'block';
    }
}

function showInfo(message) {
    if (window.TB?.ui?.Toast?.showInfo) {
        window.TB.ui.Toast.showInfo(message);
    }

    const infoPopup = document.getElementById('infoPopup');
    const infoText = document.getElementById('infoText');
    if (infoPopup && infoText) {
        infoText.textContent = message;
        infoPopup.style.display = 'block';
    }
}

// Initialize when TB is ready
if (window.TB?.once) {
    window.TB.onLoaded(setupSignup);
} else {
    // Fallback: wait for DOMContentLoaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupSignup);
    } else {
        setupSignup();
    }
}

// Export for module usage
export { setupSignup, initializeClerkSignup };
