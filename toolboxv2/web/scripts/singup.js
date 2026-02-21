// /web/scripts/singup.js
// ToolBox V2 Signup Script with Custom Auth via TB.user API

function setupSignup() {
    // Play initial animation if graphics available
    if (window.TB?.graphics?.playAnimationSequence) {
        window.TB.graphics.playAnimationSequence("Z0+12");
    }

    setTimeout(async () => {
        await initializeSignup();
    }, 100);
}

async function initializeSignup() {
    const container = document.getElementById('auth-sign-up') || document.getElementById('signupForm');

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

        if (!window.TB.user) {
            console.error('[Signup] TB.user failed to initialize');
            showError(container, 'Authentication service not available. Please refresh the page.');
            return;
        }

        // Check if user is already authenticated
        if (window.TB.user.isAuthenticated()) {
            const username = window.TB.user.getUsername();
            showSuccess(container, `Already logged in as ${username}. Redirecting...`);

            if (window.TB.graphics?.playAnimationSequence) {
                window.TB.graphics.playAnimationSequence("Z1+32:R0+50");
            }

            setTimeout(() => {
                const urlParams = new URLSearchParams(window.location.search);
                const nextUrl = urlParams.get('next') || '/web/mainContent.html';
                window.TB.router?.navigateTo(nextUrl) || (window.location.href = nextUrl);
            }, 1000);
            return;
        }

        // Auth buttons are rendered in the HTML fragment.
        // Manage loading state: mark container as ready.
        container.classList.remove('loading');
        container.classList.add('ready');
        showInfo('Create your account to get started.');

        // Listen for successful sign-in/sign-up event to redirect
        const urlParams = new URLSearchParams(window.location.search);
        const nextUrl = urlParams.get('next') || '/web/mainContent.html';

        window.TB.events?.on('user:signedIn', (data) => {
            const username = data?.username || window.TB.user?.getUsername() || 'User';
            showSuccess(container, `Welcome, ${username}! Account created successfully.`);

            if (window.TB.graphics?.playAnimationSequence) {
                window.TB.graphics.playAnimationSequence("Z1+32:R0+50");
            }

            setTimeout(() => {
                window.TB.router?.navigateTo(nextUrl) || (window.location.href = nextUrl);
            }, 1000);
        });

    } catch (error) {
        console.error('[Signup] Initialization error:', error);
        showError(container, error.message || 'Failed to initialize signup');
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
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupSignup);
    } else {
        setupSignup();
    }
}

export { setupSignup, initializeSignup };
