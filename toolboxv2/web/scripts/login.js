// /web/scripts/login.js
// ToolBox V2 Login Script with Clerk Integration

/**
 * Main function - called when TB is ready or DOM is loaded
 */
function setupLogin() {
    // Play initial animation if available
    if (window.TB?.graphics?.playAnimationSequence) {
        window.TB.graphics.playAnimationSequence("Z0+12");
    }

    // Initialize Login
    initializeClerkLogin();
}

/**
 * Initializes the Clerk Login process by delegating to TB.user
 */
async function initializeClerkLogin() {
    const container = document.getElementById('clerk-sign-in');

    if (!container) {
        console.warn('[Login] No login container found');
        return;
    }

    // Ensure TB.user is available
    if (!window.TB || !window.TB.user) {
        console.error('[Login] TB.user framework not available');
        showError(container, 'System not ready. Please refresh.');
        return;
    }

    try {
        // Check if already authenticated
        if (window.TB.user.isAuthenticated()) {
            const username = window.TB.user.getUsername();
            showSuccess(container, `Already logged in as ${username}. Redirecting...`);
            redirectUser();
            return;
        }

        // Use TB.user to mount the sign-in form
        // This handles config fetching, script loading, and the "Clerk is not a constructor" edge case
        await window.TB.user.mountSignIn(container, {
            afterSignInUrl: getRedirectUrl(),
            signUpUrl: '/web/assets/signup.html',
            appearance: {
                elements: {
                    rootBox: { width: '100%' },
                    card: {
                        background: 'transparent',
                        boxShadow: 'none',
                        border: 'none'
                    },
                    formButtonPrimary: {
                        backgroundColor: '#6366f1'
                    },
                    headerTitle: { display: 'none' }, // Hide default header as we have our own
                    headerSubtitle: { display: 'none' }
                }
            }
        });

        console.log('[Login] Clerk Sign-In mounted successfully');

        // Listen for sign-in event via TB events system
        window.TB.events.on('user:signedIn', (data) => {
            console.log('[Login] User signed in:', data.userId);
            showSuccess(container, 'Login successful! Redirecting...');
            redirectUser();
        });

    } catch (error) {
        console.error('[Login] Initialization error:', error);
        showError(container, error.message || 'Failed to load login form');

        // Add retry button if initialization failed
        const retryBtn = document.createElement('button');
        retryBtn.textContent = "Retry";
        retryBtn.className = "retry-button";
        retryBtn.onclick = () => window.location.reload();
        container.appendChild(retryBtn);
    }
}

/**
 * Helper to get the redirect URL from params or default
 */
function getRedirectUrl() {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('next') || '/web/mainContent.html';
}

/**
 * Redirects the user
 */
function redirectUser() {
    const nextUrl = getRedirectUrl();
    setTimeout(() => {
        window.TB?.router?.navigateTo(nextUrl) || (window.location.href = nextUrl);
    }, 800);
}

/**
 * UI Helper: Show Error
 */
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

    // Clear loading spinner if error occurs
    const loadingEl = container.querySelector('.loading');
    if (loadingEl) loadingEl.style.display = 'none';
}

/**
 * UI Helper: Show Success
 */
function showSuccess(container, message) {
    if (window.TB?.ui?.Toast?.showSuccess) {
        window.TB.ui.Toast.showSuccess(message);
    }

    if (window.TB?.graphics?.playAnimationSequence) {
        window.TB.graphics.playAnimationSequence("P0+21");
    }
}

// Initialization when TB is ready
if (window.TB?.once) {
    window.TB.once(setupLogin);
} else {
    // Fallback: wait for DOMContentLoaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupLogin);
    } else {
        setupLogin();
    }
}

export { setupLogin };
