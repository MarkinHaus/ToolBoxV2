// /web/scripts/login.js
// ToolBox V2 Login Script with Custom Auth via TB.user API

/**
 * Main function - called when TB is ready or DOM is loaded
 */
function setupLogin() {
    // Play initial animation if available
    if (window.TB?.graphics?.playAnimationSequence) {
        window.TB.graphics.playAnimationSequence("Z0+12");
    }

    // Initialize Login
    initializeLogin();
}

/**
 * Check server auth status and show login UI or redirect
 */
async function initializeLogin() {
    const container = document.getElementById('auth-sign-in');

    if (!container) {
        console.warn('[Login] No login container found');
        return;
    }

    if (!window.TB || !window.TB.user) {
        console.error('[Login] TB.user framework not available');
        showError(container, 'System not ready. Please refresh.');
        return;
    }

    try {
        // Check server authentication status first
        console.log('[Login] Checking authentication status with server...');

        const serverCheckResponse = await fetch('/IsValidSession', {
            method: 'GET',
            credentials: 'include',
            headers: {
                'Authorization': `Bearer ${window.TB?.state?.get('user.token') || ''}`
            }
        });

        let serverAuthenticated = false;

        if (serverCheckResponse.ok) {
            const serverCheckResult = await serverCheckResponse.json();
            serverAuthenticated = (serverCheckResult.error === "none");
        }

        console.log('[Login] Server authentication status:', serverAuthenticated);

        // If server says authenticated -> redirect
        if (serverAuthenticated) {
            console.log('[Login] Already authenticated on server, redirecting...');
            showSuccess(container, 'Already logged in. Redirecting...');
            redirectUser();
            return;
        }

        // If local TB.user says authenticated but server does not -> clear local state
        if (window.TB.user.isAuthenticated()) {
            console.warn('[Login] Local auth state inconsistent with server, clearing...');

            // Clear local state
            window.TB.state.set('user', {
                isAuthenticated: false,
                username: null,
                email: null,
                userId: null,
                userLevel: 1,
                token: null,
                userData: {},
                settings: {},
                modData: {}
            });

            // Logout via TB.user
            if (window.TB.user.logout) {
                try {
                    await window.TB.user.logout();
                    console.log('[Login] Cleared inconsistent local session');
                } catch (e) {
                    console.warn('[Login] Failed to clear local session:', e);
                }
            }
        }

        // Hide loading state and show auth container
        const loadingEl = container.querySelector('.loading');
        if (loadingEl) loadingEl.style.display = 'none';

        container.style.display = '';

        console.log('[Login] Auth UI ready');

        // Listen for sign-in event
        window.TB.events.on('user:signedIn', (data) => {
            console.log('[Login] User signed in successfully:', data.userId);
            showSuccess(container, 'Login successful! Redirecting...');
        });

        // Listen for auth errors
        window.TB.events.on('user:authError', (data) => {
            console.error('[Login] Authentication error:', data.message);
            showError(container, data.message);
        });

    } catch (error) {
        console.error('[Login] Initialization error:', error);
        showError(container, error.message || 'Failed to load login form');

        const retryBtn = document.createElement('button');
        retryBtn.textContent = "Retry";
        retryBtn.className = "retry-button";
        retryBtn.onclick = () => window.location.reload();
        container.appendChild(retryBtn);
    }
}

/**
 * Redirect user to the intended destination
 */
function redirectUser() {
    const nextUrl = getRedirectUrl();
    setTimeout(() => {
        window.TB?.router?.navigateTo(nextUrl) || (window.location.href = nextUrl);
    }, 500);
}

/**
 * Get the URL to redirect to after login
 */
function getRedirectUrl() {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('next') || '/web/mainContent.html';
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
    window.TB.onLoaded(setupLogin);
} else {
    // Fallback: wait for DOMContentLoaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupLogin);
    } else {
        setupLogin();
    }
}

export { setupLogin };
