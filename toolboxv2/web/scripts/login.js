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

    // Setup theme listener for Clerk UI
    setupThemeListener();

    // Initialize Login
    initializeClerkLogin();
}

// login.js - Ersetze initializeClerkLogin komplett

async function initializeClerkLogin() {
    const container = document.getElementById('clerk-sign-in');

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
        // WICHTIG: Prüfe Server-Status BEVOR Clerk UI
        console.log('[Login] Checking authentication status with server...');

        const serverCheckResponse = await fetch('/validateSession', {
            method: 'GET',
            credentials: 'include' // Session-Cookie mitsenden
        });

        let serverAuthenticated = false;

        if (serverCheckResponse.ok) {
            const serverCheckResult = await serverCheckResponse.json();
            serverAuthenticated = (serverCheckResult.error === "none");
        }

        console.log('[Login] Server authentication status:', serverAuthenticated);

        // Wenn Server sagt "authenticated" -> weiterleiten
        if (serverAuthenticated) {
            console.log('[Login] Already authenticated on server, redirecting...');
            showSuccess(container, 'Already logged in. Redirecting...');
            redirectUser();
            return;
        }

        // Wenn Clerk LOKAL sagt "authenticated" aber Server nicht -> AUSLOGGEN
        if (window.TB.user.isAuthenticated()) {
            console.warn('[Login] Local auth state inconsistent with server, clearing...');

            // Lokalen State löschen
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

            // Clerk ausloggen
            const clerkInstance = window.TB.user.getClerkInstance();
            if (clerkInstance?.signOut) {
                try {
                    await clerkInstance.signOut();
                    console.log('[Login] Cleared inconsistent Clerk session');
                } catch (e) {
                    console.warn('[Login] Failed to clear Clerk session:', e);
                }
            }
        }

        // Setup hash change listener für Clerk's continue route
        setupClerkContinueHandler();

        // Mount Clerk Sign-In UI
        await window.TB.user.mountSignIn(container, {
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
                    headerTitle: { display: 'none' },
                    headerSubtitle: { display: 'none' }
                }
            }
        });

        console.log('[Login] Clerk Sign-In mounted successfully');

        // Listen for sign-in event
        window.TB.events.on('user:signedIn', (data) => {
            console.log('[Login] ✓ User signed in successfully:', data.userId);
            showSuccess(container, 'Login successful! Redirecting...');
            // Redirect wird von _handlePostAuthRedirect gehandhabt
        });

        // Listen for auth errors
        window.TB.events.on('user:authError', (data) => {
            console.error('[Login] ✗ Authentication error:', data.message);
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

// Hilfsfunktion für Continue-Route bleibt gleich
function setupClerkContinueHandler() {
    checkAndHandleContinueRoute();
    window.addEventListener('hashchange', checkAndHandleContinueRoute);
}

function checkAndHandleContinueRoute() {
    const hash = window.location.hash;

    if (hash.includes('#/continue')) {
        console.log('[Login] Detected Clerk continue route, checking auth status...');

        setTimeout(() => {
            const clerkInstance = window.TB?.user?.getClerkInstance();

            if (clerkInstance?.user) {
                if (!clerkInstance.user.username) {
                    console.log('[Login] User needs to set username, redirecting to signup...');
                    const nextUrl = getRedirectUrl();
                    window.location.href = `/web/assets/signup.html?next=${encodeURIComponent(nextUrl)}`;
                } else if (window.TB?.user?.isAuthenticated()) {
                    console.log('[Login] User authenticated, redirecting...');
                    redirectUser();
                }
            }
        }, 1000);
    }
}

function redirectUser() {
    const nextUrl = getRedirectUrl();
    setTimeout(() => {
        window.TB?.router?.navigateTo(nextUrl) || (window.location.href = nextUrl);
    }, 500);
}

function getRedirectUrl() {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('next') || '/web/mainContent.html';
}

/**
 * Setup listener for theme changes to update Clerk UI
 */
function setupThemeListener() {
    // Listen for theme:changed events
    document.addEventListener('theme:changed', async (event) => {
        console.log('[Login] Theme changed to:', event.detail?.theme);

        if (window.TB?.user?.refreshClerkTheme) {
            await window.TB.user.refreshClerkTheme();
        }
    });

    // Also listen for TB.events if available
    if (window.TB?.events?.on) {
        window.TB.events.on('theme:changed', async (data) => {
            console.log('[Login] Theme changed (TB.events):', data.theme);

            if (window.TB?.user?.refreshClerkTheme) {
                await window.TB.user.refreshClerkTheme();
            }
        });
    }
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
