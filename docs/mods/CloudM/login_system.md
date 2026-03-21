# CloudM — Login System

> **File**: `LogInSystem.py` · **Version**: 0.0.6

CLI and web session management layer. No Clerk dependency. Multi-worker safe via `TBEF.DB`.

## Auth Methods

| Method | Channel | Flow |
|--------|---------|------|
| Magic Link Email | CLI + Web | Email → link → JWT stored locally |
| Device Invite Code | CLI | Running session generates 6-digit code → CLI enters → JWT |
| Web (OAuth / Passkey) | Browser | Redirect to `/web/scripts/login.html` |

## CLI Login Flow

### Magic Link

```
1. User runs: tb login
2. Prompted for email address
3. Magic link sent to email (via CloudM.Auth)
4. User clicks link in browser → callback issues JWT
5. JWT stored in ~/.data/cli_sessions/ (BlobFile, encrypted)
6. Subsequent tb commands use stored token
```

### Device Invite Code

```
1. Logged-in device runs: tb -c CloudM generate_invite
2. Returns 6-digit code (valid 5 min)
3. New device runs: tb login --code 123456
4. JWT issued and stored locally
```

## Session Storage

CLI sessions are stored on-disk using `BlobStorage` in offline mode:

```
~/.local/share/ToolBoxV2/.data/cli_sessions/
```

Tokens are **encrypted** (via `toolboxv2.utils.security.cryp.Code`). Token metadata is mirrored to `TBEF.DB` for multi-worker session validation.

## API Endpoints

```
POST /CloudM/login/magic-link          → Request magic link for email
POST /CloudM/login/magic-link/verify   → Verify token from email link
POST /CloudM/login/device-invite       → Verify device invite code
GET  /CloudM/login/web                 → Redirect to web login UI
POST /CloudM/logout                    → Invalidate current session
GET  /CloudM/session/status            → Check session validity
```

## CLI Commands

```bash
tb -c CloudM login                    # Interactive login (magic link or code)
tb -c CloudM login --code 123456      # Device invite login
tb -c CloudM logout                   # Clear local session
tb -c CloudM session status           # Show current session info
```

## Web Login UI

The web login page is served from `/web/scripts/login.html` and offers:

- Discord OAuth2
- Google OAuth2
- Passkey (WebAuthn)

After successful authentication the browser session cookie is set and the user is redirected.

## Implementation Notes

- `TBEF.DB` is the single source of truth for token validity (blacklist, expiry)
- Local BlobFile stores the token copy for offline/CLI use
- The `App.session.login()` call during module init triggers auto-login from stored credentials
- Session lifecycle: `login()` → `validate()` → `refresh()` → `logout()`

## Related

- [Auth System](auth.md) — Core JWT + provider logic
- [User Data API](user_data.md) — Data access gated by session identity
