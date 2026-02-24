"""
OAuth HTTP helpers and provider-specific user fetchers.
"""

import httpx


async def _exchange_oauth_code(config: dict, code: str) -> tuple[bool, dict]:
    """Exchange OAuth authorization code for tokens."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(config["token_url"], data={
                "client_id": config["client_id"],
                "client_secret": config["client_secret"],
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": config["redirect_uri"],
            }, headers={"Accept": "application/json"})
            if resp.status_code != 200:
                return False, {"error": f"Token exchange failed ({resp.status_code}): {resp.text}"}
            return True, resp.json()
    except Exception as e:
        return False, {"error": str(e)}


async def _get_discord_user(access_token: str) -> tuple[bool, dict]:
    """Fetch Discord user profile."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://discord.com/api/users/@me",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if resp.status_code != 200:
                return False, {"error": f"Discord user fetch failed: {resp.text}"}
            d = resp.json()
            avatar = ""
            if d.get("avatar"):
                avatar = f"https://cdn.discordapp.com/avatars/{d['id']}/{d['avatar']}.png"
            return True, {
                "provider_id": d["id"],
                "username": d["username"],
                "email": d.get("email", ""),
                "avatar": avatar,
            }
    except Exception as e:
        return False, {"error": str(e)}


async def _get_google_user(access_token: str) -> tuple[bool, dict]:
    """Fetch Google user profile."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if resp.status_code != 200:
                return False, {"error": f"Google user fetch failed: {resp.text}"}
            d = resp.json()
            return True, {
                "provider_id": d["id"],
                "username": d.get("name", d.get("email", "").split("@")[0]),
                "email": d.get("email", ""),
                "avatar": d.get("picture", ""),
            }
    except Exception as e:
        return False, {"error": str(e)}
