#!/usr/bin/env python3
"""
Registry Admin CLI — Direct server-side management tool.

Runs directly on the registry server, no authentication required.
For local admin use only — never expose this tool externally.

Usage:
    python registry_admin_cli.py [--db path/to/registry.db]
"""

import argparse
import asyncio
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# =================== Minimal DB Layer ===================

try:
    import aiosqlite
except ImportError:
    print("Error: aiosqlite required. Install with: pip install aiosqlite")
    sys.exit(1)


class DirectDB:
    """Thin async SQLite wrapper for direct admin access."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self):
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")

    async def close(self):
        if self._conn:
            await self._conn.close()

    async def fetch_all(self, query: str, params: tuple = ()) -> list:
        async with self._conn.execute(query, params) as cur:
            return await cur.fetchall()

    async def fetch_one(self, query: str, params: tuple = ()):
        async with self._conn.execute(query, params) as cur:
            return await cur.fetchone()

    async def execute(self, query: str, params: tuple = ()):
        await self._conn.execute(query, params)

    async def commit(self):
        await self._conn.commit()


# =================== Colors ===================

class C:
    R = "\033[0m"
    B = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    GREY = "\033[90m"
    MAGENTA = "\033[95m"


def colored(text: str, color: str) -> str:
    return f"{color}{text}{C.R}"


# =================== Display Helpers ===================

def banner():
    print(f"\n{C.CYAN}{'═' * 56}{C.R}")
    print(f"{C.B}  🔧  Registry Admin CLI — Direct Server Management{C.R}")
    print(f"{C.CYAN}{'═' * 56}{C.R}\n")


def print_table(headers: list[str], rows: list[list[str]], widths: list[int]):
    header_line = "  ".join(f"{C.B}{h:<{w}}{C.R}" for h, w in zip(headers, widths))
    print(f"  {header_line}")
    print(f"  {'─' * (sum(widths) + 2 * (len(widths) - 1))}")
    for row in rows:
        cells = []
        for val, w in zip(row, widths):
            cells.append(f"{val:<{w}}")
        print(f"  {'  '.join(cells)}")


def status_color(s: str) -> str:
    colors = {
        "verified": C.GREEN,
        "pending": C.YELLOW,
        "rejected": C.RED,
        "unverified": C.GREY,
        "suspended": C.MAGENTA,
    }
    return colored(s, colors.get(s, C.R))


def prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"  {C.CYAN}>{C.R} {msg}{suffix}: ").strip()
    return val or default


def confirm(msg: str) -> bool:
    return input(f"  {C.YELLOW}?{C.R} {msg} [y/N]: ").strip().lower() == "y"


def ok(msg: str):
    print(f"  {C.GREEN}✓{C.R} {msg}")


def err(msg: str):
    print(f"  {C.RED}✗{C.R} {msg}")


def info(msg: str):
    print(f"  {C.CYAN}ℹ{C.R} {msg}")


# =================== Commands ===================

async def cmd_list_users(db: DirectDB):
    """List all users."""
    rows = await db.fetch_all(
        "SELECT * FROM users ORDER BY created_at DESC"
    )
    if not rows:
        info("No users found.")
        return

    print(f"\n  {C.B}Users ({len(rows)}){C.R}\n")
    table_rows = []
    for r in rows:
        uid = (r["cloudm_user_id"] or r.get("cloudm_user_id", "?"))[:20]
        pub = r["publisher_id"] or "—"
        admin = colored("ADMIN", C.GREEN) if r["is_admin"] else "—"
        table_rows.append([uid, r["username"] or "—", r["email"] or "—", pub[:12], admin])

    print_table(
        ["User ID", "Username", "Email", "Publisher", "Admin"],
        table_rows,
        [22, 18, 28, 14, 8],
    )
    print()


async def cmd_list_publishers(db: DirectDB):
    """List all publishers."""
    rows = await db.fetch_all(
        "SELECT * FROM publishers ORDER BY created_at DESC"
    )
    if not rows:
        info("No publishers found.")
        return

    print(f"\n  {C.B}Publishers ({len(rows)}){C.R}\n")
    table_rows = []
    for r in rows:
        st = status_color(r["status"])
        table_rows.append([
            r["id"][:12],
            r["slug"],
            r["name"][:20],
            st,
            str(r["packages_count"]),
            str(r["total_downloads"]),
        ])

    print_table(
        ["ID", "Slug", "Name", "Status", "Pkgs", "DLs"],
        table_rows,
        [14, 18, 22, 14, 6, 8],
    )
    print()


async def cmd_make_publisher(db: DirectDB):
    """Promote a user to publisher."""
    username = prompt("Username or user ID")
    if not username:
        return

    # Find user
    user = await db.fetch_one(
        "SELECT * FROM users WHERE username = ? OR cloudm_user_id = ?",
        (username, username),
    )
    # Fallback: cloudm_user_id
    if not user:
        user = await db.fetch_one(
            "SELECT * FROM users WHERE cloudm_user_id = ?", (username,)
        )
    if not user:
        err(f"User '{username}' not found.")
        return

    uid = user["cloudm_user_id"] or user.get("cloudm_user_id")
    info(f"Found: {user['username']} ({user['email']})")

    if user["publisher_id"]:
        err(f"Already a publisher (ID: {user['publisher_id']})")
        return

    # Collect publisher info
    slug = prompt("Publisher slug (unique handle)", user["username"].lower().replace(" ", "-"))
    name = prompt("Display name", user["username"])
    email = prompt("Contact email", user["email"])
    website = prompt("Website (optional)")
    github = prompt("GitHub username (optional)")

    pub_id = str(uuid.uuid4())

    if not confirm(f"Create publisher @{slug} for {user['username']}?"):
        info("Cancelled.")
        return

    await db.execute(
        """
        INSERT INTO publishers (
            id, cloudm_user_id, name, slug, email, website, github,
            status, can_publish_public, can_publish_artifacts,
            max_package_size_mb, packages_count, total_downloads, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?)
        """,
        (
            pub_id, uid, name, slug, email,
            website or None, github or None,
            "unverified", False, False, 100,
            datetime.utcnow().isoformat(),
        ),
    )
    await db.execute(
        "UPDATE users SET publisher_id = ? WHERE cloudm_user_id = ?",
        (pub_id, uid),
    )
    await db.commit()
    ok(f"Publisher @{slug} created (ID: {pub_id[:12]}...)")

    # Optionally verify immediately
    if confirm("Verify immediately?"):
        await _set_status(db, pub_id, "verified", "Verified via admin CLI")
        ok("Publisher verified and can_publish_public = true.")


async def cmd_remove_publisher(db: DirectDB):
    """Remove publisher status from a user."""
    slug_or_id = prompt("Publisher slug or ID")
    if not slug_or_id:
        return

    pub = await db.fetch_one(
        "SELECT * FROM publishers WHERE slug = ? OR id = ?",
        (slug_or_id, slug_or_id),
    )
    if not pub:
        err(f"Publisher '{slug_or_id}' not found.")
        return

    info(f"Found: @{pub['slug']} ({pub['name']}) — Status: {pub['status']}")
    info(f"Packages: {pub['packages_count']}, Downloads: {pub['total_downloads']}")

    if pub["packages_count"] > 0:
        err(f"Publisher has {pub['packages_count']} packages. Remove packages first or use --force.")
        if not confirm("Force remove anyway? (publisher link on packages will break)"):
            return

    if not confirm(f"Remove publisher @{pub['slug']}? This unlinks the user."):
        info("Cancelled.")
        return

    uid = pub["cloudm_user_id"] or pub.get("cloudm_user_id")
    await db.execute("UPDATE users SET publisher_id = NULL WHERE cloudm_user_id = ?", (uid,))
    await db.execute("DELETE FROM publishers WHERE id = ?", (pub["id"],))
    await db.commit()
    ok(f"Publisher @{pub['slug']} removed, user unlinked.")


async def cmd_edit_publisher(db: DirectDB):
    """Interactively edit a publisher's fields."""
    slug_or_id = prompt("Publisher slug or ID")
    if not slug_or_id:
        return

    pub = await db.fetch_one(
        "SELECT * FROM publishers WHERE slug = ? OR id = ?",
        (slug_or_id, slug_or_id),
    )
    if not pub:
        err(f"Publisher '{slug_or_id}' not found.")
        return

    print(f"\n  {C.B}Editing: @{pub['slug']}{C.R}")
    print(f"  {C.GREY}(press Enter to keep current value){C.R}\n")

    # Editable fields
    fields = {
        "name": ("Display name", pub["name"]),
        "slug": ("Slug", pub["slug"]),
        "email": ("Email", pub["email"]),
        "website": ("Website", pub["website"] or ""),
        "github": ("GitHub", pub["github"] or ""),
        "max_package_size_mb": ("Max package size (MB)", str(pub["max_package_size_mb"])),
    }

    updates = {}
    for field_name, (label, current) in fields.items():
        new_val = prompt(f"{label}", current)
        if new_val != current:
            updates[field_name] = new_val

    if not updates:
        info("No changes.")
        return

    # Build UPDATE
    set_parts = [f"{k} = ?" for k in updates]
    values = list(updates.values())

    # max_package_size_mb is int
    if "max_package_size_mb" in updates:
        idx = list(updates.keys()).index("max_package_size_mb")
        values[idx] = int(values[idx])

    values.append(pub["id"])

    await db.execute(
        f"UPDATE publishers SET {', '.join(set_parts)} WHERE id = ?",
        tuple(values),
    )
    await db.commit()

    ok(f"Updated {len(updates)} field(s): {', '.join(updates.keys())}")


async def cmd_set_status(db: DirectDB):
    """Change a publisher's verification status."""
    slug_or_id = prompt("Publisher slug or ID")
    if not slug_or_id:
        return

    pub = await db.fetch_one(
        "SELECT * FROM publishers WHERE slug = ? OR id = ?",
        (slug_or_id, slug_or_id),
    )
    if not pub:
        err(f"Publisher '{slug_or_id}' not found.")
        return

    info(f"@{pub['slug']} — current status: {status_color(pub['status'])}")

    valid = ["unverified", "pending", "verified", "rejected", "suspended"]
    print(f"  Options: {', '.join(valid)}")
    new_status = prompt("New status")

    if new_status not in valid:
        err(f"Invalid status. Choose from: {', '.join(valid)}")
        return

    if new_status == pub["status"]:
        info("Already that status.")
        return

    notes = prompt("Notes (optional)", f"Set via admin CLI")

    if not confirm(f"Set @{pub['slug']} to {status_color(new_status)}?"):
        return

    await _set_status(db, pub["id"], new_status, notes)
    ok(f"Status updated to {new_status}.")


async def _set_status(db: DirectDB, publisher_id: str, new_status: str, notes: str):
    """Internal: update publisher status with side effects."""
    verified_at = datetime.utcnow().isoformat() if new_status == "verified" else None
    can_publish = new_status == "verified"

    await db.execute(
        """
        UPDATE publishers SET
            status = ?,
            verified_at = ?,
            verified_by = ?,
            verification_notes = ?,
            can_publish_public = ?
        WHERE id = ?
        """,
        (new_status, verified_at, "admin-cli", notes, can_publish, publisher_id),
    )
    await db.commit()


async def cmd_make_admin(db: DirectDB):
    """Toggle admin status for a user."""
    username = prompt("Username or user ID")
    if not username:
        return

    user = await db.fetch_one(
        "SELECT * FROM users WHERE username = ? OR cloudm_user_id = ?",
        (username, username),
    )
    if not user:
        user = await db.fetch_one(
            "SELECT * FROM users WHERE cloudm_user_id = ?", (username,)
        )
    if not user:
        err(f"User '{username}' not found.")
        return

    current = bool(user["is_admin"])
    new_val = not current
    action = "Grant admin" if new_val else "Revoke admin"

    info(f"{user['username']} — admin: {current} → {new_val}")
    if not confirm(f"{action} for {user['username']}?"):
        return

    uid = user["cloudm_user_id"] or user.get("cloudm_user_id")
    await db.execute(
        "UPDATE users SET is_admin = ? WHERE cloudm_user_id = ?",
        (new_val, uid),
    )
    await db.commit()
    ok(f"{action} done.")


async def cmd_sql(db: DirectDB):
    """Run raw SQL (read-only by default, write with !prefix)."""
    print(f"  {C.GREY}Enter SQL. Prefix with ! for write queries. 'q' to exit.{C.R}")
    while True:
        query = input(f"  {C.MAGENTA}SQL>{C.R} ").strip()
        if not query or query.lower() == "q":
            break

        try:
            if query.startswith("!"):
                await db.execute(query[1:])
                await db.commit()
                ok("Executed.")
            else:
                rows = await db.fetch_all(query)
                if not rows:
                    info("No results.")
                else:
                    keys = rows[0].keys()
                    print(f"  {C.B}{'  '.join(keys)}{C.R}")
                    for r in rows[:50]:
                        print(f"  {'  '.join(str(r[k])[:30] for k in keys)}")
                    if len(rows) > 50:
                        info(f"... and {len(rows) - 50} more rows")
        except Exception as e:
            err(str(e))


# =================== Main Loop ===================

COMMANDS = {
    "1": ("List users", cmd_list_users),
    "2": ("List publishers", cmd_list_publishers),
    "3": ("Make publisher (user → publisher)", cmd_make_publisher),
    "4": ("Remove publisher", cmd_remove_publisher),
    "5": ("Edit publisher", cmd_edit_publisher),
    "6": ("Set publisher status", cmd_set_status),
    "7": ("Toggle admin", cmd_make_admin),
    "8": ("Raw SQL", cmd_sql),
}


async def main_loop(db_path: str):
    db = DirectDB(db_path)
    await db.connect()
    info(f"Connected to: {db_path}")

    banner()

    while True:
        print(f"  {C.B}Commands:{C.R}")
        for key, (label, _) in COMMANDS.items():
            print(f"    {C.CYAN}{key}{C.R}  {label}")
        print(f"    {C.CYAN}q{C.R}  Quit\n")

        choice = input(f"  {C.B}→{C.R} ").strip()

        if choice.lower() == "q":
            break

        cmd = COMMANDS.get(choice)
        if not cmd:
            err("Invalid choice.")
            continue

        try:
            print()
            await cmd[1](db)
            print()
        except KeyboardInterrupt:
            print()
        except Exception as e:
            err(f"Error: {e}")
            print()

    await db.close()
    info("Bye.")


def main():
    parser = argparse.ArgumentParser(
        prog="registry-admin",
        description="Registry Admin CLI — Direct server-side management",
    )
    parser.add_argument(
        "--db",
        default="./data/registry.db",
        help="Path to registry SQLite database (default: ./data/registry.db)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        print("Use --db to specify the correct path.")
        sys.exit(1)

    asyncio.run(main_loop(str(db_path)))


if __name__ == "__main__":
    main()
