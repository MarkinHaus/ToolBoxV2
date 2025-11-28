"""
ToolBox V2 - CLI Helper Commands
Provides CLI commands for user management with Clerk integration
"""

import asyncio
from toolboxv2 import TBEF, App, Result, get_app

Name = 'helper'
export = get_app(f"{Name}.Export").tb
version = "0.2.0"


@export(mod_name=Name, name="init_system", test=False)
async def init_system(app: App):
    """
    Initializes the ToolBoxV2 system.
    With Clerk, initial user creation happens via web registration.
    This command sets up the system configuration.
    """
    print("--- ToolBoxV2 System Initialization ---")
    print("With Clerk authentication, users register via the web interface.")
    print()

    try:
        # Check if Clerk is configured
        import os
        clerk_key = os.getenv('CLERK_SECRET_KEY')
        pub_key = os.getenv('CLERK_PUBLISHABLE_KEY')

        if not clerk_key or not pub_key:
            print("⚠️  Clerk API keys not configured!")
            print()
            print("Please add the following to your .env file:")
            print("  CLERK_PUBLISHABLE_KEY=pk_test_...")
            print("  CLERK_SECRET_KEY=sk_test_...")
            print()
            print("Get your keys at: https://dashboard.clerk.com")
            return Result.default_user_error("Clerk not configured")

        print("✅ Clerk configuration detected!")
        print()
        print("To create your first admin user:")
        print("  1. Open the web interface: http://localhost:8080/web/assets/signup.html")
        print("  2. Register with your email")
        print("  3. Verify your email with the code sent to you")
        print()
        print("For CLI login after registration:")
        print("  tb login")
        print()

        return Result.ok("System initialized. Please register via web interface.")

    except (KeyboardInterrupt, EOFError):
        print("\n\nInitialization cancelled by user.")
        return Result.default_user_error("Initialization cancelled.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return Result.default_internal_error(f"An unexpected error occurred: {e}")


@export(mod_name=Name, name="login", test=False)
async def login(app: App, email: str = None):
    """
    Login to ToolBox V2 via Clerk Email + Code verification.
    No browser opening - direct code input in CLI.
    """
    app.load_mod("CloudM")

    # Import the CLI login function
    from toolboxv2.mods.CloudM.LogInSystem import cli_login

    result = await cli_login(app, email)
    return result


@export(mod_name=Name, name="logout", test=False)
async def logout(app: App):
    """Logout from the current CLI session."""
    app.load_mod("CloudM")

    from toolboxv2.mods.CloudM.LogInSystem import cli_logout

    result = await cli_logout(app)
    return result


@export(mod_name=Name, name="status", test=False)
async def status(app: App):
    """Show current authentication status."""
    app.load_mod("CloudM")

    from toolboxv2.mods.CloudM.LogInSystem import cli_status

    result = await cli_status(app)
    return result


@export(mod_name=Name, name="list-users", test=False)
def list_users_cli(app: App):
    """Lists all registered users from Clerk."""
    print("Fetching user list from Clerk...")
    app.load_mod("CloudM")

    try:
        result = app.run_any(
            TBEF.CLOUDM_AUTHCLERK.LIST_USERS,
            get_results=True
        )

        if result.is_ok():
            users = result.get()
            if not users:
                print("No users found.")
                return result

            print("--- Registered Users (Clerk) ---")
            print(f"{'ID':<30} {'Username':<20} {'Email':<30}")
            print("-" * 80)
            for user in users:
                print(f"{user.get('id', 'N/A'):<30} {user.get('username', 'N/A'):<20} {user.get('email', 'N/A'):<30}")
            print("-" * 80)
            print(f"Total: {len(users)} users")
        else:
            print("❌ Error listing users:")
            result.print()

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return Result.default_internal_error(str(e))


@export(mod_name=Name, name="delete-user", test=False)
def delete_user_cli(app: App, user_id: str):
    """
    Deletes a user from Clerk and local storage.
    Use 'list-users' to find the user ID.
    """
    print(f"Attempting to delete user '{user_id}'...")
    app.load_mod("CloudM")

    # Confirm deletion
    confirm = input(f"Are you sure you want to delete user {user_id}? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Deletion cancelled.")
        return Result.ok("Cancelled")

    try:
        result = app.run_any(
            TBEF.CLOUDM_AUTHCLERK.DELETE_USER,
            clerk_user_id=user_id,
            get_results=True
        )

        if result.is_ok():
            print(f"✅ User '{user_id}' has been deleted.")
        else:
            print(f"❌ Error deleting user: {result.info.help_text}")

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return Result.default_internal_error(str(e))


@export(mod_name=Name, name="user-info", test=False)
async def user_info(app: App):
    """Show current user information."""
    app.load_mod("CloudM")

    from toolboxv2.mods.CloudM.AuthClerk import load_local_user_data, load_session_token
    from toolboxv2.utils.clis.cli_printing import (
        print_box_header,
        print_box_content,
        print_box_footer
    )

    # Get current session
    if not app.session or not app.session.valid:
        print_box_header("Not Authenticated", "⚠")
        print_box_content("Please login first with 'tb login'", "warning")
        print_box_footer()
        return Result.default_user_error("Not authenticated")

    username = app.session.username if hasattr(app.session, 'username') else None

    if not username:
        print_box_header("No User Data", "⚠")
        print_box_content("User data not available", "warning")
        print_box_footer()
        return Result.default_user_error("No user data")

    # Try to load local data
    # Note: We need the Clerk user ID, which might be stored differently
    print_box_header("User Information", "👤")
    print_box_content(f"Username: {username}", "info")

    # Load session data
    session_data = load_session_token(username)
    if session_data:
        print_box_content(f"Email: {session_data.get('email', 'N/A')}", "info")
        print_box_content(f"User ID: {session_data.get('user_id', 'N/A')}", "info")

    print_box_footer()

    return Result.ok()


@export(mod_name=Name, name="update-settings", test=False)
async def update_settings(app: App, key: str, value: str):
    """
    Update a user setting.
    Example: tb update-settings theme dark
    """
    app.load_mod("CloudM")

    from toolboxv2.mods.CloudM.AuthClerk import load_local_user_data, save_local_user_data

    if not app.session or not app.session.valid:
        print("❌ Please login first with 'tb login'")
        return Result.default_user_error("Not authenticated")

    # Load local user data
    local_data = load_local_user_data(app.session.clerk_user_id)
    if not local_data:
        print("❌ User data not found. Please try logging in again.")
        return Result.default_user_error("User data not found")

    # Parse value (try to convert to appropriate type)
    parsed_value = value
    if value.lower() == 'true':
        parsed_value = True
    elif value.lower() == 'false':
        parsed_value = False
    elif value.isdigit():
        parsed_value = int(value)

    # Update settings
    local_data.settings[key] = parsed_value

    # Save
    if save_local_user_data(local_data):
        print(f"✅ Setting '{key}' updated to '{parsed_value}'")
        return Result.ok()
    else:
        print(f"❌ Failed to save setting")
        return Result.default_internal_error("Failed to save setting")


@export(mod_name=Name, name="sync-data", test=False)
async def sync_data(app: App):
    """
    Sync local user data with the server.
    This ensures settings and mod data are synchronized.
    """
    app.load_mod("CloudM")

    from toolboxv2.mods.CloudM.AuthClerk import (
        load_local_user_data,
        save_local_user_data,
        _db_save_user_sync_data
    )
    import time

    if not app.session or not app.session.valid:
        print("❌ Please login first with 'tb login'")
        return Result.default_user_error("Not authenticated")

    username = app.session.username

    print(f"Syncing data for {username}...")

    # Load local data
    local_data = load_local_user_data(app.session.clerk_user_id)
    if not local_data:
        print("❌ No local data to sync")
        return Result.default_user_error("No local data")

    # Update sync timestamp
    local_data.last_sync = time.time()

    # Save locally
    save_local_user_data(local_data)

    # Sync to database
    _db_save_user_sync_data(app, local_data.clerk_user_id, local_data.to_dict())

    print("✅ Data synchronized successfully")
    return Result.ok()


# Legacy compatibility - keep old function names working

@export(mod_name=Name, name="create-user", test=False)
def create_user(app: App, username: str = None, email: str = None):
    """
    [DEPRECATED] Users are created via Clerk web registration.
    Use the web interface at /web/assets/signup.html
    """
    print("⚠️  Direct user creation is deprecated with Clerk integration.")
    print()
    print("To create a new user:")
    print("  1. Open: http://localhost:8080/web/assets/signup.html")
    print("  2. Register with email")
    print("  3. Verify via email code")
    print()
    print("For CLI access after web registration:")
    print("  tb login")

    return Result.ok("Use web registration at /web/assets/signup.html")


@export(mod_name=Name, name="create-invitation", test=False)
def create_invitation(app: App, username: str = None):
    """
    [DEPRECATED] Invitations are not needed with Clerk.
    Users register directly via the web interface.
    """
    print("⚠️  Invitations are not needed with Clerk integration.")
    print()
    print("Users can register directly at:")
    print("  http://localhost:8080/web/assets/signup.html")
    print("  https://simplecore.app/web/assets/signup.html")

    return Result.ok("Direct registration available at /web/assets/signup.html")


@export(mod_name=Name, name="send-magic-link", test=False)
def send_magic_link(app: App, username: str = None):
    """
    [DEPRECATED] Magic links are replaced with email codes in Clerk Free Tier.
    Use 'tb login' for CLI authentication.
    """
    print("⚠️  Magic links are replaced with email code verification.")
    print()
    print("For CLI login:")
    print("  tb login")
    print()
    print("You will receive a 6-digit code via email.")

    return Result.ok("Use 'tb login' for email code verification")
