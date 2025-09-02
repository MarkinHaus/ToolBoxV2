from toolboxv2 import TBEF, App, Result, get_app

# Define the module name and export function
Name = 'helper'
export = get_app(f"{Name}.Export").tb
version = "0.1.0"

@export(mod_name=Name, name="init_system", test=False)
async def init_system(app: App):
    """
    Initializes the ToolBoxV2 system by creating the first administrative user.
    This is an interactive command.
    """
    print("--- ToolBoxV2 System Initialization ---")
    print("This will guide you through creating the first administrator account.")
    print("This account will have the highest permission level.\n")

    try:
        username = input("Enter the administrator's username: ").strip()
        if not username:
            print("Username cannot be empty.")
            return Result.default_user_error("Username cannot be empty.")

        email = input(f"Enter the email for '{username}': ").strip()
        if not email: # A simple check, can be improved with regex
            print("Email cannot be empty.")
            return Result.default_user_error("Email cannot be empty.")

        print(f"\nCreating user '{username}' with email '{email}'...")
        # Call the internal function to create the account
        # The 'create=True' flag likely handles the initial key generation
        result = await app.a_run_any(TBEF.CLOUDM.REGISTER_INITIAL_LOOT_USER,
                                     user_name=username,
                                     email=email,
                                     get_results=True)

        if result.is_ok():
            print("\n✅ Administrator account created successfully!")
            print("   A new cryptographic key pair has been generated for this user.")
            print("   Authentication is handled automatically using these keys.")
            print("   You can now use other CLI commands or log into the web UI.")
            return Result.ok("System initialized successfully.")
        else:
            print("\n❌ Error creating administrator account:")
            result.print()
            return result

    except (KeyboardInterrupt, EOFError):
        print("\n\nInitialization cancelled by user.")
        return Result.default_user_error("Initialization cancelled.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return Result.default_internal_error(f"An unexpected error occurred: {e}")


@export(mod_name=Name, name="create-user", test=False)
def create_user(app: App, username: str, email: str):
    """Creates a new user with a generated key pair."""
    print(f"Creating user '{username}' with email '{email}'...")
    app.load_mod("CloudM")
    # Generate an invitation on the fly
    invitation_res = app.run_any(TBEF.CLOUDM_AUTHMANAGER.GET_INVITATION,
                                 get_results=True,
                                 username=username)
    if invitation_res.is_error():
        print("❌ Error creating invitation:")
        invitation_res.print()
        return invitation_res

    result = app.run_any(TBEF.CLOUDM_AUTHMANAGER.CRATE_LOCAL_ACCOUNT,
                         get_results=True,
                         username=username,
                         email=email,
                         invitation=invitation_res.get(),
                         create=True)

    if result.is_ok():
        print(f"✅ User '{username}' created successfully.")
    else:
        print("❌ Error creating user:")
        result.print()
    return result


@export(mod_name=Name, name="delete-user", test=False)
def delete_user_cli(app: App, username: str):
    """Deletes a user and all their associated data."""
    print(f"Attempting to delete user '{username}'...")
    app.load_mod("CloudM")
    result = app.run_any(TBEF.CLOUDM_AUTHMANAGER.DELETE_USER, get_results=True, username=username)

    if result.is_ok():
        print(f"✅ User '{username}' has been deleted.")
    else:
        print(f"❌ Error deleting user: {result.info.get('help_text')}")
    return result


@export(mod_name=Name, name="list-users", test=False)
def list_users_cli(app: App):
    """Lists all registered users."""
    print("Fetching user list...")
    app.load_mod("CloudM")
    result = app.run_any(TBEF.CLOUDM_AUTHMANAGER.LIST_USERS, get_results=True)

    if result.is_ok():
        users = result.get()
        if not users:
            print("No users found.")
            return result

        print("--- Registered Users ---")
        # Simple table formatting
        print(f"{'Username':<25} {'Email':<30} {'Level'}")
        print("------------------------")
        for user in users:
            print(f"{user['username']:<25} {user['email']:<30} {user['level']}")
        print("------------------------")
    else:
        print("❌ Error listing users:")
        result.print()

    return result

@export(mod_name=Name, name="create-invitation", test=False)
def create_invitation(app: App, username: str):
    """Creates a one-time invitation code for a user to link a new device."""
    print(f"Creating invitation for user '{username}'...")
    app.load_mod("CloudM")
    result = app.run_any(TBEF.CLOUDM_AUTHMANAGER.GET_INVITATION,
                         get_results=True,
                         username=username)

    if result.is_ok():
        print(f"✅ Invitation code for '{username}': {result.get()}")
    else:
        print("❌ Error creating invitation:")
        result.print()
    return result


@export(mod_name=Name, name="send-magic-link", test=False)
def send_magic_link(app: App, username: str):
    """Sends a magic login link to the user's registered email address."""
    print(f"Sending magic link to user '{username}'...")
    app.load_mod("CloudM")
    result = app.run_any(TBEF.CLOUDM_AUTHMANAGER.GET_MAGIC_LINK_EMAIL,
                         get_results=True,
                         username=username)

    if result.is_ok():
        print(f"✅ Magic link sent successfully to the email address associated with '{username}'.")
    else:
        print("❌ Error sending magic link:")
        result.print()
    return result
