# Account Management

This document provides a guide to managing user accounts in ToolBoxV2 through the command-line interface (CLI). These commands are available through the `helper` module.

## Initial System Setup

Before any other account management commands can be used, the system must be initialized. This is done with the `init_system` command, which creates the first administrative user.

### `init_system`

This command will launch an interactive prompt to guide you through creating the first user account. This user will have the highest level of permissions.

**Usage:**

```bash
tb -c helper init_system
```

The system will prompt you for a username and an email address. Upon successful creation, a new cryptographic key pair will be generated for the user, which will be used for authentication.

## User Management

These commands allow you to create, delete, and list users.

### `create-user`

Creates a new user.

**Usage:**

```bash
tb -c helper create-user <username> <email>
```

-   `<username>`: The desired username for the new user.
-   `<email>`: The email address for the new user.

### `delete-user`

Deletes a user and all associated data, including their cryptographic keys.

**Usage:**

```bash
tb -c helper delete-user <username>
```

-   `<username>`: The username of the user to delete.

### `list-users`

Displays a list of all registered users, including their username, email, and permission level.

**Usage:**

```bash
tb -c helper list-users
```

## Device and Access Management

These commands are used to manage how users can access their accounts.

### `create-invitation`

Generates a one-time invitation code that allows a user to link a new device to their account.

**Usage:**

```bash
tb -c helper create-invitation <username>
```

-   `<username>`: The username of the user for whom to create the invitation.

### `send-magic-link`

Sends a magic login link to the user's registered email address. This link can be used to log in without a password or key.

**Usage:**

```bash
tb -c helper send-magic-link <username>
```

-   `<username>`: The username of the user to whom the magic link should be sent.
