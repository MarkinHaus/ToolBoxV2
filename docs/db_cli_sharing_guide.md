# DB CLI - Blob Sharing & Management Guide

## Overview

The DB CLI now includes full support for blob sharing, user management, and recovery features. This guide covers all the new interactive features available in the data discovery interface.

## Getting Started

Launch the interactive data discovery interface:

```bash
tb db discover
```

## Navigation

### Instance Selection
- **↑↓ or w/s**: Navigate between instances
- **Enter**: Select instance and view blobs
- **q**: Quit

**New Features:**
- 👤 **User ID Display**: Selected running instances show your Public User ID
- ✅/❌ **Status Icons**: Visual indication of running/stopped instances

### Blob List View
- **↑↓ or w/s**: Navigate between blobs
- **Enter**: View blob details
- **d**: Delete selected blob
- **r**: Refresh blob list
- **b**: Back to instance selection
- **q**: Quit

**New Features:**
- 📤 **Share Indicator**: Shows number of users blob is shared with (e.g., `📤3`)
- Displays blob ID, size, and share count

### Blob Detail View
- **e**: Export blob to file
- **d**: Delete blob
- **s**: Share blob with another user ⭐ NEW
- **v**: View all shares ⭐ NEW
- **u**: Show your User ID ⭐ NEW
- **b**: Back to blob list
- **q**: Quit

**New Features:**
- **Share Summary**: Shows first 3 users blob is shared with
- **Access Level Icons**: 
  - 👁️ Read-only access
  - ✏️ Read-write access

### Shares View ⭐ NEW
- **↑↓**: Navigate between shares
- **r**: Revoke selected share
- **b**: Back to blob detail
- **q**: Quit

Displays detailed information for each share:
- User ID
- Access level (read_only/read_write)
- Granted by (who shared it)
- Granted at (timestamp)

---

## Feature Guide

### 1. Viewing Your User ID

Your Public User ID is required for others to share blobs with you.

**Steps:**
1. Navigate to any blob detail view
2. Press **u**
3. Your User ID will be displayed
4. Share this ID with others who want to grant you access

**Example:**
```
╔════════════════════════════════════════╗
║        Your Public User ID       👤    ║
╠════════════════════════════════════════╣
║                                        ║
║  User ID: user_abc123def456789         ║
║                                        ║
║  ℹ Share this ID with others to        ║
║    receive blob access                 ║
║                                        ║
╚════════════════════════════════════════╝
```

---

### 2. Sharing a Blob

Share a blob with another user with specific access level.

**Steps:**
1. Navigate to blob detail view
2. Press **s** to share
3. Enter target User ID (e.g., `user_xyz789`)
4. Enter access level:
   - `read_only` (default) - User can only read
   - `read_write` - User can read and modify

**Example:**
```
╔════════════════════════════════════════╗
║           Share Blob           🔗      ║
╠════════════════════════════════════════╣
║                                        ║
║  Enter target User ID: user_xyz789     ║
║  Access level (read_only/read_write)   ║
║  [read_only]: read_write               ║
║                                        ║
║  ✅ Shared with user_xyz789            ║
║     (read_write)                       ║
║  ℹ Granted at: 1234567890              ║
║                                        ║
╚════════════════════════════════════════╝
```

**Access Levels:**
- **read_only**: User can download and view the blob
- **read_write**: User can download, view, and modify the blob

---

### 3. Viewing All Shares

See all users who have access to a blob.

**Steps:**
1. Navigate to blob detail view
2. Press **v** to view shares
3. Navigate with ↑↓
4. Press **r** to revoke selected share
5. Press **b** to go back

**Example:**
```
╔════════════════════════════════════════╗
║  Shares for: app/config.json     📤    ║
╠════════════════════════════════════════╣
║                                        ║
║  Total shares: 3                       ║
║                                        ║
║ ─────────────────────────────────────  ║
║  ▶ 👁️ user_abc123def456                ║
║     Access: read_only                  ║
║     Granted by: user_owner123          ║
║     Granted at: 2024-01-15 14:30:22    ║
║                                        ║
║    ✏️ user_xyz789abc123                ║
║     Access: read_write                 ║
║     Granted by: user_owner123          ║
║     Granted at: 2024-01-15 15:45:10    ║
║                                        ║
║ ─────────────────────────────────────  ║
║                                        ║
╚════════════════════════════════════════╝
```

---

### 4. Revoking Access

Remove a user's access to a blob.

**Steps:**
1. Navigate to blob detail view
2. Press **v** to view shares
3. Use ↑↓ to select the share to revoke
4. Press **r** to revoke
5. Type `yes` to confirm

**Example:**
```
⚠ Revoke access for user_abc123def456?
  Type 'yes' to confirm: yes

✅ Access revoked successfully
```

---

## Use Cases

### Use Case 1: Sharing Configuration Files

**Scenario:** Share a read-only config file with a team member

1. Create/upload config blob: `app/config.json`
2. Navigate to blob detail
3. Press **u** to get your User ID
4. Share your User ID with team member
5. Get team member's User ID
6. Press **s** to share
7. Enter team member's User ID
8. Enter `read_only` as access level

**Result:** Team member can read config but cannot modify it

---

### Use Case 2: Collaborative Editing

**Scenario:** Multiple users editing a shared document

1. Create document blob: `collab/document.json`
2. Get User IDs from all collaborators
3. For each collaborator:
   - Press **s** to share
   - Enter their User ID
   - Enter `read_write` as access level
4. Press **v** to verify all shares

**Result:** All collaborators can read and modify the document

---

### Use Case 3: Access Audit

**Scenario:** Review who has access to sensitive data

1. Navigate to sensitive blob
2. Press **v** to view all shares
3. Review each share:
   - Who has access (User ID)
   - What level (read_only/read_write)
   - When granted (timestamp)
   - Who granted it
4. Revoke unnecessary access with **r**

---

## Security Features

✅ **Server-Side Enforcement**: All access control is enforced by the Rust server  
✅ **Cannot Share with Yourself**: Prevented by server validation  
✅ **User ID System**: One party cannot have both keys  
✅ **Hash Ring Consistency**: All operations use the same server  
✅ **Audit Trail**: Track who granted access and when  

---

## Tips & Best Practices

1. **Share User IDs Securely**: User IDs are public identifiers, but share them through secure channels
2. **Use Read-Only by Default**: Only grant read-write when necessary
3. **Regular Audits**: Periodically review shares with **v** command
4. **Revoke Promptly**: Remove access immediately when no longer needed
5. **Document Shares**: Keep track of why access was granted

---

## Troubleshooting

### "BlobStorage not available"
- Ensure the instance is running
- Check that API key is properly initialized
- Restart the instance if needed

### "Failed to share: Target user does not exist"
- Verify the target User ID is correct
- Ensure the target user has created an API key (started their instance)

### "Failed to share: Cannot share with yourself"
- You entered your own User ID
- Get the correct User ID from the other user

### Share count not showing
- Shares may take a moment to load
- Press **r** to refresh the blob list
- Check server logs for errors

---

## Keyboard Reference

| Key | Action | Context |
|-----|--------|---------|
| ↑↓ | Navigate | All views |
| w/s | Navigate (alternative) | Instance & Blob list |
| Enter | Select/View | Instance & Blob list |
| **s** | **Share blob** | **Blob detail** |
| **v** | **View shares** | **Blob detail** |
| **u** | **Show User ID** | **Blob detail** |
| **r** | Revoke share / Refresh | Shares view / Blob list |
| e | Export blob | Blob detail |
| d | Delete blob | Blob detail & list |
| b | Back | All views |
| q | Quit | All views |

