#!/usr/bin/env python3
"""
Example: Blob Sharing Between Users

This example demonstrates the new sharing functionality:
- Public User IDs based on device name
- Read-only and read-write access levels
- Secure sharing (one party cannot have both keys)
- Hash ring consistency (same server for all operations)
"""

import json
from toolboxv2.utils.extras.blobs import BlobStorage, BlobFile


def example_1_basic_sharing():
    """Example 1: Basic blob sharing with read-only access"""
    print("\n=== Example 1: Basic Sharing (Read-Only) ===")
    
    # User A: Create storage and blob
    storage_a = BlobStorage(
        servers=['http://localhost:3000'],
        storage_directory='./.data/user_a_cache'
    )
    
    # Get User A's public ID
    user_a_id = storage_a.get_user_id()
    print(f"👤 User A ID: {user_a_id}")
    
    # Create a config file
    with BlobFile('shared/config.json', 'w', storage=storage_a) as f:
        f.write_json({
            'app_name': 'MyApp',
            'version': '1.0.0',
            'settings': {
                'theme': 'dark',
                'language': 'en'
            }
        })
    
    print("✅ User A created config file")
    
    # User B: Create storage (different device/user)
    storage_b = BlobStorage(
        servers=['http://localhost:3000'],
        storage_directory='./.data/user_b_cache'
    )
    
    user_b_id = storage_b.get_user_id()
    print(f"👤 User B ID: {user_b_id}")
    
    # User A shares config with User B (read-only)
    share_result = storage_a.share_blob(
        blob_id='shared/config.json',
        target_user_id=user_b_id,
        access_level='read_only'
    )
    
    print(f"🔗 User A shared config with User B: {share_result}")
    
    # User B can now read the config
    with BlobFile('shared/config.json', 'r', storage=storage_b) as f:
        config = f.read_json()
        print(f"📖 User B read config: {config}")
    
    # User B tries to write (should fail with 403 Forbidden)
    try:
        with BlobFile('shared/config.json', 'w', storage=storage_b) as f:
            f.write_json({'modified': True})
        print("❌ ERROR: User B should not be able to write!")
    except Exception as e:
        print(f"✅ User B cannot write (expected): {e}")


def example_2_readwrite_sharing():
    """Example 2: Sharing with read-write access"""
    print("\n=== Example 2: Sharing (Read-Write) ===")
    
    storage_a = BlobStorage(
        servers=['http://localhost:3000'],
        storage_directory='./.data/user_a_cache'
    )
    
    storage_b = BlobStorage(
        servers=['http://localhost:3000'],
        storage_directory='./.data/user_b_cache'
    )
    
    user_a_id = storage_a.get_user_id()
    user_b_id = storage_b.get_user_id()
    
    # User A creates a collaborative document
    with BlobFile('collab/document.json', 'w', storage=storage_a) as f:
        f.write_json({
            'title': 'Collaborative Document',
            'content': 'Initial content by User A',
            'contributors': [user_a_id]
        })
    
    print("✅ User A created collaborative document")
    
    # User A shares with User B (read-write)
    storage_a.share_blob(
        blob_id='collab/document.json',
        target_user_id=user_b_id,
        access_level='read_write'
    )
    
    print(f"🔗 User A shared document with User B (read-write)")
    
    # User B can now read AND write
    with BlobFile('collab/document.json', 'r', storage=storage_b) as f:
        doc = f.read_json()
    
    doc['content'] += '\nEdited by User B'
    doc['contributors'].append(user_b_id)
    
    with BlobFile('collab/document.json', 'w', storage=storage_b) as f:
        f.write_json(doc)
    
    print("✅ User B successfully edited the document")
    
    # User A reads the updated document
    with BlobFile('collab/document.json', 'r', storage=storage_a) as f:
        updated_doc = f.read_json()
        print(f"📖 User A sees updates: {json.dumps(updated_doc, indent=2)}")


def example_3_list_and_revoke():
    """Example 3: List shares and revoke access"""
    print("\n=== Example 3: List and Revoke Shares ===")
    
    storage_a = BlobStorage(
        servers=['http://localhost:3000'],
        storage_directory='./.data/user_a_cache'
    )
    
    # List all shares for a blob
    shares = storage_a.list_shares('shared/config.json')
    
    print(f"📋 Current shares for 'shared/config.json':")
    for share in shares:
        print(f"   - {share['user_id']}: {share['access_level']}")
        print(f"     Granted by: {share['granted_by']}")
        print(f"     Granted at: {share['granted_at']}")
    
    # Revoke access
    if shares:
        user_to_revoke = shares[0]['user_id']
        storage_a.revoke_share('shared/config.json', user_to_revoke)
        print(f"🚫 Revoked access for user: {user_to_revoke}")
        
        # Verify revocation
        updated_shares = storage_a.list_shares('shared/config.json')
        print(f"📋 Updated shares: {len(updated_shares)} users")


def example_4_security():
    """Example 4: Security features"""
    print("\n=== Example 4: Security Features ===")
    
    storage = BlobStorage(
        servers=['http://localhost:3000'],
        storage_directory='./.data/user_a_cache'
    )
    
    user_id = storage.get_user_id()
    
    # Create a blob
    with BlobFile('secure/data.json', 'w', storage=storage) as f:
        f.write_json({'secret': 'data'})
    
    # Try to share with yourself (should fail)
    try:
        storage.share_blob('secure/data.json', user_id, 'read_only')
        print("❌ ERROR: Should not be able to share with yourself!")
    except Exception as e:
        print(f"✅ Cannot share with yourself (expected): {e}")
    
    # Try to share with non-existent user (should fail)
    try:
        storage.share_blob('secure/data.json', 'user_nonexistent', 'read_only')
        print("❌ ERROR: Should not be able to share with non-existent user!")
    except Exception as e:
        print(f"✅ Cannot share with non-existent user (expected): {e}")


if __name__ == '__main__':
    print("🚀 Blob Sharing Examples")
    print("=" * 50)
    
    # Run examples
    # Uncomment the example you want to run:
    
    example_1_basic_sharing()
    # example_2_readwrite_sharing()
    # example_3_list_and_revoke()
    # example_4_security()

