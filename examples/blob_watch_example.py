#!/usr/bin/env python3
"""
Example: Advanced Blob Watching with Callbacks

This example demonstrates the new watch functionality:
- Non-blocking background watching
- Automatic callback execution
- Multiple watches with different timeouts
- Batch updates (multiple watches in one thread)
- Auto-cleanup after idle timeout
"""

import time
import json
from toolboxv2.utils.extras.blobs import BlobStorage, BlobFile


def example_1_simple_watch():
    """Example 1: Simple blob watching with callback"""
    print("\n=== Example 1: Simple Watch ===")
    
    # Initialize storage
    storage = BlobStorage(
        servers=['http://localhost:3000'],
        storage_directory='./.data/blob_cache'
    )
    
    # Define callback
    def on_config_change(blob_file: BlobFile):
        """Called when config.json changes"""
        try:
            config = blob_file.read_json()
            print(f"✅ Config updated: {config}")
        except Exception as e:
            print(f"❌ Error reading config: {e}")
    
    # Start watching
    storage.watch(
        blob_id='app/config.json',
        callback=on_config_change,
        max_idle_timeout=600  # 10 minutes
    )
    
    print("👀 Watching 'app/config.json' for changes...")
    print("   Update the blob to see the callback in action!")
    print("   Callback will auto-remove after 10 minutes without updates.")
    
    # Keep running for demo
    try:
        time.sleep(60)  # Wait 1 minute
    except KeyboardInterrupt:
        print("\n⏹️  Stopping watch...")
    finally:
        storage.stop_all_watches()


def example_2_multiple_watches():
    """Example 2: Watch multiple blobs simultaneously"""
    print("\n=== Example 2: Multiple Watches ===")
    
    storage = BlobStorage(
        servers=['http://localhost:3000'],
        storage_directory='./.data/blob_cache'
    )
    
    # Callback for config changes
    def on_config_change(blob_file: BlobFile):
        config = blob_file.read_json()
        print(f"📝 Config changed: {config.get('version', 'unknown')}")
    
    # Callback for data changes
    def on_data_change(blob_file: BlobFile):
        data = blob_file.read_json()
        print(f"📊 Data changed: {len(data)} items")
    
    # Callback for logs
    def on_log_change(blob_file: BlobFile):
        logs = blob_file.read_text()
        lines = logs.split('\n')
        print(f"📋 Logs updated: {len(lines)} lines")
    
    # Watch multiple blobs (all in one thread!)
    storage.watch('app/config.json', on_config_change, max_idle_timeout=300)
    storage.watch('app/data.json', on_data_change, max_idle_timeout=300)
    storage.watch('app/logs.txt', on_log_change, max_idle_timeout=300)
    
    print("👀 Watching 3 blobs simultaneously...")
    print("   All watches run in a single background thread!")
    
    try:
        time.sleep(120)  # Wait 2 minutes
    except KeyboardInterrupt:
        print("\n⏹️  Stopping all watches...")
    finally:
        storage.stop_all_watches()


def example_3_blobfile_watch():
    """Example 3: Using BlobFile.watch() convenience method"""
    print("\n=== Example 3: BlobFile Watch ===")
    
    storage = BlobStorage(
        servers=['http://localhost:3000'],
        storage_directory='./.data/blob_cache'
    )
    
    # Create BlobFile instance
    config_file = BlobFile('app/config.json', 'r', storage=storage)
    
    # Define callback
    def on_change(blob_file: BlobFile):
        print(f"🔔 Config file changed!")
        config = blob_file.read_json()
        print(f"   New config: {json.dumps(config, indent=2)}")
    
    # Watch using BlobFile method
    config_file.watch(on_change, max_idle_timeout=600)
    
    print("👀 Watching config file...")
    
    try:
        time.sleep(60)
    except KeyboardInterrupt:
        print("\n⏹️  Stopping watch...")
    finally:
        config_file.stop_watch()


def example_4_auto_cleanup():
    """Example 4: Demonstrate auto-cleanup after idle timeout"""
    print("\n=== Example 4: Auto-Cleanup Demo ===")
    
    storage = BlobStorage(
        servers=['http://localhost:3000'],
        storage_directory='./.data/blob_cache'
    )
    
    callback_called = {'count': 0}
    
    def on_change(blob_file: BlobFile):
        callback_called['count'] += 1
        print(f"🔔 Callback #{callback_called['count']} triggered")
    
    # Watch with short timeout for demo
    storage.watch(
        blob_id='test/data.json',
        callback=on_change,
        max_idle_timeout=30  # 30 seconds for demo
    )
    
    print("👀 Watching 'test/data.json' with 30s idle timeout...")
    print("   If no updates occur within 30s, callback will be auto-removed")
    print("   Update the blob within 30s to keep the watch active!")
    
    try:
        # Wait 60 seconds to see auto-cleanup
        for i in range(60):
            time.sleep(1)
            if i == 30:
                print("⏰ 30 seconds passed - callback should be removed soon...")
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted")
    finally:
        storage.stop_all_watches()


if __name__ == '__main__':
    print("🚀 Blob Watch Examples")
    print("=" * 50)
    
    # Run examples
    # Uncomment the example you want to run:
    
    example_1_simple_watch()
    # example_2_multiple_watches()
    # example_3_blobfile_watch()
    # example_4_auto_cleanup()

