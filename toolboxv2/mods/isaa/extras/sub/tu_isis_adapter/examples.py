"""
Usage Examples for TU ISIS Adapter

This file demonstrates various use cases for the TUIsisAdapter class.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from tu_isis_adapter import (
    TUIsisAdapter,
    quick_login,
    AuthenticationError,
    TUIsisAdapterError
)


# ============================================================================
# Example 1: Basic Authentication and Session Persistence
# ============================================================================

async def example_basic_login():
    """
    Example: Basic login with username and password
    The session is automatically saved for future use.
    """
    print("=== Example 1: Basic Login ===")
    
    async with TUIsisAdapter(headless=False) as adapter:
        await adapter.create_context()
        
        # Replace with your TU Berlin credentials
        username = "your_username"  # TU account (e.g., ab12345)
        password = "your_password"
        
        try:
            success = await adapter.login(username, password)
            if success:
                print(f"✓ Successfully logged in as: {adapter.current_user}")
                print(f"✓ Session saved for future use")
            else:
                print("✗ Login failed")
        except AuthenticationError as e:
            print(f"✗ Authentication error: {e}")


async def example_login_with_saved_state():
    """
    Example: Login using previously saved session state
    No credentials needed if a valid state exists.
    """
    print("=== Example 2: Login with Saved State ===")
    
    async with TUIsisAdapter(headless=True) as adapter:
        await adapter.create_context()
        
        # Try to load saved state (no credentials needed)
        success = await adapter.login()  # No parameters - tries to load state
        
        if success:
            print(f"✓ Successfully logged in using saved state")
            print(f"✓ User: {adapter.current_user}")
        else:
            print("✗ No valid saved state found - need credentials for initial login")


async def example_quick_login():
    """
    Example: Quick login using convenience function
    """
    print("=== Example 3: Quick Login ===")
    
    username = "your_username"
    password = "your_password"
    
    try:
        adapter = await quick_login(username, password, headless=True)
        print(f"✓ Logged in via quick_login()")
        
        # Use the adapter...
        # courses = await adapter.get_courses()
        
        # Don't forget to cleanup when done
        await adapter.teardown()
        
    except AuthenticationError as e:
        print(f"✗ Quick login failed: {e}")


# ============================================================================
# Example 2: Course Discovery
# ============================================================================

async def example_get_all_courses():
    """
    Example: Retrieve all accessible courses
    """
    print("=== Example 4: Get All Courses ===")
    
    async with TUIsisAdapter(headless=True) as adapter:
        await adapter.create_context()
        
        # Login (use saved state or credentials)
        # success = await adapter.login("username", "password")
        if not await adapter._load_saved_state():
            print("Please authenticate first")
            return
        
        try:
            courses = await adapter.get_courses()
            print(f"✓ Found {len(courses)} accessible courses")
            
            # Display first few courses
            for i, course in enumerate(courses[:5], 1):
                print(f"  {i}. {course.name} (ID: {course.course_id})")
                print(f"     URL: {course.url}")
            
        except TUIsisAdapterError as e:
            print(f"✗ Error: {e}")


async def example_get_course_details():
    """
    Example: Get detailed information about a specific course
    """
    print("=== Example 5: Get Course Details ===")
    
    async with TUIsisAdapter(headless=True) as adapter:
        await adapter.create_context()
        
        if not await adapter._load_saved_state():
            print("Please authenticate first")
            return
        
        course_id = "12345"  # Replace with actual course ID
        
        try:
            details = await adapter.get_course_details(course_id)
            print(f"✓ Course: {details.get('name')}")
            print(f"  Instructor: {details.get('instructor')}")
            print(f"  Semester: {details.get('semester')}")
            print(f"  Description: {details.get('description', '')[:100]}...")
            
        except TUIsisAdapterError as e:
            print(f"✗ Error: {e}")


# ============================================================================
# Example 3: Search Functionality
# ============================================================================

async def example_search_courses():
    """
    Example: Search for courses by name
    """
    print("=== Example 6: Search Courses ===")
    
    async with TUIsisAdapter(headless=True) as adapter:
        await adapter.create_context()
        
        if not await adapter._load_saved_state():
            print("Please authenticate first")
            return
        
        query = "Machine Learning"
        
        try:
            results = await adapter.search_courses(query)
            print(f"✓ Found {len(results)} results for '{query}'")
            
            for i, result in enumerate(results[:10], 1):
                print(f"  {i}. {result.get('name')} (ID: {result.get('course_id')})")
                
        except TUIsisAdapterError as e:
            print(f"✗ Error: {e}")


async def example_search_with_filters():
    """
    Example: Search courses with filters
    """
    print("=== Example 7: Search with Filters ===")
    
    async with TUIsisAdapter(headless=True) as adapter:
        await adapter.create_context()
        
        if not await adapter._load_saved_state():
            print("Please authenticate first")
            return
        
        query = "Data Science"
        filters = {
            'semester': '2024Wi',  # Winter 2024
            'department': 'Informatik'  # Computer Science
        }
        
        try:
            results = await adapter.search_courses(query, filters)
            print(f"✓ Found {len(results)} results for '{query}' with filters")
            
            for i, result in enumerate(results[:10], 1):
                print(f"  {i}. {result.get('name')}")
                
        except TUIsisAdapterError as e:
            print(f"✗ Error: {e}")


# ============================================================================
# Example 4: Channel/Forum Management
# ============================================================================

async def example_get_channels():
    """
    Example: Retrieve all communication channels/forums
    """
    print("=== Example 8: Get Channels ===")
    
    async with TUIsisAdapter(headless=True) as adapter:
        await adapter.create_context()
        
        if not await adapter._load_saved_state():
            print("Please authenticate first")
            return
        
        try:
            channels = await adapter.get_channels()
            print(f"✓ Found {len(channels)} channels")
            
            for i, channel in enumerate(channels[:10], 1):
                print(f"  {i}. {channel.name} (ID: {channel.channel_id})")
                
        except TUIsisAdapterError as e:
            print(f"✗ Error: {e}")


async def example_get_channel_updates():
    """
    Example: Get recent posts/updates from a channel
    """
    print("=== Example 9: Get Channel Updates ===")
    
    async with TUIsisAdapter(headless=True) as adapter:
        await adapter.create_context()
        
        if not await adapter._load_saved_state():
            print("Please authenticate first")
            return
        
        channel_id = "12345"  # Replace with actual channel ID
        
        try:
            # Get all updates
            posts = await adapter.get_channel_updates(channel_id)
            print(f"✓ Found {len(posts)} posts")
            
            # Get only recent updates (last 7 days)
            week_ago = datetime.now() - timedelta(days=7)
            recent_posts = await adapter.get_channel_updates(channel_id, week_ago)
            print(f"✓ Found {len(recent_posts)} posts in the last 7 days")
            
            for post in recent_posts[:5]:
                print(f"  - {post.get('title')}")
                print(f"    Author: {post.get('author')}")
                print(f"    Time: {post.get('timestamp')}")
                if post.get('attachments'):
                    print(f"    Attachments: {len(post['attachments'])}")
                
        except TUIsisAdapterError as e:
            print(f"✗ Error: {e}")


# ============================================================================
# Example 5: Download Functionality
# ============================================================================

async def example_download_course_materials():
    """
    Example: Download all materials from a course
    """
    print("=== Example 10: Download Course Materials ===")
    
    async with TUIsisAdapter(headless=True) as adapter:
        await adapter.create_context()
        
        if not await adapter._load_saved_state():
            print("Please authenticate first")
            return
        
        course_id = "12345"  # Replace with actual course ID
        output_dir = "my_course_downloads"
        
        try:
            downloaded = await adapter.download_course_content(course_id, output_dir)
            print(f"✓ Downloaded {len(downloaded)} files")
            print(f"  Location: {output_dir}")
            
            for file_path in downloaded[:10]:
                print(f"  - {file_path}")
                
        except TUIsisAdapterError as e:
            print(f"✗ Error: {e}")


async def example_download_channel_attachments():
    """
    Example: Download attachments from channel posts
    """
    print("=== Example 11: Download Channel Attachments ===")
    
    async with TUIsisAdapter(headless=True) as adapter:
        await adapter.create_context()
        
        if not await adapter._load_saved_state():
            print("Please authenticate first")
            return
        
        channel_id = "12345"  # Replace with actual channel ID
        
        try:
            # Download all attachments
            all_downloaded = await adapter.download_channel_attachments(channel_id)
            print(f"✓ Downloaded {len(all_downloaded)} attachments")
            
            # Download only recent attachments (last 30 days)
            month_ago = datetime.now() - timedelta(days=30)
            recent_downloaded = await adapter.download_channel_attachments(channel_id, month_ago)
            print(f"✓ Downloaded {len(recent_downloaded)} recent attachments")
            
        except TUIsisAdapterError as e:
            print(f"✗ Error: {e}")


# ============================================================================
# Example 6: Workflow - Complete Course Backup
# ============================================================================

async def example_complete_course_backup():
    """
    Example: Complete workflow - backup course materials and keep updated
    """
    print("=== Example 12: Complete Course Backup ===")
    
    async with TUIsisAdapter(headless=True) as adapter:
        await adapter.create_context()
        
        if not await adapter._load_saved_state():
            print("Please authenticate first")
            return
        
        # Step 1: Get all courses
        courses = await adapter.get_courses()
        print(f"✓ Found {len(courses)} courses to backup")
        
        backup_results = []
        
        # Step 2: Backup each course
        for course in courses[:5]:  # Limit to first 5 for demo
            print(f"\nBacking up: {course.name}")
            
            try:
                # Download course materials
                course_dir = f"backups/{course.course_id}"
                files = await adapter.download_course_content(course.course_id, course_dir)
                
                backup_results.append({
                    'course_id': course.course_id,
                    'name': course.name,
                    'files_downloaded': len(files),
                    'status': 'success'
                })
                
                print(f"  ✓ Downloaded {len(files)} files")
                
            except Exception as e:
                backup_results.append({
                    'course_id': course.course_id,
                    'name': course.name,
                    'error': str(e),
                    'status': 'failed'
                })
                print(f"  ✗ Failed: {e}")
        
        # Step 3: Summary
        print("\n=== Backup Summary ===")
        successful = sum(1 for r in backup_results if r['status'] == 'success')
        failed = len(backup_results) - successful
        
        print(f"Total: {len(backup_results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")


# ============================================================================
# Example 7: Workflow - Monitor Channel for Updates
# ============================================================================

async def example_monitor_channel():
    """
    Example: Monitor a channel for new posts and download attachments
    """
    print("=== Example 13: Monitor Channel ===")
    
    async with TUIsisAdapter(headless=True) as adapter:
        await adapter.create_context()
        
        if not await adapter._load_saved_state():
            print("Please authenticate first")
            return
        
        channel_id = "12345"  # Replace with actual channel ID
        check_interval_hours = 24
        
        # Track last checked time
        last_check = datetime.now() - timedelta(days=7)  # Start with 7 days ago
        
        print(f"Monitoring channel {channel_id} for updates...")
        print(f"Check interval: {check_interval_hours} hours")
        print(f"Starting from: {last_check}")
        
        # Simulate monitoring cycle
        for i in range(3):  # Check 3 times for demo
            print(f"\n--- Check {i+1} at {datetime.now()} ---")
            
            try:
                # Get new posts
                new_posts = await adapter.get_channel_updates(channel_id, last_check)
                
                if new_posts:
                    print(f"✓ Found {len(new_posts)} new posts")
                    
                    for post in new_posts:
                        print(f"  - {post['title']}")
                        
                        # Download attachments if any
                        if post['attachments']:
                            print(f"    Downloading {len(post['attachments'])} attachments...")
                            for url in post['attachments']:
                                # Implementation would use _download_file
                                print(f"      - Downloaded: {url}")
                    
                    # Update last check time
                    last_check = datetime.now()
                else:
                    print("✓ No new posts")
                
            except Exception as e:
                print(f"✗ Error during check: {e}")
            
            # In real monitoring, you'd sleep here:
            # await asyncio.sleep(check_interval_hours * 3600)
            print(f"Waiting {check_interval_hours} hours until next check...")
            await asyncio.sleep(1)  # Demo delay


# ============================================================================
# Example 8: Error Handling
# ============================================================================

async def example_error_handling():
    """
    Example: Proper error handling for various scenarios
    """
    print("=== Example 14: Error Handling ===")
    
    async with TUIsisAdapter(headless=True) as adapter:
        await adapter.create_context()
        
        # Handle authentication errors
        try:
            success = await adapter.login("invalid_user", "invalid_pass")
            if not success:
                print("⚠ Login failed - may need to use valid credentials")
        except AuthenticationError as e:
            print(f"✗ Authentication error: {e}")
        except Exception as e:
            print(f"✗ Unexpected error during login: {e}")
        
        # Handle network errors
        try:
            # This might fail if not authenticated
            courses = await adapter.get_courses()
        except AuthenticationError:
            print("⚠ Need to authenticate first")
        except TUIsisAdapterError as e:
            print(f"✗ Adapter error: {e}")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
        
        print("✓ Error handling example complete")


# ============================================================================
# Example 9: Advanced Configuration
# ============================================================================

async def example_advanced_config():
    """
    Example: Advanced adapter configuration
    """
    print("=== Example 15: Advanced Configuration ===")
    
    # Custom configuration
    adapter = TUIsisAdapter(
        browser_type='firefox',  # Use Firefox instead of Chromium
        headless=False,          # Show browser for debugging
        state_dir='custom_states',  # Custom state directory
        downloads_dir='custom_downloads',  # Custom downloads directory
        log_level=logging.DEBUG  # Verbose logging
    )
    
    async with adapter:
        await adapter.create_context(
            viewport={'width': 1920, 'height': 1080},  # Full HD viewport
            user_agent='Mozilla/5.0 (Custom TU ISIS Agent)'  # Custom user agent
        )
        
        print("✓ Adapter configured with:")
        print(f"  - Browser: {adapter.browser_type}")
        print(f"  - Headless: {adapter.headless}")
        print(f"  - State dir: {adapter.state_dir}")
        print(f"  - Downloads dir: {adapter.downloads_dir}")
        
        # Note: You would need to authenticate before using


# ============================================================================
# Example 10: Working with Data
# ============================================================================

async def example_data_analysis():
    """
    Example: Analyze course and channel data
    """
    print("=== Example 16: Data Analysis ===")
    
    async with TUIsisAdapter(headless=True) as adapter:
        await adapter.create_context()
        
        if not await adapter._load_saved_state():
            print("Please authenticate first")
            return
        
        # Collect data
        courses = await adapter.get_courses()
        channels = await adapter.get_channels()
        
        # Analyze courses
        print(f"\n--- Course Analysis ---")
        print(f"Total courses: {len(courses)}")
        
        # Group by potential department (based on name patterns)
        departments = {}
        for course in courses:
            dept = course.name.split()[0] if course.name.split() else "Other"
            departments[dept] = departments.get(dept, 0) + 1
        
        print(f"Departments: {len(departments)}")
        for dept, count in sorted(departments.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {dept}: {count} courses")
        
        # Analyze channels
        print(f"\n--- Channel Analysis ---")
        print(f"Total channels: {len(channels)}")
        
        # Check channels with recent activity
        channel_id = channels[0].channel_id if channels else None
        if channel_id:
            try:
                posts = await adapter.get_channel_updates(channel_id)
                print(f"Sample channel ({channels[0].name}): {len(posts)} posts")
            except:
                print("Could not retrieve channel posts")


# ============================================================================
# Main Runner
# ============================================================================

async def run_all_examples():
    """Run all examples (note: most require authentication)"""
    print("=" * 60)
    print("TU ISIS Adapter - Usage Examples")
    print("=" * 60)
    print()
    print("Note: Most examples require authentication.")
    print("Replace 'your_username' and 'your_password' with real credentials.")
    print()
    
    examples = [
        ("Error Handling", example_error_handling),
        ("Advanced Configuration", example_advanced_config),
        # Other examples require actual authentication
        # Uncomment after providing credentials:
        # ("Basic Login", example_basic_login),
        # ("Quick Login", example_quick_login),
        # ("Get All Courses", example_get_all_courses),
        # ("Search Courses", example_search_courses),
        # ("Get Channels", example_get_channels),
        # ("Download Course Materials", example_download_course_materials),
    ]
    
    for name, example_func in examples:
        print(f"\nRunning: {name}")
        print("-" * 40)
        try:
            await example_func()
        except Exception as e:
            print(f"✗ Example failed: {e}")
        print()
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    # Run a specific example:
    # asyncio.run(example_basic_login())
    # asyncio.run(example_search_courses())
    
    # Or run all examples:
    asyncio.run(run_all_examples())
