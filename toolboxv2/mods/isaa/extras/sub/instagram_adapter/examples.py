"""
Instagram Adapter - Usage Examples

This file contains practical examples for using the InstagramAdapter module.
"""

import asyncio
from datetime import datetime, timedelta
from instagram_adapter import InstagramAdapter, quick_download_saved_videos, quick_download_new_videos


# ============================================================================
# Example 1: Basic Download - All Saved Videos
# ============================================================================

async def example1_download_all_videos():
    """
    Download all saved videos from your Instagram account.
    """
    print("=" * 60)
    print("Example 1: Download All Saved Videos")
    print("=" * 60)
    
    async with InstagramAdapter(
        username="your_username",
        password="your_password",
        output_dir="downloads/all_videos"
    ) as adapter:
        # Login to Instagram
        await adapter.login()
        
        # Download all saved videos
        results = await adapter.download_all_saved_videos()
        
        print(f"\nDownload Summary:")
        print(f"  ✓ Success: {results['success']}")
        print(f"  ✗ Failed: {results['failed']}")
        print(f"  ⊘ Skipped: {results['skipped']}")
        print(f"\nVideos saved to: downloads/all_videos/videos/")


# ============================================================================
# Example 2: Login with 2FA
# ============================================================================

async def example2_login_with_2fa():
    """
    Login to Instagram with two-factor authentication.
    """
    print("=" * 60)
    print("Example 2: Login with 2FA")
    print("=" * 60)
    
    async with InstagramAdapter(
        username="your_username",
        password="your_password",
        headless=False  # Set False to see the browser for debugging
    ) as adapter:
        # Provide your 2FA code
        await adapter.login(two_factor_code="123456")
        
        print("Logged in successfully with 2FA!")


# ============================================================================
# Example 3: Quick Start - One-Liner Downloads
# ============================================================================

async def example3_quick_download():
    """
    Quick download using convenience functions.
    """
    print("=" * 60)
    print("Example 3: Quick Download")
    print("=" * 60)
    
    # Download all videos in one function call
    results = await quick_download_saved_videos(
        username="your_username",
        password="your_password",
        output_dir="downloads/quick"
    )
    
    print(f"Downloaded {results['success']} videos")
    
    # Download only new videos since last check
    results = await quick_download_new_videos(
        username="your_username",
        password="your_password",
        output_dir="downloads/quick"
    )
    
    print(f"New videos: {results['success']}")


# ============================================================================
# Example 4: Incremental Downloads (Weekly)
# ============================================================================

async def example4_weekly_download():
    """
    Download only videos saved since the last check.
    Suitable for cron jobs and automation.
    """
    print("=" * 60)
    print("Example 4: Weekly Incremental Download")
    print("=" * 60)
    
    async with InstagramAdapter(
        username="your_username",
        password="your_password",
        output_dir="downloads/weekly"
    ) as adapter:
        await adapter.login()
        
        # Run weekly download
        results = await adapter.run_weekly_download()
        
        print(f"\nWeekly Download Summary:")
        print(f"  Since: {results['since']}")
        print(f"  New videos: {results['success']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Completed at: {results['completed_at']}")


# ============================================================================
# Example 5: Get Specific Post Details
# ============================================================================

async def example5_get_post_details():
    """
    Extract detailed information from a specific Instagram post.
    """
    print("=" * 60)
    print("Example 5: Get Post Details")
    print("=" * 60)
    
    async with InstagramAdapter("your_username", "your_password") as adapter:
        await adapter.login()
        
        # Get details for a specific post
        post_url = "https://www.instagram.com/p/ABC123XYZ/"
        details = await adapter.get_post_details(post_url)
        
        print("\nPost Details:")
        print(f"  Video URL: {details.get('video_url')}")
        print(f"  Thumbnail: {details.get('thumbnail_url')}")
        print(f"  Caption: {details.get('caption', 'N/A')[:50]}...")
        print(f"  Author: {details.get('author')}")
        print(f"  Likes: {details.get('likes', 'N/A')}")


# ============================================================================
# Example 6: Download Specific Videos
# ============================================================================

async def example6_download_specific_videos():
    """
    Download videos from specific posts.
    """
    print("=" * 60)
    print("Example 6: Download Specific Videos")
    print("=" * 60)
    
    async with InstagramAdapter("your_username", "your_password") as adapter:
        await adapter.login()
        
        # List of posts to download
        post_urls = [
            "https://www.instagram.com/p/POST1/",
            "https://www.instagram.com/p/POST2/",
            "https://www.instagram.com/p/POST3/",
        ]
        
        for url in post_urls:
            print(f"\nProcessing: {url}")
            try:
                # Get post details
                details = await adapter.get_post_details(url)
                
                # Download video
                if details.get('video_url'):
                    filepath = await adapter.download_video(details)
                    print(f"  ✓ Downloaded: {filepath}")
                else:
                    print(f"  ⊘ No video found")
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")


# ============================================================================
# Example 7: Get New Videos Since Custom Date
# ============================================================================

async def example7_custom_date_range():
    """
    Download videos saved after a specific date.
    """
    print("=" * 60)
    print("Example 7: Videos Since Custom Date")
    print("=" * 60)
    
    async with InstagramAdapter("your_username", "your_password") as adapter:
        await adapter.login()
        
        # Get videos saved in the last 30 days
        last_check = datetime.now() - timedelta(days=30)
        print(f"Fetching videos since: {last_check}")
        
        new_videos = await adapter.get_new_videos_since(last_check)
        
        print(f"\nFound {len(new_videos)} new videos:")
        for video in new_videos:
            print(f"  - {video.get('author', 'unknown')}: {video.get('caption', 'N/A')[:40]}...")
            
            # Download the video
            try:
                filepath = await adapter.download_video(video)
                print(f"    ✓ Saved to: {filepath}")
            except Exception as e:
                print(f"    ✗ Error: {e}")


# ============================================================================
# Example 8: Cron Job Script
# ============================================================================

async def example8_cron_job():
    """
    Script suitable for cron job automation.
    
    To schedule this script, add to crontab:
        0 2 * * 0 /usr/bin/python3 path/to/this_script.py >> /var/log/insta_download.log 2>&1
    """
    print("=" * 60)
    print("Example 8: Cron Job Automation")
    print("=" * 60)
    
    async with InstagramAdapter(
        username="your_username",
        password="your_password",
        output_dir="/path/to/downloads",
        headless=True
    ) as adapter:
        try:
            # Login (will use saved state if available)
            await adapter.login()
            
            # Download new videos only
            results = await adapter.run_weekly_download()
            
            print(f"[{datetime.now()}] Download complete")
            print(f"  Success: {results['success']}")
            print(f"  Failed: {results['failed']}")
            
        except Exception as e:
            print(f"[{datetime.now()}] ERROR: {e}")
            raise


# ============================================================================
# Example 9: Error Handling
# ============================================================================

async def example9_error_handling():
    """
    Proper error handling for Instagram adapter.
    """
    print("=" * 60)
    print("Example 9: Error Handling")
    print("=" * 60)
    
    from instagram_adapter import (
        LoginError,
        RateLimitError,
        VideoDownloadError,
        InstagramAdapterError
    )
    
    try:
        async with InstagramAdapter("username", "password") as adapter:
            await adapter.login()
            results = await adapter.download_all_saved_videos()
            print(f"Downloaded {results['success']} videos")
            
    except LoginError as e:
        print(f"Login failed: {e}")
        print("Check credentials and try again.")
        
    except RateLimitError as e:
        print(f"Rate limited: {e}")
        print("Wait 24-48 hours before retrying.")
        
    except VideoDownloadError as e:
        print(f"Download error: {e}")
        print("Some videos may have been skipped.")
        
    except InstagramAdapterError as e:
        print(f"General error: {e}")
        
    except Exception as e:
        print(f"Unexpected error: {e}")


# ============================================================================
# Example 10: Get Saved Posts List Only
# ============================================================================

async def example10_list_saved_posts():
    """
    Get a list of saved posts without downloading.
    """
    print("=" * 60)
    print("Example 10: List Saved Posts")
    print("=" * 60)
    
    async with InstagramAdapter("your_username", "your_password") as adapter:
        await adapter.login()
        
        # Get list of saved posts
        posts = await adapter.get_saved_posts()
        
        print(f"\nFound {len(posts)} saved posts:")
        for i, post in enumerate(posts[:10], 1):
            print(f"  {i}. {post['url']}")
            print(f"     Collected: {post['collected_at']}")
        
        if len(posts) > 10:
            print(f"  ... and {len(posts) - 10} more")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Run a specific example
    examples = {
        "1": example1_download_all_videos,
        "2": example2_login_with_2fa,
        "3": example3_quick_download,
        "4": example4_weekly_download,
        "5": example5_get_post_details,
        "6": example6_download_specific_videos,
        "7": example7_custom_date_range,
        "8": example8_cron_job,
        "9": example9_error_handling,
        "10": example10_list_saved_posts,
    }
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in examples:
        asyncio.run(examples[sys.argv[1]]())
    else:
        print("Usage: python examples.py <example_number>")
        print("Available examples:")
        for num, func in examples.items():
            print(f"  {num}: {func.__doc__.strip()}")
