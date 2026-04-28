import json
import instaloader
import requests
import os
import shutil # Needed for cleanup
from datetime import datetime

# --- Constants ---
POST_DOWNLOAD_DIR = "static/temp_posts" # Temporary folder for post images

# --- load_instaloader_with_cookies (Unchanged) ---
def load_instaloader_with_cookies(cookie_file, username):
    # ... (Keep this function exactly as it was) ...
    """Load Instaloader with cookies."""
    L = instaloader.Instaloader(
        download_pictures=True, download_videos=False, download_video_thumbnails=False,
        download_geotags=False, download_comments=False, save_metadata=False,
        compress_json=False, filename_pattern="{date_utc}_UTC_{profile}" # Basic pattern
    )

    # Configure Instaloader to download into our temp dir FOR POSTS ONLY
    # We will handle profile pic download separately
    L.dirname_pattern = os.path.join(POST_DOWNLOAD_DIR, "{profile}")


    try:
        with open(cookie_file, "r") as f:
            cookies = json.load(f)
        
        # Import cookies into session
        for cookie in cookies:
             # Ensure required fields exist
            if 'name' in cookie and 'value' in cookie:
                L.context._session.cookies.set(
                    cookie['name'], cookie['value'],
                    domain=cookie.get('domain'), path=cookie.get('path', '/'),
                    secure=cookie.get('secure', False), expires=cookie.get('expiry') 
                )
                
        # Attempt to resume session using loaded cookies
        L.context.username = username # Set username for session context
        print(f"Attempting to verify login for user: {username}")
        # Verify login by fetching own profile
        profile = instaloader.Profile.from_username(L.context, username)
        if profile.userid:
             print(f"✅ Login session verified for: {profile.username}")
             return L, profile
        else:
             print("[ERROR] Cookie login failed verification.")
             return None, None
            
    except FileNotFoundError:
        print(f"[ERROR] Cookie file not found: {cookie_file}")
        return None, None
    except (instaloader.exceptions.ConnectionException, instaloader.exceptions.LoginRequiredException) as e:
        print(f"[ERROR] Login failed: {e}")
        return None, None
    except Exception as e:
        print(f"[ERROR] Unexpected error during login: {e}")
        return None, None

# --- fetch_instagram_profile (Modified to download profile pic manually) ---
def fetch_instagram_profile(loader, target_username, login_username):
    L = loader
    # login_profile = instaloader.Profile.from_username(L.context, login_username) # Not strictly needed here anymore

    if not L:
        return None

    print(f"[INFO] Fetching profile metadata for @{target_username}...")
    try:
        target = instaloader.Profile.from_username(L.context, target_username)
    except instaloader.exceptions.ProfileNotFoundError:
        print(f"[ERROR] Profile @{target_username} not found.")
        return None
    except instaloader.exceptions.ConnectionException as e:
        print(f"[ERROR] Connection error fetching profile @{target_username}: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to fetch profile metadata @{target_username}: {e}")
        return None

    # --- Manually Download Profile Picture ---
    profile_pic_path = None
    profile_pic_dir = os.path.join("static", "insta_profiles")
    os.makedirs(profile_pic_dir, exist_ok=True)
    
    if target.profile_pic_url:
        try:
            profile_pic_path_temp = os.path.join(profile_pic_dir, f"{target_username}_profile.jpg")
            img_data = requests.get(target.profile_pic_url, timeout=10).content
            with open(profile_pic_path_temp, 'wb') as handler:
                handler.write(img_data)
            profile_pic_path = profile_pic_path_temp # Assign only on success
            print(f"[INFO] Profile picture saved to: {profile_pic_path}")
        except requests.exceptions.RequestException as e:
            print(f"[WARNING] Failed to download profile picture URL {target.profile_pic_url}: {e}")
        except Exception as e:
             print(f"[WARNING] Failed to save profile picture: {e}")
    else:
        print(f"[WARNING] No profile picture URL found for @{target_username}")


    # --- (Mutual friends estimation removed for simplicity/reliability) ---

    return {
        "username": target.username,
        "full_name": target.full_name or "", # Ensure strings
        "biography": target.biography or "", # Ensure strings
        "profile_pic": f"/{profile_pic_path.replace(os.path.sep, '/')}" if profile_pic_path else None, # Use forward slash for web path
        # "mutual_friends": [] # Removed
    }

# --- ★★★ NEW FUNCTION ★★★ ---
def fetch_recent_post_images(loader, target_username, max_posts=12):
    """
    Downloads images from the most recent posts of a target profile.
    Returns a list of file paths to the downloaded images.
    """
    L = loader
    if not L:
        return []

    print(f"[INFO] Fetching recent posts for @{target_username}...")
    downloaded_files = []
    
    try:
        profile = instaloader.Profile.from_username(L.context, target_username)
        
        post_count = 0
        # Iterate through posts, newest first
        for post in profile.get_posts():
            if post_count >= max_posts:
                print(f"[INFO] Reached max_posts limit ({max_posts}).")
                break
                
            # Skip videos
            if post.is_video:
                print(f"[INFO] Skipping video post {post.shortcode}")
                continue

            print(f"[INFO] Downloading image(s) from post {post.shortcode}...")
            
            try:
                # Instaloader handles download based on L.dirname_pattern set during init
                # It might download multiple images if it's a carousel post
                success = L.download_post(post, profile.username) 
                
                if success:
                    post_count += 1
                    # Instaloader doesn't easily return the exact filenames,
                    # so we'll scan the directory it *should* have used.
                    # This is a bit fragile but necessary.
                    expected_dir = os.path.join(POST_DOWNLOAD_DIR, profile.username)
                    if os.path.exists(expected_dir):
                        # Find files matching the post's date (or close to it)
                        # Our filename pattern is {date_utc}_UTC_{profile}.jpg
                        # Example: 2025-10-27_14-30-00_UTC_someuser.jpg
                        post_datetime_str = post.date_utc.strftime("%Y-%m-%d_%H-%M-%S")
                        
                        for filename in os.listdir(expected_dir):
                             if filename.startswith(post_datetime_str) and filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                                file_path = os.path.join(expected_dir, filename)
                                # Check if added recently to avoid duplicates across runs if dir not cleaned
                                if os.path.getmtime(file_path) > (datetime.now().timestamp() - 60): # Added in last 60s
                                     downloaded_files.append(file_path)
                                     print(f"   -> Found downloaded image: {filename}")
                else:
                    print(f"[WARNING] Failed to download post {post.shortcode}")

            except (instaloader.exceptions.InstaloaderException, requests.exceptions.RequestException) as dl_err:
                 print(f"[ERROR] Error downloading post {post.shortcode}: {dl_err}")
            except Exception as e:
                print(f"[ERROR] Unexpected error processing post {post.shortcode}: {e}")

    except instaloader.exceptions.ProfileNotFoundError:
        print(f"[ERROR] Profile @{target_username} not found for fetching posts.")
        return []
    except instaloader.exceptions.PrivateProfileNotFollowedException:
         print(f"[WARNING] Profile @{target_username} is private or requires login. Cannot fetch posts.")
         return []
    except (instaloader.exceptions.ConnectionException, instaloader.exceptions.LoginRequiredException) as e:
        print(f"[ERROR] Connection or Login error fetching posts for @{target_username}: {e}")
        return []
    except Exception as e:
        print(f"[ERROR] Unexpected error fetching posts for @{target_username}: {e}")
        return []

    print(f"[INFO] Finished fetching posts. Downloaded images for {post_count} posts.")
    # Return relative paths suitable for web use
    relative_paths = [f"/{f.replace(os.path.sep, '/')}" for f in downloaded_files]
    return relative_paths


# --- ★★★ NEW FUNCTION ★★★ ---
def cleanup_temp_posts():
    """Removes the temporary post download directory."""
    if os.path.exists(POST_DOWNLOAD_DIR):
        print(f"[INFO] Cleaning up temporary post directory: {POST_DOWNLOAD_DIR}")
        shutil.rmtree(POST_DOWNLOAD_DIR, ignore_errors=True)


# --- Example Usage (Optional - Keep for testing) ---
if __name__ == "__main__":
    
    # --- Test fetch_instagram_profile ---
    print("\n--- Testing Profile Fetch ---")
    loader, profile = load_instaloader_with_cookies("instagram_cookies.json", "panda.debmalya") # Replace username
    if loader:
        target = "instagram" # Test with a known public profile
        result = fetch_instagram_profile(loader, target, "panda.debmalya") # Replace username
        if result:
            print(f"\n✅ Profile Data for @{result['username']}")
            print(f"Name: {result['full_name']}")
            print(f"Bio: {result['biography']}")
            print(f"Profile Picture Path: {result['profile_pic']}")
        else:
            print(f"❌ Failed to fetch profile data for {target}.")
            
        # --- Test fetch_recent_post_images ---
        print("\n--- Testing Post Fetch ---")
        cleanup_temp_posts() # Clean before test
        post_image_paths = fetch_recent_post_images(loader, target, max_posts=3)
        if post_image_paths:
            print(f"\n✅ Downloaded {len(post_image_paths)} post images:")
            for path in post_image_paths:
                print(f"   -> {path}")
            # cleanup_temp_posts() # Optionally clean up after test
        else:
            print(f"❌ Failed to download post images for {target}.")
    else:
        print("❌ Login failed, cannot run tests.")
