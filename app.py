import atexit
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import json
import shutil
import pickle
import numpy as np

# --- Core Matching ---
# Import necessary functions from matcher. find_all_matches uses get_augmented_embeddings internally.
from utils.matcher import clean_temp_folder, find_all_matches, annotate_uploaded_image, cosine_similarity 
# --- AI Detection ---
from utils.ai_detector import get_ai_detection_score
# --- Text Similarity ---
from utils.text_similarity import get_text_similarity
# --- Image Similarity (CLIP) ---
from utils.image_similarity import get_clip_embedding, calculate_clip_similarity
# --- Database ---
from utils.db import get_db_connection
# --- Instagram Scraping ---
from utils.insta_scraper import (
    fetch_instagram_profile,
    load_instaloader_with_cookies,
    fetch_recent_post_images, 
    cleanup_temp_posts        
)
# --- Clustering ---
from utils.clusterer import get_all_embeddings_and_filenames, cluster_faces, organize_clusters
from sklearn.cluster import DBSCAN # Required for cluster_faces if not imported there


UPLOAD_FOLDER = 'static/uploads'
MATCH_FOLDER = 'static/matches'
DATASET_FOLDER = 'faces_dataset' # Path relative to app.py

app = Flask(__name__)
# Allow uploads up to 16 megabytes
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MATCH_FOLDER'] = MATCH_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# --- Load Metadata ---
try:
    # Ensure UTF-8 encoding is used
    with open('dataset_metadata.json', 'r', encoding='utf-8') as f:
        DATASET_METADATA = json.load(f)
except FileNotFoundError:
    print("[WARNING] dataset_metadata.json not found. No details will be shown.")
    DATASET_METADATA = {}
except json.JSONDecodeError as e:
    print(f"[ERROR] Failed to decode dataset_metadata.json: {e}")
    DATASET_METADATA = {} # Use empty dict on error


# --- Ensure static folders exist ---
# Use os.path.join for cross-platform compatibility
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MATCH_FOLDER, exist_ok=True)
os.makedirs(os.path.join('static', 'insta_profiles'), exist_ok=True) 
os.makedirs(os.path.join('static', 'temp_posts'), exist_ok=True) 
os.makedirs(os.path.join('static', 'temp_enhanced'), exist_ok=True) # Ensure this exists too


# --- Routes ---

@app.route('/')
def index():
    """Renders the main homepage."""
    # Pass flag to template to conditionally show cluster link
    return render_template('index.html', show_cluster_link=True)

@app.route('/upload', methods=['POST'])
def upload():
    """Handles file or webcam upload for face matching against the local dataset."""
    filename = None
    filepath = None
    user_image_url = None # Initialize

    try:
        # Check if webcam image was submitted (base64-encoded PNG)
        if 'webcam_image' in request.form and request.form['webcam_image'].startswith("data:image"):
            base64_data = request.form['webcam_image'].split(',')[1]
            image_data = base64.b64decode(base64_data)
            image = Image.open(BytesIO(image_data))
            filename = "webcam_capture.png" # Secure filename not strictly needed here
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)
            print(f"[INFO] Webcam image saved to: {filepath}")
        # Check if file was uploaded
        elif 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return 'No selected file'
            # Secure the filename before saving
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"[INFO] File image saved to: {filepath}")
        else:
            return 'No file part found in request'

        # Ensure filepath is set before proceeding
        if not filepath or not os.path.exists(filepath):
             return f"Error saving uploaded file: {filename}"

        # Construct web path correctly with leading slash
        user_image_url = f"/{UPLOAD_FOLDER}/{filename}".replace("\\", "/")

        # --- AI Check ---
        print(f"[INFO] Running AI detection on: {filepath}")
        ai_score = get_ai_detection_score(filepath)
        threshold_ai = 0.95 # AI detection threshold

        if ai_score > threshold_ai:
            print(f"[INFO] AI score {ai_score:.4f} exceeds threshold {threshold_ai}. Flagging as AI-generated.")
            return render_template(
                'fake_result.html',
                user_image=user_image_url
            )
        else:
            print(f"[INFO] AI score {ai_score:.4f} is below threshold. Proceeding.")

        print("[INFO] Image passed check. Proceeding with face matching...")
        # find_all_matches uses the database via updated matcher.py
        matches = find_all_matches(filepath, DATASET_FOLDER) 

        match_images = []
        for match in matches:
            match_filename = match['filename']
            confidence = match['confidence']
            details = DATASET_METADATA.get(match_filename, {})

            match_images.append({
                'url': match['web_path'], # find_all_matches returns web_path starting with '/'
                'confidence': confidence,
                'name': details.get('name', 'N/A'),
                'bio': details.get('bio', 'No details available.')
            })

        # Annotate the uploaded image AFTER matching
        print(f"[INFO] Annotating uploaded image: {filepath}")
        annotated_img_path_rel = annotate_uploaded_image(filepath) # Returns relative path like 'static/annotated_uploads/...'
        annotated_img_url = f"/{annotated_img_path_rel}".replace("\\", "/") # Ensure leading slash for web

        return render_template(
            'result.html',
            user_image=user_image_url,
            annotated_image=annotated_img_url,
            matches=match_images,
            ai_score=f"{ai_score:.4f}"
        )

    except Exception as e:
        print(f"[ERROR] Exception during upload/matching: {e}")
        # Consider logging the full traceback here for debugging
        # import traceback
        # traceback.print_exc()
        return f"An error occurred: {e}", 500
    finally:
        # Clean up temporary enhanced images regardless of success/failure
        clean_temp_folder()


# --- Cleanup functions ---
@atexit.register
def cleanup_on_exit():
    """Register cleanup functions to run when the app exits gracefully."""
    print("[INFO] Running cleanup on exit...")
    clean_temp_folder() # Enhanced images used by matcher
    cleanup_temp_posts() # Downloaded post images
    # Clean up match summary CSV
    csv_path = os.path.join("static", "match_log.csv")
    if os.path.exists(csv_path):
        try:
            os.remove(csv_path)
            print("[INFO] Removed match_log.csv")
        except OSError as e:
            print(f"[ERROR] Could not remove match_log.csv: {e}")
    # Clean up uploaded files? Optional, maybe keep for debugging.
    # upload_dir = app.config['UPLOAD_FOLDER']
    # if os.path.exists(upload_dir):
    #     print(f"[INFO] Cleaning up upload directory: {upload_dir}")
    #     shutil.rmtree(upload_dir, ignore_errors=True)
    #     os.makedirs(upload_dir, exist_ok=True) # Recreate empty dir
        
    print("[INFO] Cleanup finished.")


@app.route('/search_instagram')
def search_instagram_form():
    """Renders the form to search for an Instagram profile."""
    # Clean up any leftover post downloads from a previous search *before* rendering the form
    cleanup_temp_posts()
    return render_template('search_instagram.html')

# --- ENHANCED INSTAGRAM SEARCH ROUTE ---
@app.route('/process_instagram_search', methods=['POST'])
def process_instagram_search():
    """Handles Instagram profile search, scrapes data, performs hybrid matching."""
    target_username = request.form.get('username')
    if not target_username:
        return "Please enter an Instagram username.", 400
        
    MAX_POSTS_TO_CHECK = 5 # Limit number of posts to scrape/check

    # --- 1. Fetch Profile Data & Load Instaloader ---
    # Use a try...finally block to ensure cleanup happens
    try:
        # Replace 'panda.debmalya' with the actual username used to generate cookies, or load dynamically
        logged_in_user = "panda.debmalya" 
        loader, _ = load_instaloader_with_cookies("instagram_cookies.json", logged_in_user) 
        if not loader: return "❌ Login to Instagram failed. Cookies might be invalid or expired. Run 'run_local_login_server.py' again."

        profile_data = fetch_instagram_profile(loader, target_username, logged_in_user) 
        if not profile_data: return f"❌ Could not fetch profile metadata for: @{target_username}. It might be private or non-existent."

        profile_pic_rel_path = profile_data.get('profile_pic') # Should start with '/'
        profile_pic_abs_path = None
        if profile_pic_rel_path:
             profile_pic_abs_path = profile_pic_rel_path.lstrip('/') # Get FS path (e.g., static/insta_profiles/...)
             if not os.path.exists(profile_pic_abs_path):
                 print(f"[WARNING] Profile picture file specified but not found at: {profile_pic_abs_path}")
                 profile_pic_abs_path = None # Treat as if no profile pic was downloaded

        # --- 2. AI Check (Profile Pic Only - skip if no pic) ---
        ai_score = 0.0 # Default score if no pic
        if profile_pic_abs_path:
            print(f"[INFO] Running AI detection on profile picture: {profile_pic_abs_path}")
            ai_score = get_ai_detection_score(profile_pic_abs_path)
            threshold_ai = 0.70
            if ai_score > threshold_ai:
                print(f"[INFO] Profile picture AI score {ai_score:.4f} exceeds threshold {threshold_ai}. Flagging.")
                # No need to cleanup posts here as they haven't been downloaded yet
                return render_template('fake_result.html', user_image=profile_data.get('profile_pic', ''))
            else:
                 print(f"[INFO] Profile picture AI score {ai_score:.4f} is below threshold.")
        else:
             print("[INFO] No profile picture found or downloaded. Skipping AI check on profile pic.")


        # --- 3. Fetch Recent Post Images ---
        print(f"[INFO] Fetching up to {MAX_POSTS_TO_CHECK} recent post images for @{target_username}...")
        # fetch_recent_post_images returns RELATIVE web paths (e.g., /static/temp_posts/...)
        post_image_web_paths = fetch_recent_post_images(loader, target_username, max_posts=MAX_POSTS_TO_CHECK)
        # Convert web paths back to absolute filesystem paths for local processing
        post_image_abs_paths = [p.lstrip('/') for p in post_image_web_paths if p] # Ensure p is not None
        print(f"[INFO] Found {len(post_image_abs_paths)} post images locally.")

        # --- 4. Process Profile Pic + Post Images for Matches ---
        all_potential_face_matches = {} # Dict to store best face match per dataset file {filename: {confidence, web_path}}
        all_potential_clip_matches = {} # Dict to store best clip match per dataset file {filename: {confidence, web_path}}

        # Combine profile pic path (if exists) and post image paths into one list for processing
        images_to_process = []
        if profile_pic_abs_path:
             images_to_process.append(profile_pic_abs_path)
        images_to_process.extend(post_image_abs_paths)

        # Check if there are any images at all to process
        if not images_to_process:
             # cleanup_temp_posts() # Cleanup happens in finally block
             return f"❌ No profile picture or recent post images were successfully downloaded for @{target_username}. Cannot perform matching."

        print(f"[INFO] Processing {len(images_to_process)} images for matches (profile pic + posts)...")

        # --- Fetch DB CLIP embeddings once before the loop ---
        db_clip = get_db_connection()
        all_db_clip_embeddings = [] # List of tuples: (filename, deserialized_embedding)
        if db_clip:
            cursor_clip = db_clip.cursor()
            try:
                cursor_clip.execute("SELECT filename, clip_embedding FROM clip_embeddings")
                db_results = cursor_clip.fetchall()
                print(f"[INFO] Fetched {len(db_results)} CLIP embeddings from DB.")
                # Deserialize embeddings here to avoid doing it repeatedly in the loop
                for db_filename, db_serialized_embedding in db_results:
                     try:
                         db_clip_embedding = pickle.loads(db_serialized_embedding)
                         all_db_clip_embeddings.append((db_filename, db_clip_embedding))
                     except (pickle.UnpicklingError, TypeError, ValueError) as load_err:
                          print(f"[ERROR] Could not deserialize DB CLIP embedding for {db_filename}: {load_err}")

            except Exception as db_err:
                 print(f"[ERROR] Failed fetching/deserializing CLIP embeddings: {db_err}")
            finally:
                cursor_clip.close()
                db_clip.close()
        else:
            print("[ERROR] Database connection failed for CLIP search. Cannot perform visual similarity.")
        # --- End fetch DB CLIP ---

        # --- Loop through each downloaded image (profile pic + posts) ---
        for i, image_path in enumerate(images_to_process):
            print(f"\n--- Processing image {i+1}/{len(images_to_process)}: {os.path.basename(image_path)} ---")
            if not os.path.exists(image_path):
                print(f"[WARNING] Image path not found during processing loop, skipping: {image_path}")
                continue

            # === 4a. Face Matching for this image ===
            print("[INFO] Performing Face Matching...")
            # find_all_matches uses the database, needs dataset_folder for constructing web_path if needed
            current_face_matches = find_all_matches(image_path, DATASET_FOLDER) 
            print(f"[INFO] Found {len(current_face_matches)} potential face matches for this image.")
            # Aggregate the best match found so far for each dataset person
            for match in current_face_matches:
                filename = match['filename']
                confidence = match['confidence']
                if filename not in all_potential_face_matches or confidence > all_potential_face_matches[filename]['confidence']:
                    all_potential_face_matches[filename] = {'confidence': confidence, 'web_path': match['web_path']}

            # === 4b. CLIP Matching for this image ===
            print("[INFO] Performing CLIP Visual Similarity Matching...")
            current_clip_embedding = get_clip_embedding(image_path)
            # Define CLIP threshold (can be adjusted)
            CLIP_SIMILARITY_THRESHOLD = 0.88 # Lowered this in previous step, confirm final value - using 0.88 from example

            if current_clip_embedding is not None and all_db_clip_embeddings:
                # Compare against pre-fetched & deserialized DB embeddings
                for db_filename, db_clip_embedding in all_db_clip_embeddings:
                    try:
                        similarity = calculate_clip_similarity(current_clip_embedding, db_clip_embedding)
                        similarity_percent = round(similarity * 100, 2)

                        if similarity > CLIP_SIMILARITY_THRESHOLD:
                            # Update if this is a better visual match for this dataset file
                            if db_filename not in all_potential_clip_matches or similarity_percent > all_potential_clip_matches[db_filename]['confidence']:
                                # Ensure the image file exists in static/matches for display
                                original_dataset_path = os.path.join(DATASET_FOLDER, db_filename)
                                match_dir = os.path.join('static', 'matches')
                                os.makedirs(match_dir, exist_ok=True)
                                dest_path = os.path.join(match_dir, db_filename)
                                web_path = f"/static/matches/{db_filename}" # Web path relative to static
                                
                                # Copy from dataset to matches if not already there
                                if not os.path.exists(dest_path) and os.path.exists(original_dataset_path):
                                     try:
                                          shutil.copyfile(original_dataset_path, dest_path)
                                     except IOError as copy_err:
                                          print(f"[ERROR] Could not copy match file {db_filename} to static/matches: {copy_err}")
                                          web_path = None # Indicate error?
                                elif not os.path.exists(original_dataset_path):
                                     print(f"[WARNING] Original dataset file missing for CLIP match, cannot display: {original_dataset_path}")
                                     web_path = None # Cannot display if original is gone

                                # Only add if we have a valid web_path
                                if web_path:
                                     all_potential_clip_matches[db_filename] = {'confidence': similarity_percent, 'web_path': web_path}

                    except Exception as comp_err:
                        # Catch potential errors during similarity calculation
                        print(f"[ERROR] Unexpected error during CLIP comparison for {db_filename}: {comp_err}")
                        
            elif current_clip_embedding is None:
                 print("[WARNING] Could not get CLIP embedding for current image, skipping visual match.")
            # End CLIP comparison loop for one DB entry
        # End loop for processing one image (profile or post)

        # --- 5. Consolidate and Sort Final Matches from aggregated results ---
        # Convert dictionaries to lists and sort by confidence
        final_face_matches = sorted(
            [{'filename': fn, **data} for fn, data in all_potential_face_matches.items()],
            key=lambda x: x['confidence'], reverse=True
        )
        final_clip_matches = sorted(
            [{'filename': fn, **data} for fn, data in all_potential_clip_matches.items() if data.get('web_path')], # Ensure web_path exists
            key=lambda x: x['confidence'], reverse=True
        )
        print(f"\n--- Aggregated Results ---")
        print(f"[INFO] Total unique face matches found across all images: {len(final_face_matches)}")
        print(f"[INFO] Total unique visual matches found across all images: {len(final_clip_matches)}")

        # --- 6. Text Similarity Alert (Based on BEST OVERALL FACE Match) ---
        alert = False
        name_sim_score = 0.0
        bio_sim_score = 0.0
        TEXT_SIMILARITY_THRESHOLD = 0.6 # Threshold for triggering alert

        if final_face_matches: # Check only if at least one face match was found
            best_face_match = final_face_matches[0] # The list is sorted, first is best
            matched_filename = best_face_match['filename']
            metadata = DATASET_METADATA.get(matched_filename)

            if metadata:
                real_name = metadata.get('name', '')
                real_bio = metadata.get('bio', '')
                # Get scraped data, ensuring keys exist
                scraped_name = profile_data.get('full_name', '')
                scraped_bio = profile_data.get('biography', '') # Key from insta_scraper

                name_sim_score = get_text_similarity(real_name, scraped_name)
                bio_sim_score = get_text_similarity(real_bio, scraped_bio)
                print(f"[DEBUG] Text Similarity (vs best face match {matched_filename}) -> Name: {name_sim_score:.4f}, Bio: {bio_sim_score:.4f}")

                # Trigger alert if BOTH name and bio similarity are high enough
                if name_sim_score > TEXT_SIMILARITY_THRESHOLD and bio_sim_score > TEXT_SIMILARITY_THRESHOLD:
                    alert = True
                    print("[INFO] Doppelgänger Alert triggered!")

        # --- 7. Render Template ---
        return render_template(
            "instagram_result.html",
            profile=profile_data,
            face_matches=final_face_matches, # Pass FINAL aggregated face matches
            clip_matches=final_clip_matches, # Pass FINAL aggregated clip matches
            alert=alert,
            ai_score=f"{ai_score:.4f}",
            name_sim = f"{name_sim_score:.4f}",
            bio_sim = f"{bio_sim_score:.4f}"
        )

    except Exception as e:
         print(f"[ERROR] An unexpected error occurred in /process_instagram_search: {e}")
         # Consider logging traceback here
         # import traceback
         # traceback.print_exc()
         # Return a generic error message to the user
         return f"An internal server error occurred processing the request for @{target_username}.", 500
         
    finally:
        # --- ★★★ Ensure temporary post images are ALWAYS cleaned up ★★★ ---
        cleanup_temp_posts()

# --- Cluster Analysis Route ---
# Make sure cluster imports are here
from utils.clusterer import get_all_embeddings_and_filenames, cluster_faces, organize_clusters
# from sklearn.cluster import DBSCAN # Should be imported in clusterer.py

@app.route('/analyze_clusters')
def analyze_clusters():
    """Fetches all embeddings, runs clustering, and displays the results."""
    try:
        # 1. Get Data
        filenames, embeddings_array = get_all_embeddings_and_filenames()

        if not filenames or embeddings_array is None or embeddings_array.size == 0:
            return "Error: Could not fetch embeddings from the database, no data found, or data format issue."

        # 2. Run Clustering
        # eps=0.70 is the distance threshold; adjust if needed
        labels = cluster_faces(embeddings_array, eps=0.70) 

        # 3. Organize Results
        clusters_dict = organize_clusters(filenames, labels)

        # --- ★★★ FIX: Copy images to static/matches for display ★★★ ---
        # The HTML looks for images in 'static/matches', but they are in 'faces_dataset'.
        # We verify and copy them so the browser can load them.
        for cluster_id, files in clusters_dict.items():
            for filename in files:
                src_path = os.path.join(DATASET_FOLDER, filename)
                dest_path = os.path.join(app.config['MATCH_FOLDER'], filename)

                # Check if the source exists
                if os.path.exists(src_path):
                    # Only copy if it's not already there to save time
                    if not os.path.exists(dest_path):
                        try:
                            shutil.copyfile(src_path, dest_path)
                        except IOError as e:
                            print(f"[WARNING] Could not copy {filename} for clustering display: {e}")
        # --- ★★★ END FIX ★★★ ---

        return render_template('clusters.html', clusters=clusters_dict)
        
    except Exception as e:
        print(f"[ERROR] An error occurred during cluster analysis: {e}")
        # import traceback
        # traceback.print_exc()
        return f"An internal server error occurred during cluster analysis.", 500

# --- Main execution ---
if __name__ == '__main__':
    # Models should be loaded when their respective utils/*.py files are imported
    print("[INFO] Starting Flask Application...")
    # debug=True enables auto-reloading and traceback pages
    # Use host='0.0.0.0' to make it accessible on your local network if needed
    app.run(debug=True, host='127.0.0.1', port=5000)

