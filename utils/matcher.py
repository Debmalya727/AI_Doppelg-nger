# utils/matcher.py
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
import shutil
import csv
import pickle  # <-- IMPORT PICKLE
from utils.db import get_db_connection  # <-- IMPORT DB CONNECTION
from utils.enhancer import enhance_image
from typing import List
from collections import defaultdict

# Initialize model (remains global for the main app)
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

def clean_temp_folder():
    temp_path = os.path.join("static", "temp_enhanced")
    if os.path.exists(temp_path):
        for f in os.listdir(temp_path):
            os.remove(os.path.join(temp_path, f))

def get_augmented_embeddings(image_path):
    """
    This function is now self-contained.
    It enhances, finds faces, and gets embeddings.
    """
    enhanced_path = enhance_image(image_path)
    img = cv2.imread(enhanced_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    embeddings = []

    # Angles + scaling to simulate variations
    transforms = [(-15, 1.0), (0, 1.0), (15, 1.0), (0, 0.95), (0, 1.05)]

    for angle, scale in transforms:
        M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, scale)
        transformed = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        
        # Use the globally defined 'app' model
        faces = app.get(transformed) 

        for face in faces:
            if face.embedding.shape == (512,):
                embeddings.append(face.embedding.copy())

    return embeddings

def cosine_similarity(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    if a.shape[0] != 512 or b.shape[0] != 512:
        return 0.0

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def annotate_uploaded_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot load uploaded image: {image_path}")
        return image_path

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb) # Uses global 'app'

    for i, face in enumerate(faces):
        box = face.bbox.astype(int)
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 20)
        label = f"Face {i + 1}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 5
        thickness = 5
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = y1 - 15 if y1 - 15 > 15 else y1 + text_height + 10
        label_x = x1
        cv2.rectangle(img, (label_x - 5, label_y - text_height - 10), (label_x + text_width + 5, label_y + 5), (0, 0, 0), -1)
        cv2.putText(img, label, (label_x, label_y), font, font_scale, (255, 255, 255), thickness)

    os.makedirs("static/annotated_uploads", exist_ok=True)
    filename = os.path.basename(image_path)
    output_path = os.path.join("static", "annotated_uploads", filename)
    cv2.imwrite(output_path, img)
    return output_path


def save_match_image_for_web(original_path):
    static_match_dir = os.path.join("static", "matches")
    os.makedirs(static_match_dir, exist_ok=True)
    filename = os.path.basename(original_path)
    dest_path = os.path.join(static_match_dir, filename)
    shutil.copyfile(original_path, dest_path)
    return f"/static/matches/{filename}"

# --- THIS IS THE MAINLY MODIFIED FUNCTION ---
def find_all_matches(uploaded_image_path, dataset_folder, threshold=0.30, same_person_threshold=0.45):
    clean_temp_folder()

    uploaded_embeddings = get_augmented_embeddings(uploaded_image_path)
    if not uploaded_embeddings:
        print("[ERROR] No faces found in uploaded image.")
        return []

    # Clean static/matches folder
    match_folder = os.path.join("static", "matches")
    if os.path.exists(match_folder):
        for f in os.listdir(match_folder):
            os.remove(os.path.join(match_folder, f))

    # CSV log setup
    csv_log_path = os.path.join("static", "match_log.csv")
    csv_rows = [["Face Index", "Matched File", "Similarity (%)"]]

    all_candidates = []  # temporary matches before filtering
    
    # --- NEW: Fetch all embeddings from MySQL Database ---
    print("[INFO] Fetching embeddings from MySQL...")
    db = get_db_connection()
    if not db:
        print("[ERROR] Database connection failed, cannot perform match.")
        return []
    
    cursor = db.cursor()
    cursor.execute("SELECT filename, embedding FROM faces_embeddings")
    all_dataset_faces = cursor.fetchall()
    cursor.close()
    db.close()
    print(f"[INFO] Loaded {len(all_dataset_faces)} embeddings from database.")
    # --- END OF NEW DB LOGIC ---

    # --- MODIFIED LOOP: Iterate over DB results, not file system ---
    for (filename, serialized_embedding) in all_dataset_faces:
        
        # Deserialize the list of embeddings from the BLOB
        try:
            dataset_embeddings = pickle.loads(serialized_embedding)
        except Exception as e:
            print(f"[ERROR] Could not deserialize embedding for {filename}: {e}")
            continue

        if not dataset_embeddings:
            print(f"[WARNING] No face found in dataset image: {filename}")
            continue
            
        dataset_path = os.path.join(dataset_folder, filename) # Still need this path

        for idx, upload_embedding in enumerate(uploaded_embeddings):
            best_sim = 0.0
            for data_embedding in dataset_embeddings:
                sim = cosine_similarity(upload_embedding, data_embedding)
                sim_percent = round(sim * 100, 2)
                if sim > best_sim:
                    best_sim = sim
            csv_rows.append([idx + 1, filename, round(best_sim * 100, 2)])

            if best_sim > threshold:
                all_candidates.append({
                    'filename': filename,
                    'confidence': round(best_sim * 100, 2),
                    'dataset_embedding': dataset_embeddings[0],  # first face assumed
                    'web_path': save_match_image_for_web(dataset_path)
                })
    # --- END OF MODIFIED LOOP ---

    # (Filtering logic is unchanged)
    filtered_matches = []
    used = set()

    for i in range(len(all_candidates)):
        if i in used:
            continue
        group = [all_candidates[i]]
        used.add(i)
        for j in range(i + 1, len(all_candidates)):
            if j in used:
                continue
            sim = cosine_similarity(
                all_candidates[i]['dataset_embedding'],
                all_candidates[j]['dataset_embedding']
            )
            if sim > same_person_threshold:
                group.append(all_candidates[j])
                used.add(j)

        best = max(group, key=lambda x: x['confidence'])
        filtered_matches.append({
            'filename': best['filename'],
            'confidence': best['confidence'],
            'web_path': best['web_path']
        })

    # Save CSV
    with open(csv_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    filtered_matches = sorted(filtered_matches, key=lambda x: x['confidence'], reverse=True)
    print(f"[RESULT] Unique persons matched: {len(filtered_matches)}")

    return filtered_matches