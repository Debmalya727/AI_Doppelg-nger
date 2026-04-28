# ingest_clip_dataset.py
import os
import pickle
import numpy as np
from utils.db import get_db_connection
from utils.image_similarity import get_clip_embedding # Import the new function

DATASET_FOLDER = 'faces_dataset'

def populate_clip_database():
    """
    Scans the dataset folder, generates CLIP embeddings,
    and stores them in the MySQL 'clip_embeddings' table.
    """
    db = get_db_connection()
    if not db:
        print("[ERROR] Cannot connect to database. Aborting CLIP ingestion.")
        return
    
    cursor = db.cursor()
    
    # Optional: Clear the table for a fresh start? Or update existing?
    # For simplicity, let's clear it. Add an 'UPDATE' logic later if needed.
    cursor.execute("TRUNCATE TABLE clip_embeddings")
    print("[INFO] Emptied 'clip_embeddings' table.")
    
    file_count = 0
    processed_files = set() # To avoid duplicates if script runs multiple times

    for file in os.listdir(DATASET_FOLDER):
        # Skip non-image files and duplicates
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')) or file in processed_files:
            continue

        print(f"Processing CLIP for: {file}...")
        dataset_path = os.path.join(DATASET_FOLDER, file)
        
        try:
            # Generate the CLIP embedding for the whole image
            clip_embedding_array = get_clip_embedding(dataset_path) 
            
            if clip_embedding_array is None:
                print(f"[WARNING] Could not get CLIP embedding for {file}, skipping.")
                continue

            # Serialize the numpy array into a binary object using pickle
            serialized_embedding = pickle.dumps(clip_embedding_array)
            
            # Insert into the database
            sql = "INSERT INTO clip_embeddings (filename, clip_embedding) VALUES (%s, %s)"
            cursor.execute(sql, (file, serialized_embedding))
            processed_files.add(file)
            file_count += 1

        except Exception as e:
            print(f"[ERROR] Failed to process CLIP for {file}: {e}")

    db.commit()
    cursor.close()
    db.close()
    
    print(f"\n✅ [SUCCESS] CLIP Ingestion complete.")
    print(f"Total images processed and stored in 'clip_embeddings': {file_count}")

if __name__ == "__main__":
    populate_clip_database()
