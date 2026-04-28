# ingest_dataset.py
import os
import pickle  # We'll use pickle to serialize the list of embeddings
from utils.db import get_db_connection
from utils.matcher import get_augmented_embeddings
from insightface.app import FaceAnalysis # We need this for the app.prepare

DATASET_FOLDER = 'faces_dataset'

def populate_database():
    """
    Scans the dataset folder, generates embeddings,
    and stores them in the MySQL database.
    """
    db = get_db_connection()
    if not db:
        print("[ERROR] Cannot connect to database. Aborting ingestion.")
        return
    
    cursor = db.cursor()
    
    # Clear the table for a fresh start
    cursor.execute("TRUNCATE TABLE faces_embeddings")
    print("[INFO] Emptied 'faces_embeddings' table.")
    
    # We must initialize the model here just for this script
    print("[INFO] Loading InsightFace model for ingestion...")
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    print("[INFO] Model loaded.")

    file_count = 0
    for file in os.listdir(DATASET_FOLDER):
        if "enhanced" in file.lower() or not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        print(f"Processing: {file}...")
        dataset_path = os.path.join(DATASET_FOLDER, file)
        
        # Get the list of augmented embeddings
        # NOTE: This uses the get_augmented_embeddings from matcher.py
        # which now needs the 'app' model passed to it.
        # We will modify matcher.py in the next step.
        # For now, let's assume a temporary function.
        
        # --- Temporary plan: Let's simplify matcher.py first.
        # This script needs to call the embedding function.
        # Let's use a simplified approach for this script.
        
        # We need to get the embeddings. Let's borrow from matcher.py
        try:
            # We call the *real* function from matcher.py
            embeddings_list = get_augmented_embeddings(dataset_path) 
            
            if not embeddings_list:
                print(f"[WARNING] No face found in {file}, skipping.")
                continue

            # Serialize the list of embeddings into a binary object
            serialized_embeddings = pickle.dumps(embeddings_list)
            
            # Insert into the database
            sql = "INSERT INTO faces_embeddings (filename, embedding) VALUES (%s, %s)"
            cursor.execute(sql, (file, serialized_embeddings))
            file_count += 1

        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")

    db.commit()
    cursor.close()
    db.close()
    
    print(f"\n✅ [SUCCESS] Ingestion complete.")
    print(f"Total images processed and stored: {file_count}")

if __name__ == "__main__":
    populate_database()