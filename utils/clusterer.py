import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from utils.db import get_db_connection
from collections import defaultdict

def get_all_embeddings_and_filenames():
    """
    Fetches all filenames and their embeddings from the database.
    Calculates the AVERAGE embedding for each image.
    Returns:
        - List of filenames.
        - NumPy array where each row is the average embedding for that filename.
    """
    print("[INFO] Fetching all embeddings from DB for clustering...")
    db = get_db_connection()
    if not db:
        print("[ERROR] Database connection failed for clustering.")
        return [], None
    
    cursor = db.cursor()
    cursor.execute("SELECT filename, embedding FROM faces_embeddings")
    all_dataset_faces = cursor.fetchall()
    cursor.close()
    db.close()

    filenames = []
    average_embeddings = []

    if not all_dataset_faces:
        print("[WARNING] No embeddings found in the database.")
        return [], None

    print(f"[INFO] Processing {len(all_dataset_faces)} entries for averaging...")
    for (filename, serialized_embedding) in all_dataset_faces:
        try:
            # Deserialize the list of embeddings
            embeddings_list = pickle.loads(serialized_embedding)
            
            if not embeddings_list:
                print(f"[WARNING] No embeddings data found for {filename}, skipping.")
                continue

            # --- ★★★ NEW: Calculate the average embedding ★★★ ---
            if len(embeddings_list) > 0:
                # Stack embeddings into a NumPy array and calculate the mean along axis 0
                avg_embedding = np.mean(np.array(embeddings_list), axis=0)
                
                # Normalize the average embedding (important for cosine distance)
                norm = np.linalg.norm(avg_embedding)
                if norm == 0: 
                    # Avoid division by zero if embedding is all zeros (unlikely)
                    print(f"[WARNING] Zero norm embedding for {filename}, skipping.")
                    continue 
                normalized_avg_embedding = avg_embedding / norm

                average_embeddings.append(normalized_avg_embedding)
                filenames.append(filename)
            else:
                 print(f"[WARNING] Empty embedding list for {filename}, skipping.")
            # --- ★★★ END OF NEW LOGIC ★★★ ---

        except Exception as e:
            print(f"[ERROR] Could not process embedding for {filename}: {e}")
            continue
            
    if not average_embeddings:
        print("[ERROR] No valid embeddings could be processed.")
        return [], None

    print(f"[INFO] Successfully processed {len(filenames)} embeddings for clustering.")
    return filenames, np.array(average_embeddings) # Return NumPy array


def cluster_faces(embeddings_array, eps=0.5, min_samples=2):
    """
    Performs DBSCAN clustering on the provided embeddings.
    Args:
        embeddings_array: NumPy array of embeddings.
        eps: The maximum distance (cosine distance) between samples for one to be considered as in the neighborhood of the other.
        min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    Returns:
        NumPy array of cluster labels (-1 means noise).
    """
    print(f"[INFO] Running DBSCAN clustering with eps={eps}, min_samples={min_samples}...")
    # Use cosine distance because embeddings are normalized
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine') 
    db.fit(embeddings_array)
    labels = db.labels_
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"[INFO] Clustering complete. Found {n_clusters} clusters and {n_noise} noise points.")
    
    return labels

def organize_clusters(filenames, labels):
    """
    Organizes filenames into a dictionary based on cluster labels.
    Args:
        filenames: List of filenames corresponding to the embeddings.
        labels: NumPy array of cluster labels from DBSCAN.
    Returns:
        Dictionary where keys are cluster IDs (0, 1, 2...) and values are lists of filenames in that cluster.
        Excludes noise points (label -1).
    """
    clusters = defaultdict(list)
    for filename, label in zip(filenames, labels):
        if label != -1:  # Ignore noise points
            clusters[label].append(filename)
    
    # Sort clusters by size (descending) for better display
    sorted_clusters = dict(sorted(clusters.items(), key=lambda item: len(item[1]), reverse=True))
    return sorted_clusters

