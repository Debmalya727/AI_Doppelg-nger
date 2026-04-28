from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import os

# Suppress console warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Load the CLIP model ONCE when the app starts ---
MODEL_ID = "openai/clip-vit-base-patch32"
print(f"[INFO] Loading CLIP model: {MODEL_ID}...")
try:
    # Use GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    print(f"[INFO] CLIP model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Could not load CLIP model: {e}")
    model = None
    processor = None
    device = "cpu" # Default to CPU on error
# -----------------------------------------------------------

def get_clip_embedding(image_path):
    """
    Generates a CLIP embedding vector for the entire image.
    Returns a numpy array or None if failed.
    """
    if not model or not processor:
        print("[WARNING] CLIP model not loaded. Cannot generate embedding.")
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        
        # Process the image and run it through the CLIP model
        with torch.no_grad():
            inputs = processor(text=None, images=image, return_tensors="pt", padding=True).to(device)
            image_features = model.get_image_features(**inputs)
            
        # Move embedding to CPU and convert to numpy array
        embedding = image_features.cpu().numpy().flatten()
        return embedding

    except Exception as e:
        print(f"[ERROR] Could not process image for CLIP embedding ({image_path}): {e}")
        return None

def calculate_clip_similarity(embedding1, embedding2):
    """
    Calculates cosine similarity between two CLIP embedding vectors.
    Assumes embeddings are numpy arrays.
    """
    if embedding1 is None or embedding2 is None:
        return 0.0
        
    # Normalize vectors before dot product for cosine similarity
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0 # Avoid division by zero
        
    # Cosine similarity is dot product of normalized vectors
    similarity = np.dot(embedding1 / norm1, embedding2 / norm2)
    
    # Clip similarity to be between 0 and 1 (sometimes float errors cause > 1)
    return np.clip(similarity, 0.0, 1.0)
