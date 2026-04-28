from transformers import AutoProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import os

# Suppress console warnings from Hugging Face
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# This is the standard model for AI image detection
MODEL_ID = "umm-maybe/AI-image-detector"

# --- These lines load the model ONCE when the app starts ---
print(f"[INFO] Loading AI detector model: {MODEL_ID}...")
try:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    print(f"[INFO] AI detector model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Could not load AI detector model: {e}")
    processor = None
    model = None
# -----------------------------------------------------------


def get_ai_detection_score(image_path):
    """
    Checks an image and returns its 'fake' score (0.0 to 1.0).
    """
    if not model or not processor:
        print("[WARNING] AI detector is not loaded. Skipping check.")
        return 0.0 # Return a score of 0

    try:
        # Open the image
        image = Image.open(image_path).convert("RGB")
        
        # Process the image and run it through the model
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            
        logits = outputs.logits
        
        # The model gives two scores: one for 'real' and one for 'fake'
        # We apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get the score for the 'fake' class
        # model.config.id2label[1] is 'fake' for this model
        fake_score = probs[0][1].item()
        
        print(f"[INFO] AI detection score: {fake_score:.4f}")

        # Return the raw score
        return fake_score

    except Exception as e:
        print(f"[ERROR] Could not process image for AI detection: {e}")
        return 0.0 # Return a score of 0