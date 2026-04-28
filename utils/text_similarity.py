import spacy
import os

# Suppress console warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Load the model ONCE when the app starts ---
print("[INFO] Loading spaCy NLP model (en_core_web_md)...")
try:
    # Use en_core_web_md because it has word vectors for similarity
    nlp = spacy.load("en_core_web_md")
    print("[INFO] spaCy model loaded successfully.")
except OSError:
    print("[ERROR] Could not load spaCy model 'en_core_web_md'.")
    print("Please run: python -m spacy download en_core_web_md")
    nlp = None
# -------------------------------------------------

def get_text_similarity(text1, text2):
    """
    Calculates the semantic similarity between two texts
    using spaCy. Returns a score from 0.0 to 1.0.
    """
    if not nlp:
        print("[WARNING] spaCy model not loaded. Skipping text similarity.")
        return 0.0

    # Handle empty or None inputs
    if not text1 or not text2:
        return 0.0

    # Process the texts
    doc1 = nlp(text1.lower())
    doc2 = nlp(text2.lower())

    # Calculate and return the similarity score
    return doc1.similarity(doc2)