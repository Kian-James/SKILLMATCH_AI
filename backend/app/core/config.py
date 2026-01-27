from app.core.constants import DEFAULT_ALGORITHM

# Main configuration class
class Config:
    # Algorithm selection: "sbert" or "knn"
    MATCHING_ALGORITHM: str = DEFAULT_ALGORITHM

    # Hugging Face models
    SBERT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    OCR_MODEL_NAME: str = "microsoft/trocr-base-handwritten"

    # Firestore info
    FIRESTORE_PROJECT_ID: str = "your-project-id"

    # Other global settings
    DEBUG: bool = True
