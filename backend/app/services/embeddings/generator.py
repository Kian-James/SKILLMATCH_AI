import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY missing")

class EmbeddingGenerator:
    def __init__(self):
        self.client = InferenceClient(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token=HF_API_KEY
        )

    def generate_text_embedding(self, text: str):
        if not text.strip():
            return None

        embedding = self.client.feature_extraction(text)
        return embedding
