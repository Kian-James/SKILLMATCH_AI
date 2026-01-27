# backend/app/repositories/firestore/certifications.py
from app.repositories.firestore.client import db
from typing import List, Dict, Any

CERTIFICATIONS_COLLECTION = "certifications"
CERT_EMBEDDINGS_COLLECTION = "certification_embeddings"

class CertificationRepository:
    """
    Firestore access for certifications and learning paths
    """

    def get_all_certifications(self) -> List[Dict[str, Any]]:
        """
        Returns list of all certifications
        """
        docs = db.collection(CERTIFICATIONS_COLLECTION).stream()
        return [doc.to_dict() for doc in docs]

    def get_learning_resources(self, cert_name: str) -> List[Dict[str, str]]:
        """
        Returns learning resources for a specific certification
        Example resource dict: {"name": "AWS Certified Developer Guide", "url": "https://..."}
        """
        doc_ref = db.collection(CERTIFICATIONS_COLLECTION).document(cert_name.lower())
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            return data.get("learning_resources", [])
        return []
