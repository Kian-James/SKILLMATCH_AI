import uuid
from firebase_admin import firestore
from sklearn.metrics.pairwise import cosine_similarity
from app.database.firestore import db
from app.services.nlp.embeddings import embed

THRESHOLD = 0.8

def assign_cluster(skill: str):
    vector = embed([skill])[0]

    for doc in db.collection("skill_clusters").stream():
        data = doc.to_dict()
        score = cosine_similarity([vector], [data["embedding"]])[0][0]
        if score >= THRESHOLD:
            db.collection("skill_clusters").document(doc.id).update({
                "skills": firestore.ArrayUnion([skill]),
                "usage": firestore.Increment(1)
            })
            return doc.id

    cluster_id = str(uuid.uuid4())
    db.collection("skill_clusters").document(cluster_id).set({
        "name": skill,
        "skills": [skill],
        "embedding": vector.tolist(),
        "usage": 1
    })
    return cluster_id
