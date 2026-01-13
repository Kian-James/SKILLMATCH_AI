import re
from firebase_admin import firestore
from app.database.firestore import db
from app.services.nlp.skill_cluster import assign_cluster

def extract_phrases(text):
    return list(set(re.findall(r"\b[a-z][a-z ]{2,40}\b", text)))

def skill_confidence(phrase):
    doc = db.collection("global_skills").document(phrase).get()
    if not doc.exists:
        return 0.0
    data = doc.to_dict()
    return min((data.get("resume", 1) + data.get("job", 0)) / 20, 1)

def extract_skills(text):
    skills = []

    for phrase in extract_phrases(text):
        ref = db.collection("global_skills").document(phrase)
        ref.set({"resume": firestore.Increment(1)}, merge=True)

        confidence = skill_confidence(phrase)
        if confidence < 0.3:
            continue

        cluster_id = assign_cluster(phrase)
        skills.append({
            "skill": phrase,
            "cluster_id": cluster_id,
            "confidence": round(confidence, 2)
        })

    return skills
