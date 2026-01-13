import requests
from bs4 import BeautifulSoup
from app.services.nlp.skill_extractor import extract_skills
from app.database.firestore import db

def scrape_job(url):
    text = BeautifulSoup(requests.get(url).text, "html.parser").get_text(" ")
    skills = extract_skills(text)

    db.collection("jobs").add({
        "title": url,
        "description": text,
        "skills": [s["skill"] for s in skills],
        "skill_clusters": [s["cluster_id"] for s in skills]
    })
