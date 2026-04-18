import firebase_admin
from firebase_admin import credentials, firestore
import json

# Initialize Firebase
cred = credentials.Certificate("firebase_service_account.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

def export_firestore():
    data = {}

    for col in db.collections():
        data[col.id] = []
        for doc in col.stream():
            data[col.id].append({
                "id": doc.id,
                **doc.to_dict()
            })

    with open("firestore_export.json", "w") as f:
        json.dump(data, f, indent=2)

    print("✅ Export complete: firestore_export.json")

if __name__ == "__main__":
    export_firestore()