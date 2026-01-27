import json
import firebase_admin
from firebase_admin import credentials, firestore
import os

# ===== CONFIG =====
SERVICE_ACCOUNT_FILE = '../../skillmatch-ai-27c75-firebase-adminsdk-fbsvc-03e7fa5061.json'
OUTPUT_DIR = 'firestore_backup'  # Folder to save JSON files
COLLECTIONS = ['jobs']

# ===== INIT FIREBASE =====
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ===== EXPORT FUNCTION =====
def export_collection(collection_name):
    print(f'Exporting {collection_name}...')
    docs = db.collection(collection_name).stream()
    data = {doc.id: doc.to_dict() for doc in docs}
    file_path = os.path.join(OUTPUT_DIR, f'{collection_name}.json')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f'Saved {len(data)} documents to {file_path}')

# ===== RUN EXPORT =====
for coll in COLLECTIONS:
    export_collection(coll)

print('✅ Backup complete!')
