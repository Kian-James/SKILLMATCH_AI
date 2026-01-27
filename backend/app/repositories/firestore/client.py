# backend/app/repositories/firestore/client.py
import os
import json
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

load_dotenv()

# Load JSON from env variable
service_account_json = os.getenv("FIREBASE_CREDENTIALS")
cred_dict = json.loads(service_account_json)

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()
