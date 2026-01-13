from firebase_admin import auth
from fastapi import Header, HTTPException

def verify_firebase_token(authorization: str = Header(...)):
    try:
        token = authorization.replace("Bearer ", "")
        decoded = auth.verify_id_token(token)
        return decoded["uid"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
