from fastapi import APIRouter, Depends
from app.core.security import verify_firebase_token

router = APIRouter(prefix="/users")

@router.get("/me")
def me(user_id=Depends(verify_firebase_token)):
    return {"user_id": user_id}
