import asyncio
import sys
from pathlib import Path

# Point to backend so imports work
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from services.firebase_service import init_firebase
from ai_modules.faiss_index import FAISSIndex

async def main():
    init_firebase()
    idx = FAISSIndex()
    await idx.rebuild()
    print("Done! Check backend/faiss_index/ folder.")

asyncio.run(main())