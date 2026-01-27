from pydantic import BaseSettings

class Settings(BaseSettings):
    matching_algorithm: str = "sbert"
    firestore_project_id: str
    debug: bool = True

    class Config:
        env_file = ".env"

settings = Settings()
