from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    MONGODB_URL = os.getenv("MONGODB_URL")
    DB_NAME = os.getenv("DB_NAME")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

settings = Settings()
