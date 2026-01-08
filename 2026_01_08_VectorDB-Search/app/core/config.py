from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MONGO_URI: str
    DB_NAME: str = "vector_search"
    CHUNKS_COLLECTION: str = "document_chunks"

    VECTOR_INDEX_NAME: str = "vector_index"

    EMBEDDING_MODEL: str = "mxbai-embed-large:latest"
    LLM_MODEL: str = "llama3.2:latest"

    class Config:
        env_file = ".env"


settings = Settings()
