from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    
    mongo_url : str
    db_name : str
    env : str = 'development'

settings = Settings()