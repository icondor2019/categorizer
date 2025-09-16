import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv('.env')


class Settings(BaseSettings):
    ENVIRONMENT: str = os.getenv('ENVIRONMENT', 'dev')

    # FastAPI docs
    OPEN_API_PATH: str = "/swagger-ui"
    OPEN_API_JSON: str = "/openapi.json"

    # MLFLOW setup
    MLFLOW_TRACKING_URI: str = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000/')

    # Postgres setup
    POSTGRES_SCHEMA: str = os.getenv('POSTGRES_SCHEMA', 'postgres')
    POSTGRES_USER: str = os.getenv('POSTGRES_USER', 'postgres')
    POSTGRES_DB: str = os.getenv('POSTGRES_DB', 'postgres')
    POSTGRES_PASSWORD: str = os.getenv('POSTGRES_PASSWORD', 'postgres')
    POSTGRES_HOST: str = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT: str = os.getenv('POSTGRES_PORT', '5432')

    class Config():
        case_sensitive = True


settings = Settings()
