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

    class Config():
        case_sensitive = True


settings = Settings()
