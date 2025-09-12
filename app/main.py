from fastapi import FastAPI
from loguru import logger
from app.configuration.settings import settings
from app.controllers import api_router

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML models and allocate the resources
    logger.info("Starting app...")
    yield
    # Clean up the ML models and release the resources
    logger.info("Closing app...")

app = FastAPI(root_path='/categorizer',
              title='categorizer service',
              version='1.0.0',
              description='categorizer service for product classification',
              docs_url=settings.OPEN_API_PATH,
              openapi_url=settings.OPEN_API_JSON,
              lifespan=lifespan
              )


def setup_application():
    app.include_router(api_router)


setup_application()
