from fastapi import APIRouter

from app.controllers import (
    health_controller,
    training_controller
)

api_router = APIRouter()
api_router.include_router(health_controller.api)
api_router.include_router(training_controller.api)
