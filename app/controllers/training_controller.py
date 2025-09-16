from loguru import logger
from fastapi import APIRouter
from threading import Thread
from app.services.training_service import RFCategorizer

api = APIRouter()


def run_training_job():
    __rf = RFCategorizer()
    __rf.main()
    logger.info("rf_categorizer training process finished ✅")


@api.get(path="/training")
def run_training_process():
    # # Lanzar en un hilo separado
    thread = Thread(target=run_training_job, daemon=True)
    thread.start()
    return {"message": "rf_categorizer training process running"}
