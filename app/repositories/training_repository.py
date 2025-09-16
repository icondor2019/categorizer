from loguru import logger
from sqlalchemy import text
from app.configuration.database import db
from sqlalchemy.exc import SQLAlchemyError
from app.repositories.queries.training_query import (
    GET_INPUT_RF_CATEGORIZER_QUERY
)


class TrainingRepository:
    """Manage training records"""

    def get_input_rf_categorizer(self):
        try:
            result = db.execute(text(GET_INPUT_RF_CATEGORIZER_QUERY))
            rows = result.fetchall()
            columns = result.keys()
            return rows, columns
        except SQLAlchemyError as ex:
            logger.error(f"Error getting input rf categorizer. Error: {ex}")
