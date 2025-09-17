import mlflow
import pickle
from loguru import logger
from mlflow import MlflowClient

from app.configuration.settings import settings


class InferenceService:
    def __init__(self):
        # Set the tracking URI to the MLflow server
        self.mlflow_uri = settings.MLFLOW_TRACKING_URI
        self.model_name = 'RF_Categorizer_Model'
        self.model_version_alias = "champion"
        self.prod_model_uri = f"models:/{self.model_name}@{self.model_version_alias}"
        self.artifact_path = "preprocessors/bow_vectorizer.pkl"
        mlflow.set_tracking_uri(uri=self.mlflow_uri)
        self.client = MlflowClient()

    def load_model(self):
        # Load the model from MLflow Model Registry
        loaded_model = mlflow.sklearn.load_model(self.prod_model_uri)
        return loaded_model

    def load_artifact(self):
        # Get information about the model
        model_info = self.client.get_model_version_by_alias(self.model_name, self.model_version_alias)
        run_id = model_info.run_id
        artifact_file = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=self.artifact_path)
        with open(artifact_file, "rb") as f:
            bow_vec = pickle.load(f)
        return bow_vec

    def predict(self, product_list: list) -> list:
        logger.debug("Starting prediction...")
        try:
            if not hasattr(self, 'model'):
                logger.debug("Loading model for the first time...")
                self.model = self.load_model()
            if not hasattr(self, 'bow_vec'):
                logger.debug("Loading artifact for the first time...")
                self.bow_vec = self.load_artifact()

            # Transform the product descriptions using the loaded artifact
            transformed_data = self.bow_vec.transform(product_list).toarray()

            # Make predictions using the loaded model
            predicted_category = self.model.predict(transformed_data)
            logger.debug("Prediction completed successfully.")

            return predicted_category
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None

    def setup(self):
        logger.info("Setting up Inference Service...")
        self.model = self.load_model()
        logger.info("Model loaded successfully.")

        self.bow_vec = self.load_artifact()
        logger.info("Artifact loaded successfully.")
