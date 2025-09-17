import re
import nltk
import mlflow
import pickle
import traceback
import pandas as pd
from loguru import logger
from app.configuration.settings import settings
from app.responses.mlflow_response import RFCategorizerMetrics
from app.repositories.training_repository import TrainingRepository

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


class RFCategorizer:
    def __init__(self):
        """Initialize the RF Categorizer with necessary components for training"""
        self.__training_repository = TrainingRepository()
        self.vectorizer = CountVectorizer(max_features=2000)
        self.optimal_learners = None
        self.model = None

        # MLflow settings
        self.mlflow_uri = settings.MLFLOW_TRACKING_URI
        self.mlflow_experiment = 'RF_Categorizer_Experiment'
        self.registered_model_name = settings.MLFLOW_RF_MODEL
        self.artifact_name = 'Random Forest Classifier CV 5'
        self.vectorizer_file = settings.MLFLOW_RF_VECTORIZER
        self.artifact_path = settings.MLFLOW_ARTIFACT_PATH

    def get_input_data(self):
        """Fetch input data from the database"""
        rows, columns = self.__training_repository.get_input_rf_categorizer()
        df = pd.DataFrame(rows, columns=columns)
        return df

    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        stop_words = set(stopwords.words('spanish')).union(set(stopwords.words('english')))
        lemmatizer = WordNetLemmatizer()

        text = text.lower()
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

        words = re.split(r'\s+', text)
        words = [word for word in words if word not in stop_words and len(word) >= 3]
        words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize words
        return words

    def load_and_preprocess_data(self):
        """Load data and apply preprocessing steps"""
        # Get data from porstgres database
        raw_data = self.get_input_data()

        # Apply preprocessing
        raw_data['processed_name'] = raw_data['name'].apply(self.preprocess_text)
        raw_data['clean_name'] = raw_data['processed_name'].apply(lambda x: ' '.join(x))

        # Create feature matrix
        df_processed = raw_data[['category', 'clean_name']].copy()
        features = self.vectorizer.fit_transform(df_processed['clean_name'])
        features = features.toarray()

        # Create category mapping
        all_labels = df_processed['category'].tolist()
        self.category_mapping = {category: idx for idx, category in enumerate(set(all_labels))}

        return features, all_labels

    def find_optimal_learners(self, X_train, y_train, max_learners=25):
        """Find the optimal number of learners through cross-validation"""
        base_learners = list(range(1, max_learners + 1))
        cv_scores = []

        for n_learners in base_learners:
            clf = RandomForestClassifier(n_estimators=n_learners)
            scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
            cv_scores.append(scores.mean())

        # Calculate error and find optimal number of learners
        errors = [1 - x for x in cv_scores]
        self.optimal_learners = base_learners[errors.index(min(errors))]

        return self.optimal_learners

    def train_model(self, features, labels, test_size=0.3, random_state=42):
        """Train the RF model with optimal parameters"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state
        )

        # Find optimal number of learners if not already determined
        if self.optimal_learners is None:
            self.find_optimal_learners(X_train, y_train)

        # Train with optimal parameters
        params = {
            'n_estimators': self.optimal_learners,
            'n_jobs': 9
        }

        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train, y_train)

        # Generate predictions and classification report
        y_pred = self.model.predict(X_test)
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        # compare with production model
        return {
            'metrics': report_dict,
            'params': params,
            'X_test': X_test,
            'y_test': y_test
        }

    def compare_with_production_model(self, X_test, y_test, report_dict):
        prod_model_uri = f"models:/{self.registered_model_name}@champion"
        prod_model = mlflow.pyfunc.load_model(prod_model_uri)

        # --- Evaluar modelo en producción sobre los mismos datos ---
        y_pred_prod = prod_model.predict(X_test)
        prod_report_dict = classification_report(y_test, y_pred_prod, output_dict=True)

        # --- Comparativa ---
        f1_new = report_dict['macro avg']['f1-score']
        f1_prod = prod_report_dict['macro avg']['f1-score']
        improvement_threshold = 0.03  # Umbral de mejora

        logger.debug(f"F1 Score nuevo modelo: {f1_new:.4f}")
        logger.debug(f"F1 Score modelo producción (champion): {f1_prod:.4f}")

        if f1_new > f1_prod + improvement_threshold:
            logger.debug("✅ El nuevo modelo supera al champion, se puede promocionar.")
            return True
        else:
            logger.debug("⚠️ El champion sigue siendo mejor, no promocionar.")
            return False

    def log_metrics_to_mlflow(self, training_results):
        """Log metrics and model to MLflow"""
        mlflow.set_tracking_uri(uri=self.mlflow_uri)
        mlflow.set_experiment(experiment_name=self.mlflow_experiment)
        report_dict = training_results['metrics']
        params = training_results['params']
        prod_challenger = self.compare_with_production_model(
            training_results['X_test'],
            training_results['y_test'],
            report_dict)

        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            metrics = RFCategorizerMetrics(
                accuracy=report_dict['accuracy'],
                f1_score_macro=report_dict['macro avg']['f1-score'],
                recall_macro=report_dict['macro avg']['recall'],
                precision_macro=report_dict['macro avg']['precision'],
                recall_abc=report_dict['Aseo, Baños y Cocinas']['recall'],
                recall_electricos=report_dict['Electricos']['recall'],
                recall_ferreteria=report_dict['Ferretería']['recall'],
                recall_herramientas=report_dict['Herramientas']['recall'],
                recall_pinturas=report_dict['Pinturas']['recall'],
                recall_plomeria=report_dict['Plomería']['recall']
            )

            mlflow.log_metrics(metrics.model_dump())

            # Log model
            if prod_challenger:
                logger.info('Logging and registering new production model in MLflow')
                mlflow.sklearn.log_model(
                    sk_model=self.model,
                    name=self.artifact_name,
                    registered_model_name=self.registered_model_name)

                with open(self.vectorizer_file, "wb") as file:
                    pickle.dump(self.vectorizer, file)
                mlflow.log_artifact(self.vectorizer_file, artifact_path=self.artifact_path)
            else:
                logger.info('Logging model in MLflow without registering (not a challenger)')
                mlflow.sklearn.log_model(
                    sk_model=self.model,
                    name=self.artifact_name)

    def main(self):
        """Main method to execute the entire training pipeline"""
        try:
            # 1. Load and preprocess data
            logger.info("Loading and preprocessing data...")
            features, labels = self.load_and_preprocess_data()

            # 2. Train the model
            logger.info("Training model with cross-validation...")
            training_results = self.train_model(features, labels)

            # 3. Log metrics and model to MLflow
            logger.info("Logging metrics and model to MLflow...")
            self.log_metrics_to_mlflow(training_results)

            # 4. Return results
            logger.info("Training completed successfully!")
            summary = {
                'success': True,
                'metrics': training_results['metrics'],
                'model': self.model,
                'category_mapping': self.category_mapping
            }

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            logger.error(traceback.format_exc())
            summary = {
                'success': False,
                'error': str(e)
            }
        logger.debug(summary)
