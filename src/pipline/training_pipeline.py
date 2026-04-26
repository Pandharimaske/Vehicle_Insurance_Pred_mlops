import os
import sys
import mlflow
import dagshub

from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
)
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
)
from src.constants import (
    MLFLOW_EXPERIMENT_NAME,
    DAGSHUB_REPO_OWNER,
    DAGSHUB_REPO_NAME,
)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    # ------------------------------------------------------------------
    # DagsHub / MLflow initialisation
    # ------------------------------------------------------------------
    @staticmethod
    def _setup_mlflow() -> None:
        """
        Initialise DagsHub remote tracking if DAGSHUB_REPO_OWNER and
        DAGSHUB_REPO_NAME env vars are set; otherwise fall back to local
        MLflow tracking (./mlruns).

        DagsHub automatically sets the MLflow tracking URI so that all
        mlflow.log_* calls are forwarded to the DagsHub MLflow server,
        making experiments and artifacts visible at
        https://dagshub.com/<owner>/<repo>/experiments.
        """
        owner = os.environ.get("DAGSHUB_REPO_OWNER", DAGSHUB_REPO_OWNER).strip()
        repo = os.environ.get("DAGSHUB_REPO_NAME", DAGSHUB_REPO_NAME).strip()

        if owner and repo:
            logging.info(f"Initialising DagsHub tracking: {owner}/{repo}")
            dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)
            logging.info("DagsHub MLflow tracking URI set.")
        else:
            logging.info(
                "DAGSHUB_REPO_OWNER / DAGSHUB_REPO_NAME not set — "
                "using local MLflow tracking (./mlruns)."
            )
            mlflow.set_tracking_uri("./mlruns")

        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ------------------------------------------------------------------
    # Individual stage starters
    # ------------------------------------------------------------------
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """Fetch data from MongoDB and split into train/test CSV files."""
        try:
            logging.info("Entered start_data_ingestion method of TrainPipeline class")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Exited start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """Validate schema and column presence."""
        logging.info("Entered start_data_validation method of TrainPipeline class")
        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Exited start_data_validation method of TrainPipeline class")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        """Apply feature engineering, scaling, and SMOTEENN resampling."""
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact,
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise MyException(e, sys)

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        """Run Optuna HPO (50 trials, TPE) over XGBoost + LightGBM; log SHAP."""
        try:
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise MyException(e, sys)

    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        """Champion / Challenger: auto-promote to MLflow Production if ΔAUC > gate."""
        try:
            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )
            return model_evaluation.initiate_model_evaluation()
        except Exception as e:
            raise MyException(e, sys)

    def start_model_pusher(
        self, model_evaluation_artifact: ModelEvaluationArtifact
    ) -> ModelPusherArtifact:
        """Push accepted model to AWS S3 bucket."""
        try:
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=self.model_pusher_config,
            )
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise MyException(e, sys)

    # ------------------------------------------------------------------
    # Full pipeline orchestration
    # ------------------------------------------------------------------
    def run_pipeline(self) -> None:
        """
        Runs the complete training pipeline inside a single MLflow run:
          data_ingestion → data_validation → data_transformation
          → model_trainer (Optuna HPO + SHAP) → model_evaluation
          (champion/challenger AUC gate + MLflow registry promotion)
          → model_pusher (S3)

        All params, metrics, and artifacts are tracked on DagsHub via MLflow.
        """
        try:
            self._setup_mlflow()

            with mlflow.start_run(run_name="vehicle_insurance_training") as run:
                logging.info(
                    f"MLflow run started — ID: {run.info.run_id} | "
                    f"Experiment: {MLFLOW_EXPERIMENT_NAME}"
                )

                # Stage 1 — Data Ingestion
                data_ingestion_artifact = self.start_data_ingestion()

                # Stage 2 — Data Validation
                data_validation_artifact = self.start_data_validation(
                    data_ingestion_artifact=data_ingestion_artifact
                )

                # Stage 3 — Data Transformation (feature engineering + SMOTEENN)
                data_transformation_artifact = self.start_data_transformation(
                    data_ingestion_artifact=data_ingestion_artifact,
                    data_validation_artifact=data_validation_artifact,
                )

                # Stage 4 — Model Training (Optuna HPO + SHAP logging)
                model_trainer_artifact = self.start_model_trainer(
                    data_transformation_artifact=data_transformation_artifact
                )

                # Stage 5 — Model Evaluation (AUC gate + MLflow registry)
                model_evaluation_artifact = self.start_model_evaluation(
                    data_ingestion_artifact=data_ingestion_artifact,
                    model_trainer_artifact=model_trainer_artifact,
                )

                if not model_evaluation_artifact.is_model_accepted:
                    logging.info(
                        "Model not accepted by champion/challenger gate. "
                        "S3 push skipped — Production model unchanged."
                    )
                    return None

                # Stage 6 — Model Pusher (to AWS S3)
                self.start_model_pusher(
                    model_evaluation_artifact=model_evaluation_artifact
                )

                logging.info(
                    f"Pipeline complete. Run ID: {run.info.run_id} logged to "
                    f"experiment '{MLFLOW_EXPERIMENT_NAME}'."
                )

        except Exception as e:
            raise MyException(e, sys)