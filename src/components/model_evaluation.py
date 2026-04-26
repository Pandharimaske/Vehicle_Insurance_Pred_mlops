import sys
import os
from typing import Optional

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.metrics import roc_auc_score, f1_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
)
from src.exception import MyException
from src.constants import (
    TARGET_COLUMN, AUC_PROMOTION_GATE, MLFLOW_MODEL_NAME
)
from src.logger import logging
from src.utils.main_utils import load_object
from src.entity.s3_estimator import Proj1Estimator
from dataclasses import dataclass


@dataclass
class EvaluateModelResponse:
    """Carries the champion-vs-challenger comparison result."""
    trained_model_auc: float        # challenger AUC
    best_model_auc: float           # champion (Production) AUC; 0.0 if no prod model
    is_model_accepted: bool         # True  → challenger promoted to Production
    auc_improvement: float          # challenger AUC − champion AUC
    # F1 retained for backward-compatibility with downstream artifacts
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]


class ModelEvaluation:

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    # ------------------------------------------------------------------
    # Champion model retrieval (AWS S3 + MLflow registry)
    # ------------------------------------------------------------------
    def get_best_model(self) -> Optional[Proj1Estimator]:
        """Return the current Production model from S3, or None if absent."""
        try:
            estimator = Proj1Estimator(
                bucket_name=self.model_eval_config.bucket_name,
                model_path=self.model_eval_config.s3_model_key_path,
            )
            if estimator.is_model_present(model_path=self.model_eval_config.s3_model_key_path):
                return estimator
            return None
        except Exception as e:
            raise MyException(e, sys)

    def _get_production_auc_from_registry(self) -> float:
        """
        Query the MLflow Model Registry for the AUC-ROC of the current
        Production model version.  Returns 0.0 if no Production version exists.
        """
        try:
            client = MlflowClient()
            prod_versions = client.get_latest_versions(
                MLFLOW_MODEL_NAME, stages=["Production"]
            )
            if not prod_versions:
                logging.info("No Production version found in MLflow registry — treating champion AUC as 0.0")
                return 0.0
            run = client.get_run(prod_versions[0].run_id)
            prod_auc = run.data.metrics.get("auc_roc", 0.0)
            logging.info(f"Champion (Production) AUC from MLflow registry: {prod_auc:.4f}")
            return prod_auc
        except Exception as exc:
            logging.warning(f"Could not fetch production AUC from MLflow registry: {exc}")
            return 0.0

    # ------------------------------------------------------------------
    # Feature preprocessing helpers (unchanged from original)
    # ------------------------------------------------------------------
    def _map_gender_column(self, df):
        df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1}).astype(int)
        return df

    def _create_dummy_columns(self, df):
        return pd.get_dummies(df, drop_first=True)

    def _rename_columns(self, df):
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years",
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype("int")
        return df

    def _drop_id_column(self, df):
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        return df

    # ------------------------------------------------------------------
    # Core evaluation — champion / challenger on AUC-ROC
    # ------------------------------------------------------------------
    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Champion / Challenger evaluation:
          - Challenger AUC = AUC-ROC logged by model_trainer (Optuna best)
          - Champion AUC   = AUC-ROC of the Production model in MLflow registry
          - Gate            = AUC_PROMOTION_GATE (default 0.005)
          - Decision        = challenger accepted  iff  ΔAUC > gate
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x = test_df.drop(TARGET_COLUMN, axis=1)
            y = test_df[TARGET_COLUMN]

            x = self._map_gender_column(x)
            x = self._drop_id_column(x)
            x = self._create_dummy_columns(x)
            x = self._rename_columns(x)

            # --- Challenger metrics (from model_trainer artifact) ---
            challenger_auc = self.model_trainer_artifact.metric_artifact.roc_auc
            challenger_f1 = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"Challenger AUC-ROC: {challenger_auc:.4f} | F1: {challenger_f1:.4f}")

            # --- Champion AUC from MLflow Production registry ---
            champion_auc = self._get_production_auc_from_registry()

            # Also evaluate champion F1 on live test set (if model exists in S3)
            champion_f1 = None
            best_model = self.get_best_model()
            if best_model is not None:
                try:
                    y_hat = best_model.predict(x)
                    champion_f1 = f1_score(y, y_hat)
                    logging.info(f"Champion F1 on test set: {champion_f1:.4f}")
                except Exception:
                    logging.warning("Could not evaluate champion on test set.")

            # --- Gate decision ---
            auc_improvement = challenger_auc - champion_auc
            is_accepted = auc_improvement > AUC_PROMOTION_GATE
            logging.info(
                f"Champion AUC: {champion_auc:.4f} | Challenger AUC: {challenger_auc:.4f} | "
                f"ΔAUC: {auc_improvement:.4f} | Gate: {AUC_PROMOTION_GATE} | "
                f"Accepted: {is_accepted}"
            )

            # Log gate decision to MLflow
            mlflow.log_metrics({
                "champion_auc": champion_auc,
                "challenger_auc": challenger_auc,
                "auc_improvement": auc_improvement,
                "promotion_gate": AUC_PROMOTION_GATE,
            })
            mlflow.log_param("model_accepted", str(is_accepted))

            return EvaluateModelResponse(
                trained_model_auc=challenger_auc,
                best_model_auc=champion_auc,
                is_model_accepted=is_accepted,
                auc_improvement=auc_improvement,
                trained_model_f1_score=challenger_f1,
                best_model_f1_score=champion_f1,
            )

        except Exception as e:
            raise MyException(e, sys)

    # ------------------------------------------------------------------
    # MLflow Model Registry promotion
    # ------------------------------------------------------------------
    def _promote_to_production(self) -> None:
        """
        Register the current MLflow run's model artifact and transition it
        to the 'Production' stage, archiving any prior Production versions.
        """
        try:
            active_run = mlflow.active_run()
            if active_run is None:
                logging.warning("No active MLflow run — skipping registry promotion.")
                return

            run_id = active_run.info.run_id
            model_uri = f"runs:/{run_id}/model"
            logging.info(f"Registering model from run {run_id} → '{MLFLOW_MODEL_NAME}'")

            mv = mlflow.register_model(model_uri=model_uri, name=MLFLOW_MODEL_NAME)

            client = MlflowClient()
            client.transition_model_version_stage(
                name=MLFLOW_MODEL_NAME,
                version=mv.version,
                stage="Production",
                archive_existing_versions=True,   # auto-archive old Production
            )
            logging.info(
                f"✅ Model v{mv.version} promoted to Production "
                f"(ΔAUC={mlflow.active_run().data.metrics.get('auc_improvement', 0):.4f})"
            )
        except Exception as e:
            logging.warning(f"MLflow registry promotion failed (non-critical for S3 push): {e}")

    # ------------------------------------------------------------------
    # Pipeline entry point
    # ------------------------------------------------------------------
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        1. Run champion / challenger AUC comparison with defined gate.
        2. If challenger wins → register + promote to Production in MLflow Model Registry.
        3. Return ModelEvaluationArtifact (is_model_accepted drives model_pusher).
        """
        try:
            print("-" * 96)
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()

            if evaluate_model_response.is_model_accepted:
                logging.info("Challenger accepted — promoting to MLflow Production stage …")
                self._promote_to_production()
            else:
                logging.info(
                    f"Challenger rejected — ΔAUC={evaluate_model_response.auc_improvement:.4f} "
                    f"≤ gate={AUC_PROMOTION_GATE}. Production model unchanged."
                )

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.auc_improvement,
            )
            logging.info(f"ModelEvaluationArtifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise MyException(e, sys) from e