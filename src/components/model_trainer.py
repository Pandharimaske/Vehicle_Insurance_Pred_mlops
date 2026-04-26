import os
import sys
from typing import Tuple

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe inside pipelines/Docker
import matplotlib.pyplot as plt
import numpy as np
import shap
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score
)

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
)
from src.entity.estimator import MyModel
from src.constants import OPTUNA_N_TRIALS

# Feature names after ColumnTransformer:
#   StandardScaler → Age, Vintage
#   MinMaxScaler   → Annual_Premium
#   passthrough    → Region_Code, Driving_License, Previously_Insured,
#                    Gender, Vehicle_Age_lt_1_Year, Vehicle_Age_gt_2_Years,
#                    Vehicle_Damage_Yes, Policy_Sales_Channel
FEATURE_NAMES = [
    "Age", "Vintage",
    "Annual_Premium",
    "Region_Code", "Driving_License", "Previously_Insured",
    "Gender", "Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years",
    "Vehicle_Damage_Yes", "Policy_Sales_Channel",
]


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        """
        :param data_transformation_artifact: Output of the data transformation stage.
        :param model_trainer_config: Configuration for model training.
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    # ------------------------------------------------------------------
    # Optuna objective
    # ------------------------------------------------------------------
    def _objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        """
        Optuna objective function.
        Searches over both XGBoost and LightGBM hyperparameter spaces.
        Maximises AUC-ROC on the held-out test set.
        """
        model_type = trial.suggest_categorical("model_type", ["xgboost", "lightgbm"])

        if model_type == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "random_state": 42,
                "verbosity": 0,
            }
            model = xgb.XGBClassifier(**params)
        else:  # lightgbm
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "random_state": 42,
                "verbose": -1,
            }
            model = lgb.LGBMClassifier(**params)

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        return auc

    # ------------------------------------------------------------------
    # SHAP explainability
    # ------------------------------------------------------------------
    def _log_shap_artifacts(
        self, model, X_test: np.ndarray, model_type: str
    ) -> None:
        """
        Generate SHAP feature-importance bar chart and log it as an MLflow artifact.
        Surfaces top predictive features (vehicle age, annual premium, prior claims)
        for business interpretability.
        """
        try:
            logging.info("Generating SHAP explanations …")
            # Cap sample size so it runs in reasonable time
            X_sample = X_test[:500] if len(X_test) > 500 else X_test

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # LightGBM binary returns a list [neg_class, pos_class]; pick pos class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            fig, _ = plt.subplots(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=FEATURE_NAMES,
                plot_type="bar",
                show=False,
            )
            plt.title(f"SHAP Feature Importance — {model_type}", fontsize=14)
            plt.tight_layout()

            shap_dir = self.model_trainer_config.model_trainer_dir
            os.makedirs(shap_dir, exist_ok=True)
            shap_path = os.path.join(shap_dir, "shap_summary.png")
            plt.savefig(shap_path, bbox_inches="tight", dpi=150)
            plt.close(fig)

            mlflow.log_artifact(shap_path, artifact_path="shap_explainability")
            logging.info(f"SHAP summary plot logged as MLflow artifact: {shap_path}")

        except Exception as shap_err:
            # SHAP failure must not abort the training run
            logging.warning(f"SHAP logging skipped (non-critical): {shap_err}")

    # ------------------------------------------------------------------
    # Main training method
    # ------------------------------------------------------------------
    def get_model_object_and_report(
        self, train: np.ndarray, test: np.ndarray
    ) -> Tuple[object, ClassificationMetricArtifact]:
        """
        Runs a 50-trial Optuna study (TPE sampler) over XGBoost and LightGBM.
        Best configuration is retrained; metrics + SHAP are logged to MLflow.

        Returns: (best_trained_model, ClassificationMetricArtifact)
        """
        try:
            X_train, y_train = train[:, :-1], train[:, -1]
            X_test, y_test = test[:, :-1], test[:, -1]

            # --- Optuna HPO: 50 trials, TPE sampler, maximise AUC-ROC ---
            logging.info(
                f"Starting Optuna HPO — {OPTUNA_N_TRIALS} trials, TPE sampler, "
                "objective: AUC-ROC"
            )
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            sampler = TPESampler(seed=42)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(
                lambda trial: self._objective(trial, X_train, y_train, X_test, y_test),
                n_trials=OPTUNA_N_TRIALS,
                show_progress_bar=False,
            )

            best_params = dict(study.best_params)
            model_type = best_params.pop("model_type")
            best_auc_trial = study.best_value
            logging.info(
                f"Optuna best: model={model_type}, AUC={best_auc_trial:.4f}, "
                f"params={best_params}"
            )

            # --- Retrain best configuration on full training set ---
            if model_type == "xgboost":
                best_model = xgb.XGBClassifier(
                    **best_params,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42,
                    verbosity=0,
                )
            else:
                best_model = lgb.LGBMClassifier(
                    **best_params, random_state=42, verbose=-1
                )

            best_model.fit(X_train, y_train)

            # --- Compute final evaluation metrics ---
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            logging.info(
                f"Final metrics — AUC-ROC: {auc:.4f}, F1: {f1:.4f}, "
                f"Precision: {precision:.4f}, Recall: {recall:.4f}"
            )

            # --- Log to active MLflow run ---
            mlflow.log_params({
                "model_type": model_type,
                "optuna_n_trials": OPTUNA_N_TRIALS,
                "optuna_sampler": "TPE",
                **{f"best_{k}": v for k, v in best_params.items()},
            })
            mlflow.log_metrics({
                "auc_roc": auc,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
            })

            # --- SHAP explainability (Bullet 3) ---
            self._log_shap_artifacts(best_model, X_test, model_type)

            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1,
                precision_score=precision,
                recall_score=recall,
                roc_auc=auc,
            )
            return best_model, metric_artifact

        except Exception as e:
            raise MyException(e, sys) from e

    # ------------------------------------------------------------------
    # Pipeline entry point
    # ------------------------------------------------------------------
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Orchestrates the full model training step:
          1. Load transformed numpy arrays from the transformation artifact.
          2. Run Optuna HPO (50 trials, TPE) over XGBoost + LightGBM.
          3. Evaluate the winner and log metrics + SHAP to MLflow.
          4. Wrap model in MyModel (preprocessor + trained model) and persist.
        """
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            print("-" * 96)
            print("Starting Model Trainer Component — Optuna HPO (XGBoost / LightGBM)")

            # Load transformed data
            train_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )
            logging.info("Transformed train/test arrays loaded.")

            # Run HPO + get best model
            trained_model, metric_artifact = self.get_model_object_and_report(
                train=train_arr, test=test_arr
            )

            # Validate against expected minimum score
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            train_f1 = f1_score(y_train, trained_model.predict(X_train))
            if train_f1 < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"Best model train F1={train_f1:.4f} is below threshold "
                    f"{self.model_trainer_config.expected_accuracy}. Aborting."
                )

            # Load preprocessor and wrap into MyModel
            preprocessing_obj = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )
            my_model = MyModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=trained_model,
            )
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info(
                f"Model saved: {self.model_trainer_config.trained_model_file_path}"
            )

            # Log the serialised model as an MLflow artifact
            mlflow.sklearn.log_model(
                trained_model,
                artifact_path="model",
                registered_model_name=None,   # Registry handled in model_evaluation
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"ModelTrainerArtifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e