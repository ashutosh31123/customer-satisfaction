import logging
from typing import Annotated

import mlflow
import pandas as pd
from model.model_dev import ModelTrainer
from sklearn.base import RegressorMixin
from zenml import ArtifactConfig, step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_type: str = "lightgbm",
    do_fine_tuning: bool = True,
) -> Annotated[
    RegressorMixin,
    ArtifactConfig(name="sklearn_regressor", is_model_artifact=True),
]:
    """
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
        model_type: str - available options ["lightgbm", "randomforest", "xgboost"]
        do_fine_tuning: Should full training run or only fine tuning
    Returns:
        model: RegressorMixin
    """
    model = None
    try:
        model_training = ModelTrainer(x_train, y_train, x_test, y_test)

        if model_type == "lightgbm":
            mlflow.lightgbm.autolog()
            model = model_training.lightgbm_trainer(fine_tuning=do_fine_tuning)
        elif model_type == "randomforest":
            mlflow.sklearn.autolog()
            model = model_training.random_forest_trainer(fine_tuning=do_fine_tuning)
        elif model_type == "xgboost":
            mlflow.xgboost.autolog()
            model = model_training.xgboost_trainer(fine_tuning=do_fine_tuning)
        else:
            raise ValueError("Model type not supported")
        
    except Exception as e:
        logging.error(e)
        raise e
    
    if model is None:
        raise ValueError("Model training did not return a valid model.")
    
    return model
