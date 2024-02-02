import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from src.evaluation import MSE, R2, RMSE
import mlflow
from mlflow.tracking import MlflowClient as Client

@step
def evaluate_model(
    model: RegressorMixin,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    experiment_tracker: Annotated[str, "Experiment tracker name"]
) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"],
]:
    try:
        prediction = model.predict(x_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("mse", mse)
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        rmse_class = RMSE()  # Assuming RMSE is defined in src.evaluation
        rmse = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("rmse", rmse)

        return r2, rmse
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e
