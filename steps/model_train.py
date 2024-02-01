import logging
import pandas as pd
from zenml import step
from sklearn.linear_model import LinearRegression
from .config import ModelNameConfig

@step
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
) -> LinearRegression:  # Change the return type to LinearRegression
    try:
        model = None
        if config.model_name == "LinearRegression":
            model = LinearRegression()
            model.fit(x_train, y_train)
            logging.info("Model training completed")
            return model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e
