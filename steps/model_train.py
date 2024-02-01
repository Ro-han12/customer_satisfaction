import logging 
import pandas as pd 
from zenml import step 
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
@step 
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> RegressorMixin:
    try:
        
        model=None
        if config.model_name=="LinearRegression":
            model=LinearRegressionModel()
            trained_model=model.train(x_train,y_train)
            return  trained_model
        else:
            raise ValueError("model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error("error in training model: {}".format(e))
        raise e
            

    