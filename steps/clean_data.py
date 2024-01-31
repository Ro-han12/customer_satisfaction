import logging 
import pandas as pd 
from zenml import step 
from src.data_cleaning import DataCleaning,DataSplitStrategy,DataPreProcesStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_data(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    try:
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        divide_strategy = DataSplitStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("data cleaning completed")
    except Exception as e:
        logging.error("error in cleaning data: {}".format(e))
        raise e
        