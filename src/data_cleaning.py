import logging 
from abc import ABC, abstractmethod
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from typing import Union

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass 
    
class DataPreProcesStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.drop(["order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_customer_date", "order_purchase_timestamp"], axis=1)
            
            data["product_weight_g"].fillna(data["product_weight_g"].median(),inplace=True) 
            data["product_length_cm"].fillna(data["product_length_cm"].median(),inplace=True) 
            data["product_height_cm"].fillna(data["product_height_cm"].median(),inplace=True) 
            data["product_width_cm"].fillna(data["product_width_cm"].median(),inplace=True) 
            data["review_comment_message"].fillna("no review",inplace=True) 
            
            data=data.select_dtypes(include=[np.number])
            cols_to_drop=['customer_zip_code_prefix','order_item_id']
            data=data.drop(cols_to_drop,axis=1)
            return data
        
        except Exception as e:
            logging.error(f"An error occurred during data processing: {e}")
            # Handle the exception or raise it if needed
            raise
        
        
class  DataSplitStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            x=data.drop(['review_score'],axis=1)
            y=data['review_score']
            x_train,x_test,y_train,y_test
        except Exception as e:
            logging.error("error in dividing data: {}".fomat(e))
            raise e 
        
        
class DataCleaning:
    def  __init__(self,data: pd.DataFrame,strategy:DataStrategy):
        self.data=data 
        self.strategy= strategy 
    
    def handle_data(self)-> Union[pd.DataFrame,pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error while cleaning data: {}".format(e))
            raise e 
        
# if __name__=='__main__':
#     data=pd.read_csv("/Users/rohansridhar/Desktop/customer_satisfaction/data/olist_customers_dataset.csv")
#     data_cleaning=DataCleaning(data,DataPreProcesStrategy())
#     data_cleaning.handle_data()