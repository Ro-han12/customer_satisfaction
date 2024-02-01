import logging 
from abc import ABC, abstractmethod 
from sklearn.linear_model import LinearRegression

class Model(ABC):
    def train(self,x_train,y_train):
        pass 
    
class LinearRegressionModel(Model):
    def train(self,x_train,y_train,**kwargs):
        try:
            reg=LinearRegressionModel(**kwargs)
            reg.fit(x_train,y_train)
            logging.info("model training completed")
            return reg 
        except Exception as e:
            logging.error("Error in model training: {}".format(e))
            raise e