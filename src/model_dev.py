import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    @abstractmethod
    def train(self, x_train, y_train):
        pass

class LinearRegressionModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LinearRegression(**kwargs)

    def train(self, x_train, y_train):
        try:
            self.model.fit(x_train, y_train)
            logging.info("Model training completed")
            return self
        except Exception as e:
            logging.error("Error in model training: {}".format(e))
            raise e
