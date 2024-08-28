from Customer_churn.exception import CustomException
from Customer_churn.config.configuration import ConfigurationManager
from Customer_churn.components.model_prediction import ModelPrediction
import sys










class PredictionPipeline:
    def __init__(self):
        pass

    def main (self,features):
        try:
            config = ConfigurationManager()
            model_prediction_config = config.get_model_prediction_config()
            model_prediction = ModelPrediction(model_prediction_config=model_prediction_config)
            
        
        except Exception as e:
            raise CustomException(e,sys)