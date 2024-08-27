
from Customer_churn.components.data_preprocessing import DataTransformation
from Customer_churn.config.configuration import ConfigurationManager
from Customer_churn.components.model_training import ModelTrainer




class DataPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        train,test= data_transformation.initiate_data_transformation()
        return train,test
        
