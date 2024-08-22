
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
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.initiate_model_trainer(train,test)
