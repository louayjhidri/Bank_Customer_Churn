
from Customer_churn.components.data_preprocessing import DataTransformation
from Customer_churn.config.configuration import ConfigurationManager
from Customer_churn.components.model_training import ModelTrainer


class ModelTrainingPipeline:
    def __init__(self): 
        pass

    def main(self,train,test):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.initiate_model_trainer(train,test)