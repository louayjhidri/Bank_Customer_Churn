
from Customer_churn.components.data_preprocessing import DataTransformation
from Customer_churn.config.configuration import ConfigurationManager
from Customer_churn.components.model_training import ModelTrainer
from Customer_churn.logging import logger
import numpy as np

STAGE_NAME = "Model Training stage"
class ModelTrainingPipeline:
    def __init__(self): 
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.initiate_model_trainer()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e