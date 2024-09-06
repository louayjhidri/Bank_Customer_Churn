
from Customer_churn.components.data_preprocessing import DataTransformation
from Customer_churn.config.configuration import ConfigurationManager
from Customer_churn.logging import logger


STAGE_NAME = "Data Preprocessing stage"

class DataPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        train,test= data_transformation.initiate_data_transformation()
        return train,test
        
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e