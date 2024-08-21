from Customer_churn.logging import logger
from Customer_churn.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from Customer_churn.pipeline.stage_02_data_preprocessing import DataPreprocessingPipeline


# STAGE_NAME = "Data Ingestion Stage"

# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

#     data_ingestion = DataIngestionPipeline()

#     data_ingestion.main()

#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

# except Exception as e:
#     logger.exception(e)
#     raise e

STAGE_NAME = "Data Preprocessing Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    data_transformation = DataPreprocessingPipeline()
    data_transformation.main()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e