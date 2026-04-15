import sys
from networkSecurity.components.data_ingestion import DataIngestion
from networkSecurity.components.data_validation import DataValidation
from networkSecurity.components.data_transformation import DataTransformation
from networkSecurity.logging.logger import logging
from networkSecurity.exception.exception import NetworkSecurityException
from networkSecurity.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig


if __name__ == "__main__":
    try:
        logging.info("Initiate data ingestion")        
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion_obj = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion_obj.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")

        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation_obj = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
        logging.info("Initiated Data Validation")
        data_validation_artifact = data_validation_obj.initiate_data_validation()
        logging.info("Data Validation Completed")

        logging.info("Initiated Data Transformation")   
        data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation_obj = DataTransformation(data_transformation_config=data_transformation_config, data_validation_artifact=data_validation_artifact)
        data_transformation_artifact = data_transformation_obj.initiate_data_transformation()
        logging.info("Data Transformation Complete")
        print(data_transformation_artifact)


    except Exception as e:
        raise NetworkSecurityException(e,sys)