import sys
from networkSecurity.components.data_ingestion import DataIngestion
from networkSecurity.logging.logger import logging
from networkSecurity.exception.exception import NetworkSecurityException
from networkSecurity.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig


if __name__ == "__main__":
    try:
        logging.info("Initiate data ingestion")        
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion_obj = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion_obj.initiate_data_ingestion()
        print(data_ingestion_artifact)
    except Exception as e:
        raise NetworkSecurityException(e,sys)