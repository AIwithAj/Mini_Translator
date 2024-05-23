from src.Mini_Translator.config.configuration import ConfigurationManager
from src.Mini_Translator.components.data_ingestion import DataIngestion
try:
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.initiate_data_ingestion()
except Exception as e:
    raise e