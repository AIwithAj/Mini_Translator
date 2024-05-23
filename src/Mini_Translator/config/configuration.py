from src.Mini_Translator.constants import *
from src.Mini_Translator.utils.common import read_yaml,create_directories
from src.Mini_Translator.entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelConfig

class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):
        
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) ->DataIngestionConfig:
        config=self.config.data_ingestion

        create_directories([config.root_dir,config.raw_path])

        data_ingestion_config=DataIngestionConfig(
            root_dir=config.root_dir,
            raw_path=config.raw_path,
            dataset_name=config.dataset_name,
            raw_data=config.data_files.raw_data,
            train=config.data_files.train,
            valid=config.data_files.validation,
            test=config.data_files.test
            
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params=self.params

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_1=config.tokenizer_1,
            tokenizer_2=config.tokenizer_2,
            lang1=params.lang1,
            lang2=params.lang2,
            sos_token=params.sos_token,
            eos_token=params.eos_token,
            max_length=params.max_length,
            lower=params.lower,
            data_loader=config.data_loader,
            batch_size=params.batch_size
        )

        return data_transformation_config
    def get_model_config(self) -> ModelConfig:
        config = self.config.base_Model
        params=self.params

        create_directories([config.root_dir])

        model_config=ModelConfig(root_dir=config.root_dir,
                                 encoder_embedding_dim =params.encoder_embedding_dim,
                        decoder_embedding_dim = params.decoder_embedding_dim,
                        hidden_dim = params.hidden_dim,
                        n_layers= params.n_layers,
                        encoder_dropout = params.encoder_dropout,
                        decoder_dropout = params.decoder_dropout)
        
        return model_config


        
        