from Mini_Translator.config.configuration import ConfigurationManager
from Mini_Translator.components.prepare_base_model import Base_Model
from Mini_Translator.logging import logger

STAGE_NAME="Prepare Base Model"
class BaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            get_model_config = config.get_model_config()
            base_model = Base_Model(config=get_model_config)
            base_model.initiate_prepare_base_model()
        except Exception as e:
            raise e


        
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = BaseModelPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

