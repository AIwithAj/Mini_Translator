from src.Mini_Translator.logging import logger

import os
import json

class DataIngestion:
    def __init__(self, config):
        self.config = config

    def convert_dataset_to_serializable(self, dataset):
        return [example for example in dataset]

    def initiate_data_ingestion(self):
        logger.info("Initiating dataIngestion..")
        try:
            dataset = datasets.load_dataset(self.config.dataset_name)
        except Exception as e:
            logger.info("incorrect dataset")
            raise e

        serializable_dataset = {
            "train": list(dataset["train"]),
            "test": list(dataset["test"]),
            "validation": list(dataset["validation"])
        }

        # Save the serializable dataset to a JSON file
        with open(self.config.raw_data, 'w') as json_file:
            json.dump(serializable_dataset, json_file, indent=4)

        train_data, valid_data, test_data = dataset["train"], dataset["validation"], dataset["test"]

        train_data_serializable = self.convert_dataset_to_serializable(train_data)
        valid_data_serializable = self.convert_dataset_to_serializable(valid_data)
        test_data_serializable = self.convert_dataset_to_serializable(test_data)

        logger.info("saving train, valid, test dataset..")
        with open(self.config.train, 'w') as train_file:
            json.dump(train_data_serializable, train_file, indent=4)
        with open(self.config.valid, 'w') as valid_file:
            json.dump(valid_data_serializable, valid_file, indent=4)
        with open(self.config.test, 'w') as test_file:
            json.dump(test_data_serializable, test_file, indent=4)

        logger.info("data_ingestion successfully saved")
