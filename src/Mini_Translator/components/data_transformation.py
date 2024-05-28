import os
from Mini_Translator.logging import logger
import spacy
from datasets import load_dataset, load_from_disk
import json
import pandas as pd
from datasets import Dataset, DatasetDict
import torchtext
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn
import torch
from Mini_Translator.constants import *
from Mini_Translator.utils.common import read_yaml
from Mini_Translator.config.configuration import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig, config_filepath=CONFIG_FILE_PATH):
        self.config = config
        os.system(f"python -m spacy download {config.tokenizer_1}")
        os.system(f"python -m spacy download {config.tokenizer_2}")
        self.config2 = read_yaml(config_filepath)

    def tokenize_example(self, example, en_nlp, de_nlp, max_length, lower, sos_token, eos_token):
        en_tokens = [token.text for token in en_nlp.tokenizer(example[self.config.lang1])][:max_length]
        de_tokens = [token.text for token in de_nlp.tokenizer(example[self.config.lang2])][:max_length]
        if lower:
            en_tokens = [token.lower() for token in en_tokens]
            de_tokens = [token.lower() for token in de_tokens]
        en_tokens = [sos_token] + en_tokens + [eos_token]
        de_tokens = [sos_token] + de_tokens + [eos_token]
        return {f"{self.config.lang1}_tokens": en_tokens, f"{self.config.lang2}_tokens": de_tokens}
    
    def numericalize_example(self, example, en_vocab, de_vocab):
        en_ids = en_vocab.lookup_indices(example[f"{self.config.lang1}_tokens"])
        de_ids = de_vocab.lookup_indices(example[f"{self.config.lang2}_tokens"])
        return {f"{self.config.lang1}_ids": en_ids, f"{self.config.lang2}_ids": de_ids}
    
    def get_collate_fn(self, pad_index):
        def collate_fn(batch):
            batch_en_ids = [example[f"{self.config.lang1}_ids"] for example in batch]
            batch_de_ids = [example[f"{self.config.lang2}_ids"] for example in batch]
            batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
            batch_de_ids = nn.utils.rnn.pad_sequence(batch_de_ids, padding_value=pad_index)
            return {f"{self.config.lang1}_ids": batch_en_ids, f"{self.config.lang2}_ids": batch_de_ids}
        return collate_fn
    
    def get_data_loader(self, dataset, batch_size, pad_index, shuffle=False):
        collate_fn = self.get_collate_fn(pad_index)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
        )
        return data_loader

    def initiate_tokenization(self):
        en_nlp = spacy.load(self.config.tokenizer_1)
        de_nlp = spacy.load(self.config.tokenizer_2)
        ingestion_config = self.config2.data_ingestion

        with open(ingestion_config.data_files.train, 'r') as json_file:
            train_data = json.load(json_file)

        with open(ingestion_config.data_files.validation, 'r') as json_file:
            valid_data = json.load(json_file)

        with open(ingestion_config.data_files.test, 'r') as json_file:
            test_data = json.load(json_file)

        # Convert the loaded lists of dictionaries to datasets.Dataset objects
        train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
        valid_dataset = Dataset.from_pandas(pd.DataFrame(valid_data))
        test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))

        # Tokenize using map
        fn_kwargs = {
            "en_nlp": en_nlp,
            "de_nlp": de_nlp,
            "max_length": self.config.max_length,
            "lower": self.config.lower,
            "sos_token": self.config.sos_token,
            "eos_token": self.config.eos_token
        }

        def tokenize_wrapper(example):
            return self.tokenize_example(example, **fn_kwargs)

        train_dataset = train_dataset.map(tokenize_wrapper)
        valid_dataset = valid_dataset.map(tokenize_wrapper)
        test_dataset = test_dataset.map(tokenize_wrapper)

        logger.info(f"After tokenization, type of train_data: {type(train_dataset)}")

        # Building vocabulary
        min_freq = 2
        unk_token = "<unk>"
        pad_token = "<pad>"
        special_tokens = [unk_token, pad_token, self.config.sos_token, self.config.eos_token]

        en_vocab = build_vocab_from_iterator(
            (x[f"{self.config.lang1}_tokens"] for x in train_dataset),
            min_freq=min_freq,
            specials=special_tokens,
        )
        de_vocab = build_vocab_from_iterator(
            (x[f"{self.config.lang2}_tokens"] for x in train_dataset),
            min_freq=min_freq,
            specials=special_tokens,
        )
  
        #         torch.save(en_vocab, os.path.join(self.config.root_dir,'vocab/en_vocab.pth'))
        # torch.save(de_vocab, os.path.join(self.config.root_dir,'vocab/de_vocab.pth'))

        assert en_vocab[unk_token] == de_vocab[unk_token]
        assert en_vocab[pad_token] == de_vocab[pad_token]
        unk_index = en_vocab[unk_token]
        pad_index = en_vocab[pad_token]

        en_vocab.set_default_index(en_vocab[unk_token])
        de_vocab.set_default_index(de_vocab[unk_token])
        torch.save(en_vocab, 'en_vocab.pth')
        torch.save(de_vocab, 'de_vocab.pth')






        logger.info(f"English vocab size: {len(en_vocab)}")
        logger.info(f"German vocab size: {len(de_vocab)}")

        with open("metadata.json", 'w') as file:
            json.dump({
                'en_vocab': len(en_vocab),
                'de_vocab': len(de_vocab),
                'pad_index': pad_index,
                'unk_index': unk_index,
                
            }, file, indent=4)

        fn_kwargs = {"en_vocab": en_vocab, "de_vocab": de_vocab}

        def numericalize_wrapper(example):
            return self.numericalize_example(example, **fn_kwargs)

        train_dataset = train_dataset.map(numericalize_wrapper)
        valid_dataset = valid_dataset.map(numericalize_wrapper)
        test_dataset = test_dataset.map(numericalize_wrapper)

        data_type = "torch"
        format_columns = [f"{self.config.lang1}_ids", f"{self.config.lang2}_ids"]
        train_dataset = train_dataset.with_format(type=data_type, columns=format_columns, output_all_columns=True)
        valid_dataset = valid_dataset.with_format(type=data_type, columns=format_columns, output_all_columns=True)
        test_dataset = test_dataset.with_format(type=data_type, columns=format_columns, output_all_columns=True)

        logger.info(type(train_dataset[0][f"{self.config.lang2}_ids"]))

        root_dir = self.config.root_dir
        train_dataset.save_to_disk(os.path.join(root_dir, "train_dataset"))
        valid_dataset.save_to_disk(os.path.join(root_dir, "valid_dataset"))
        test_dataset.save_to_disk(os.path.join(root_dir, "test_dataset"))

        # Create dataloaders (batches)
        batch_size = self.config.batch_size
        train_data_loader = self.get_data_loader(train_dataset, batch_size, pad_index, shuffle=True)
        valid_data_loader = self.get_data_loader(valid_dataset, batch_size, pad_index)
        test_data_loader = self.get_data_loader(test_dataset, batch_size, pad_index)

        logger.info(f"Length of train data loader: {len(train_data_loader)}")
        logger.info(f"Length of valid data loader: {len(valid_data_loader)}")
        logger.info(f"Length of test data loader: {len(test_data_loader)}")

        # Saving dataloaders as list of batches
        root_dir=self.config.root_dir
        torch.save(list(train_data_loader), os.path.join(root_dir, "train_data_loader.pth"))  # Changed line
        torch.save(list(valid_data_loader), os.path.join(root_dir, "valid_data_loader.pth"))  # Changed line
        torch.save(list(test_data_loader), os.path.join(root_dir, "test_data_loader.pth"))    # Changed line

        logger.info(f"Data loaders saved to {root_dir}")
        logger.info("Data transformation successfully completed")

