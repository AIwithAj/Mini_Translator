import torch
import numpy as np
import json
import torch.nn as nn
from Mini_Translator.logging import logger
import spacy
import pandas as pd
from datasets import Dataset, DatasetDict
import evaluate
from Mini_Translator.constants import *
from Mini_Translator.utils.common import read_yaml, create_directories
from Mini_Translator.config.configuration import model_eval_config
from Mini_Translator.components import Seq2Seq,Encoder,Decoder
import os
import tqdm



class Evaluation:
    def __init__(self,config:model_eval_config,config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH):

        
        self.params = read_yaml(params_filepath)
        self.config=config
        self.config2=read_yaml(config_filepath)

 
    
    def evaluate_fn(self,model,data_loader,criterion,device):
        model.eval()
        epoch_loss=0
        i=4
        with torch.no_grad():
            for batch in data_loader:
                if i<4:
                    i=i+1
                    continue

                print(i)
                i=i+1
                src=batch["de_ids"].to(device)
                trg=batch["en_ids"].to(device)
                output=model(src,trg,0)
                output_dim=output.shape[-1]
                output=output[1:].view(-1,output_dim)
                trg=trg[1:].view(-1)
                loss=criterion(output,trg)
                epoch_loss+=loss.item()
        return epoch_loss / len(data_loader)
    
    def translate_sentence(self,
    sentence,
    model,
    en_nlp,
    de_nlp,
    en_vocab,
    de_vocab,
    lower,
    sos_token,
    eos_token,
    device,
    max_output_length=25,
    ):
        model.eval()
        with torch.no_grad():
            if isinstance(sentence, str):
                tokens = [token.text for token in de_nlp.tokenizer(sentence)]
            else:
                tokens = [token for token in sentence]
            if lower:
                tokens = [token.lower() for token in tokens]
            tokens = [sos_token] + tokens + [eos_token]
            ids = de_vocab.lookup_indices(tokens)
            tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
            hidden, cell = model.encoder(tensor)
            inputs = en_vocab.lookup_indices([sos_token])
            for _ in range(max_output_length):
                inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
                output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)
                predicted_token = output.argmax(-1).item()
                inputs.append(predicted_token)
                if predicted_token == en_vocab[eos_token]:
                    break
            tokens = en_vocab.lookup_tokens(inputs)
        return tokens
    

    def initiate_model_Evaluation(self):
        root_dir =self.config2.data_transformation.root_dir

        test_data_loader_path = os.path.join(root_dir, "test_data_loader.pth")
        test_data_loader = torch.load(test_data_loader_path)
        model_path = os.path.join(self.config2.base_Model.root_dir, 'complete_model.pth')
        model = torch.load(model_path)

        

        with open("metadata.json",'r') as file:
            f=json.load(file)
            pad_index=f['pad_index']
            device=f['device']
        criterion=nn.CrossEntropyLoss(ignore_index=pad_index)

        model.load_state_dict(torch.load("artifacts/trained_model/tut1-model.pt"))

        test_loss = self.evaluate_fn(model, test_data_loader, criterion, device)

        print(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")


        en_nlp = spacy.load(self.config2.data_transformation.tokenizer_1)
        de_nlp = spacy.load(self.config2.data_transformation.tokenizer_2)


        en_vocab = torch.load('en_vocab.pth')
        de_vocab = torch.load('de_vocab.pth')

        with open(self.config2.data_ingestion.data_files.test, 'r') as json_file:
            test_data = json.load(json_file)

        test_data = Dataset.from_pandas(pd.DataFrame(test_data))
        
        translations = [
            self.translate_sentence(
                example["de"],
                model,
                en_nlp,
                de_nlp,
                en_vocab,
                de_vocab,
                self.params.lower,
                self.params.sos_token,
                self.params.eos_token,
                device,
            )
            for example in tqdm.tqdm(test_data)
        ]

        bleu = evaluate.load("bleu")
        logger.info("blueu loaded ")

        predictions = [" ".join(translation[1:-1]) for translation in translations]

        references = [[example["en"]] for example in test_data]

        def get_tokenizer_fn(nlp, lower):
            def tokenizer_fn(s):
                tokens = [token.text for token in nlp.tokenizer(s)]
                if lower:
                    tokens = [token.lower() for token in tokens]
                return tokens

            return tokenizer_fn
        tokenizer_fn = get_tokenizer_fn(en_nlp, self.params.lower)

        results = bleu.compute(
            predictions=predictions, references=references, tokenizer=tokenizer_fn
        )

        json_path = os.path.join(self.config.root_dir,"results.json")

        # Save the results to a JSON file
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)

        logger.info(f"Results saved to {json_path}")