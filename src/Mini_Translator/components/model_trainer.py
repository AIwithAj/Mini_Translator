import torch.nn as nn
import torch
import tqdm
import numpy as np
import json
import torch.optim as optim
import os
from Mini_Translator.logging import logger
from Mini_Translator.config.configuration import model_trainer_config
from Mini_Translator.constants import *
from Mini_Translator.utils.common import read_yaml
from Mini_Translator.components import Seq2Seq,Encoder,Decoder

class modelTrainer:
    def __init__(self,config:model_trainer_config,config_filepath = CONFIG_FILE_PATH):
        self.config=config
        self.config2= read_yaml(config_filepath)

    def train_fn(self,model,data_loader,optimizer,criterion,clip,teacher_forcing_ratio, device):
        model.train()
        epoch_loss=0
        i=0
        for  batch in data_loader:
            if i <220:
                i=i+1
                continue
            print(i)
            i=i+1
            src=batch["de_ids"].to(device)
            #src=[src length , batch size]
            trg=batch["en_ids"].to(device)
            #trg=[trg length ,batch size]
            optimizer.zero_grad()
            output=model(src,trg,teacher_forcing_ratio)
            #output=[trg length , batch size, trg vocab size]
            output_dim=output.shape[-1]
            output=output[1:].view(-1,output_dim)
            #output=[(trg length -1) * batch size ,trg vocab size]
            trg=trg[1:].view(-1)
            loss=criterion(output,trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
            optimizer.step()
            epoch_loss +=loss.item()
        return epoch_loss / len(data_loader)
    
    def evaluate_fn(self,model,data_loader,criterion,device):
        model.eval()
        epoch_loss=0
        i=0
        with torch.no_grad():
            for batch in data_loader:
                if i<4:
                    i=i+1
                    continue
                src=batch["de_ids"].to(device)
                trg=batch["en_ids"].to(device)
                output=model(src,trg,0)
                output_dim=output.shape[-1]
                output=output[1:].view(-1,output_dim)
                trg=trg[1:].view(-1)
                loss=criterion(output,trg)
                epoch_loss+=loss.item()
        return epoch_loss / len(data_loader)
    
    

    def initiate_model_trainer(self):
        root_dir = self.config2.data_transformation.root_dir
        train_data_loader_path = os.path.join(root_dir, "train_data_loader.pth")
        valid_data_loader_path = os.path.join(root_dir, "valid_data_loader.pth")

        # Load the DataLoader objects
        train_data_loader = torch.load(train_data_loader_path)
        valid_data_loader = torch.load(valid_data_loader_path)


        model_path = os.path.join(self.config2.base_Model.root_dir, 'complete_model.pth')

        model = torch.load(model_path)

        optimizer=optim.Adam(model.parameters())

        with open("metadata.json",'r') as file:
            f=json.load(file)
            pad_index=f["pad_index"]
            device=f["device"]
        criterion=nn.CrossEntropyLoss(ignore_index=pad_index)

        n_epochs=1
        clip=self.config.clip
        teacher_forcing_ratio= self.config.teacher_forcing_ratio
        best_valid_loss=float("inf")
        for epoch in tqdm.tqdm(range(n_epochs)):
            train_loss=self.train_fn(model,train_data_loader,optimizer,criterion,clip,teacher_forcing_ratio,device)
            valid_loss=self.evaluate_fn(model,valid_data_loader,criterion,device)
            if valid_loss < best_valid_loss:
                best_valid_loss=valid_loss
                torch.save(model.state_dict(),os.path.join(self.config.root_dir,"tut1-model.pt"))
        # Calculate perplexities
        train_ppl = np.exp(train_loss)
        valid_ppl = np.exp(valid_loss)

        # Print the losses and perplexities
        print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {train_ppl:7.3f}")
        print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {valid_ppl:7.3f}")

        # Create a dictionary to hold the information
        results = {
            "train_loss": train_loss,
            "train_ppl": train_ppl,
            "valid_loss": valid_loss,
            "valid_ppl": valid_ppl
        }

        # Specify the path to the JSON file
        json_path = os.path.join(self.config.root_dir,"results.json")

        # Save the results to a JSON file
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)

        logger.info(f"Results saved to {json_path}")
