import json
import os
import torch
import torch.nn as nn
from Mini_Translator.logging import logger
from Mini_Translator.config.configuration import ModelConfig
from Mini_Translator.components import Seq2Seq,Encoder,Decoder

class Base_Model:
    def __init__(self, config: ModelConfig):
        self.config = config

    def initiate_prepare_base_model(self):
        with open('metadata.json', 'r') as file:
            f = json.load(file)
            en_vocab = f['en_vocab']
            de_vocab = f['de_vocab']

        input_dim = de_vocab
        output_dim = en_vocab

        encoder_embedding_dim = self.config.encoder_embedding_dim
        decoder_embedding_dim = self.config.decoder_embedding_dim
        hidden_dim = self.config.hidden_dim
        n_layers = self.config.n_layers
        encoder_dropout = self.config.encoder_dropout
        decoder_dropout = self.config.decoder_dropout

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder = Encoder(input_dim, encoder_embedding_dim, hidden_dim, n_layers, encoder_dropout)
        decoder = Decoder(output_dim, decoder_embedding_dim, hidden_dim, n_layers, decoder_dropout)
        model = Seq2Seq(encoder, decoder, device).to(device)

        def init_weights(m):
            for name, param in m.named_parameters():
                nn.init.uniform_(param.data, -0.08, 0.08)
        model.apply(init_weights)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"The Model has {count_parameters(model):,} trainable parameters")

        root_dir = self.config.root_dir

        # Save the complete model
        model_path = os.path.join(root_dir, 'complete_model.pth')
        torch.save(model, model_path)
        logger.info(f"Complete model saved to {model_path}")

        # Update metadata.json with device information
        metadata_path = 'metadata.json'
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)

        metadata['device'] = str(device)

        with open(metadata_path, 'w') as file:
            json.dump(metadata, file, indent=4)
        logger.info(f"Device information saved to {metadata_path}")

