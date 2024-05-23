from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    raw_path:Path
    dataset_name:str
    raw_data:Path
    train:Path
    valid:Path
    test:Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list



@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_1: str
    tokenizer_2:str
    lang1:str
    lang2:str
    sos_token: str
    eos_token: str
    max_length : int
    lower: bool
    data_loader:Path
    batch_size: int


@dataclass(frozen=True)
class ModelConfig:
    root_dir: Path
    encoder_embedding_dim: int
    decoder_embedding_dim: int
    hidden_dim: int
    n_layers: int
    encoder_dropout: float
    decoder_dropout: float
    

    

