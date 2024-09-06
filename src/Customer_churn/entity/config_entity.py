from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path



@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    raw_data_path: Path
    preprocessor_obj_file_path: Path



@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    data_path: Path
    model_path: Path
    



@dataclass(frozen=True)
class ModelPredictionConfig:
    preprocessor_obj_file_path: Path
    model_path: Path