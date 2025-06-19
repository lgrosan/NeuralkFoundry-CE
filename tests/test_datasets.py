from neuralk_foundry_ce.datasets import get_data_config, LocalDataConfig
from dataclasses import dataclass
import pandas as pd
import tempfile
import os


@dataclass
class DataConfig(LocalDataConfig):
    name: str='fake_dataset'
    task: str = 'classification'
    target: str = 'target'
    file_path: str = "./my_dataset.parquet"


def test_registration_dataset():
    # Create fake parquet
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': ['a', 'b', 'c'],
        'target': [0, 1, 0]
    })
    temp_dir = tempfile.TemporaryDirectory()
    file_path = os.path.join(temp_dir.name, "my_dataset.parquet")
    df.to_parquet(file_path, index=False)

    # Check that the dataset is well imported
    dataset = get_data_config('fake_dataset')()
    dataset.file_path = file_path
    dataset.load()

    temp_dir.cleanup()
