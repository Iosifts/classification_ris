# scripts/data_utils.py

import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from hydra.utils import to_absolute_path

def preprocess_data(file_path, text_column, label_column, label_encoder, fit=False):
    """
    Preprocess the data by reading the CSV, encoding labels, and converting to Hugging Face Dataset.
    
    Args:
        file_path (str): Path to the CSV file.
        text_column (str): Name of the column containing text data.
        label_column (str): Name of the column containing labels.
        label_encoder (LabelEncoder): An instance of LabelEncoder.
        fit (bool): Whether to fit the label encoder on the data.
    
    Returns:
        Dataset: Hugging Face Dataset object with tokenized inputs and encoded labels.
    """
    absolute_path = to_absolute_path(file_path)
    df = pd.read_csv(absolute_path)
    df[label_column] = df[label_column].astype(str).str.strip()
    if fit:
        label_encoder.fit(df[label_column])
    df['labels'] = label_encoder.transform(df[label_column])
    return Dataset.from_pandas(df)
